"""Merged Qwen3-ASR server: ONE model instance serving both
  - POST /transcribe  : file upload, short/long multipath, optional forced-aligned
                        word timestamps (timestamps=true). NON-streaming.
  - WS-less streaming  : /api/start -> /api/chunk (float32 16k PCM) -> /api/finish
                        incremental text, NO timestamps. Uses the same instance's
                        streaming_transcribe (aligner is simply ignored on this path).

Isolated + deferred. Does NOT touch production (~/ASR/Qwen3ASR, ports 8002/8003).
No webhook, local logs only. Default port 8006.

Run:  ./run_merged.sh    (preflight VRAM guard)  or
      ALIGN/GPU envs + uvicorn server_merged:app --host 0.0.0.0 --port 8006
"""
import asyncio
import json
import os
import re
import threading
import tempfile
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from qwen_asr import Qwen3ASRModel
from silero_vad import load_silero_vad, get_speech_timestamps, VADIterator

# --- config (env-overridable) ---
MODEL_NAME = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_NAME = os.environ.get("ASR_FORCED_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.40"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))  # serves /transcribe; streaming stops at EOS early
# Decode-time script mask: only Hangul/jamo + ASCII + punct/symbol/space tokens
# are sampleable (vLLM allowed_token_ids). Applied to model.sampling_params at
# startup, so ALL decode paths (transcribe/lm/streaming, anchor via backend)
# inherit it. Aligner + text-LM untouched. Rollback: RESTRICT_SCRIPTS=0.
RESTRICT_SCRIPTS = os.environ.get("RESTRICT_SCRIPTS", "1") != "0"

# /transcribe_lm fusion config
LM_MODEL = os.environ.get("LM_MODEL", "Qwen/Qwen3-0.6B")
TORN_P = float(os.environ.get("TORN_P", "0.85"))  # consult the LM when the chosen token prob < this
LM_MAX_FIXES = int(os.environ.get("LM_MAX_FIXES", "3"))
# LM_PIPE: "hybrid" (default) = v1 token fusion on non-homophone tears + homophone sweep;
#          "sweep" = plain context decode + sweep only (no fusion);
#          "v1" = the original token-fusion path (instant rollback, no file change)
LM_PIPE = os.environ.get("LM_PIPE", "hybrid")
GATE_NATS = float(os.environ.get("GATE_NATS", "1.0"))  # sweep: classmate must win by this much (total nats)
HOMOPHONES_PATH = os.environ.get(
    "HOMOPHONES_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "homophones.json"))

SAMPLE_RATE = 16000
TARGET_CHUNK_SEC = 60
BATCH_MAX_SIZE = 32
# Long-audio path releases _model_lock every N chunks so a waiting short clip's
# micro-batch can interleave instead of blocking on the whole job. N = the engine's
# inference batch width keeps each locked group at full GPU efficiency.
LONG_LOCK_GROUP = int(os.environ.get("LONG_LOCK_GROUP", "8"))
BATCH_WAIT_SEC = 0.01
TRANSCRIBE_TIMEOUT = 1800

# streaming params (init_streaming_state)
ST_UNFIXED_CHUNK_NUM = int(os.environ.get("ST_UNFIXED_CHUNK_NUM", "4"))
ST_UNFIXED_TOKEN_NUM = int(os.environ.get("ST_UNFIXED_TOKEN_NUM", "5"))
ST_CHUNK_SIZE_SEC = float(os.environ.get("ST_CHUNK_SIZE_SEC", "1.0"))


def _qparam(ws, name, default, cast):
    """Per-connection override of a streaming knob from the WS query string (falls back to env default)."""
    v = ws.query_params.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


SESSION_TTL_SEC = 10 * 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

vad_model = None
model = None
lm_model = None   # Qwen3-0.6B text LM for /transcribe_lm fusion
lm_tok = None
_model_lock = threading.Lock()  # serialize ALL model calls (transcribe + streaming) on the single engine
# Cooperative priority for short clips: a short-clip batch bumps _short_pending
# while it waits for / holds _model_lock. The long-audio path checks this between
# chunk-groups and yields the lock (threading.Lock isn't fair, so a bare
# release/re-acquire would just starve the waiter — the long thread wins the race).
_pending_lock = threading.Lock()
_short_pending = 0
# Serialize long-audio jobs (one at a time) so their preprocessing (VAD/resample/
# chunk-writing) can't thundering-herd and starve interactive short clips.
_long_job_sem = asyncio.Semaphore(1)

_req_logger = logging.getLogger("asr.merged")
_req_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(os.path.join(BASE_DIR, "logs", "requests.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_req_logger.addHandler(_fh)


def _allowed_token_ids(tok):
    """Whitelist for SamplingParams.allowed_token_ids: Hangul/jamo + ASCII +
    punctuation/symbols/whitespace. Bans other scripts AND raw byte-fragment
    tokens (which could compose CJK; every real Korean syllable has a whole
    token, only never-in-speech ones need byte pieces).
    Added/special ids (<asr_text>, im_end, eos, ...) are always allowed."""
    import unicodedata

    def _ok(ch):
        o = ord(ch)
        return (0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF
                or 0x3130 <= o <= 0x318F or ch.isascii()
                or unicodedata.category(ch)[0] in "PSZ")

    ids = [t for t in range(tok.vocab_size)
           if (s := tok.decode([t])) and "\ufffd" not in s and all(map(_ok, s))]
    return ids + list(range(tok.vocab_size, len(tok)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_queue, batch_loop_task, vad_model, model, lm_model, lm_tok
    vad_model = load_silero_vad()
    model = Qwen3ASRModel.LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        max_new_tokens=MAX_NEW_TOKENS,
        max_inference_batch_size=8,
        forced_aligner=ALIGNER_NAME,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    if RESTRICT_SCRIPTS:
        model.sampling_params.allowed_token_ids = _allowed_token_ids(model.processor.tokenizer)
        _req_logger.info(f"RESTRICT_SCRIPTS on: {len(model.sampling_params.allowed_token_ids)} allowed token ids")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lm_tok = AutoTokenizer.from_pretrained(LM_MODEL)
    lm_model = AutoModelForCausalLM.from_pretrained(LM_MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    batch_queue = asyncio.Queue()
    batch_loop_task = asyncio.create_task(batch_loop())
    yield
    batch_loop_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ======================= FILE PATH (short/long multipath) =======================

@dataclass
class BatchItem:
    audio_path: str
    context: str
    language: Optional[str]
    timestamps: bool
    logprobs: bool = False
    token_logprobs: bool = False
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


batch_queue: "asyncio.Queue[BatchItem]" = None
batch_loop_task: asyncio.Task = None


async def batch_loop():
    while True:
        items: list[BatchItem] = []
        item = await batch_queue.get()
        items.append(item)
        deadline = time.monotonic() + BATCH_WAIT_SEC
        while len(items) < BATCH_MAX_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(batch_queue.get(), timeout=remaining)
                items.append(item)
            except asyncio.TimeoutError:
                break

        groups = {}
        for i in items:
            groups.setdefault((i.timestamps, i.logprobs, i.token_logprobs), []).append(i)
        for group in groups.values():
            if not group:
                continue
            try:
                results = await asyncio.get_event_loop().run_in_executor(None, _transcribe_batch, group)
                for it, result in zip(group, results):
                    it.future.set_result(result)
            except Exception as e:
                for it in group:
                    if not it.future.done():
                        it.future.set_exception(e)


def _transcribe_batch(items: list[BatchItem]) -> list[dict]:
    audio_paths = [i.audio_path for i in items]
    contexts = [i.context for i in items]
    languages = [i.language for i in items]
    timestamps = items[0].timestamps
    # _short_pending is now managed by the /transcribe handler (bumped as soon as a short
    # request arrives, before it is even queued), so a running long job yields sooner.
    with _model_lock:
        _infer_t0 = time.perf_counter()
        results = model.transcribe(
            audio=audio_paths, context=contexts, language=languages,
            return_time_stamps=timestamps, return_token_logprobs=items[0].token_logprobs,
            return_language_confidence=items[0].logprobs,
        )
        _infer_ms = (time.perf_counter() - _infer_t0) * 1000.0
    try:
        _ntok = sum(len(getattr(r, "tokens", []) or []) for r in results) or None
    except Exception:
        _ntok = None
    _req_logger.info(
        f"INFER model_ms={_infer_ms:.1f} batch={len(items)} "
        f"per_item_ms={_infer_ms/max(1,len(items)):.1f}"
        + (f" tokens={_ntok} ms_per_tok={_infer_ms/_ntok:.2f}" if _ntok else "")
    )
    want_lp = items[0].logprobs
    want_tok = items[0].token_logprobs
    out = []
    for r in results:
        resp = {"text": r.text, "language": r.language}
        if timestamps and r.time_stamps is not None:
            resp["timestamps"] = merge_segments(merge_word_timestamps(r.text, list(r.time_stamps)))
        if want_lp:
            resp["confidence"] = r.confidence
            resp["min_token_confidence"] = r.min_token_confidence
            if r.language_confidence is not None:
                resp["language_confidence"] = r.language_confidence
                resp["language_alternatives"] = r.language_alternatives
        if want_tok:
            resp["tokens"] = r.tokens
        out.append(resp)
    return out


def load_and_resample(path: str) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def vad_chunk(wav: torch.Tensor) -> list[tuple[int, int]]:
    speech_ts = get_speech_timestamps(
        wav, vad_model, sampling_rate=SAMPLE_RATE, threshold=0.5,
        min_speech_duration_ms=250, min_silence_duration_ms=100,
        speech_pad_ms=30, return_seconds=False,
    )
    if not speech_ts:
        return [(0, len(wav))]
    target_samples = TARGET_CHUNK_SEC * SAMPLE_RATE
    chunks = []
    chunk_start = speech_ts[0]["start"]
    chunk_end = speech_ts[0]["end"]
    for seg in speech_ts[1:]:
        if seg["end"] - chunk_start > target_samples:
            chunks.append((chunk_start, chunk_end))
            chunk_start = seg["start"]
        chunk_end = seg["end"]
    chunks.append((chunk_start, chunk_end))
    return chunks


def merge_word_timestamps(text, items):
    words = text.split()
    ts_list = []
    idx = 0
    for word in words:
        clean = word.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
        start = items[idx].start_time if idx < len(items) else 0
        end = start
        merged = ""
        while idx < len(items) and len(merged) < len(clean):
            merged += items[idx].text
            end = max(end, items[idx].end_time)
            idx += 1
        if start >= end:
            end = items[idx].start_time if idx < len(items) else start + 0.08
        ts_list.append({"text": word, "start": round(start, 3), "end": round(end, 3)})
    return ts_list


MERGE_GAP = 2.0


def merge_segments(word_ts: list[dict]) -> list[dict]:
    if not word_ts:
        return []
    segments = []
    cur = {"text": word_ts[0]["text"], "start": word_ts[0]["start"], "end": word_ts[0]["end"]}
    for w in word_ts[1:]:
        if cur["text"][-1] in ".?!" or w["start"] - cur["end"] >= MERGE_GAP:
            segments.append(cur)
            cur = {"text": w["text"], "start": w["start"], "end": w["end"]}
        else:
            cur["text"] += " " + w["text"]
            cur["end"] = w["end"]
    segments.append(cur)
    return segments


# ======================= /transcribe_lm (ASR x LM fusion) =======================
# Greedy decode with the engine's per-token top-5 logprobs; at a "torn" step
# (chosen prob < TORN_P) the small LM reads the same context + transcript-so-far,
# ASR and LM probabilities are multiplied over the top-5 options, and if the
# product picks a different token the decode restarts from the corrected prefix
# (vLLM prefix caching makes the re-decode cheap). Max LM_MAX_FIXES corrections.

_ASR_TEXT_MARKER = "<asr_text>"


def _lm_next_probs(text: str) -> torch.Tensor:
    # Bound the LM prompt: a runaway ASR decode can make sofar thousands of
    # tokens, and full-sequence logits (seq x vocab) then OOM the shared GPU.
    ids = lm_tok(text, return_tensors="pt").input_ids[:, -512:].to(lm_model.device)
    with torch.no_grad():
        return torch.softmax(lm_model(ids).logits[0, -1].float(), dim=-1)


def fused_decode(wav_np: np.ndarray, context: str, language: Optional[str]):
    import math
    prompt = model._build_text_prompt(context=context, force_language=language or None)
    consults = fixes = 0
    prefix = ""  # committed corrected transcript text (after the language header)
    detected_language = language or ""
    for _round in range(LM_MAX_FIXES + 1):
        with _model_lock:
            out = model.model.generate(
                [{"prompt": prompt + prefix, "multi_modal_data": {"audio": [wav_np]}}],
                model.sampling_params, use_tqdm=False,
            )[0].outputs[0]
        text = out.text
        if _ASR_TEXT_MARKER in text:
            header, gen_text = text.split(_ASR_TEXT_MARKER, 1)
            if "language " in header:
                detected_language = header.split("language ", 1)[1].strip() or detected_language
        else:
            gen_text = text
        header_len = len(text) - len(gen_text)

        cum = 0
        corrected = False
        for tid, lps in zip(out.token_ids, out.logprobs or []):
            chosen = lps.get(tid) if lps else None
            if chosen is None:
                break
            piece = chosen.decoded_token or ""
            if cum + len(piece) <= header_len:
                cum += len(piece)
                continue
            p = math.exp(chosen.logprob)
            if p < TORN_P and len(lps) > 1 and fixes < LM_MAX_FIXES:
                consults += 1
                sofar = prefix + gen_text[: max(0, cum - header_len)]
                lm_probs = _lm_next_probs((context + " " + sofar.strip()).strip())
                best_piece, best_score = piece, -1.0
                for _atid, alp in lps.items():
                    apiece = alp.decoded_token or ""
                    lm_ids = lm_tok(apiece, add_special_tokens=False).input_ids
                    # First-token approximation for multi-token pieces: a hard 0.0
                    # made them unelectable and biased repair toward common
                    # single-token words.
                    p_lm = float(lm_probs[lm_ids[0]]) if lm_ids else 0.0
                    score = math.exp(alp.logprob) * p_lm
                    if score > best_score:
                        best_score, best_piece = score, apiece
                if best_piece != piece:
                    # homophone-pair flips belong to the gated sweep, not fusion:
                    # inside a sound class the ASR x LM product double-counts text
                    # priors, so fusion only handles different-sounding tears here
                    if _are_homophones(piece, best_piece):
                        cum += len(piece)
                        continue
                    fixes += 1
                    prefix = sofar + best_piece
                    corrected = True
                    break
            cum += len(piece)
        if not corrected:
            return (prefix + gen_text).strip(), detected_language, consults, fixes
    return (prefix + gen_text).strip(), detected_language, consults, fixes


# ---- homophone sweep (LM_PIPE="sweep"): plain context decode, then every word
# sitting in a sound-identical class gets its classmates proposed and ONE batched
# LM pass scores all variants. Within an exact class the audio provably cannot
# distinguish the spellings, so the ASR gets no vote — but the incumbent spelling
# wins unless a classmate is decisively better (total-sentence gain > GATE_NATS).

_HOMO_SETS = None


def _homo_key(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower()).replace("'", "")


def _homo_sets():
    global _HOMO_SETS
    if _HOMO_SETS is None:
        try:
            groups = json.load(open(HOMOPHONES_PATH))
        except Exception:
            groups = {}
        _HOMO_SETS = {}
        for _k, words in groups.items():
            ws = {_homo_key(w) for w in words}
            for w in ws:
                _HOMO_SETS.setdefault(w, set()).update(ws)
    return _HOMO_SETS


def _are_homophones(a: str, b: str) -> bool:
    ka, kb = _homo_key(a.strip(" .,!?")), _homo_key(b.strip(" .,!?"))
    if not ka or not kb or ka == kb:
        return False
    return kb in _homo_sets().get(ka, ())


def _lm_sum_lp_batch(ctx: str, conts: list) -> list:
    """Total logprob of each continuation of ctx — one padded LM pass.
    Scores from where tok(ctx+cont) diverges from tok(ctx), not from len(tok(ctx)):
    ctx ends with a space that the tokenizer merges into the continuation's first
    word, which otherwise lands before the counting boundary and is scored for
    free — sentence-initial homophone flips were judged on everything except the
    word being compared (I->Ai, row 84988)."""
    if not conts:
        return []
    ctx_ids = lm_tok(ctx).input_ids
    lm_tok.pad_token = lm_tok.pad_token or lm_tok.eos_token
    enc = lm_tok([ctx + c for c in conts], return_tensors="pt", padding=True).to(lm_model.device)
    with torch.no_grad():
        logits = lm_model(**enc).logits.float()
    outs = []
    for b in range(len(conts)):
        L = int(enc["attention_mask"][b].sum())
        ids = enc["input_ids"][b]
        n = 0
        while n < len(ctx_ids) and n < L and int(ids[n]) == ctx_ids[n]:
            n += 1
        outs.append(sum(float(torch.log_softmax(logits[b, p - 1], dim=-1)[ids[p]])
                        for p in range(n, L)))
    return outs


def homophone_sweep(text: str, context: str, max_flips: int = 3):
    """Returns (text, lm_passes, flips)."""
    t = text.strip()
    base = context.strip() + " "
    flips = passes = 0
    for _ in range(max_flips):
        words = t.split()
        variants = []
        for i, w in enumerate(words):
            m = re.match(r"^(.*?)([.,!?]*)$", w)
            core = m.group(1)
            for alt in _homo_sets().get(_homo_key(core), ()):
                if not alt or alt == _homo_key(core):
                    continue
                rep = alt.capitalize() if core[:1].isupper() else alt
                variants.append(" ".join(words[:i] + [rep + m.group(2)] + words[i + 1:]))
        if not variants:
            return t, passes, flips
        sums = _lm_sum_lp_batch(base, [t] + variants)
        passes += 1
        best_i = max(range(len(variants)), key=lambda i: sums[i + 1])
        if sums[best_i + 1] - sums[0] > GATE_NATS:
            t = variants[best_i]
            flips += 1
        else:
            return t, passes, flips
    return t, passes, flips


def plain_decode(wav_np: np.ndarray, context: str, language: Optional[str]):
    """One engine decode with context; returns (gen_text, detected_language)."""
    prompt = model._build_text_prompt(context=context, force_language=language or None)
    with _model_lock:
        out = model.model.generate(
            [{"prompt": prompt, "multi_modal_data": {"audio": [wav_np]}}],
            model.sampling_params, use_tqdm=False,
        )[0].outputs[0]
    text = out.text
    detected_language = language or ""
    if _ASR_TEXT_MARKER in text:
        header, gen_text = text.split(_ASR_TEXT_MARKER, 1)
        if "language " in header:
            detected_language = header.split("language ", 1)[1].strip() or detected_language
    else:
        gen_text = text
    return gen_text.strip(), detected_language


def v1_decode(wav_np: np.ndarray, context: str, language: Optional[str]):
    """LM_PIPE="v1": token fusion only. Returns (text, lang, consults, fusion_fixes, sweep_flips)."""
    text, det_lang, consults, fixes = fused_decode(wav_np, context, language)
    return text, det_lang, consults, fixes, 0


def sweep_decode(wav_np: np.ndarray, context: str, language: Optional[str]):
    """LM_PIPE="sweep": plain context decode + homophone sweep."""
    gen_text, det_lang = plain_decode(wav_np, context, language)
    text, passes, flips = homophone_sweep(gen_text, context)
    return text, det_lang, passes, 0, flips


def hybrid_decode(wav_np: np.ndarray, context: str, language: Optional[str]):
    """LM_PIPE="hybrid" (default): v1 token fusion restricted to non-homophone
    tears (recovers dropped words like bbq's "party"), then the gated homophone
    sweep for same-sound respells. Each tier handles only its own case."""
    text, det_lang, consults, fixes = fused_decode(wav_np, context, language)
    text, passes, flips = homophone_sweep(text, context)
    return text, det_lang, consults + passes, fixes, flips


@app.post("/transcribe_lm")
async def transcribe_lm(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(""),
    language: str | None = Form("English"),
):
    _start = time.perf_counter()
    _client = request.client.host if request.client else "-"

    if not context or not context.strip():
        return JSONResponse(status_code=400, content={"detail": "context is required"})
    if language is not None:
        supported = model.get_supported_languages()
        if language not in supported:
            return JSONResponse(status_code=400, content={
                "detail": f"Unsupported language: {language}", "supported": supported})

    audio_bytes = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        try:
            wav = load_and_resample(path)
        except Exception:
            return JSONResponse(status_code=400, content={
                "detail": f"Unsupported or invalid audio file: {file.filename}"})
        duration_sec = len(wav) / SAMPLE_RATE
        if duration_sec > TARGET_CHUNK_SEC * 1.5:
            return JSONResponse(status_code=400, content={
                "detail": "transcribe_lm supports clips up to 90s; use /transcribe"})
        _req_logger.info(f"{_client} LM file={file.filename!r} lang={language} dur={duration_sec:.1f}s START")

        global _short_pending
        with _pending_lock:
            _short_pending += 1
        try:
            wav_np = wav.numpy()
            loop = asyncio.get_event_loop()
            _decode = {"sweep": sweep_decode, "v1": v1_decode}.get(LM_PIPE, hybrid_decode)
            text, det_lang, consults, ffix, sflip = await asyncio.wait_for(
                loop.run_in_executor(None, _decode, wav_np, context, language),
                timeout=TRANSCRIBE_TIMEOUT,
            )
        finally:
            with _pending_lock:
                _short_pending -= 1
        # Echo gate (see /transcribe): impossible words/sec => context echo.
        _echo = False
        _wc = len(text.split())
        if _wc >= 6 and _wc / max(duration_sec, 0.1) > 5.0:
            text, det_lang = await loop.run_in_executor(None, plain_decode, wav_np, "", language)
            _echo = True
            _req_logger.info(f"{_client} LM echo-gate fired ({_wc} words / {duration_sec:.1f}s); bare re-decode")
        _req_logger.info(
            f"{_client} LM status=200 pipe={LM_PIPE} consults={consults} fusion={ffix} sweep={sflip} time={time.perf_counter()-_start:.2f}s")
        return {"text": text, "language": det_lang, "lm_consults": consults, "lm_fixes": ffix + sflip,
                "fusion_fixes": ffix, "sweep_flips": sflip, "pipe": LM_PIPE, "echo_gate": _echo}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(""),
    language: str | None = Form(None),
    timestamps: bool = Form(False),
    logprobs: bool = Form(False),
    token_logprobs: bool = Form(False),
):
    _start = time.perf_counter()
    _client = request.client.host if request.client else "-"

    if language is not None:
        supported = model.get_supported_languages()
        if language not in supported:
            return JSONResponse(status_code=400, content={
                "detail": f"Unsupported language: {language}", "supported": supported})

    audio_bytes = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        try:
            wav = load_and_resample(path)
        except Exception:
            os.unlink(path)
            return JSONResponse(status_code=400, content={
                "detail": f"Unsupported or invalid audio file: {file.filename}"})
        duration_sec = len(wav) / SAMPLE_RATE
        _req_logger.info(f"{_client} file={file.filename!r} lang={language} ts={timestamps} dur={duration_sec:.1f}s START")

        # Short audio: micro-batch queue. Bump _short_pending immediately (before
        # queueing) so a running long job yields to us as early as possible.
        if duration_sec <= TARGET_CHUNK_SEC * 1.5:
            global _short_pending
            with _pending_lock:
                _short_pending += 1
            try:
                item = BatchItem(audio_path=path, context=context,
                                 language=language or None, timestamps=timestamps,
                                 logprobs=logprobs, token_logprobs=token_logprobs)
                await batch_queue.put(item)
                result = await asyncio.wait_for(item.future, timeout=TRANSCRIBE_TIMEOUT)
            finally:
                with _pending_lock:
                    _short_pending -= 1
            # Echo gate: a context-conditioned transcript claiming an impossible
            # speech rate (>=6 words at >5 words/sec of audio) is context echo;
            # re-decode without context and return that instead.
            if context and isinstance(result, dict):
                _wc = len((result.get("text") or "").split())
                if _wc >= 6 and _wc / max(duration_sec, 0.1) > 5.0:
                    _g_text, _g_lang = await asyncio.get_event_loop().run_in_executor(
                        None, plain_decode, wav.numpy(), "", language or None)
                    _req_logger.info(f"{_client} echo-gate fired ({_wc} words / {duration_sec:.1f}s); bare re-decode")
                    result = {"text": _g_text, "language": _g_lang, "echo_gate": True}
            _req_logger.info(f"{_client} file={file.filename!r} status=200 time={time.perf_counter()-_start:.2f}s")
            return result

        # Long audio: VAD chunk -> batch -> merge (with timestamp offsetting).
        # Serialize long jobs (one at a time) so their preprocessing can't starve short clips.
        await _long_job_sem.acquire()
        chunk_paths = []
        try:
            chunks = vad_chunk(wav)
            for cs, ce in chunks:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ctmp:
                    sf.write(ctmp.name, wav[cs:ce].numpy(), SAMPLE_RATE)
                    chunk_paths.append(ctmp.name)
            lang = language or None

            def _locked_transcribe():
                # Transcribe in groups of LONG_LOCK_GROUP chunks, re-acquiring the
                # lock per group. Between groups, if any short clips are waiting, drain
                # them first (the lock is free while we spin here) — this bounds a short
                # clip's wait to one group instead of the whole long job. Order is
                # preserved (chunks in order, results extended in order).
                out = []
                for g in range(0, len(chunk_paths), LONG_LOCK_GROUP):
                    # Defer to any waiting short clip BEFORE (re)acquiring the lock —
                    # including before the first group — so a queued long job can't jump
                    # ahead of it. With LONG_LOCK_GROUP=1 this bounds a short clip's wait
                    # to a single in-flight chunk.
                    while True:
                        with _pending_lock:
                            if _short_pending == 0:
                                break
                        time.sleep(0.003)
                    grp = chunk_paths[g:g + LONG_LOCK_GROUP]
                    with _model_lock:
                        out.extend(model.transcribe(audio=grp, context=context,
                                                    language=lang, return_time_stamps=timestamps,
                                                    return_token_logprobs=token_logprobs,
                                                    return_language_confidence=logprobs))
                return out

            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _locked_transcribe),
                timeout=TRANSCRIBE_TIMEOUT)

            all_text, all_timestamps, detected_lang = [], [], None
            all_conf = []
            all_min_conf = []
            all_tokens = []
            lang_conf = None
            lang_alts = None
            for i, r in enumerate(results):
                if not detected_lang:
                    detected_lang = r.language
                all_text.append(r.text)
                if r.confidence is not None:
                    all_conf.append(r.confidence)
                if r.min_token_confidence is not None:
                    all_min_conf.append(r.min_token_confidence)
                if r.tokens:
                    all_tokens.extend(r.tokens)
                if lang_conf is None and r.language_confidence is not None:
                    lang_conf = r.language_confidence
                    lang_alts = r.language_alternatives
                if timestamps and r.time_stamps is not None:
                    offset = chunks[i][0] / SAMPLE_RATE
                    for wt in merge_word_timestamps(r.text, list(r.time_stamps)):
                        wt["start"] = round(wt["start"] + offset, 3)
                        wt["end"] = round(wt["end"] + offset, 3)
                        all_timestamps.append(wt)
        finally:
            for p in chunk_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            _long_job_sem.release()

        resp = {"text": " ".join(all_text), "language": detected_lang, "chunks": len(chunks)}
        if timestamps:
            resp["timestamps"] = merge_segments(all_timestamps)
        if logprobs:
            resp["confidence"] = (sum(all_conf) / len(all_conf)) if all_conf else None
            resp["min_token_confidence"] = min(all_min_conf) if all_min_conf else None
            if lang_conf is not None:
                resp["language_confidence"] = lang_conf
                resp["language_alternatives"] = lang_alts
        if token_logprobs:
            resp["tokens"] = all_tokens
        _req_logger.info(f"{_client} file={file.filename!r} status=200 chunks={len(chunks)} time={time.perf_counter()-_start:.2f}s")
        return resp
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ======================= STREAMING PATH (text-only, same instance) =======================

@dataclass
class Session:
    state: object
    last_seen: float


SESSIONS: Dict[str, Session] = {}


def _gc_sessions():
    now = time.time()
    for sid in [s for s, v in SESSIONS.items() if now - v.last_seen > SESSION_TTL_SEC]:
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


def _state_payload(state) -> dict:
    return {"language": getattr(state, "language", "") or "",
            "text": getattr(state, "text", "") or ""}


@app.post("/api/start")
async def api_start(language: str | None = None):
    if language is not None and language not in model.get_supported_languages():
        return JSONResponse(status_code=400,
                            content={"error": f"Unsupported language: {language}"})
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: model.init_streaming_state(
            language=language or None,
            unfixed_chunk_num=ST_UNFIXED_CHUNK_NUM,
            unfixed_token_num=ST_UNFIXED_TOKEN_NUM,
            chunk_size_sec=ST_CHUNK_SIZE_SEC))
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(state=state, last_seen=time.time())
    return {"session_id": sid}


@app.post("/api/chunk")
async def api_chunk(request: Request, session_id: str):
    s = _get_session(session_id)
    if not s:
        return JSONResponse(status_code=400, content={"error": "invalid session_id"})
    raw = await request.body()
    if len(raw) % 4 != 0:
        return JSONResponse(status_code=400, content={"error": "float32 bytes not multiple of 4"})
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)  # expects 16k float32 mono

    def _step():
        with _model_lock:
            model.streaming_transcribe(wav, s.state)
        return _state_payload(s.state)

    return await asyncio.get_event_loop().run_in_executor(None, _step)


@app.post("/api/finish")
async def api_finish(session_id: str):
    s = _get_session(session_id)
    if not s:
        return JSONResponse(status_code=400, content={"error": "invalid session_id"})

    def _finish():
        with _model_lock:
            model.finish_streaming_transcribe(s.state)
        return _state_payload(s.state)

    out = await asyncio.get_event_loop().run_in_executor(None, _finish)
    SESSIONS.pop(session_id, None)
    return out


@app.websocket("/stream")
async def ws_stream(ws: WebSocket):
    """WebSocket streaming: client sends binary float32 16k mono PCM chunks, then
    a text frame {"type":"eos"}. Server replies {"type":"partial"|"final","text","language"}.

    Optional VAD endpointing: ?endpoint=vad[&silence_ms=700] -> when end-of-speech is
    detected the current utterance is finalized ({"type":"final"}) and a fresh streaming
    state is started; the socket stays open for the next utterance (continuous mode).
    Without the param, behaviour is unchanged (finalize only on client {"type":"eos"}).
    Same instance / same _model_lock as /transcribe."""
    await ws.accept()
    loop = asyncio.get_event_loop()
    req_lang = ws.query_params.get("language") or None
    if req_lang is not None and req_lang not in model.get_supported_languages():
        await ws.send_json({"type": "error",
                            "detail": f"Unsupported language: {req_lang}"})
        await ws.close()
        return

    use_vad = (ws.query_params.get("endpoint") or "").lower() == "vad"
    try:
        silence_ms = int(ws.query_params.get("silence_ms") or 700)
    except (TypeError, ValueError):
        silence_ms = 700
    try:
        max_utterance_ms = int(ws.query_params.get("max_utterance_ms") or 20000)
    except (TypeError, ValueError):
        max_utterance_ms = 20000
    try:
        vad_threshold = float(ws.query_params.get("vad_threshold") or 0.5)
    except (TypeError, ValueError):
        vad_threshold = 0.5
    if not (0.0 < vad_threshold < 1.0):
        vad_threshold = 0.5

    def _new_state():
        return model.init_streaming_state(
            language=req_lang,
            unfixed_chunk_num=_qparam(ws, "unfixed_chunk_num", ST_UNFIXED_CHUNK_NUM, int),
            unfixed_token_num=_qparam(ws, "unfixed_token_num", ST_UNFIXED_TOKEN_NUM, int),
            chunk_size_sec=_qparam(ws, "chunk_size_sec", ST_CHUNK_SIZE_SEC, float))

    state = await loop.run_in_executor(None, _new_state)

    vad_iter = None
    vad_buf = np.zeros((0,), dtype=np.float32)
    speech_seen = False
    if use_vad:
        vad_iter = VADIterator(vad_model, sampling_rate=SAMPLE_RATE,
                               threshold=vad_threshold,
                               min_silence_duration_ms=silence_ms)
    VAD_WIN = 512  # silero requires 512-sample windows at 16k
    max_utt_samples = int(max_utterance_ms / 1000 * SAMPLE_RATE) if max_utterance_ms > 0 else 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                raw = msg["bytes"]
                if len(raw) % 4 != 0:
                    await ws.send_json({"type": "error", "detail": "float32 bytes not multiple of 4"})
                    continue
                wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

                def _step():
                    with _model_lock:
                        model.streaming_transcribe(wav, state)
                    return _state_payload(state)

                await ws.send_json({"type": "partial", **(await loop.run_in_executor(None, _step))})

                if vad_iter is not None:
                    vad_buf = np.concatenate([vad_buf, wav])
                    endpointed = False
                    while vad_buf.shape[0] >= VAD_WIN:
                        win = vad_buf[:VAD_WIN]
                        vad_buf = vad_buf[VAD_WIN:]
                        ev = vad_iter(torch.from_numpy(win), return_seconds=True)
                        if ev:
                            if "start" in ev:
                                speech_seen = True
                            elif "end" in ev and speech_seen:
                                endpointed = True
                                break
                    if (not endpointed and max_utt_samples
                            and state.audio_accum.shape[0] >= max_utt_samples):
                        endpointed = True  # safety cap: bound re-decode cost on no-pause speech
                    if endpointed:
                        def _finish_cur():
                            with _model_lock:
                                model.finish_streaming_transcribe(state)
                            return _state_payload(state)
                        payload = await loop.run_in_executor(None, _finish_cur)
                        if (payload.get("text") or "").strip():
                            await ws.send_json({"type": "final", **payload})
                        # start a fresh utterance; keep the socket open
                        state = await loop.run_in_executor(None, _new_state)
                        vad_iter.reset_states()
                        vad_buf = np.zeros((0,), dtype=np.float32)
                        speech_seen = False
            elif msg.get("text") is not None:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") == "eos":
                    def _finish():
                        with _model_lock:
                            model.finish_streaming_transcribe(state)
                        return _state_payload(state)

                    await ws.send_json({"type": "final", **(await loop.run_in_executor(None, _finish))})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/stream_full")
async def ws_stream_full(ws: WebSocket):
    """Streams incremental partials (like /stream) AND, on {"type":"end"} (or "eos"),
    returns BOTH the finalized streaming transcript and a FRESH full ASR pass over the
    whole audio, so you can compare incremental vs single-pass on the same clip:
        partials: {"type":"partial","text",...}
        final   : {"type":"final","stream_text","full_text","language",
                   "full_pass_ms","buffered_sec"}
    Optional ?language=.. . No timestamps."""
    await ws.accept()
    loop = asyncio.get_event_loop()
    req_lang = ws.query_params.get("language") or None
    if req_lang is not None and req_lang not in model.get_supported_languages():
        await ws.send_json({"type": "error", "detail": f"Unsupported language: {req_lang}"})
        await ws.close()
        return

    def _new_state():
        return model.init_streaming_state(
            language=req_lang,
            unfixed_chunk_num=_qparam(ws, "unfixed_chunk_num", ST_UNFIXED_CHUNK_NUM, int),
            unfixed_token_num=_qparam(ws, "unfixed_token_num", ST_UNFIXED_TOKEN_NUM, int),
            chunk_size_sec=_qparam(ws, "chunk_size_sec", ST_CHUNK_SIZE_SEC, float))

    state = await loop.run_in_executor(None, _new_state)
    raw = []  # keep every chunk for the full-audio pass
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) % 4 != 0:
                    await ws.send_json({"type": "error", "detail": "float32 bytes not multiple of 4"})
                    continue
                wav = np.frombuffer(b, dtype=np.float32).reshape(-1)
                raw.append(wav)

                def _step():
                    with _model_lock:
                        model.streaming_transcribe(wav, state)
                    return _state_payload(state)

                await ws.send_json({"type": "partial", **(await loop.run_in_executor(None, _step))})
            elif msg.get("text") is not None:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") in ("end", "eos"):
                    # 1) finalize the streaming (incremental) transcript
                    def _finish():
                        with _model_lock:
                            model.finish_streaming_transcribe(state)
                        return _state_payload(state)

                    stream_payload = await loop.run_in_executor(None, _finish)

                    # 2) fresh full ASR pass over ALL accumulated audio (via batch queue)
                    full_text = ""
                    full_ms = None
                    dur = 0.0
                    if raw:
                        allwav = np.concatenate(raw)
                        dur = len(allwav) / SAMPLE_RATE
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            sf.write(tmp.name, allwav, SAMPLE_RATE)
                            path = tmp.name
                        t0 = time.perf_counter()
                        global _short_pending
                        with _pending_lock:
                            _short_pending += 1
                        try:
                            item = BatchItem(audio_path=path, context="", language=req_lang, timestamps=False)
                            await batch_queue.put(item)
                            full = await asyncio.wait_for(item.future, timeout=TRANSCRIBE_TIMEOUT)
                        finally:
                            with _pending_lock:
                                _short_pending -= 1
                            try:
                                os.unlink(path)
                            except Exception:
                                pass
                        full_text = full.get("text", "")
                        full_ms = round((time.perf_counter() - t0) * 1000)

                    await ws.send_json({
                        "type": "final",
                        "stream_text": stream_payload.get("text", ""),
                        "full_text": full_text,
                        "language": stream_payload.get("language"),
                        "full_pass_ms": full_ms,
                        "buffered_sec": round(dur, 2),
                    })
                    # keep the socket open for the next utterance: reset stream + buffer
                    state = await loop.run_in_executor(None, _new_state)
                    raw = []
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/stream_smart")
async def ws_stream_smart(ws: WebSocket):
    """Early-lock smart streaming. Streams force-EN partials; after ~arb_after_sec of
    SPEECH (energy-gated) it decides the utterance language ONCE and locks it:
      - force-EN kept Hangul -> Korean (passthrough)
      - all-Latin -> batched EN+KO decode + PER guard(:8010) argmin; if Korean wins, the
        stream switches to force-KO and replays so the rest streams in Korean.
    Emits {"type":"lock",...}; partials carry "locked". Short utterances are arbitrated
    on end. Continuous (stays open). ?arb_after_sec=1.5 tunes the decision point.
    ?skip_final=1 skips the final finish_streaming_transcribe decode on end (returns the
    last streaming partial as the final) -> faster end, may drop the trailing <chunk audio.
    NOTE: can false-positive on Korean-accented English."""
    import re as _re
    import httpx as _httpx
    HANGUL = _re.compile(r"[가-힣]")
    try:
        arb_after = float(ws.query_params.get("arb_after_sec") or 1.5)
    except (TypeError, ValueError):
        arb_after = 1.5
    skip_final = (ws.query_params.get("skip_final") or "").lower() in ("1", "true", "yes")
    SPEECH_THR = 0.01

    await ws.accept()
    loop = asyncio.get_event_loop()

    def _new_state(lang):
        return model.init_streaming_state(
            language=lang, unfixed_chunk_num=_qparam(ws, "unfixed_chunk_num", ST_UNFIXED_CHUNK_NUM, int),
            unfixed_token_num=_qparam(ws, "unfixed_token_num", ST_UNFIXED_TOKEN_NUM, int), chunk_size_sec=_qparam(ws, "chunk_size_sec", ST_CHUNK_SIZE_SEC, float))

    state = await loop.run_in_executor(None, _new_state, "English")
    raw = []
    speech_samps = 0
    locked = None
    last_payload = {}
    last_arb = {}

    async def _dec(path, lang):
        global _short_pending
        with _pending_lock:
            _short_pending += 1
        try:
            it = BatchItem(audio_path=path, context="", language=lang, timestamps=False)
            await batch_queue.put(it)
            r = await asyncio.wait_for(it.future, timeout=TRANSCRIBE_TIMEOUT)
        finally:
            with _pending_lock:
                _short_pending -= 1
        return r.get("text", "")

    async def _arbitrate():
        allwav = np.concatenate(raw)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, allwav, SAMPLE_RATE)
            path = tmp.name
        try:
            en_text, ko_text = await asyncio.gather(_dec(path, "English"), _dec(path, "Korean"))
            per_en = per_ko = None
            try:
                async with _httpx.AsyncClient(timeout=30) as c:
                    with open(path, "rb") as fh:
                        gr = await c.post("http://127.0.0.1:8010/guard",
                                          files={"file": ("c.wav", fh, "audio/wav")},
                                          data={"en_text": en_text, "ko_text": ko_text})
                    if gr.status_code == 200:
                        gj = gr.json()
                        per_en = gj.get("per_en")
                        per_ko = gj.get("per_ko")
            except Exception as e:
                _req_logger.warning(f"stream_smart guard failed: {e}")
            chose = "korean" if (per_ko is not None and per_en is not None and per_ko <= per_en) else "english"
            return chose, en_text, ko_text, per_en, per_ko
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    async def _switch_to_korean():
        nonlocal state, last_payload
        ko_state = await loop.run_in_executor(None, _new_state, "Korean")
        catchup = np.concatenate(raw)

        def _replay():
            with _model_lock:
                model.streaming_transcribe(catchup, ko_state)
            return _state_payload(ko_state)

        kp = await loop.run_in_executor(None, _replay)
        state = ko_state
        last_payload = kp
        return kp

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) % 4 != 0:
                    await ws.send_json({"type": "error", "detail": "float32 bytes not multiple of 4"})
                    continue
                wav = np.frombuffer(b, dtype=np.float32).reshape(-1)
                raw.append(wav)
                speech_samps += int(np.count_nonzero(np.abs(wav) > SPEECH_THR))

                def _step():
                    with _model_lock:
                        model.streaming_transcribe(wav, state)
                    return _state_payload(state)

                payload = await loop.run_in_executor(None, _step)
                last_payload = payload
                await ws.send_json({"type": "partial", "locked": locked, **payload})

                if locked is None and speech_samps >= arb_after * SAMPLE_RATE:
                    if HANGUL.search(payload.get("text", "")):
                        locked = "Korean"
                        await ws.send_json({"type": "lock", "language": "Korean",
                                            "path": "hangul_passthrough", "per_en": None, "per_ko": None})
                    else:
                        chose, en_text, ko_text, per_en, per_ko = await _arbitrate()
                        last_arb = {"en_text": en_text, "ko_text": ko_text,
                                    "per_en": per_en, "per_ko": per_ko, "chose": chose}
                        if chose == "korean":
                            locked = "Korean"
                            kp = await _switch_to_korean()
                            await ws.send_json({"type": "lock", "language": "Korean", "path": "argmin",
                                                **last_arb})
                            await ws.send_json({"type": "partial", "locked": "Korean", **kp})
                        else:
                            locked = "English"
                            await ws.send_json({"type": "lock", "language": "English", "path": "argmin",
                                                **last_arb})
            elif msg.get("text") is not None:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") in ("end", "eos"):
                    if locked is None and raw:  # short utterance ended before trigger
                        if HANGUL.search(last_payload.get("text", "")):
                            locked = "Korean"
                        else:
                            chose, en_text, ko_text, per_en, per_ko = await _arbitrate()
                            last_arb = {"en_text": en_text, "ko_text": ko_text,
                                        "per_en": per_en, "per_ko": per_ko, "chose": chose}
                            if chose == "korean":
                                locked = "Korean"
                                await _switch_to_korean()
                            else:
                                locked = "English"

                    if skip_final:
                        fp = last_payload
                    else:
                        def _finish():
                            with _model_lock:
                                model.finish_streaming_transcribe(state)
                            return _state_payload(state)

                        fp = await loop.run_in_executor(None, _finish)

                    await ws.send_json({"type": "final", "text": fp.get("text", ""),
                                        "language": locked or fp.get("language"),
                                        "locked": locked, "skip_final": skip_final, **last_arb})
                    state = await loop.run_in_executor(None, _new_state, "English")
                    raw = []
                    speech_samps = 0
                    locked = None
                    last_payload = {}
                    last_arb = {}
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "aligner": ALIGNER_NAME,
            "languages": model.get_supported_languages() if model else None}

"""Anchor (ARCH-004) — evidence-anchored transcription as a standalone service (port 8004).

Composes the merged backend (:8006) and ZIPA guard (:8010); touches neither.
Spec + validation: ASR-Hub ARCH-004 entry and the 91-clip scoreboard
(87/91; drill 46/50, silence 41/41). Benchmarked p50 ≈ 103 ms shape
(one decode + one guard call on the happy path).

Stages: force-EN decode (context + labeled expected append, only when the
context lacks the target) -> base-ZIPA phone evidence -> thin-evidence rescue
lane (conf >= TAU or corroborating PFER) -> Hangul per_ko verify -> lazy EN
PFER check -> force-KO argmin rescue -> anchored homophone sweep
(Qwen3-0.6B, CPU, lazy-loaded) -> final filters (filler-only, repeat-collapse).
"""
import asyncio
import json
import logging
import os
import re
import time
import uuid

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

# The wrapper's Supabase/ES/CDN logger is imported lazily inside the handler:
# wrapper_merged also imports this module to mount the endpoint on :8002, and a
# top-level import here would make that circular in one direction.

BACKEND = os.getenv("ANCHOR_BACKEND_URL", "http://127.0.0.1:8006")
GUARD = os.getenv("ANCHOR_GUARD_URL", "http://127.0.0.1:8010")
HOMOPHONES_PATH = os.getenv(
    "HOMOPHONES_PATH", "/home/ailab/ASR/Qwen3ASR-streaming/homophones_curated.json")
LM_MODEL = os.getenv("ANCHOR_LM_MODEL", "Qwen/Qwen3-0.6B")
TAU_CONF = float(os.getenv("ANCHOR_TAU_CONF", "0.9"))
PFER_ACCEPT = float(os.getenv("ANCHOR_PFER_ACCEPT", "0.55"))
PER_KO_ECHO = float(os.getenv("ANCHOR_PER_KO_ECHO", "0.75"))
GATE_NATS = float(os.getenv("ANCHOR_GATE_NATS", "1.0"))
THIN_PHONES = int(os.getenv("ANCHOR_THIN_PHONES", "2"))
FOLDED_BAND = float(os.getenv("ANCHOR_FOLDED_BAND", "0.70"))
FOLDED_MARGIN = float(os.getenv("ANCHOR_FOLDED_MARGIN", "0.0"))
FOLDED_MIN_WORDS = int(os.getenv("ANCHOR_FOLDED_MIN_WORDS", "3"))
FOLDED_SHORT_SUPPORT = float(os.getenv("ANCHOR_FOLDED_SHORT_SUPPORT", "0.75"))
FOLDED_ONLY = os.getenv("ANCHOR_FOLDED_ONLY", "1") == "1"  # drop the weighted trigger
FOLDED_UNIVERSAL = os.getenv("ANCHOR_UNIVERSAL_FLOOR", "0") == "1"  # apply KO support floor to ALL word counts
XEUS_FALLBACK = os.getenv("ANCHOR_XEUS_FALLBACK", "1") == "1"  # re-score KO with Xeus in the ambiguous ZIPA band
XEUS_SUPPORT = float(os.getenv("ANCHOR_XEUS_SUPPORT", "0.70"))
XEUS_ARBITRATE = os.getenv("ANCHOR_XEUS_ARBITRATE", "1") == "1"  # Xeus scores BOTH EN+KO, argmin + reject if both high
XEUS_REJECT = float(os.getenv("ANCHOR_XEUS_REJECT", "0.75"))
XEUS_URL = os.getenv("ANCHOR_XEUS_URL", "http://192.168.1.239:8001/xeus")
EN_ECHO_MIN_WORDS = int(os.getenv("ANCHOR_EN_ECHO_MIN_WORDS", "3"))  # 2026-09-11: was 5; 3 fixes quoted-context echoes on short answers (row 163700)
EN_ECHO_SIM = float(os.getenv("ANCHOR_EN_ECHO_SIM", "0.35"))

FILLER = {"hm", "hmm", "hmmm", "mm", "mmm", "um", "umm", "uh", "uhh"}

HANGUL = re.compile(r"[가-힯]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("anchor")
app = FastAPI()
_client = httpx.AsyncClient(timeout=1800)

# ---------------- homophone sweep (LM lazy-loaded, CPU) ----------------
_lm = _tok = None
_lm_lock = asyncio.Lock()
SETS: dict[str, list[str]] = {}
for _k, _words in json.load(open(HOMOPHONES_PATH)).items():
    for _w in _words:
        SETS.setdefault(re.sub(r"[^a-z0-9']", "", _w.lower()).replace("'", ""), _words)


def _key(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower()).replace("'", "")


_lm_device = "cpu"


def _load_lm():
    global _lm, _tok, _lm_device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(LM_MODEL)
    _tok.pad_token = _tok.pad_token or _tok.eos_token
    if torch.cuda.is_available() and os.getenv("ANCHOR_SWEEP_GPU", "1") == "1":
        try:
            _lm = AutoModelForCausalLM.from_pretrained(LM_MODEL, dtype=torch.float16).eval().to("cuda")
            _lm_device = "cuda"
            log.info("sweep LM loaded (GPU fp16)")
            return
        except Exception as e:
            log.warning(f"sweep LM GPU load failed ({e}); falling back to CPU")
            torch.cuda.empty_cache()
    _lm = AutoModelForCausalLM.from_pretrained(LM_MODEL, dtype=torch.float32).eval()
    _lm_device = "cpu"
    log.info("sweep LM loaded (CPU)")


def _lm_sum(ctx: str, conts: list[str]) -> list[float]:
    import torch
    ctx_ids = _tok(ctx).input_ids
    _tok.pad_token = _tok.pad_token or _tok.eos_token
    enc = _tok([ctx + c for c in conts], return_tensors="pt", padding=True).to(_lm_device)
    with torch.no_grad():
        logits = _lm(**enc).logits.float().cpu()
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


def _sweep_candidates(text: str, expected: str):
    words = text.strip().split()
    anchored = len(words) < 3
    if anchored and not expected:
        return []
    variants = []
    for i, w in enumerate(words):
        m = re.match(r"^(.*?)([.,!?]*)$", w)
        core = m.group(1)
        for alt in SETS.get(_key(core), []):
            if _key(alt) == _key(core):
                continue
            if anchored and _key(alt) != _key(expected.strip()):
                continue
            rep = alt.capitalize() if core[:1].isupper() else alt
            variants.append(" ".join(words[:i] + [rep + m.group(2)] + words[i + 1:]))
    return variants


async def sweep(text: str, context: str, expected: str) -> tuple[str, bool]:
    variants = _sweep_candidates(text, expected)
    if not variants:
        return text, False
    async with _lm_lock:
        if _lm is None:
            await asyncio.get_event_loop().run_in_executor(None, _load_lm)
        sums = await asyncio.get_event_loop().run_in_executor(
            None, _lm_sum, context.strip() + " ", [text.strip()] + variants)
    best = max(range(len(variants)), key=lambda i: sums[i + 1])
    if sums[best + 1] - sums[0] > GATE_NATS:
        return variants[best], True
    return text, False


# ---------------- helpers ----------------
OPENAI_MODE = os.getenv("ANCHOR_BACKEND_MODE", "") == "openai"
_ASR_RE = re.compile(r"language ([A-Za-z]+)<asr_text>(.*)", re.S)

# Stock Qwen3-ASR template drops assistant turns entirely, so vllm-serve's
# continue_final_message can't see the 'language X<asr_text>' prefill. This
# copy is identical except it renders a final assistant message verbatim --
# giving byte parity with the toolkit's _build_text_prompt forcing suffix.
_CHAT_TEMPLATE = """{%- set ns = namespace(system_text="") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.system_text = ns.system_text + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and (c.text is defined) -%}
          {%- set ns.system_text = ns.system_text + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}
{%- set ns2 = namespace(audio_tokens="") -%}
{%- for m in messages -%}
  {%- if m.content is not string -%}
    {%- for c in m.content -%}
      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}
        {%- set ns2.audio_tokens = ns2.audio_tokens + "<|audio_start|><|audio_pad|><|audio_end|>" -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}
{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}
{%- if messages[-1].role == 'assistant' -%}
{{- '<|im_start|>assistant\\n' -}}
{%- if messages[-1].content is string -%}
{{- messages[-1].content -}}
{%- else -%}
{%- for c in messages[-1].content -%}{%- if c.type == 'text' and (c.text is defined) -%}{{- c.text -}}{%- endif -%}{%- endfor -%}
{%- endif -%}
{%- elif add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}"""


async def _backend_decode_openai(audio: bytes, filename: str, context: str,
                                 language: str | None, logprobs: bool = False) -> dict:
    """Shim: same contract as backend_decode, but against a vllm-serve OpenAI
    endpoint (continuous batching). Mirrors the toolkit's prompt exactly:
    context as the system message, audio as the user turn, language forcing via
    an assistant prefill 'language X<asr_text>' with continue_final_message.
    ponytail: RESTRICT_SCRIPTS token mask not passed - bench copy only."""
    import base64 as _b64, math as _math
    mime = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
    msgs = [{"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio_url", "audio_url": {
                "url": f"data:{mime};base64,{_b64.b64encode(audio).decode()}"}}]}]
    body = {"messages": msgs, "temperature": 0, "chat_template": _CHAT_TEMPLATE}
    if language:
        msgs.append({"role": "assistant", "content": f"language {language}<asr_text>"})
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    if logprobs:
        body["logprobs"] = True
    r = await _client.post(f"{BACKEND}/v1/chat/completions", json=body)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = ch["message"]["content"] or ""
    full = (f"language {language}<asr_text>{content}") if language else content
    m = _ASR_RE.match(full)
    lang, text = (m.group(1), m.group(2)) if m else (language or "", full)
    out = {"text": text.strip(), "language": lang}
    lp = (ch.get("logprobs") or {}).get("content") or []
    if logprobs and lp:
        vals = [t["logprob"] for t in lp]
        out["min_token_confidence"] = float(_math.exp(min(vals)))
        out["confidence"] = float(_math.exp(sum(vals) / len(vals)))
    return out


async def backend_decode(audio: bytes, filename: str, context: str,
                         language: str | None, logprobs: bool = False) -> dict:
    if OPENAI_MODE:
        return await _backend_decode_openai(audio, filename, context, language, logprobs)
    data = {"context": context}
    if language:
        data["language"] = language
    if logprobs:
        data.update(token_logprobs="true", logprobs="true")
    r = await _client.post(f"{BACKEND}/transcribe",
                           files={"file": (filename, audio)}, data=data)
    r.raise_for_status()
    return r.json()


async def guard(audio: bytes, filename: str, en_text: str = "", ko_text: str = "") -> dict:
    data = {"en_text": en_text or "x"}
    if ko_text:
        data["ko_text"] = ko_text
    r = await _client.post(f"{GUARD}/guard", files={"file": (filename, audio)},
                           data=data, timeout=60)
    return r.json() if r.status_code == 200 else {}


async def xeus_ipa(audio: bytes, filename: str):
    """Xeus phone recognizer on the 3060 server — cleaner Korean than ZIPA."""
    try:
        r = await _client.post(XEUS_URL, files={"files": (filename, audio)}, timeout=10)
        return r.json()[0]["ipa"] if r.status_code == 200 else None
    except Exception:
        return None

import unicodedata

def _fold(sx: str) -> str:
    """Notation-fold an IPA string so ZIPA-vs-G2P symbol conventions stop counting
    as errors (validated 2026-09-02: folded PER separates translation/echo outputs
    from honest accented speech at >0.85 on long texts; raw PFER cannot)."""
    sx = unicodedata.normalize("NFD", sx)
    sx = "".join(c for c in sx if not unicodedata.combining(c))
    sx = sx.replace("ʰ", "").replace("ː", "").replace("˞", "")
    V = {"a":"a","ɑ":"a","ʌ":"a","ə":"a","æ":"e","e":"e","ɛ":"e","i":"i","ɪ":"i","y":"i",
         "o":"o","ɔ":"o","œ":"o","u":"u","ʊ":"u","ɯ":"u","ʉ":"u"}
    C = {"b":"p","p":"p","d":"t","t":"t","g":"k","k":"k","ɡ":"k","z":"s","s":"s","ɕ":"c","ʃ":"c","ʂ":"c","c":"c","ʈ":"t",
         "r":"l","ɾ":"l","l":"l","ɹ":"l","m":"N","n":"N","ŋ":"N","j":"j","w":"w","h":"","ʔ":"","ð":"t","θ":"t","v":"p","f":"p"}
    out = []
    for ch in sx:
        if ch in V: out.append(V[ch])
        elif ch in C:
            if C[ch]: out.append(C[ch])
        elif ch.isalpha(): out.append(ch)
    r = "".join(out).replace("tc", "c")
    return re.sub(r"(.)\1+", r"\1", r)


def folded_per(en_ipa: str, audio_ipa: str) -> float | None:
    R, H = _fold(en_ipa or ""), _fold(audio_ipa or "")
    if not R: return None
    dp = list(range(len(H) + 1))
    for i in range(1, len(R) + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, len(H) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (R[i - 1] != H[j - 1]))
            prev = cur
    return dp[-1] / len(R)


_QUOTE = re.compile(r"['\"‘’“”]([^'\"‘’“”]{4,})['\"‘’“”]")

def _strip(x: str) -> str:
    return re.sub(r"[\s.,!?~…'\"]+", "", str(x or "").lower())

def ctx_quote(text: str, context: str) -> bool:
    """True when text appears verbatim inside a quoted span of context (or the
    whole context) — the signature of a drill that quotes its target answer."""
    t = _strip(text)
    if len(t) < 4:
        return False
    spans = [m.group(1) for m in _QUOTE.finditer(context)] + [context]
    return any(t in _strip(sp) for sp in spans)

def _bigram_sim(a: str, b: str) -> float:
    a, b = _strip(a), _strip(b)
    if not a or not b:
        return 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(1, len(A | B))


def ctx_echo(text: str, context: str) -> bool:
    """Verbatim overlap: the text (sans spacing/punct) appears inside the context.
    per_ko alone cannot separate echo from honest short Korean (measured: innocent
    "초밥 먹었어요" scored 0.857, the real echo 0.833) — echo means COPYING."""
    t = re.sub(r"[\s.,!?~…]+", "", text)
    c = re.sub(r"[\s.,!?~…]+", "", context)
    return bool(t) and t in c


def filler_only(t: str) -> bool:
    toks = re.sub(r"[^a-z ]", " ", t.lower()).split()
    return bool(toks) and all(x in FILLER for x in toks)


def repeat_pattern(text: str, expected: str) -> int:
    """N >= 2 if text is the expected token sequence repeated N times."""
    toks = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    exp = re.sub(r"[^a-z' ]", " ", expected.lower()).split()
    if not exp or len(toks) < 2 * len(exp) or len(toks) % len(exp):
        return 0
    n = len(toks) // len(exp)
    return n if toks == exp * n else 0


# ---------------- endpoint ----------------
@app.post("/transcribe_anchor")
async def transcribe_anchor(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(""),
    expected: str = Form(""),
    origin: str = Form(""),
    debug: str = Form(""),
):
    t0 = time.perf_counter()
    client_ip = request.client.host if request.client else "-"
    if not context.strip():
        return JSONResponse(status_code=400, content={"detail": "context is required"})
    audio = await file.read()
    filename = file.filename or "audio.wav"
    ctx0 = re.sub(r"^\s*튜터:\s*", "", context).strip()
    exp = expected.strip()
    # conditional labeled append: inject only information the context lacks
    full_ctx = ctx0 if (not exp or exp.lower() in ctx0.lower()) else f"{ctx0}\nexpected: {exp}"

    d = await backend_decode(audio, filename, full_ctx, "English", logprobs=True)
    text = d.get("text", "")
    minconf = d.get("min_token_confidence")
    dbg = {"decode_en": text, "min_token_confidence": minconf}

    rejected_ko = None
    g = await guard(audio, filename, en_text=text)
    audio_ipa = (g.get("audio_ipa") or "").replace(" ", "")
    phones = len(audio_ipa)
    pfer = g.get("en_pfer_w")
    dbg.update(audio_ipa=audio_ipa, audio_phones=phones, en_pfer_w=pfer, en_ipa=g.get("en_ipa"),
               fold_en=_fold(g.get("en_ipa") or ""), fold_audio=_fold(g.get("audio_ipa") or ""))
    tier = "clean"
    language = "English"

    if phones <= THIN_PHONES:
        short = len(text.split()) <= 2
        conf_ok = minconf is not None and minconf >= TAU_CONF
        pfer_ok = phones > 0 and pfer is not None and pfer <= PFER_ACCEPT
        if short and (conf_ok or pfer_ok):
            tier = "rescue-ship"
        else:
            text, tier = "", "no_speech"
    elif HANGUL.search(text):
        g2 = await guard(audio, filename, ko_text=text)
        perko = g2.get("per_ko")
        dbg.update(branch="hangul", per_ko=perko, ko_ipa=g2.get("ko_ipa"), ctx_copy=ctx_echo(text, ctx0),
                   fold_ko=_fold(g2.get("ko_ipa") or ""), fold_audio=_fold(g2.get("audio_ipa") or ""))
        # echo = phones in the fabrication band AND a verbatim copy of the context;
        # per_ko alone false-rejects honest short Korean (ZIPA onset drops).
        if perko is not None and perko >= PER_KO_ECHO and ctx_echo(text, ctx0):
            rejected_ko, text, tier = text, "", "ko-echo-reject"
        else:
            tier, language = "hangul-verified", "Korean"
    else:
        # EN echo check: a long Latin output that verbatim-copies a context-quoted
        # string is a fabrication the prefix-weighted PFER can't see. Confirm with
        # a no-context decode: if it diverges hard, the context manufactured the
        # text -> ship the no-context/KO result if coherent, else no_speech.
        # (measured 2026-09-03: 5/5 flags across ~700 clips were real fabrications.)
        en_echo = False
        if (len(text.split()) >= EN_ECHO_MIN_WORDS and ctx_quote(text, ctx0)):
            nc = (await backend_decode(audio, filename, "", "English")).get("text", "")
            if _bigram_sim(nc, text) < EN_ECHO_SIM:
                # fabrication confirmed. try the Korean reading; ship it iff it
                # holds Hangul (a real answer), otherwise no_speech.
                kd = await backend_decode(audio, filename, ctx0, "Korean")
                ko = (kd.get("text") or "").strip()
                if ko and HANGUL.search(ko) and not ctx_quote(ko, ctx0):
                    text, tier, language = ko, "en-echo-reject+ko", "Korean"
                else:
                    text, tier = "", "en-echo-reject"
                en_echo = True
        # widened mismatch: prefix-weighted PFER, or a long output whose folded
        # phone-rate is in/above the suspect band (covert-translation/echo family;
        # rev3: band rows enter the argmin too, judged by FOLDED per with an
        # incumbent margin — honest band clips keep EN, translations flip to KO)
        fper = None if en_echo else folded_per(g.get("en_ipa"), g.get("audio_ipa"))
        dbg.update(branch="latin", folded_en=fper, en_echo=en_echo)
        n_words = len(text.split())
        long_mismatch = (n_words >= FOLDED_MIN_WORDS and fper is not None and fper > FOLDED_BAND)
        weighted_trig = (not FOLDED_ONLY) and (pfer is not None and pfer > PFER_ACCEPT)
        if not en_echo and (weighted_trig or long_mismatch):
            kd = await backend_decode(audio, filename, ctx0, "Korean")
            ko = (kd.get("text") or "").strip()
            if ko:
                g3 = await guard(audio, filename, en_text=text, ko_text=ko)
                if long_mismatch:
                    # folded-scored argmin with incumbent margin
                    au3 = g3.get("audio_ipa")
                    fe = folded_per(g3.get("en_ipa"), au3)
                    fk = folded_per(g3.get("ko_ipa"), au3)
                    dbg.update(ko_candidate=ko, ko_ipa=g3.get("ko_ipa"), folded_en_arg=fe, folded_ko_arg=fk,
                               fold_en_arg=_fold(g3.get("en_ipa") or ""), fold_ko_arg=_fold(g3.get("ko_ipa") or ""), fold_audio_arg=_fold(au3 or ""))
                    # measured: honest EN keeps outright (KO scores worse); real
                    # translations win by 0.02–0.4 — margin 0 with tie→EN suffices
                    # the argmin's only job is EN-vs-Korean: a force-KO decode
                    # that returns Latin text is just a second English decode, and
                    # arbitrating two English renderings only ships jitter
                    # ("These days"->"This day") — keep the incumbent then.
                    # 3-word tier additionally demands the KO candidate be
                    # actually SUPPORTED (folded < FOLDED_SHORT_SUPPORT), not
                    # merely less-bad: short garbled evidence makes less-bad a
                    # coin flip that swaps in context fragments ("Let me
                    # see."->"말씀."). The floor is 0.75: measured, real
                    # loanword-heavy Korean (프로그램/시스템/진전) folds to
                    # 0.60-0.667 even when it is plainly the answer (ZIPA garbles
                    # the loanwords), while the context-fragment echoes (말씀,
                    # 낯가렸-prompt) fold at 0.818-0.833 — a wide clean gap.
                    # (weighted PFER cannot split them: 낯가렸 frag 0.279 sits
                    # inside the good cases' 0.236-0.297.)
                    handled = False
                    if XEUS_ARBITRATE and fk is not None and 0.6 <= fk <= 0.85:
                        xe = await xeus_ipa(audio, filename)
                        fex = folded_per(g3.get("en_ipa"), xe) if xe else None
                        fkx = folded_per(g3.get("ko_ipa"), xe) if xe else None
                        dbg.update(xeus_ipa=xe, xeus_en=fex, xeus_ko=fkx, fold_xeus=_fold(xe or ""))
                        if fex is not None and fkx is not None:
                            handled = True
                            if fkx < XEUS_SUPPORT and HANGUL.search(ko):   # KO genuinely good (absolute) -> swap
                                text, tier, language = ko, "ko-argmin+xeus", "Korean"
                            elif fex >= XEUS_REJECT:                        # KO not good AND EN also bad -> fabrication -> reject
                                text, tier = "", "xeus-reject"
                            else:                                          # KO not good but EN is fine -> keep EN
                                tier = "en-argmin+xeus"
                    if not handled:
                        used_xeus = False
                        if XEUS_FALLBACK and fk is not None and 0.6 <= fk <= 0.85:
                            xe = await xeus_ipa(audio, filename)
                            fkx = folded_per(g3.get("ko_ipa"), xe) if xe else None
                            dbg.update(xeus_ipa=xe, xeus_ko=fkx, fold_xeus=_fold(xe or ""))
                            if fkx is not None:
                                short_ok = fkx < XEUS_SUPPORT
                                used_xeus = True
                            else:
                                short_ok = n_words >= 4 or fk < FOLDED_SHORT_SUPPORT
                        else:
                            short_ok = ((fk is not None and fk < FOLDED_SHORT_SUPPORT) if FOLDED_UNIVERSAL
                                        else (n_words >= 4 or (fk is not None and fk < FOLDED_SHORT_SUPPORT)))
                        if (fe is not None and fk is not None and fk < fe - FOLDED_MARGIN
                                and HANGUL.search(ko) and short_ok):
                            text, language = ko, "Korean"
                            tier = "ko-argmin+xeus" if used_xeus else "ko-argmin+folded"
                        else:
                            tier = "en-argmin+xeus" if used_xeus else "en-argmin+folded"
                else:
                    enw, kow = g3.get("en_pfer_w"), g3.get("ko_pfer_w")
                    if enw is not None and kow is not None and kow <= enw:
                        text, tier, language = ko, "ko-argmin", "Korean"
                    else:
                        tier = "en-argmin"
        if language == "English" and text:
            swept, flipped = await sweep(text, full_ctx, exp)
            if flipped:
                text, tier = swept, tier + "+sweep"

    # final filters on every text exit
    if text and language == "English":
        if filler_only(text):
            text, tier = "", tier + "+filler-drop"
        elif exp:
            n = repeat_pattern(text, exp)
            if n >= 2:
                ge = await guard(audio, filename, en_text=exp)
                need = len((ge.get("en_ipa") or "").replace(" ", ""))
                heard = len((ge.get("audio_ipa") or "").replace(" ", ""))
                if need and heard < 1.5 * need:
                    text, tier = exp[0].upper() + exp[1:] + ".", tier + "+repeat-collapse"

    out = {
        "text": text,
        "language": language if text else None,
        "no_speech": not bool(text),
        "path": tier,
        "min_token_confidence": minconf,
        "en_pfer_w": pfer,
        "audio_phones": phones,
        "request_id": str(uuid.uuid4()),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }
    if debug.strip():
        dbg["tier"] = tier
        dbg["final_text"] = text
        dbg["language"] = language if text else None
        out["debug"] = dbg
    log.info(f"anchor origin={origin or '-'} path={tier} text={text[:40]!r} "
             f"conf={minconf} phones={phones} pfer={pfer} {out['latency_ms']}ms")
    ext = os.path.splitext(filename)[1] or ".wav"
    if os.getenv("ANCHOR_NO_LOG", "") == "1":
        return out
    from wrapper_merged import upload_and_log  # lazy: see note at top
    asyncio.create_task(upload_and_log(
        audio_bytes=audio,
        extension=ext,
        client_ip=client_ip,
        language=None,
        detected_language=out["language"],
        text=text,
        context=context,
        latency=(time.perf_counter() - t0),
        provider="Qwen3ASR-anchor",
        environment="live",  # always archive audio: every anchor row is replay-dataset material
        endpoint="anchor",
        path=tier,
        request_id=out["request_id"],
        origin=origin or None,
        expected=exp or None,
        ko_text=rejected_ko,
        zipa_ipa=audio_ipa or None,
        pfer={"en_pfer_w": pfer} if pfer is not None else None,
    ))
    return out


@app.get("/health")
async def health():
    return {"ok": True, "sweep_lm_loaded": _lm is not None,
            "homophone_classes": len(set(map(id, SETS.values())))}

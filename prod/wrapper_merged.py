"""
Minimal reverse proxy: forwards to ASR backend, falls back to ElevenLabs if it's down.
Handles all DB logging.
"""

import asyncio
import ftplib
import io
import json
import os
import time
import uuid
import logging

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse

# Load secrets from the prod env file regardless of cwd (new isolated wrapper).
load_dotenv(os.getenv("WRAPPER_ENV_FILE", "/home/ailab/ASR/Qwen3ASR/.env"))

# Merged backend (transcribe + streaming) instead of the old batched backend.
ASR_BACKEND_URL = os.getenv("ASR_BACKEND_URL", "http://127.0.0.1:8006")
# /transcribe_lm guard stage: accept the EN text when its weighted PFER vs the
# audio phones is at or below this; above it, decode Korean and arbitrate.
# Measured on real learner clips: correct accented English scores up to ~0.43
# (know1 0.333, letsgo 0.433); fabricated text on garbage audio 0.81. The
# threshold sits between the bands; calibrate from the pfer logged per LM row.
GUARD_ACCEPT_PFER = float(os.getenv("GUARD_ACCEPT_PFER", "0.55"))
# KO replaces EN only when it wins the arbitration by this much (incumbent
# advantage: the EN candidate already survived sweep/fusion scoring).
GUARD_KO_MARGIN = float(os.getenv("GUARD_KO_MARGIN", "0.10"))
BACKEND_WS_URL = ASR_BACKEND_URL.replace("https://", "wss://").replace("http://", "ws://") + "/stream"
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
ES_URL = os.getenv("ES_URL")  # e.g. https://es.weaversmind.com — ES dual-write off when unset
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX = os.getenv("ES_INDEX", "stt_generation_log")
FTP_HOST = os.environ["FTP_HOST"]
FTP_USER = os.environ["FTP_USER"]
FTP_PASSWORD = os.environ["FTP_PASSWORD"]
FTP_CDN_URL = os.environ["FTP_CDN_URL"]

logger = logging.getLogger("asr.proxy")
app = FastAPI()

# Qwen uses full names, ElevenLabs uses ISO codes
QWEN_TO_ELEVENLABS = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "zh",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}
# Main client (backend decodes + guard + fallbacks): generous pool because
# decode calls can hold connections for the whole GPU queue during bursts.
_http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=200, max_keepalive_connections=40))
# Dedicated client for background logging (Supabase) so the log path never
# competes with user-path calls for pool slots.
_log_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=5))

# Second-opinion phone read for smart arbitration rows (Xeus on the 3060 box).
# Logged only — never part of the decision path; failure -> null column.
XEUS_IPA_URL = os.getenv("XEUS_IPA_URL", "http://192.168.1.239:8001/xeus")
# Winning weighted-PFER at/above this is indistinguishable from chance: 30 unrelated
# Korean words score 0.41-0.61 against a typical ZIPA read (mean 0.515), so an
# "argmin" in that band is a coin flip. Such rows go to scribe_v2 instead.
SCRIBE_PFER_FLOOR = float(os.getenv("SCRIBE_PFER_FLOOR", "0.4"))


async def _expected_match(expected: str, text: str) -> tuple[bool | None, str | None]:
    """Homophone check via the guard's /g2p: (match, expected_ipa). match=True
    when `text` is phonetically identical to `expected` (know/No, right/write...).
    match=None = guard unavailable (callers then leave the transcript untouched)."""
    if not expected or not text:
        return None, None
    try:
        r = await _http_client.post("http://127.0.0.1:8010/g2p",
                                    data={"a": expected, "b": text}, timeout=5)
        if r.status_code == 200:
            j = r.json()
            return bool(j.get("match")), j.get("a") or None
    except Exception:
        pass
    return None, None


async def _scribe_text(audio_bytes: bytes, filename: str) -> str | None:
    """scribe_v2 second opinion for smart rows the guard could not call.
    Deliberately no language_code: the reason we are here is that we could not
    tell which language it was. Returns None on any failure (caller keeps the
    arbitration result), so this can never take the endpoint down."""
    try:
        r = await _http_client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"file": (filename, audio_bytes)},
            data={"model_id": "scribe_v2"},
            timeout=30,
        )
        r.raise_for_status()
        return ((r.json() or {}).get("text") or "").strip() or None
    except Exception as e:
        logger.warning(f"scribe fallback failed: {e}")
        return None


async def _xeus_ipa(audio_bytes: bytes, filename: str) -> str | None:
    try:
        r = await _log_client.post(XEUS_IPA_URL,
                                   files={"files": (filename, audio_bytes)}, timeout=20)
        if r.status_code == 200:
            return (r.json() or [{}])[0].get("ipa")
    except Exception:
        pass
    return None


def _upload_ftp(audio_bytes: bytes, extension: str) -> str:
    """Upload audio to FTP, return CDN URL."""
    filename = f"stt/{uuid.uuid4().hex}{extension}"
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASSWORD)
    try:
        ftp.mkd("stt")
    except ftplib.error_perm:
        pass
    ftp.storbinary(f"STOR {filename}", io.BytesIO(audio_bytes))
    ftp.quit()
    return f"{FTP_CDN_URL}/{filename}"


async def upload_to_cdn(audio_bytes: bytes, extension: str = ".wav") -> str | None:
    try:
        return await asyncio.to_thread(_upload_ftp, audio_bytes, extension)
    except Exception as e:
        logger.error(f"FTP upload failed: {e}")
        return None


async def upload_and_log(
    audio_bytes: bytes,
    extension: str,
    client_ip: str,
    language: str | None,
    detected_language: str | None,
    text: str,
    context: str,
    latency: float,
    provider: str,
    environment: str,
    timestamps: list | None = None,
    endpoint: str | None = None,
    path: str | None = None,
    request_id: str | None = None,
    origin: str | None = None,
    expected: str | None = None,
    expected_ipa: str | None = None,
    expected_match: bool | None = None,
    en_text: str | None = None,
    ko_text: str | None = None,
    ko_ipa: str | None = None,
    en_ipa: str | None = None,
    zipa_ipa: str | None = None,
    xeus_ipa: str | None = None,
    ko_per: float | None = None,
    en_per: float | None = None,
    pfer: dict | None = None,  # {en_pfer_w, en_pfer_u, ko_pfer_w, ko_pfer_u}
):
    # Only upload to FTP in live environment
    audio_url = None
    if environment == "live":
        audio_url = await upload_to_cdn(audio_bytes, extension)

    # Then log to DB with the URL
    try:
        row = {
            "client_ip": client_ip,
            "language": language,
            "detected_language": detected_language,
            "text": text,
            "context": context or None,
            "latency": round(latency, 3),
            "timestamps": timestamps,
            "provider": provider,
            "audio_url": audio_url,
            "origin": origin,
            "expected": expected,
            "expected_ipa": expected_ipa,
            "expected_match": expected_match,
        }
        # Smart-endpoint IPA/PER columns (only present when passed by /transcribe_smart).
        if endpoint is not None:
            row.update({
                "endpoint": endpoint,
                "path": path,
                "request_id": request_id,
                "en_text": en_text,
                "ko_text": ko_text,
                "ko_ipa": ko_ipa,
                "en_ipa": en_ipa,
                "zipa_ipa": zipa_ipa,
                "xeus_ipa": xeus_ipa,
                "ko_per": ko_per,
                "en_per": en_per,
                **{k: (pfer or {}).get(k) for k in ("en_pfer_w", "en_pfer_u", "ko_pfer_w", "ko_pfer_u")},
            })
        resp = await _log_client.post(
            f"{SUPABASE_URL}/rest/v1/stt_generation_log",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
                "Accept-Profile": "weavers_tts",
                "Content-Profile": "weavers_tts",
            },
            json=row,
            timeout=10,
        )
        inserted = resp.json()
        if inserted:
            row = inserted[0]  # id + created_at + db defaults; keeps ES doc identical to the Supabase row
    except Exception as e:
        logger.warning(f"DB log failed: {e}")
    # Dual-write mirror to Elasticsearch; _id = Supabase id so re-migration upserts cleanly.
    if ES_URL:
        try:
            doc_path = f"/{ES_INDEX}/_doc/{row['id']}" if row.get("id") is not None else f"/{ES_INDEX}/_doc"
            r = await _log_client.request(
                "PUT" if row.get("id") is not None else "POST",
                f"{ES_URL}{doc_path}",
                auth=(ES_USER, ES_PASSWORD),
                json=row,
                timeout=10,
            )
            if r.status_code >= 300:
                logger.warning(f"ES log failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"ES log failed: {e}")


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(""),
    language: str | None = Form(None),
    timestamps: bool = Form(False),
    logprobs: bool = Form(False),
    token_logprobs: bool = Form(False),
    environment: str = Form("dev"),
    origin: str = Form(""),
    expected: str = Form(""),
):
    _start = time.perf_counter()
    _client_ip = request.client.host if request.client else "-"
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    provider = "Qwen3ASR"
    result = None

    # Try backend
    try:
        resp = await _http_client.post(
            f"{ASR_BACKEND_URL}/transcribe",
            files={"file": (filename, audio_bytes)},
            data={"context": context, "timestamps": str(timestamps).lower(), "logprobs": str(logprobs).lower(), "token_logprobs": str(token_logprobs).lower(), **({"language": language} if language else {})},
            timeout=1800,
        )
        if resp.status_code == 400:
            return JSONResponse(status_code=400, content=resp.json())
        if resp.status_code == 200:
            result = resp.json()
        else:
            raise Exception(f"backend returned {resp.status_code}")
    except Exception:
        # Fallback to ElevenLabs
        provider = "ElevenLabs"
        try:
            resp = await _http_client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files={"file": (filename, audio_bytes)},
                data={"model_id": "scribe_v2", **({"language_code": QWEN_TO_ELEVENLABS.get(language, language)} if language else {})},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            result = {"text": data.get("text", ""), "language": data.get("language_code", language)}
            if timestamps and data.get("words"):
                words = [w for w in data["words"] if w.get("type") == "word"]
                segments = []
                cur = None
                for w in words:
                    if cur is None:
                        cur = {"text": w["text"], "start": w["start"], "end": w["end"]}
                    elif cur["text"][-1] in ".?!" or w["start"] - cur["end"] >= 2.0:
                        segments.append(cur)
                        cur = {"text": w["text"], "start": w["start"], "end": w["end"]}
                    else:
                        cur["text"] += " " + w["text"]
                        cur["end"] = w["end"]
                if cur:
                    segments.append(cur)
                result["timestamps"] = segments
        except Exception as e:
            logger.error(f"ElevenLabs fallback failed: {e}")
            return JSONResponse(status_code=503, content={"detail": "Both ASR backend and ElevenLabs unavailable."})

    elapsed = time.perf_counter() - _start
    if isinstance(result, dict):
        result["latency_ms"] = round(elapsed * 1000)
    # Expected-word homophone respelling (repeat drills: expected="know", ASR
    # wrote "No."). Spelling-only change — raw decode preserved in asr_text.
    _exp_ipa = None; _exp_m = None
    if expected and isinstance(result, dict) and result.get("text"):
        _exp_m, _exp_ipa = await _expected_match(expected, result["text"])
        if _exp_m is not None:
            result["expected_match"] = _exp_m
            if _exp_m:
                result["asr_text"] = result["text"]
                result["text"] = expected
    ext = os.path.splitext(filename)[1] or ".wav"
    asyncio.create_task(upload_and_log(
        audio_bytes=audio_bytes,
        extension=ext,
        client_ip=_client_ip,
        language=language,
        detected_language=result.get("language"),
        text=result.get("text", ""),
        context=context,
        latency=elapsed,
        provider=provider,
        environment=environment,
        timestamps=result.get("timestamps"),
        origin=origin or None,
        expected=expected or None, expected_ipa=_exp_ipa, expected_match=_exp_m,
    ))
    return result


@app.post("/transcribe_lm")
async def transcribe_lm(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(""),
    language: str | None = Form(None),
    environment: str = Form("dev"),
    origin: str = Form(""),
    expected: str = Form(""),
):
    """ASR x LM fusion transcribe. Requires context (400 without). No ElevenLabs
    fallback (the fallback cannot do fusion). Logged with path='LM'."""
    _start = time.perf_counter()
    _client_ip = request.client.host if request.client else "-"
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"

    try:
        # Free-speech routing (2026-08-27): without an expected answer the
        # fusion prior is unanchored and rewrites learner speech (measured:
        # memorize-deletion, only/fully flip, Starbucks). Inject-only (plain
        # decode + context) is the free-speech operating point.
        # Strip a leading speaker label: "튜터: ..." makes the prompt read as a
        # dialogue transcript to continue (echo-flavored decodes, row 87862).
        _c = (context or "").lstrip()
        if _c.startswith("튜터:"):
            context = _c[3:].lstrip()
        _lm_route = bool(expected and expected.strip())
        # Free-speech first pass is force-EN (smart-style): the guard stage below
        # arbitrates EN vs a force-KO decode by weighted PFER, so Korean answers
        # are rescued while EN answers avoid auto-detect language drift.
        _fwd_lang = language or (None if _lm_route else "English")
        resp = await _http_client.post(
            f"{ASR_BACKEND_URL}" + ("/transcribe_lm" if _lm_route else "/transcribe"),
            files={"file": (filename, audio_bytes)},
            data={"context": context, **({"language": _fwd_lang} if _fwd_lang else {})},
            timeout=1800,
        )
        if resp.status_code == 400:
            return JSONResponse(status_code=400, content=resp.json())
        if resp.status_code != 200:
            raise Exception(f"backend returned {resp.status_code}")
        result = resp.json()
    except Exception as e:
        logger.error(f"transcribe_lm backend failed: {e}")
        return JSONResponse(status_code=503, content={"detail": "ASR backend unavailable (no fallback for transcribe_lm)."})

    # ---- guard stage: verify the produced text against the audio's phones.
    # Clean match (en_pfer_w <= GUARD_ACCEPT_PFER) -> return as-is (~20ms).
    # Mismatch -> one force-KO decode, guard arbitrates EN vs KO by weighted
    # PFER (argmin, tie -> Korean, same rule as /transcribe_smart); KO winning
    # replaces the text, so a fusion hallucination never reaches the client.
    # Fail-safe: guard or KO decode unavailable -> text passes through untouched.
    _gj = {}; _guard_tier = None; _ko_text = None
    if result.get("text", "").strip():
        try:
            gr = await _http_client.post("http://127.0.0.1:8010/guard",
                files={"file": (filename, audio_bytes)},
                data={"en_text": result["text"]}, timeout=30)
            if gr.status_code == 200:
                _gj = gr.json()
        except Exception:
            _gj = {}
        _pfer_en = _gj.get("en_pfer_w")
        # No detectable speech: ZIPA returned zero phones, and empty-vs-text
        # scores a PERFECT 0.0 under prefix-weighted PFER — a context echo of a
        # quoted target sails through (the toto.wav case). No phones + text
        # -> return empty; a re-prompt beats a fabricated pass.
        if "audio_ipa" in _gj and not (_gj.get("audio_ipa") or "").strip():
            result["en_text_rejected"] = result["text"]
            result["text"] = ""
            result["no_speech"] = True
            _guard_tier = "guard-empty"
        elif _pfer_en is not None and _pfer_en > GUARD_ACCEPT_PFER:
            try:
                ko_resp = await _http_client.post(f"{ASR_BACKEND_URL}/transcribe",
                    files={"file": (filename, audio_bytes)},
                    data={"language": "Korean", **({"context": context} if context else {})},
                    timeout=1800)
                _ko_text = (ko_resp.json().get("text") or "").strip() if ko_resp.status_code == 200 else None
            except Exception:
                _ko_text = None
            if _ko_text:
                try:
                    gr2 = await _http_client.post("http://127.0.0.1:8010/guard",
                        files={"file": (filename, audio_bytes)},
                        data={"en_text": result["text"], "ko_text": _ko_text}, timeout=30)
                    if gr2.status_code == 200:
                        _gj = gr2.json()
                except Exception:
                    pass
                _en_w, _ko_w = _gj.get("en_pfer_w"), _gj.get("ko_pfer_w")
                _same = _ko_text.strip().lower().rstrip(".!?") == result["text"].strip().lower().rstrip(".!?")
                if (_en_w is not None and _ko_w is not None and not _same
                        and _ko_w < _en_w - GUARD_KO_MARGIN):
                    result["en_text_rejected"] = result["text"]
                    result["text"] = _ko_text
                    result["language"] = "Korean"
                    _guard_tier = "guard-ko"
                else:
                    _guard_tier = "guard-en"
        result["en_pfer_w"] = _pfer_en

    elapsed = time.perf_counter() - _start
    result["latency_ms"] = round(elapsed * 1000)
    _exp_ipa = None; _exp_m = None; _respelled = False
    if expected and result.get("text"):
        _exp_m, _exp_ipa = await _expected_match(expected, result["text"])
        if _exp_m is not None:
            result["expected_match"] = _exp_m
            if _exp_m:
                _respelled = result["text"].strip() != expected.strip()
                result["asr_text"] = result["text"]
                result["text"] = expected
    # path records which tier(s) actually changed the transcript:
    # LM-clean / LM-fusion / LM-sweep / LM-fusion+sweep, +respell if the
    # expected-match rewrite fired on top.
    _tiers = []
    if result.get("fusion_fixes"):
        _tiers.append("fusion")
    if result.get("sweep_flips"):
        _tiers.append("sweep")
    if _guard_tier:
        _tiers.append(_guard_tier)
    if _respelled:
        _tiers.append("respell")
    _pfx = "LM-" if _lm_route else "INJ-"
    _path = _pfx + "+".join(_tiers) if _tiers else _pfx + "clean"
    _req_id = str(uuid.uuid4())
    result["request_id"] = _req_id
    ext = os.path.splitext(filename)[1] or ".wav"
    asyncio.create_task(upload_and_log(
        audio_bytes=audio_bytes,
        extension=ext,
        client_ip=_client_ip,
        language=language,
        detected_language=result.get("language"),
        text=result.get("text", ""),
        context=context,
        latency=elapsed,
        provider="Qwen3ASR-lm",
        environment="live",  # always upload audio: every LM row is replay-dataset material
        timestamps=None,
        endpoint="lm",
        path=_path,
        request_id=_req_id,
        origin=origin or None,
        expected=expected or None, expected_ipa=_exp_ipa, expected_match=_exp_m,
        en_text=result.get("en_text_rejected") or result.get("text"),
        ko_text=_ko_text,
        en_ipa=_gj.get("en_ipa"), ko_ipa=_gj.get("ko_ipa") or None, zipa_ipa=_gj.get("audio_ipa"),
        en_per=_gj.get("per_en"), ko_per=_gj.get("per_ko"),
        pfer={k: _gj.get(k) for k in ("en_pfer_w", "en_pfer_u", "ko_pfer_w", "ko_pfer_u")} if _gj else None,
    ))
    return result


async def _log_stream_final(text: str, language: str | None, client_ip: str, latency: float):
    """Best-effort DB log of a finished stream. Never raises into the WS path."""
    try:
        await upload_and_log(
            audio_bytes=b"", extension=".wav", client_ip=client_ip,
            language=None, detected_language=language, text=text or "",
            context="", latency=latency, provider="Qwen3ASR-stream",
            environment="dev",  # no CDN upload for streams
            timestamps=None,
        )
    except Exception as e:
        logger.warning(f"stream log failed: {e}")


@app.get("/")
def index():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic.html")
    return HTMLResponse(open(p, encoding="utf-8").read())


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Transparent WS proxy to the merged backend's /stream.
    Client <-> wrapper(8002) <-> backend(/stream). Thin passthrough: no ElevenLabs
    fallback, best-effort final-transcript logging only."""
    await ws.accept()
    _client_ip = ws.client.host if ws.client else "-"
    _start = time.perf_counter()
    final_text, final_lang = None, None
    try:
        # Forward the client query string (e.g. ?language=English) to the backend;
        # otherwise language forcing on :8002 is silently dropped.
        _qs = ws.url.query
        _backend_url = BACKEND_WS_URL + ("?" + _qs if _qs else "")
        async with websockets.connect(_backend_url, max_size=None) as backend:

            async def client_to_backend():
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        await backend.close()
                        return
                    if msg.get("bytes") is not None:
                        await backend.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await backend.send(msg["text"])

            pump = asyncio.create_task(client_to_backend())
            try:
                async for m in backend:  # backend sends JSON text frames
                    await ws.send_text(m if isinstance(m, str) else m.decode("utf-8"))
                    try:
                        d = json.loads(m)
                        if d.get("type") == "final":
                            final_text, final_lang = d.get("text"), d.get("language")
                    except Exception:
                        pass
            finally:
                pump.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"/stream proxy error: {e}")
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if final_text:
            asyncio.create_task(_log_stream_final(final_text, final_lang, _client_ip, time.perf_counter() - _start))
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/stream_full")
async def stream_full(ws: WebSocket):
    """Transparent WS proxy to the merged backend's /stream_full (live partials +
    dual final: stream_text & full_text). Thin passthrough; best-effort logs full_text."""
    await ws.accept()
    _client_ip = ws.client.host if ws.client else "-"
    _start = time.perf_counter()
    full_text, full_lang = None, None
    backend_ws_url = ASR_BACKEND_URL.replace("https://", "wss://").replace("http://", "ws://") + "/stream_full"
    try:
        _qs = ws.url.query
        _backend_url = backend_ws_url + ("?" + _qs if _qs else "")
        async with websockets.connect(_backend_url, max_size=None) as backend:

            async def client_to_backend():
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        await backend.close()
                        return
                    if msg.get("bytes") is not None:
                        await backend.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await backend.send(msg["text"])

            pump = asyncio.create_task(client_to_backend())
            try:
                async for m in backend:
                    await ws.send_text(m if isinstance(m, str) else m.decode("utf-8"))
                    try:
                        d = json.loads(m)
                        if d.get("type") == "final":
                            full_text, full_lang = d.get("full_text"), d.get("language")
                    except Exception:
                        pass
            finally:
                pump.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"/stream_full proxy error: {e}")
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if full_text:
            asyncio.create_task(_log_stream_final(full_text, full_lang, _client_ip, time.perf_counter() - _start))
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/stream_smart")
async def stream_smart(ws: WebSocket):
    """Transparent WS proxy to the merged backend's /stream_smart (force-EN partials +
    per-utterance language arbitration on end). Best-effort logs the chosen final text."""
    await ws.accept()
    _client_ip = ws.client.host if ws.client else "-"
    _start = time.perf_counter()
    fin_text, fin_lang = None, None
    backend_ws_url = ASR_BACKEND_URL.replace("https://", "wss://").replace("http://", "ws://") + "/stream_smart"
    try:
        _qs = ws.url.query
        _backend_url = backend_ws_url + ("?" + _qs if _qs else "")
        async with websockets.connect(_backend_url, max_size=None) as backend:

            async def client_to_backend():
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        await backend.close()
                        return
                    if msg.get("bytes") is not None:
                        await backend.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await backend.send(msg["text"])

            pump = asyncio.create_task(client_to_backend())
            try:
                async for m in backend:
                    await ws.send_text(m if isinstance(m, str) else m.decode("utf-8"))
                    try:
                        d = json.loads(m)
                        if d.get("type") == "final":
                            fin_text, fin_lang = d.get("text"), d.get("language")
                    except Exception:
                        pass
            finally:
                pump.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"/stream_smart proxy error: {e}")
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if fin_text:
            asyncio.create_task(_log_stream_final(fin_text, fin_lang, _client_ip, time.perf_counter() - _start))
        try:
            await ws.close()
        except Exception:
            pass


@app.post("/transcribe_smart")
async def transcribe_smart(request: Request, file: UploadFile = File(...),
                           context: str = Form(""), origin: str = Form(""),
                           expected: str = Form(""), debug: bool = Form(False)):
    """Decode force-EN. If it kept Hangul, return it (1 decode). If all-Latin, decode force-KO
    and pick whichever matches the audio (argmin PER, 2 decodes). Response: {text, language}.
    Pass debug=true for path / decodes / candidate PERs."""
    import re as _re
    _start = time.perf_counter()
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"
    _client_ip = request.client.host if request.client else "-"
    # Client-side correlation id: returned in the response AND written to the log row,
    # so the caller can look its row up without us waiting on the (backgrounded) insert.
    _req_id = str(uuid.uuid4())
    # Expected-word metadata for logging (filled by _apply_expected).
    _exp_meta = {"expected": expected or None, "expected_ipa": None, "expected_match": None}

    def _save(_out, path=None, ipa=None):
        # Smart path: always archive audio to CDN + log to DB (fire-and-forget).
        # path = which decision branch/PER outcome ran; ipa carries the guard's
        # IPA/PER fields when the arbitration path ran.
        _out["request_id"] = _req_id
        ipa = ipa or {}
        _ext = os.path.splitext(filename)[1] or ".wav"

        async def _bg():
            # Xeus second opinion only where there's a ZIPA read to compare against.
            xeus = await _xeus_ipa(audio_bytes, filename) if ipa.get("zipa_ipa") else None
            await upload_and_log(
                audio_bytes=audio_bytes, extension=_ext, client_ip=_client_ip,
                language=None, detected_language=_out.get("language"),
                text=_out.get("text", ""), context=context,
                latency=time.perf_counter() - _start, provider="Qwen3ASR-smart",
                environment="live", timestamps=None,
                endpoint="smart", path=path, request_id=_req_id, origin=origin or None,
                expected=_exp_meta["expected"], expected_ipa=_exp_meta["expected_ipa"],
                expected_match=_exp_meta["expected_match"],
                en_text=_out.get("en_text"), ko_text=_out.get("ko_text"),
                ko_ipa=ipa.get("ko_ipa"), en_ipa=ipa.get("en_ipa"), zipa_ipa=ipa.get("zipa_ipa"),
                xeus_ipa=xeus,
                ko_per=ipa.get("ko_per"), en_per=ipa.get("en_per"), pfer=ipa.get("pfer"),
            )
        asyncio.create_task(_bg())

    def _lang(t):
        ko = bool(_re.search(r"[\uac00-\ud7a3]", t)); en = bool(_re.search(r"[A-Za-z]", t))
        return "mixed" if (ko and en) else ("Korean" if ko else ("English" if en else "unknown"))

    async def _decode(lang):
        t = time.perf_counter()
        r = await _http_client.post(f"{ASR_BACKEND_URL}/transcribe",
            files={"file": (filename, audio_bytes)},
            data={"context": context, "language": lang}, timeout=1800)
        return (r.json().get("text", "") if r.status_code == 200 else ""), (time.perf_counter()-t)*1000

    async def _apply_expected(_out):
        # Expected-word homophone respelling (spelling only; raw decodes stay
        # in en_text/ko_text and the replaced original in asr_text).
        if not expected:
            return
        m, eipa = await _expected_match(expected, _out.get("text", ""))
        _exp_meta["expected_ipa"] = eipa
        if m is None:
            return
        # The chosen candidate may not match while the OTHER one does — on
        # homophone-class rows arbitration is a coin flip (row 67437: chose 노,
        # but the en candidate "No." == expected "know"). When the ASR itself
        # produced a candidate phonetically identical to the declared target,
        # the caller's knowledge beats the metric. Exact equality only.
        if not m:
            for alt in (_out.get("en_text"), _out.get("ko_text")):
                if alt and alt != _out.get("text"):
                    m2, _ = await _expected_match(expected, alt)
                    if m2:
                        m = True
                        break
        _exp_meta["expected_match"] = m
        _out["expected_match"] = m
        if m:
            _out["asr_text"] = _out["text"]
            _out["text"] = expected
            _out["language"] = _lang(expected)

    en_text, en_ms = await _decode("English")
    if _re.search(r"[\uac00-\ud7a3]", en_text):
        _path = "hangul-passthrough"
        out = {"text": en_text, "language": _lang(en_text), "en_text": en_text, "ko_text": None,
               "latency_ms": round((time.perf_counter()-_start)*1000)}
        if debug:
            out.update({"path": _path, "decodes": 1, "en_ms": round(en_ms)})
        await _apply_expected(out)
        _save(out, path=_path)
        return out

    ko_text, ko_ms = await _decode("Korean")
    # If the Korean-forced pass produced no Hangul, there is no Korean candidate to
    # arbitrate -- the audio is English. Skip the guard/PER test entirely.
    if not _re.search(r"[\uac00-\ud7a3]", ko_text):
        _path = "latin-passthrough"
        out = {"text": en_text, "language": _lang(en_text), "en_text": en_text, "ko_text": ko_text,
               "latency_ms": round((time.perf_counter()-_start)*1000)}
        if debug:
            out.update({"path": _path, "decodes": 2,
                        "en_ms": round(en_ms), "ko_ms": round(ko_ms)})
        await _apply_expected(out)
        _save(out, path=_path)
        return out

    per_en = per_ko = None; en_ipa = ko_ipa = zipa_ipa = None; pfer = {}; gms = 0.0
    try:
        tg = time.perf_counter()
        gr = await _http_client.post("http://127.0.0.1:8010/guard",
            files={"file": (filename, audio_bytes)},
            data={"en_text": en_text, "ko_text": ko_text}, timeout=30)
        gms = (time.perf_counter()-tg)*1000
        if gr.status_code == 200:
            gj = gr.json()
            per_en = gj.get("per_en"); per_ko = gj.get("per_ko")
            en_ipa = gj.get("en_ipa"); ko_ipa = gj.get("ko_ipa"); zipa_ipa = gj.get("audio_ipa")
            pfer = {k: gj.get(k) for k in ("en_pfer_w", "en_pfer_u", "ko_pfer_w", "ko_pfer_u")}
    except Exception:
        pass
    # Decision + path: weighted-PFER argmin (tie -> Korean) — switched from PER
    # argmin 2026-08-11 after the feature-weighted metric judged every
    # truth-labeled disagreement correctly. PER is still computed/logged for
    # auditing, and remains the fallback if the guard returns no PFER.
    # guard-fail -> English.
    #
    # [ko-codeswitch — tried 2026-08-11, removed same day; notes for revival]
    # Idea: on this branch en_text is all-Latin, so if ko_text is code-switched
    # (Hangul + Latin words) the audio contained Korean that force-EN translated
    # away — ko_text is the superset transcript; always choose it, path
    # "ko-codeswitch", skipping the metric.
    # Record: 7/8 correct on genuine code-switch (rows 66490, 66558, 66597-601:
    # force-EN dropped 그러니까/음 etc.). Failure = row 66709: audio was ENGLISH
    # ("iced americano always hits the spot"); force-KO transliterated it into
    # Hangul loanwords (아이스 아메리카노...) -> looked code-switched -> rule
    # shipped the transliteration. Acoustics can't tell a loanword transliteration
    # from the original (same class as Fishing/피싱) — needs a non-acoustic prior
    # (request context, decode logprobs, product rule).
    # If revived, add the superset check that separates the two classes:
    # genuine code-switch has latin(ko_text) ≈ en_text (Hangul = ADDED content);
    # transliteration has latin(ko_text) ⊂⊂ en_text (Hangul REPLACED content).
    # Note pfer-w alone decided all 7 genuine rows correctly anyway; the override
    # only adds value for code-switched audio that STARTS with the shared English
    # part (prefix window can't see the Korean tail) — not yet observed in traffic.
    en_w = pfer.get("en_pfer_w"); ko_w = pfer.get("ko_pfer_w")
    if en_w is not None and ko_w is not None:
        if ko_w < en_w:
            chose, _path = "korean", "pfer-ko"
        elif ko_w > en_w:
            chose, _path = "english", "pfer-en"
        else:
            chose, _path = "korean", "pfer-tie"
    elif per_en is not None and per_ko is not None:
        if per_ko < per_en:
            chose, _path = "korean", "per-ko"
        elif per_ko > per_en:
            chose, _path = "english", "per-en"
        else:
            chose, _path = "korean", "per-tie"
    else:
        chose, _path = "english", "guard-fail"
    chosen = ko_text if chose == "korean" else en_text

    # --- scribe fallback: arbitration was not actually a decision ------------
    # Two ways the PFER argmin is meaningless even though it returned a number:
    #   winner >= SCRIBE_PFER_FLOOR -> both candidates score like random words
    #   empty audio_ipa             -> per() returns 0.0, so BOTH look perfect
    #                                  on zero evidence and it ties to Korean
    # Guard-fail (no scores at all) is deliberately NOT included: if the guard
    # service dies, that would send every request to a paid API.
    _scribe = None
    if en_w is not None and ko_w is not None:
        _no_audio_ipa = not (zipa_ipa or "").strip()
        if min(en_w, ko_w) >= SCRIBE_PFER_FLOOR or _no_audio_ipa:
            _scribe = await _scribe_text(audio_bytes, filename)
            if _scribe:
                _asr_text = chosen
                chosen = _scribe
                _path = "scribe-fallback"
    out = {"text": chosen, "language": _lang(chosen), "en_text": en_text, "ko_text": ko_text,
           "latency_ms": round((time.perf_counter()-_start)*1000)}
    if _scribe:
        # what the argmin would have returned, kept for comparison
        out["asr_text"] = _asr_text
    if debug:
        out.update({"path": _path, "decodes": 2, "chose": chose, "scribe": _scribe,
                    "per_en": per_en, "per_ko": per_ko,
                    "en_ipa": en_ipa, "ko_ipa": ko_ipa, "zipa_ipa": zipa_ipa, **pfer,
                    "en_ms": round(en_ms), "ko_ms": round(ko_ms), "guard_ms": round(gms)})
    await _apply_expected(out)
    _save(out, path=_path, ipa={"en_ipa": en_ipa, "ko_ipa": ko_ipa, "zipa_ipa": zipa_ipa,
                                "en_per": per_en, "ko_per": per_ko, "pfer": pfer})
    return out


@app.post("/transcribe_dual")
async def transcribe_dual(
    request: Request,
    file: UploadFile = File(...),
    language_a: str = Form(...),
    language_b: str = Form(...),
    context: str = Form(""),
    timestamps: bool = Form(False),
    logprobs: bool = Form(False),
    token_logprobs: bool = Form(False),
    environment: str = Form("dev"),
    origin: str = Form(""),
):
    """Force-transcribe the same audio as two languages in parallel, return both.

    Fires two independent /transcribe calls to the Qwen backend (one per forced
    language) concurrently and returns each result plus a combined transcript.
    No ElevenLabs fallback here: if a language fails (e.g. unsupported, or the
    backend is down) that sub-result carries an "error" field and the other still
    returns.
    """
    _start = time.perf_counter()
    _client_ip = request.client.host if request.client else "-"
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"

    async def _one(lang: str) -> dict:
        try:
            resp = await _http_client.post(
                f"{ASR_BACKEND_URL}/transcribe",
                files={"file": (filename, audio_bytes)},
                data={"context": context, "timestamps": str(timestamps).lower(), "logprobs": str(logprobs).lower(), "token_logprobs": str(token_logprobs).lower(), "language": lang},
                timeout=1800,
            )
            if resp.status_code == 200:
                return resp.json()
            ct = resp.headers.get("content-type", "")
            return {"error": resp.json() if ct.startswith("application/json") else resp.text,
                    "status": resp.status_code}
        except Exception as e:
            return {"error": str(e)}

    res_a, res_b = await asyncio.gather(_one(language_a), _one(language_b))

    def _block(lang: str, res: dict) -> dict:
        b = {"language": lang, "text": res.get("text", "")}
        if "error" in res:
            b["error"] = res["error"]
        if logprobs:
            b["confidence"] = res.get("confidence")
            b["min_token_confidence"] = res.get("min_token_confidence")
        if token_logprobs:
            b["tokens"] = res.get("tokens")
        return b

    # Translation guard: force-EN is the default; suggest "korean" only when the
    # Korean-forced output matches the audio phonetically better (force-EN translated).
    # Calls the local ZIPA guard service; fail-safe (suggestion=None if unavailable).
    suggestion = None
    try:
        en_txt = res_a.get("text", "") if language_a == "English" else res_b.get("text", "")
        ko_txt = res_b.get("text", "") if language_b == "Korean" else res_a.get("text", "")
        gr = await _http_client.post(
            "http://127.0.0.1:8010/guard",
            files={"file": (filename, audio_bytes)},
            data={"en_text": en_txt, "ko_text": ko_txt},
            timeout=30,
        )
        if gr.status_code == 200:
            suggestion = gr.json().get("suggestion")
    except Exception:
        suggestion = None

    # Log one row per dual request (fire-and-forget, like smart). text = the
    # suggested candidate when the guard resolved, else the language_a (primary)
    # result; the forced pair goes in `language`, the guard outcome in `path`.
    import re as _re
    _req_id = str(uuid.uuid4())
    if suggestion == "korean" and ko_txt:
        _text = ko_txt
    elif suggestion == "english" and en_txt:
        _text = en_txt
    else:
        _text = res_a.get("text", "")
    _ko = bool(_re.search(r"[가-힣]", _text)); _en = bool(_re.search(r"[A-Za-z]", _text))
    _detected = "mixed" if (_ko and _en) else ("Korean" if _ko else ("English" if _en else "unknown"))
    asyncio.create_task(upload_and_log(
        audio_bytes=audio_bytes, extension=os.path.splitext(filename)[1] or ".wav",
        client_ip=_client_ip, language=f"{language_a}+{language_b}",
        detected_language=_detected, text=_text, context=context,
        latency=time.perf_counter() - _start, provider="Qwen3ASR-dual",
        environment=environment, timestamps=None,
        endpoint="dual", path=(f"suggest-{suggestion}" if suggestion else "guard-fail"),
        request_id=_req_id, origin=origin or None,
    ))

    return {
        "language_a": _block(language_a, res_a),
        "language_b": _block(language_b, res_b),
        "suggestion": suggestion,
        "request_id": _req_id,
        "latency_ms": round((time.perf_counter() - _start) * 1000),
    }


# ---- ARCH-004 Anchor endpoint (same handler as the standalone :8004 service) ----
import anchor_service as _anchor
app.post("/transcribe_anchor")(_anchor.transcribe_anchor)

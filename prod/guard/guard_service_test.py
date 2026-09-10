"""EN/KO arbitration guard — wav2vec2-XLS-R-300M KO/EN backend (replaces ZIPA-large PER).

Same HTTP contract as the old ZIPA guard so transcribe_smart / transcribe_dual need no
changes:  POST /guard  (file, en_text, ko_text) -> {suggestion, per_en, per_ko, audio_text}

How it routes: run the code-switch-native wav2vec2 model on the audio to get a reference
transcript, then compare it (normalized edit distance) against the two Qwen force-decodes.
Routing keys off SCRIPT match (Hangul vs Latin), which is easy even when the small model's
transcription is rough — the cross-script contrast dominates the distance.
"""
import os, re, sys, tempfile, unicodedata
import numpy as np, torch, soundfile as sf, librosa
from fastapi import FastAPI, File, Form, UploadFile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL = os.environ.get("GUARD_MODEL", "raniczan/wav2vec2-xls-r-300m-korean-english")
MARGIN = 0.10  # switch to Korean only when it clearly matches better (dual relies on this)
_want = os.environ.get("GUARD_DEVICE", "cuda")

_proc = Wav2Vec2Processor.from_pretrained(MODEL)
_model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
DEVICE = "cpu"
if _want == "cuda" and torch.cuda.is_available():
    try:
        _model = _model.to("cuda"); DEVICE = "cuda"
    except RuntimeError as e:  # OOM on a tight GPU -> stay on CPU
        print("guard: cuda load failed, using cpu:", e, flush=True)
print(f"guard: wav2vec2 KO/EN loaded on {DEVICE} ({MODEL})", flush=True)


def _norm(s):
    # lowercase, drop everything but letters/digits/Hangul so the distance is script-vs-script
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def _cer(ref, cand):
    a, b = _norm(ref), _norm(cand)
    if not a and not b: return 0.0
    if not a or not b: return 1.0
    n = len(b); prev = list(range(n + 1))
    for i, ch in enumerate(a, 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != b[j - 1]))
        prev = cur
    return prev[n] / max(len(a), len(b))


def audio_text(path):
    a, sr = sf.read(path)
    if a.ndim > 1: a = a[:, 0]
    if sr != 16000:
        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=16000)
    a = a.astype(np.float32)
    iv = _proc(a, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        ids = _model(iv).logits.argmax(-1)
    return unicodedata.normalize("NFC", _proc.batch_decode(ids)[0])


# warm once so the first real request doesn't eat graph/allocator setup
try:
    audio_text_warm = audio_text
    _iv = _proc(np.zeros(16000, dtype=np.float32), sampling_rate=16000,
                return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        _model(_iv)
    print("guard: warmed", flush=True)
except Exception as e:
    print("guard: warm skipped:", e, flush=True)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL}


@app.post("/guard")
async def guard(file: UploadFile = File(...), en_text: str = Form(""), ko_text: str = Form("")):
    # identical forced outputs -> nothing to arbitrate, keep English (their default)
    if en_text.strip() == ko_text.strip():
        return {"suggestion": "english", "reason": "identical", "per_en": None, "per_ko": None}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read()); path = tmp.name
    try:
        ref = audio_text(path)
    finally:
        try: os.unlink(path)
        except OSError: pass
    if not ref.strip():  # model heard nothing -> fail safe to English
        return {"suggestion": "english", "reason": "empty_audio_text", "per_en": None, "per_ko": None}
    pe, pk = _cer(ref, en_text), _cer(ref, ko_text)
    suggestion = "korean" if pk < pe - MARGIN else "english"
    return {"suggestion": suggestion, "per_en": round(pe, 3), "per_ko": round(pk, 3), "audio_text": ref}

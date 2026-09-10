import os, re, sys, subprocess, tempfile
import numpy as np, torch, onnxruntime as ort, soundfile as sf, librosa
from fastapi import FastAPI, File, Form, UploadFile
import epitran
from phonemizer.backend import EspeakBackend
_ESPK = EspeakBackend('en-us', with_stress=False)
sys.path.insert(0, '/home/ailab/zipa_test')
from utils import load_tokens, ctc_greedy_decode, get_fbank_extractor
from pfer import prefix_pfer, tokenize_ipa

BASE = '/home/ailab/zipa_test'
MARGIN = 0.10
EPI = epitran.Epitran('kor-Hang')
EXT = get_fbank_extractor()
VOCAB = load_tokens(f'{BASE}/tokens.txt')
SESS = ort.InferenceSession(f'{BASE}/model.fp16.onnx',
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print("guard: ZIPA-large fp16 loaded on", SESS.get_providers()[0], flush=True)

def clean(s):
    for c in 'ˈˌːʰʲ ˑ̃▁': s = s.replace(c, '')
    return re.sub(r'\s+', '', s)

def espeak(t):
    return clean(_ESPK.phonemize([t])[0])

def g2p(text):
    # group maximal same-script runs (Korean or Latin, incl. spaces) so each English
    # RUN is a single espeak spawn instead of one per word; preserves spoken order.
    out = []
    for m in re.finditer(r"[가-힣][가-힣\s]*|[A-Za-z'][A-Za-z'\s]*", text):
        chunk = m.group()
        out.append(clean(EPI.transliterate(chunk)) if re.match(r'[가-힣]', chunk) else espeak(chunk))
    return ''.join(out)

def per(a, b):
    # a = 2s-window audio IPA (partial), b = full candidate G2P.
    # prefix-PER: min over the last DP row = audio aligned to the best prefix of b.
    if not a: return 0.0
    if not b: return 1.0
    n = len(b); prev = list(range(n+1))
    for i, ch in enumerate(a, 1):
        cur = [i] + [0]*n
        for j in range(1, n+1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1] + (ch != b[j-1]))
        prev = cur
    return min(prev) / len(a)

WIN_SAMPLES = 3 * 16000
PRE_ROLL = int(0.5 * 16000)  # back the window up before the energy onset so soft
                             # attacks (fricatives etc.) under the threshold aren't clipped

def _energy_onset(a, thr=0.02, fr=512):
    for i in range(0, len(a) - fr, fr):
        if np.sqrt(np.mean(a[i:i+fr] ** 2)) > thr:
            return i
    return 0

def _zipa_seg(seg):
    feat = EXT.extract_batch([torch.from_numpy(seg).float().unsqueeze(0)], sampling_rate=16000)[0].unsqueeze(0)
    out = SESS.run(None, {'x': feat.numpy(), 'x_lens': np.array([feat.shape[1]], dtype=np.int64)})[0][0]
    return clean(''.join(ctc_greedy_decode(out, VOCAB)))

def audio_ipa(path):
    a, sr = sf.read(path)
    if a.ndim > 1: a = a[:, 0]
    if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    a = a.astype(np.float32)
    on = max(0, _energy_onset(a) - PRE_ROLL)
    seg = a[on:on + WIN_SAMPLES]
    if len(seg) < WIN_SAMPLES:
        seg = np.pad(seg, (0, WIN_SAMPLES - len(seg)))
    return _zipa_seg(seg)

# warm the fixed 2s shape once so subsequent calls skip graph capture
for _ in range(2):
    _zipa_seg(np.zeros(WIN_SAMPLES, dtype=np.float32))
print("guard: fixed-2s ZIPA warmed", flush=True)

app = FastAPI()

@app.get('/health')
def health():
    return {"ok": True, "provider": SESS.get_providers()[0]}

@app.post('/g2p')
async def g2p_compare(a: str = Form(''), b: str = Form('')):
    """Phonetic equality of two texts (homophone check, e.g. expected-word vs
    ASR transcript: 'know' vs 'No.'). Mixed-script safe via the same g2p()."""
    ga, gb = g2p(a), g2p(b)
    return {"a": ga, "b": gb, "match": bool(ga) and ga == gb}

@app.post('/guard')
async def guard(file: UploadFile = File(...), en_text: str = Form(''), ko_text: str = Form('')):
    # identical forced outputs -> nothing to arbitrate, keep English (their default)
    if en_text.strip() == ko_text.strip():
        return {"suggestion": "english", "reason": "identical", "per_en": None, "per_ko": None}
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await file.read()); path = tmp.name
    try:
        aud = audio_ipa(path)
        en_ipa, ko_ipa = g2p(en_text), g2p(ko_text)
        pe, pk = per(aud, en_ipa), per(aud, ko_ipa)
        # PFER diagnostics (phone-level prefix metric; weighted = 20-feature
        # substitution cost, unweighted = binary). Logged, not used for the decision.
        taud, ten, tko = tokenize_ipa(aud), tokenize_ipa(en_ipa), tokenize_ipa(ko_ipa)
        pfer = {"en_pfer_w": round(prefix_pfer(taud, ten, True), 3),
                "en_pfer_u": round(prefix_pfer(taud, ten, False), 3),
                "ko_pfer_w": round(prefix_pfer(taud, tko, True), 3),
                "ko_pfer_u": round(prefix_pfer(taud, tko, False), 3)}
    finally:
        try: os.unlink(path)
        except OSError: pass
    # switch to Korean only when it clearly matches the audio better (force-EN translated)
    suggestion = "korean" if pk < pe - MARGIN else "english"
    return {"suggestion": suggestion, "per_en": round(pe, 3), "per_ko": round(pk, 3),
            "audio_ipa": aud, "en_ipa": en_ipa, "ko_ipa": ko_ipa, **pfer}

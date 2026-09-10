import sys, asyncio, json
sys.path.insert(0, "/home/ailab/zipa_test")
sys.path.insert(0, "/home/ailab/ASR/Qwen3ASR-streaming")
import numpy as np, soundfile as sf, librosa
import guard_service as G          # loads ZIPA model + g2p + audio_ipa
import anchor_service as A         # folded_per, backend_decode

def _energy_offset(a, fr=400, thr=None):
    if thr is None:
        thr = 0.15 * np.max(np.abs(a)) if a.size else 0
    for i in range(len(a) - fr, 0, -fr):
        if np.sqrt(np.mean(a[i:i+fr] ** 2)) > thr:
            return min(len(a), i + fr)
    return len(a)

def recover_ipa(path, reps=3, lead=0.5, gap=0.15):
    a, sr = sf.read(path)
    if a.ndim > 1: a = a[:, 0]
    if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    a = a.astype(np.float32)
    on = max(0, G._energy_onset(a))
    off = _energy_offset(a)
    speech = a[on:off] if off > on else a[on:on + 16000]
    lead_s = np.zeros(int(lead * 16000), dtype=np.float32)
    gap_s = np.zeros(int(gap * 16000), dtype=np.float32)
    seg = lead_s.copy()
    for _ in range(reps):
        seg = np.concatenate([seg, speech, gap_s])
    seg = seg[:G.WIN_SAMPLES]
    if len(seg) < G.WIN_SAMPLES:
        seg = np.pad(seg, (0, G.WIN_SAMPLES - len(seg)))
    return G._zipa_seg(seg), (off - on) / 16000

async def main():
    bench = json.load(open("/tmp/benchmark50.json"))[:10]
    print(f"{'decode':26} {'splen':5} {'base_ipa':22} {'rec_ipa':26} {'foldB':6} {'foldR':6}")
    for b in bench:
        fn = b["audio_url"].split("/")[-1]; path = f"/tmp/b50_{fn}"
        import os
        if not os.path.exists(path): continue
        audio = open(path, "rb").read()
        en = (await A.backend_decode(audio, fn, (b.get("context") or "-"), "English")).get("text", "").strip()
        base = G.audio_ipa(path)
        rec, splen = recover_ipa(path)
        eg = G.g2p(en)
        fb = A.folded_per(eg, base); fr = A.folded_per(eg, rec)
        print(f"{en[:26]:26} {splen:.2f}s {base[:22]:22} {rec[:26]:26} {fb if fb is None else round(fb,3):6} {fr if fr is None else round(fr,3):6}")

asyncio.run(main())

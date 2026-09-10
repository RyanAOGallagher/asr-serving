import asyncio, json, re, sys
sys.path.insert(0, "/home/ailab/zipa_test")
import anchor_service as A
from pfer import prefix_pfer, tokenize_ipa

# full-length variant: same DP as prefix_pfer but return the CORNER cell
# (prev[-1]) instead of min(prev) — charges the candidate's overrun.
def full_pfer(heard, candidate, weighted):
    from pfer import normalize, feature_distance
    a = [normalize(t) for t in heard]; b = [normalize(t) for t in candidate]
    if not a: return 0.0
    if not b: return 1.0
    m = len(b); prev = list(range(m + 1))
    for ph in a:
        cur = [prev[0] + 1] + [0.0] * m
        for j in range(1, m + 1):
            sub = feature_distance(ph, b[j-1]) if weighted else (0.0 if ph == b[j-1] else 1.0)
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + sub)
        prev = cur
    return prev[-1] / max(len(a), len(b))   # normalize by longer side (symmetric)

async def metrics(path, ctx):
    ctx0 = re.sub(r"^\s*튜터:\s*", "", ctx).strip()
    audio = open(path, "rb").read(); fn = path.split("/")[-1]
    en = (await A.backend_decode(audio, fn, ctx0, "English")).get("text", "").strip()
    g = await A.guard(audio, fn, en_text=en, ko_text="x")
    au, ei = g.get("audio_ipa") or "", g.get("en_ipa") or ""
    taud, ten = tokenize_ipa(au), tokenize_ipa(ei)
    return {
        "en": en,
        "w_prefix": round(prefix_pfer(taud, ten, True), 3),   # current en_pfer_w
        "w_full":   round(full_pfer(taud, ten, True), 3),     # full-length weighted
        "u_prefix": round(prefix_pfer(taud, ten, False), 3),
        "u_full":   round(full_pfer(taud, ten, False), 3),
        "folded":   round(A.folded_per(ei, au) or 0, 3),
    }

# label sets
TRANS = [  # Korean audio, force-EN mistranslates -> SHOULD flag
    ("/tmp/caseR1.wav", "직원 교육 프로그램. a staff training program."),
    ("/tmp/caseR2.wav", "새로운 시스템. a new system."),
    ("/tmp/caseR3.wav", "진전을 이루다, make a progress"),
]

async def main():
    bench = json.load(open("/tmp/benchmark50.json"))
    honest = []
    for b in bench:
        fn = b["audio_url"].split("/")[-1]
        honest.append((f"/tmp/b50_{fn}", b.get("context") or "-"))

    print("=== TRANSLATIONS (should flag high) ===")
    tvals = {k: [] for k in ("w_prefix", "w_full", "u_prefix", "u_full", "folded")}
    for p, c in TRANS:
        m = await metrics(p, c)
        for k in tvals: tvals[k].append(m[k])
        print(f"  {m['en'][:26]:28} wP={m['w_prefix']} wF={m['w_full']} uP={m['u_prefix']} uF={m['u_full']} fold={m['folded']}")

    print("\n=== HONEST ENGLISH (should stay low) ===")
    hvals = {k: [] for k in tvals}
    import os
    n = 0
    for p, c in honest:
        if not os.path.exists(p) or n >= 12: continue
        m = await metrics(p, c)
        if len(m["en"].split()) < 3: continue   # ≥3-word population only
        n += 1
        for k in hvals: hvals[k].append(m[k])
        print(f"  {m['en'][:30]:32} wP={m['w_prefix']} wF={m['w_full']} uP={m['u_prefix']} uF={m['u_full']} fold={m['folded']}")

    print("\n=== SEPARATION (trans min vs honest max — bigger gap = cleaner single-threshold) ===")
    for k in ("w_prefix", "w_full", "u_prefix", "u_full", "folded"):
        tmin = min(tvals[k]); hmax = max(hvals[k]) if hvals[k] else 0
        gap = tmin - hmax
        print(f"  {k:9} trans[{min(tvals[k])}-{max(tvals[k])}]  honest[{min(hvals[k]) if hvals[k] else '-'}-{hmax}]  gap={round(gap,3)}  {'CLEAN' if gap>0 else 'OVERLAP'}")

asyncio.run(main())

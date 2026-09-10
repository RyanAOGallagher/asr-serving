import asyncio, json
import anchor_service as A

cases = [
    (1, "직원 교육 프로그램. a staff training program.", "직원 교육 프로그램"),
    (2, "새로운 시스템. a new system.", "새로운 시스템"),
    (3, "진전을 이루다, make a progress", "진전을 이루다"),
]

async def diag(n, ctx, ko_truth):
    audio = open(f"/tmp/caseR{n}.wav", "rb").read()
    fn = f"caseR{n}.wav"
    import re
    ctx0 = re.sub(r"^\s*튜터:\s*", "", ctx).strip()
    en = (await A.backend_decode(audio, fn, ctx0, "English")).get("text", "").strip()
    ko = (await A.backend_decode(audio, fn, ctx0, "Korean")).get("text", "").strip()
    ko_nc = (await A.backend_decode(audio, fn, "", "Korean")).get("text", "").strip()
    auto = (await A.backend_decode(audio, fn, ctx0, None)).get("text", "").strip()
    g = await A.guard(audio, fn, en_text=en, ko_text=ko)
    au = g.get("audio_ipa")
    fe = A.folded_per(g.get("en_ipa"), au)
    fk = A.folded_per(g.get("ko_ipa"), au)
    has_h = bool(A.HANGUL.search(ko))
    nw = len(en.split())
    short_ok = nw >= 4 or (fk is not None and fk < A.FOLDED_SHORT_SUPPORT)
    would_swap = (fe is not None and fk is not None and fk < fe - A.FOLDED_MARGIN and has_h and short_ok)
    print(f"CASE {n}  ko_truth={ko_truth!r}")
    print(f"  force-EN   : {en!r}  (nwords={nw})")
    print(f"  force-KO   : {ko!r}  hasHangul={has_h}")
    print(f"  KO no-ctx  : {ko_nc!r}")
    print(f"  auto       : {auto!r}")
    print(f"  folded  fe(EN)={fe}  fk(KO)={fk}   fk<fe? {None if (fe is None or fk is None) else fk<fe}")
    print(f"  en_pfer_w={g.get('en_pfer_w')}  ko_pfer_w={g.get('ko_pfer_w')}")
    print(f"  short_ok={short_ok} (need nw>=4 or fk<{A.FOLDED_SHORT_SUPPORT})  -> WOULD SWAP TO KO: {would_swap}")
    print(f"  audio_ipa={au}")
    print(f"  en_ipa   ={g.get('en_ipa')}")
    print(f"  ko_ipa   ={g.get('ko_ipa')}")
    print()

async def main():
    A._load_lm  # noqa
    for n, ctx, kt in cases:
        await diag(n, ctx, kt)

asyncio.run(main())

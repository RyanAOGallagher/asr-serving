import asyncio, json, re
import anchor_service_trial as A

rows = {r["file"]: r for r in json.load(open("/tmp/h2h_rows.json"))}
targets = [
    ("말씀 (BAD frag)", "39078b7f376a403082f8bb41a14bb0cf.wav"),
    ("낯가렸 (BAD frag)", "f9ec09f3f6c04054b3e9ca75338adac8.wav"),
    ("낯가렸b (BAD frag)", "e50d72738a9a470eb236936a343e9a45.wav"),
    ("오두시 (GOOD?)", "f70f22250ff64048bd3ca79a47b7e9b6.wav"),
]

async def one(label, fn):
    r = rows.get(fn, {})
    ctx = r.get("context", "")
    ctx0 = re.sub(r"^\s*튜터:\s*", "", ctx).strip()
    audio = open(f"/tmp/b400/{fn}", "rb").read()
    en = (await A.backend_decode(audio, fn, ctx0, "English")).get("text", "").strip()
    ko = (await A.backend_decode(audio, fn, ctx0, "Korean")).get("text", "").strip()
    g = await A.guard(audio, fn, en_text=en, ko_text=ko)
    au = g.get("audio_ipa")
    fe = A.folded_per(g.get("en_ipa"), au)
    fk = A.folded_per(g.get("ko_ipa"), au)
    # is KO candidate a verbatim context fragment?
    frag = A.ctx_echo(ko, ctx0) if hasattr(A, "ctx_echo") else None
    print(f"{label}")
    print(f"   ctx={ctx0[:50]!r}")
    print(f"   EN={en!r}  KO={ko!r}")
    print(f"   fe={fe:.3f} fk={fk:.3f}  en_pfer_w={g.get('en_pfer_w')}  ko_pfer_w={g.get('ko_pfer_w')}  ctx_echo(ko)={frag}")
    print()

async def main():
    for label, fn in targets:
        await one(label, fn)
    print("--- GOOD loanword cases (from earlier diag): ko_pfer_w = case1 0.263, case2 0.236, case3 0.297 ---")

asyncio.run(main())

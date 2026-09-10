#!/usr/bin/env python3
"""One-off: copy weavers_tts.stt_generation_log (Supabase) -> ES index stt_generation_log."""
import json
import sys
import urllib.request
import base64
from pathlib import Path

HERE = Path(__file__).parent
env = dict(
    line.strip().split("=", 1)
    for line in (Path("/home/ailab/ASR/Qwen3ASR/.env")).read_text().splitlines()
    if "=" in line
)
SB_URL = env["SUPABASE_URL"].strip()
SB_KEY = env["SUPABASE_ANON_KEY"].strip()

ES_URL = "https://es.weaversmind.com"
ES_AUTH = base64.b64encode(b"ailab:weavers!AI").decode()
INDEX = "stt_generation_log"
PAGE = 1000

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def sb_page(after_id):
    url = (f"{SB_URL}/rest/v1/stt_generation_log?select=*"
           f"&id=gt.{after_id}&order=id.asc&limit={PAGE}")
    req = urllib.request.Request(url, headers={
        "apikey": SB_KEY,
        "Authorization": f"Bearer {SB_KEY}",
        "Accept-Profile": "weavers_tts",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry after {e}", flush=True)


def es_max_id():
    req = urllib.request.Request(
        f"{ES_URL}/{INDEX}/_search",
        data=json.dumps({"size": 0, "aggs": {"m": {"max": {"field": "id"}}}}).encode(),
        headers={"Authorization": f"Basic {ES_AUTH}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        v = json.loads(r.read())["aggregations"]["m"]["value"]
    return int(v) if v else 0


def es_bulk(rows):
    lines = []
    for row in rows:
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": row["id"]}}))
        lines.append(json.dumps(row, ensure_ascii=False))
    body = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(
        f"{ES_URL}/_bulk",
        data=body,
        headers={
            "Authorization": f"Basic {ES_AUTH}",
            "Content-Type": "application/x-ndjson",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        resp = json.loads(r.read())
    if resp.get("errors"):
        errs = [i["index"]["error"] for i in resp["items"]
                if i["index"].get("error")][:3]
        print("BULK ERRORS (first 3):", json.dumps(errs, indent=2))
        sys.exit(1)


last_id = es_max_id()
print(f"resuming after id {last_id}")
total = 0
while True:
    rows = sb_page(last_id)
    if not rows:
        break
    es_bulk(rows)
    last_id = rows[-1]["id"]
    total += len(rows)
    print(f"  {total} docs (last id {last_id})", flush=True)

print(f"DONE: {total} docs migrated")

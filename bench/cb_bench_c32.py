#!/usr/bin/env python3
"""Benchmark stock qwen-asr-serve (vLLM continuous batching) on the filtered test set.

Prereq: ssh localasr '~/ASR/Qwen3ASR-streaming/cb_swap.sh start'  (takes prod down!)
Usage:  python3 scripts/cb_bench.py
After:  ssh localasr '~/ASR/Qwen3ASR-streaming/cb_swap.sh stop'
Writes: dataset/cb_results.json
"""
import base64, json, os, statistics, time
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = 'http://127.0.0.1:8021/v1/chat/completions'
LEVELS = [32, 64]

rows = json.load(open('/workspace/qwenasr_bench/stress_test_500.json'))
clips = []
for s in rows:
    p = f"/workspace/qwenasr_bench/stress_clips/{s['audio_url'].split('/')[-1]}"
    if os.path.exists(p):
        mime = 'audio/mpeg' if p.endswith('.mp3') else 'audio/wav'
        clips.append((s['id'], f"data:{mime};base64,{base64.b64encode(open(p, 'rb').read()).decode()}"))
print(f'{len(clips)} clips loaded')


def one(c):
    cid, uri = c
    t0 = time.perf_counter()
    try:
        r = requests.post(URL, json={'messages': [{'role': 'user', 'content': [
            {'type': 'audio_url', 'audio_url': {'url': uri}}]}]}, timeout=300)
        ok = r.status_code == 200
        txt = r.json()['choices'][0]['message']['content'] if ok else None
        return {'id': cid, 'ms': (time.perf_counter() - t0) * 1000, 'status': r.status_code, 'raw': txt}
    except Exception as e:
        return {'id': cid, 'ms': (time.perf_counter() - t0) * 1000, 'status': 0, 'error': str(e)[:100]}


out = {}
for lvl in LEVELS:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=lvl) as ex:
        res = list(ex.map(one, clips))
    wall = time.perf_counter() - t0
    lat = sorted(r['ms'] for r in res if r['status'] == 200)
    stats = {'wall_s': round(wall, 1), 'rps': round(len(lat) / wall, 2), 'n_ok': len(lat),
             'n_err': len(res) - len(lat), 'p50_ms': round(lat[len(lat) // 2]),
             'p95_ms': round(lat[int(len(lat) * .95)]), 'mean_ms': round(statistics.mean(lat))} if lat else {'n_err': len(res)}
    out[f'c{lvl}'] = {'stats': stats, 'results': res}
    print(f'c{lvl}: {stats}', flush=True)

json.dump(out, open('/workspace/qwenasr_bench/cb_results_5090.json', 'w'), ensure_ascii=False)
print('saved -> dataset/cb_results.json')

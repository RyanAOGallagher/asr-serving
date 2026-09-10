import json, os, re, statistics, time
from concurrent.futures import ThreadPoolExecutor
import requests

B = '/workspace/qwenasr_bench'
rows = json.load(open(f'{B}/stress_test_500.json'))
strip = lambda c: re.sub(r'튜터\s*:\s*', '', str(c or '')).strip()
clips = [(s, f"{B}/stress_clips/{s['audio_url'].split('/')[-1]}") for s in rows]
clips = [(s, p) for s, p in clips if os.path.exists(p)]
print(len(clips), 'clips')

def one(sp):
    s, p = sp
    t0 = time.perf_counter()
    with open(p, 'rb') as f:
        r = requests.post('http://127.0.0.1:8006/transcribe',
                          data={'context': strip(s.get('context')) or '-', 'language': 'English'},
                          files={'file': (os.path.basename(p), f)}, timeout=300)
    return (time.perf_counter() - t0) * 1000, r.status_code

out = {}
for lvl in (4, 8, 16):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=lvl) as ex:
        res = list(ex.map(one, clips))
    wall = time.perf_counter() - t0
    lat = sorted(ms for ms, st in res if st == 200)
    stats = {'wall_s': round(wall, 1), 'rps': round(len(lat) / wall, 2), 'n_ok': len(lat),
             'n_err': len(res) - len(lat), 'p50_ms': round(lat[len(lat) // 2]),
             'p95_ms': round(lat[int(len(lat) * .95)]), 'mean_ms': round(statistics.mean(lat))}
    out[f'c{lvl}'] = stats
    print(f'c{lvl}:', stats, flush=True)
json.dump(out, open(f'{B}/merged_results_5090.json', 'w'))

import json, os, re, statistics, time, collections, sys
from concurrent.futures import ThreadPoolExecutor
import requests

B = '/workspace/qwenasr_bench'
rows = json.load(open(f'{B}/stress_test_500.json'))
strip = lambda c: re.sub(r'튜터\s*:\s*', '', str(c or '')).strip()
clips = [(s, f"{B}/stress_clips/{s['audio_url'].split('/')[-1]}") for s in rows]
clips = [(s, p) for s, p in clips if os.path.exists(p)]

def one(sp):
    s, p = sp
    data = {'context': strip(s.get('context')) or '-', 'origin': 'shim-bench'}
    if (s.get('expected') or '').strip(): data['expected'] = s['expected'].strip()
    t0 = time.perf_counter()
    try:
        with open(p, 'rb') as f:
            r = requests.post('http://127.0.0.1:8004/transcribe_anchor', data=data,
                              files={'file': (os.path.basename(p), f)}, timeout=300)
        j = r.json() if r.status_code == 200 else {}
        return {'ms': (time.perf_counter()-t0)*1000, 'st': r.status_code, 'path': j.get('path'), 'srv': j.get('latency_ms'), 'text': j.get('text')}
    except Exception as e:
        return {'ms': (time.perf_counter()-t0)*1000, 'st': 0, 'err': str(e)[:80]}

if sys.argv[1:] == ['smoke']:
    for s, p in clips[:4]:
        r = one((s, p))
        print(r['st'], round(r['ms']), r.get('path'), repr((r.get('text') or '')[:50]))
    raise SystemExit

out = {}
for lvl in (4, 8, 16):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=lvl) as ex:
        res = list(ex.map(one, clips))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r['st'] == 200]
    lat = sorted(r['ms'] for r in ok)
    stats = {'wall_s': round(wall,1), 'rps': round(len(ok)/wall,2), 'n_ok': len(ok), 'n_err': len(res)-len(ok),
             'p50_ms': round(lat[len(lat)//2]), 'p95_ms': round(lat[int(len(lat)*.95)]), 'mean_ms': round(statistics.mean(lat))}
    out[f'c{lvl}'] = stats
    print(f'c{lvl}:', stats, flush=True)
    by = collections.defaultdict(list)
    for r in ok: by[r.get('path') or '?'].append(r['ms'])
    for k, v in sorted(by.items(), key=lambda x: -statistics.mean(x[1]))[:4]:
        print(f'   {k:24s} n={len(v):3d} mean={statistics.mean(v):5.0f}ms', flush=True)
json.dump(out, open(f'{B}/anchor_shim_results.json','w'))

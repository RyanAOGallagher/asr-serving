#!/usr/bin/env python3
"""Test the merged ASR stack: POST /transcribe and WS /stream, with latency.

Usage:
  ./.venv/bin/python asr_latency_test.py --audio /tmp/cs2.wav
  ./.venv/bin/python asr_latency_test.py --base http://127.0.0.1:8006   # hit backend directly
  ./.venv/bin/python asr_latency_test.py --realtime                     # pace chunks like a live mic
"""
import argparse, asyncio, json, time, statistics, sys

import numpy as np
import soundfile as sf
import httpx
import websockets


def load_audio_16k(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torch, torchaudio
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    return data.astype("<f4")


def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def test_transcribe(base, path, timestamps):
    url = base.rstrip("/") + "/transcribe"
    with open(path, "rb") as f:
        audio = f.read()
    t0 = time.perf_counter()
    r = httpx.post(url, files={"file": (path.split("/")[-1], audio)},
                   data={"timestamps": str(timestamps).lower()}, timeout=600)
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    print("\n=== POST /transcribe ===")
    print(f"  latency        : {dt:8.1f} ms  (request -> full response)")
    print(f"  language       : {d.get('language')}")
    print(f"  timestamps segs: {len(d.get('timestamps', []))}")
    print(f"  text           : {d.get('text','')[:100]}")
    return dt


async def test_stream(base, audio, chunk_ms, realtime):
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/stream"
    step = int(16000 * chunk_ms / 1000)
    audio_dur = len(audio) / 16000
    rtts, first_partial_ms, sent_samples = [], None, 0
    t_start = time.perf_counter()
    async with websockets.connect(ws_url, max_size=None) as ws:
        for i in range(0, len(audio), step):
            if realtime and i > 0:
                await asyncio.sleep(chunk_ms / 1000)
            chunk = audio[i:i + step].tobytes()
            t0 = time.perf_counter()
            await ws.send(chunk)
            msg = json.loads(await ws.recv())
            rtt = (time.perf_counter() - t0) * 1000
            rtts.append(rtt)
            sent_samples += len(audio[i:i + step])
            if first_partial_ms is None:
                first_partial_ms = (time.perf_counter() - t_start) * 1000
        t_eos = time.perf_counter()
        await ws.send(json.dumps({"type": "eos"}))
        final = json.loads(await ws.recv())
        eos_ms = (time.perf_counter() - t_eos) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000
    print("\n=== WS /stream ===")
    print(f"  mode             : {'realtime-paced' if realtime else 'throughput (chunks sent back-to-back)'}")
    print(f"  audio duration   : {audio_dur:8.2f} s   chunk={chunk_ms}ms  chunks={len(rtts)}")
    print(f"  time-to-first    : {first_partial_ms:8.1f} ms  (1st chunk sent -> 1st partial)")
    print(f"  per-chunk RTT    : mean {statistics.mean(rtts):6.1f}  median {statistics.median(rtts):6.1f}  p95 {pct(rtts,95):6.1f} ms")
    print(f"  eos -> final     : {eos_ms:8.1f} ms")
    print(f"  total wall       : {total_ms:8.1f} ms   RTF={total_ms/1000/audio_dur:.3f} (lower=faster than realtime)")
    print(f"  final text       : {final.get('text','')[:100]}")
    return total_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002", help="wrapper(8002) or backend(8006)")
    ap.add_argument("--audio", default="/tmp/cs2.wav")
    ap.add_argument("--chunk-ms", type=int, default=1000)
    ap.add_argument("--timestamps", action="store_true", default=True)
    ap.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    ap.add_argument("--realtime", action="store_true", help="pace chunks to wall-clock like a live mic")
    ap.add_argument("--only", choices=["transcribe", "stream"], help="run just one")
    args = ap.parse_args()

    print(f"target base: {args.base}   audio: {args.audio}")
    audio = load_audio_16k(args.audio)
    if args.only != "stream":
        test_transcribe(args.base, args.audio, args.timestamps)
    if args.only != "transcribe":
        asyncio.run(test_stream(args.base, audio, args.chunk_ms, args.realtime))


if __name__ == "__main__":
    main()

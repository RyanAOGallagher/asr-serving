import asyncio, json, numpy as np, soundfile as sf, websockets

WAV = "/home/ailab/ASR/Qwen3ASR/audio/long/out-01076649220-7609-20240801-100700-1722474420.3562.wav"
d, sr = sf.read(WAV, dtype="float32")
if d.ndim > 1:
    d = d[:, 0]
if sr != 16000:
    n = int(len(d) * 16000 / sr)
    d = np.interp(np.linspace(0, len(d) - 1, n), np.arange(len(d)), d).astype("float32")
d = d[:16000 * 30]  # first 30s

URL = "ws://127.0.0.1:8002/stream?endpoint=vad&language=Korean&silence_ms=700"


async def main():
    finals = []
    npart = 0
    async with websockets.connect(URL, max_size=None) as ws:
        FR = 3200  # 200ms frames like mic.html

        async def sender():
            for i in range(0, len(d), FR):
                await ws.send(d[i:i + FR].tobytes())
            await ws.send(json.dumps({"type": "eos"}))

        task = asyncio.create_task(sender())
        while True:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 60))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            t = m.get("type")
            if t == "partial":
                npart += 1
            elif t == "final":
                finals.append(m.get("text", ""))
                lang = m.get("language")
                txt = (m.get("text", "") or "")[:80]
                print(f"[FINAL #{len(finals)}] ({lang}) {txt}")
        await task
    print(f"--- partials={npart}, finals(utterances)={len(finals)} ---")


asyncio.run(main())

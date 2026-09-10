#!/usr/bin/env python3
"""Add optional VAD endpointing (continuous mode) to server_merged.py's /stream."""
import time, sys

P = "server_merged.py"
s = open(P, encoding="utf-8").read()

# 1) import VADIterator
imp_old = "from silero_vad import load_silero_vad, get_speech_timestamps"
imp_new = "from silero_vad import load_silero_vad, get_speech_timestamps, VADIterator"
assert s.count(imp_old) == 1, f"import anchor count={s.count(imp_old)}"
s = s.replace(imp_old, imp_new)

# 2) replace the whole ws_stream function
start_marker = '@app.websocket("/stream")'
end_marker = '@app.get("/health")'
i = s.index(start_marker)
j = s.index(end_marker)
assert i != -1 and j != -1 and j > i, "function markers not found"

new_fn = '''@app.websocket("/stream")
async def ws_stream(ws: WebSocket):
    """WebSocket streaming: client sends binary float32 16k mono PCM chunks, then
    a text frame {"type":"eos"}. Server replies {"type":"partial"|"final","text","language"}.

    Optional VAD endpointing: ?endpoint=vad[&silence_ms=700] -> when end-of-speech is
    detected the current utterance is finalized ({"type":"final"}) and a fresh streaming
    state is started; the socket stays open for the next utterance (continuous mode).
    Without the param, behaviour is unchanged (finalize only on client {"type":"eos"}).
    Same instance / same _model_lock as /transcribe."""
    await ws.accept()
    loop = asyncio.get_event_loop()
    req_lang = ws.query_params.get("language") or None
    if req_lang is not None and req_lang not in model.get_supported_languages():
        await ws.send_json({"type": "error",
                            "detail": f"Unsupported language: {req_lang}"})
        await ws.close()
        return

    use_vad = (ws.query_params.get("endpoint") or "").lower() == "vad"
    try:
        silence_ms = int(ws.query_params.get("silence_ms") or 700)
    except (TypeError, ValueError):
        silence_ms = 700

    def _new_state():
        return model.init_streaming_state(
            language=req_lang,
            unfixed_chunk_num=ST_UNFIXED_CHUNK_NUM,
            unfixed_token_num=ST_UNFIXED_TOKEN_NUM,
            chunk_size_sec=ST_CHUNK_SIZE_SEC)

    state = await loop.run_in_executor(None, _new_state)

    vad_iter = None
    vad_buf = np.zeros((0,), dtype=np.float32)
    speech_seen = False
    if use_vad:
        vad_iter = VADIterator(vad_model, sampling_rate=SAMPLE_RATE,
                               min_silence_duration_ms=silence_ms)
    VAD_WIN = 512  # silero requires 512-sample windows at 16k

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                raw = msg["bytes"]
                if len(raw) % 4 != 0:
                    await ws.send_json({"type": "error", "detail": "float32 bytes not multiple of 4"})
                    continue
                wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

                def _step():
                    with _model_lock:
                        model.streaming_transcribe(wav, state)
                    return _state_payload(state)

                await ws.send_json({"type": "partial", **(await loop.run_in_executor(None, _step))})

                if vad_iter is not None:
                    vad_buf = np.concatenate([vad_buf, wav])
                    endpointed = False
                    while vad_buf.shape[0] >= VAD_WIN:
                        win = vad_buf[:VAD_WIN]
                        vad_buf = vad_buf[VAD_WIN:]
                        ev = vad_iter(torch.from_numpy(win), return_seconds=True)
                        if ev:
                            if "start" in ev:
                                speech_seen = True
                            elif "end" in ev and speech_seen:
                                endpointed = True
                                break
                    if endpointed:
                        def _finish_cur():
                            with _model_lock:
                                model.finish_streaming_transcribe(state)
                            return _state_payload(state)
                        payload = await loop.run_in_executor(None, _finish_cur)
                        if (payload.get("text") or "").strip():
                            await ws.send_json({"type": "final", **payload})
                        # start a fresh utterance; keep the socket open
                        state = await loop.run_in_executor(None, _new_state)
                        vad_iter.reset_states()
                        vad_buf = np.zeros((0,), dtype=np.float32)
                        speech_seen = False
            elif msg.get("text") is not None:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") == "eos":
                    def _finish():
                        with _model_lock:
                            model.finish_streaming_transcribe(state)
                        return _state_payload(state)

                    await ws.send_json({"type": "final", **(await loop.run_in_executor(None, _finish))})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


'''

s = s[:i] + new_fn + s[j:]

bak = f"{P}.bak.vadendpoint.{int(time.time())}"
# back up the ORIGINAL from disk (s is already mutated in memory), then write.
import shutil
shutil.copyfile(P, bak)
open(P, "w", encoding="utf-8").write(s)
print("patched. backup:", bak)

# Qwen3-ASR Streaming (isolated, deferred launch)

Isolated from production batched deployment in `~/ASR/Qwen3ASR` (ports 8002 wrapper / 8003 model).
Nothing here autostarts. Production is untouched.

## What this is
Fresh clone of github.com/QwenLM/Qwen3-ASR (HEAD c17a131). The repo ships a real
**streaming** server, vLLM-backend only:
- console script `qwen-asr-demo-streaming` -> `qwen_asr/cli/demo_streaming.py`
- HTTP API: `POST /api/start` -> `POST /api/chunk` (float32 16kHz PCM, octet-stream) -> `POST /api/finish`
- wraps `init_streaming_state` / `streaming_transcribe` / `finish_streaming_transcribe`
- `GET /` serves a browser mic demo UI
- streaming output is TEXT-ONLY: no word timestamps, forced aligner NOT used
  (alignment needs the whole segment; mutually exclusive with streaming).

## Run (DEFERRED -- needs GPU headroom)
The 3090 Ti is ~95% used by the batched service, so this CANNOT run alongside it
as-is. `run_streaming.sh` preflight-checks free VRAM (>= 6000 MiB) and refuses
otherwise, to protect production.

To actually launch, pick ONE:
  A) Free this GPU: stop batched (`supervisorctl -c ~/ASR/Qwen3ASR/supervisord.conf stop all`)
     or lower its `gpu_memory_utilization` (server_batched.py, 0.45 -> lower) and restart it.
  B) Run on another GPU host (dev RunPod 2x5090 / B200) -- copy this dir, same script.

Then:
  ./run_streaming.sh                 # port 8005, gpu_mem 0.30, chunk 1.0s
  # tunables via env: PORT, GPU_MEM_UTIL, CHUNK_SIZE_SEC, UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM

Open http://<host>:8005/ for the mic demo, or drive /api/start|chunk|finish directly.

## Streaming params
- chunk_size_sec: audio fed per step (lower = lower latency, more GPU calls)
- unfixed_chunk_num / unfixed_token_num: how many recent tokens stay tentative
  (rolled back + re-decoded) before being committed -- the stabilization window.

## Optional: run under supervisor
`supervisord.streaming.conf` is provided with autostart=false and its own sock/pidfile
(separate from production). It will not start anything until you explicitly do so.

## Word-level Viterbi forced aligner (word_aligner/)
Brought from research(runpod) /tmp/tts_stack_bundle/aligners/word__stream_align_server.py.
- Model: KoEn fine-tuned XLS-R char-CTC (align_xlsr_best, 1.2G). vocab=410: Latin a-z +
  62 Korean conjoining jamo (U+1100) -> genuinely bilingual. Hand-rolled online Viterbi
  over CTC log-probs (NOT torchaudio). Word level only.
- WS server /stream: client sends {"type":"text",...} then PCM16 mono chunks then {"type":"eos"};
  server returns {"type":"audio",...,"words":[{word,startS,endS}]} as words settle. Text-first
  (forced alignment, transcript known up front).
- Run (deferred, VRAM guard >=2500MiB): ./word_aligner/run_word_aligner.sh   (port 8009)
  env: PORT, ALIGN_DEVICE (cuda|cpu), ALIGN_CKPT, CUDA_VISIBLE_DEVICES, MIN_FREE_MIB.
  CPU mode works (validated): ALIGN_DEVICE=cpu ./word_aligner/run_word_aligner.sh
- Patched one line: DEV reads ALIGN_DEVICE env (was hardcoded "cuda"); original saved as
  word_align_server.py.orig.
- Validated on localasr (CPU): load 6.3s, tokenized "hello 안녕하세요 world", Viterbi emitted
  word-level spans for all 3 incl. the Korean word.

## Merged server (server_merged.py) -- ONE Qwen instance, both paths
New file, does not touch production or the other scripts. Built WITH forced_aligner.
- POST /transcribe (file): short/long multipath ported from prod server_batched.py
  (short <=90s -> micro-batch queue; long -> VAD ~60s chunks -> batch -> merge w/ offset).
  timestamps=true -> Qwen forced-aligned word timestamps; timestamps=false -> text only.
  NON-streaming. (Forced-align window is 180s, ASR window 1200s -> long audio chunked.)
- POST /api/start -> /api/chunk (float32 16k PCM) -> /api/finish: streaming text, NO
  timestamps, same model instance (streaming_transcribe ignores the aligner).
- One _model_lock serializes ALL model calls (transcribe + streaming) on the single
  vLLM engine: correct but a long transcribe will briefly stall streaming chunks (single GPU).
- Differs from prod: no Mattermost webhook, local logs (./logs), env-configurable
  (GPU_MEM_UTIL, MAX_NEW_TOKENS=1024, ST_CHUNK_SIZE_SEC/UNFIXED_*). Port 8006.
- Deps added to venv: silero-vad. Run: ./run_merged.sh (VRAM guard >=12000MiB). DEFERRED.
- Validated: py_compile OK; imports clean (model load deferred to lifespan); routes present.

## Wrapper stack (wrapper_merged.py on 8002 + server_merged.py on 8006)
New files; prod stt_wrapper.py + server_batched.py untouched.
- 8002 wrapper_merged.py (no GPU): POST /transcribe -> proxies ASR_BACKEND_URL (default :8006)
  + ElevenLabs fallback + Supabase/CDN logging (loads /home/ailab/ASR/Qwen3ASR/.env).
  + WS /stream -> transparent proxy to backend ws /stream; thin passthrough, no EL fallback,
    best-effort final-transcript DB log. Run: ./run_wrapper.sh (PORT/ASR_BACKEND_URL envs).
- 8006 server_merged.py (GPU): one Qwen instance. /transcribe (short/long multipath + optional
  forced-align timestamps), /api/start|chunk|finish, WS /stream (binary f32 16k chunks + {type:eos}).
- Verified end-to-end via 8002 with cs2.wav: /transcribe text+timestamps; WS /stream partials+final.
- NOTE: 8002 is also prod stt_wrapper port -> they cannot run at once. Prod Qwen3-ASR currently STOPPED.
  Revert to prod: pkill wrapper_merged + server_merged, then supervisorctl -c ~/ASR/Qwen3ASR/supervisord.conf start all.

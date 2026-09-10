# ASR serving + interactive-latency fix — reference notes

_Last updated: 2026-07-03. Lives at `/home/ailab/ASR/Qwen3ASR-streaming/INTERACTIVE_LATENCY_NOTES.md`._

## 1. Serving chain (how a request reaches the model)

```
edge fn / client
      │  POST https://asr.ryangallagher.dev/stt   (multipart: file, language, timestamps)
      │  header: X-Proxy-Key: <secret in /etc/asr-proxy.env>
      ▼
Cloudflare  ──(named tunnel "asr", systemd: asr-tunnel.service)──▶ gpu-225
      ▼
asr-proxy.service  (Flask, 127.0.0.1:6767, code in /home/ailab/asr-proxy/)
      │  - checks X-Proxy-Key (401 if wrong)
      │  - transcodes m4a/aac/webm/etc -> 16k wav via ffmpeg (libsndfile can't read those);
      │    wav/opus/flac/mp3 pass straight through (fast path)
      ▼
wrapper_merged  (uvicorn, :8002)  - forwards to backend; falls back to ElevenLabs if backend down
      ▼
server_merged   (uvicorn, :8006)  - THE MODEL: vLLM (Qwen3-ASR-1.7B) + Qwen3-ForcedAligner-0.6B
                                    + streaming.  All inference serialized by one _model_lock.
```

- Secret (X-Proxy-Key / PROXY_SECRET) lives ONLY in `/etc/asr-proxy.env` (root, chmod 600). Not in code.
- `asr-proxy.service` and `asr-tunnel.service` are systemd, enabled, survive reboot.
- **`server_merged` (:8006) is NOT systemd** — it's a manual daemon started by `run_merged.sh`. See restart section.
- LAN clients can skip the whole public chain: `POST http://192.168.1.225:8002/transcribe` (~0.5s), or `:8006` directly.
- **Always send Opus** — a 1 MB wav → ~32 KB opus, ~2× faster over the tunnel, identical transcript.

## 2. The interactive-latency fix (2026-07-03)

**Problem:** an hourly cron (~:00–:10) submits long transcription jobs. They held the model lock in
groups of 8 chunks, so interactive short clips waited **20–33 s** behind them.

**Root cause:** a long job of ≤8 chunks = one un-yielded group → a short clip waits the *whole* job;
multiple queued long jobs stacked → 30 s. The GPU op in flight can't be interrupted.

**Fix (in `server_merged.py` + `run_merged.sh`):**
1. `LONG_LOCK_GROUP=1` (set in `run_merged.sh`) — long jobs release the lock after every chunk.
2. `_long_job_sem = asyncio.Semaphore(1)` around the long-audio branch — serializes long jobs so their
   VAD/resample/chunk-writing can't thundering-herd and starve short clips.
3. `_short_pending` is bumped in the `/transcribe` handler the instant a short clip arrives (moved out
   of `_transcribe_batch`) so a running long job yields sooner.
4. Long-audio loop (`_locked_transcribe`) defers to a waiting short clip BEFORE every group, incl. the first.

**Measured results (short clip = 2s audio, ~0.15s pure compute; rest is lock wait):**

| Situation                              | Before   | After        |
|----------------------------------------|----------|--------------|
| Idle short clip                        | ~0.5 s   | ~0.11 s      |
| Short clip during heavy long-job load  | 20–33 s  | ~0.8–1.6 s   |
| First clip of a cold burst             | ~28–33 s | ~6 s         |
| Public edge path (contended)           | —        | ~1.2 s       |
| Timestamps (forced aligner)            | works    | still works  |
| Streaming (/api, /stream)              | works    | untouched    |

**Cost:** long jobs are now ~3× slower (33-min file: ~26 s → ~81 s). This was intentional — Ryan does
not care about long-job speed. WATCH: if the hourly cron's queue could back up or must finish within
:00–:10, 3× slower may make it overrun. If so, see Tuning.

**Backups of the pre-fix files:** `server_merged.py.bak.pre-lockfix`, `run_merged.sh.bak.pre-lockfix`.

## 3. Tuning (if long jobs overrun, or short clips need to be even faster)

- **Ease the long-job penalty:** set `LONG_LOCK_GROUP=2` or `3` in `run_merged.sh` (restart). Long-job
  penalty drops to ~1.5–2×; short clips rise to ~2–3 s. It's the balance dial.
- **Have both (adaptive):** make `_locked_transcribe` use group=8 when `_short_pending==0` and drop to 1
  only while a short clip waits → full long-job speed when idle, fast short clips during overlap. ~15 lines;
  not yet implemented.
- **Short clips ~0.15 s EVEN during long jobs:** requires rewriting the inference core to vLLM's
  `AsyncLLMEngine` (continuous batching). `qwen_asr` is local code using the offline vLLM `LLM` + a
  separate forced aligner + streaming state, so this is a **days-long rewrite** with regression risk to
  timestamps + streaming. Only worth it if ~1.5 s during the cron window is genuinely unacceptable.

## 4. Restarting server_merged (:8006) — DO THIS CAREFULLY

It's a manual daemon (PPID 1), not systemd. `run_merged.sh` has a VRAM guard (refuses to start if
< 12000 MiB free).

1. **Only restart OUTSIDE the :00–:10 cron window** (reload takes ~1–2 min; racing the cron for GPU
   caused a ~15-min outage on 2026-07-03).
2. **Kill uvicorn AND its vLLM child** — the `VLLM::EngineCore` child ORPHANS if you only kill uvicorn,
   keeps ~9 GB, and starves the relaunch:
   ```bash
   kill $(pgrep -f "uvicorn server_merged:app") $(pgrep -f "VLLM::EngineCore")
   sleep 6
   # verify: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
   # the ~2.4 GB `tm_stt` celery worker is SEPARATE — leave it. If a VLLM orphan survives, kill -9 it
   #   and confirm GPU frees to > 12 GB before relaunching.
   nvidia-smi --query-gpu=memory.free --format=csv,noheader   # want > 12000 MiB
   ```
3. **Relaunch (do NOT wrap in `timeout` — it'd kill mid-load and orphan vLLM):**
   ```bash
   cd /home/ailab/ASR/Qwen3ASR-streaming
   nohup setsid ./run_merged.sh > /tmp/merged.log 2>&1 </dev/null & disown
   ```
4. **Wait for ready:** `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8006/docs` == 200
   (~1–2 min). NOTE: the **first request after load takes ~7 s** (vLLM CUDA-graph warmup) — this is
   normal; the 2nd+ are ~0.11 s.
5. During the reload the backend is down, so `wrapper_merged` (:8002) auto-falls-back to ElevenLabs.

## 5. Rollback (undo the fix)

```bash
cd /home/ailab/ASR/Qwen3ASR-streaming
cp server_merged.py.bak.pre-lockfix server_merged.py
cp run_merged.sh.bak.pre-lockfix   run_merged.sh
# then restart per section 4
```

## 6. Troubleshooting crashes / timeouts

- **Is the model up?** `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8006/docs` (want 200).
  If not 200: check `pgrep -f "uvicorn server_merged:app"` and `tail -50 /tmp/merged*.log`.
- **Public endpoint 530 / timeouts:** check `asr-tunnel.service` and `asr-proxy.service`
  (`systemctl status asr-tunnel asr-proxy`).
- **A short clip took a long time:** check GPU — `nvidia-smi`. If a long job is mid-flight you'll see
  ~1–1.5 s (expected). If it's minutes, check for a **vLLM orphan** (two `VLLM::EngineCore` holding VRAM)
  or that the cron is overrunning past :10.
- **Timeouts during a restart window:** expected for ~1–2 min while the model reloads; requests fall back
  to ElevenLabs via the wrapper.
- **After any crash/timeout, useful to capture:** the exact time, `nvidia-smi`
  (util + `--query-compute-apps`), the last lines of `logs/requests.log`, and `/tmp/merged*.log`.

## Key facts / gotchas cheat-sheet
- server_merged :8006 = model (lock fix here). wrapper :8002 = fwd + ElevenLabs fallback. proxy :6767 =
  auth + ffmpeg transcode. tunnel = public.
- Restart :8006 only outside :00–:10, kill the vLLM child too, don't use `timeout`, expect a 7 s warmup.
- `tm_stt` celery worker (~2.4 GB) on the GPU is separate — don't kill it.
- Lock fix persists across restarts (it's in `run_merged.sh`).

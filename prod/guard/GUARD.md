# Translation Guard for `/transcribe_dual`

Detects when **force-English mistranslated** Korean audio, and suggests using the
Korean-forced output instead. Added 2026-07-09.

## What it does
`/transcribe_dual` (wrapper on `:8002`) force-decodes the same audio as English
(`language_a`) and Korean (`language_b`). The guard compares each output to what the
audio *actually sounds like* (phone recognition) and returns a `suggestion`:

- `"suggestion": "korean"`  → force-EN translated the audio; use `language_b` (Korean).
- `"suggestion": "english"` → keep the forced-English output (`language_a`). Default.

It is a **suggestion only** — the endpoint still returns both transcripts unchanged.

## How it decides
1. Audio → IPA via **ZIPA-large fp16** (CTC phone recognizer, `anyspeech/zipa-large-crctc-ns-800k`), on the GPU.
2. G2P each candidate text → IPA (Korean via `epitran`, English via `espeak-ng`, interleaved by script run).
3. **PER** (phone error rate) of each candidate vs the audio IPA.
4. `suggestion = "korean"` iff `PER(korean) < PER(english) - 0.10`, else `"english"`.
   (Identical EN/KO texts short-circuit to `"english"`.)

Rationale: a *translation* adds phones that were never spoken (e.g. "여행 가고 싶어요" →
"I want to travel"), so its PER blows up; a faithful transcription tracks the audio.
Validated: `travel` (translation) → PER EN 0.93 vs KO 0.43 → `korean`. Non-translations tie → `english`.

### Scope / known limits
- Reliable for the target case: **pure/near-pure Korean that force-EN randomly translates.**
- NOT a general code-switch language picker — on HiKE code-switch it's ~baseline (most
  differences are Latin-vs-Hangul *spelling* of identical-sounding loanwords, which phonetics
  can't decide). That's why it's a hint, not an override.

## Services & files (all on localasr / gpu-225)
| piece | where | port |
|---|---|---|
| wrapper (`/transcribe_dual`) | `~/ASR/Qwen3ASR-streaming/wrapper_merged.py` | 8002 |
| ASR backend (force decode) | `~/ASR/Qwen3ASR-streaming/server_merged.py` | 8006 |
| **guard service** | `~/zipa_test/guard_service.py` | 8010 |
| ZIPA model | `~/zipa_test/model.fp16.onnx` + `tokens.txt` | — |
| guard venv | `~/zipa_test/.venv` (onnxruntime-gpu, lhotse, epitran) | — |

## Start / restart
Guard service:
```bash
cd ~/zipa_test
# kill by port (do NOT pkill -f 'wrapper_merged' — it matches your own shell):
kill $(ss -tlnp | grep ':8010 ' | grep -oP 'pid=\K[0-9]+')
setsid nohup ./run_guard.sh > guard.log 2>&1 < /dev/null &   # run_guard.sh sets LD_LIBRARY_PATH for CUDA
```
Wrapper:
```bash
cd ~/ASR/Qwen3ASR-streaming
kill $(ss -tlnp | grep ':8002 ' | grep -oP 'pid=\K[0-9]+')
setsid nohup ./run_wrapper.sh > logs/wrapper.log 2>&1 < /dev/null &
```
Backend (`:8006`) needs ≥12 GB free VRAM; kill its uvicorn AND the orphaned `VLLM::EngineCore` first.

## ⚠️ Persistence
All run under `setsid nohup` (parent = init), no supervisor / no systemd / no auto-restart on crash.
- **guard (`:8010`)**: survives reboot via `@reboot` crontab (`crontab -l`) that nohup-launches `run_guard.sh`.
- **wrapper (`:8002`) + backend (`:8006`)**: do **NOT** survive reboot — manual relaunch needed (see Start/restart).
Nothing auto-restarts on a crash; add systemd units if that's needed later.

## Latency
Guard adds ~30–38 ms (warm) to the dual call. First call after a guard restart is ~1.2 s
(ONNX CUDA-graph warmup). Fail-safe: if `:8010` is down, `suggestion` is `null` and the
endpoint behaves as before.

## Test
```bash
curl -s -X POST http://192.168.1.225:8002/transcribe_dual \
  -F "file=@/path/to/audio.wav" \
  -F "language_a=English" -F "language_b=Korean" | python3 -m json.tool --no-ensure-ascii
```
Expect `"suggestion": "korean"` for Korean audio force-EN mistranslated, else `"english"`.

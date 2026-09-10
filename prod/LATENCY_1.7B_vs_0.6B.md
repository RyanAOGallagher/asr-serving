# Latency: Qwen3-ASR 1.7B vs 0.6B

Measured 2026-07-07 on gpu-225 (RTX 3090 Ti), vLLM 0.14, backend `:8006` direct
(`server_merged`), via `asr_latency_test.py`. Same clip both runs:
`/tmp/clip6.wav` (~6 s, Korean). Warm numbers (first request per model is a ~7.8 s
torch.compile cold-start and is excluded).

## Latency

| Metric                        | 1.7B    | 0.6B    | Δ      |
|-------------------------------|---------|---------|--------|
| Transcribe **w/ timestamps**  | 292 ms  | 194 ms  | −34%   |
| Transcribe **no timestamps**  | 271 ms  | 173 ms  | −36%   |
| Streaming **TTFP**            | 61 ms   | 45 ms   | −26%   |
| Per-chunk RTT (mean)          | 73 ms   | 42 ms   | −43%   |
| Total wall (stream)           | 446 ms  | 259 ms  | −42%   |
| RTF                           | 0.074   | 0.043   | −42%   |

Timestamps (forced aligner) add ~20 ms on both.
0.6B is ~35% faster on request latency, ~42% faster on streaming throughput.

## Accuracy (same clip — the trade-off)

- **1.7B:** 여보세요. 안녕하세요. 존버존 일본어 스피키 맥스입니다. 황민호 고객님 맞으세요.
- **0.6B:** 여보세요. 안녕하세요. 존버든 일본어 스피게맥스입니다. 황미노 고객님 맞으세요.

0.6B errors on this one clip: brand "스피키 맥스"→"스피게맥스", and the customer
**name 황민호→황미노**. Smaller model degrades most on proper nouns / names —
the part that matters most for sales/call-center use.

## Caveats

- **n = 1 clip**, warm, single clean Korean utterance. This is a latency probe,
  NOT a WER verdict. For a real speed/accuracy decision, run a batch of
  accented / code-switch clips through both and compare WER.
- Both models cold-start ~7.8 s on first request (torch.compile) — irrelevant to
  steady state.
- Test harness: `asr_latency_test.py --base http://127.0.0.1:8006 --audio /tmp/clip6.wav`.

## How to switch models

Model is env-configurable (`server_merged.py:37`, `ASR_MODEL`), no code change:

```bash
# 0.6B
ASR_MODEL=Qwen/Qwen3-ASR-0.6B nohup ./run_merged.sh > /tmp/asr_0.6b.log 2>&1 &
# 1.7B (default)
nohup ./run_merged.sh > /tmp/asr_1.7b.log 2>&1 &
```

Only one instance fits on the 24 GB card at a time (1.7B stack ~20 GB). The
`:8006` service is a bare process (no systemd/supervisor); killing the uvicorn
leaves the vLLM `EngineCore` subprocess orphaned — kill it explicitly by PID too.
Poll readiness with `curl :8006/health` (returns the live `model`).

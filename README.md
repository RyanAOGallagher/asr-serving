# asr-serving

Everything custom around the Qwen3-ASR serving stack, pulled together 2026-09-10 from the boxes it lived on. Upstream `Qwen3-ASR` (qwen_asr package etc.) is not here — these files sit on top of a checkout of it.

| dir | from | what |
|---|---|---|
| `prod/` | localasr `~/ASR/Qwen3ASR-streaming/` (live, 2026-09-10) | `server_merged.py` (:8006 micro-batcher), `wrapper_merged.py` (:8002), `anchor_service.py` (`/transcribe_anchor`, includes the ko-guard gather kept from the 2026-09-08 test), run scripts, notes. |
| `prod/env.example` | localasr `~/ASR/Qwen3ASR/.env` | Key names only (secrets stripped) plus the ANCHOR overrides live in prod. **Prod behaviour = code + this file**: e.g. the 2026-09-10 context-echo floor 5→3 is `ANCHOR_EN_ECHO_MIN_WORDS=3` here, the code default is still 5. |
| `prod/guard/` | localasr `~/zipa_test/` | ZIPA guard service (:8010) + GUARD.md. |
| `shim/` | 5090 box `/workspace/qwenasr_bench/streaming/` | `anchor_service.py` = pristine pre-gather anchor + the vLLM-serve shim (`ANCHOR_BACKEND_MODE=openai` → `_backend_decode_openai`, custom `_CHAT_TEMPLATE`). `cb_swap.sh` swaps `server_merged` ↔ `qwen-asr-serve :8021`. |
| `bench/` | 5090 box `/workspace/qwenasr_bench/` | load-test scripts + results. `stress_test_500.json` = 363-clip set (<5 s, sampled from live traffic; rows tagged `origin=stress-500`). Audio clips not included. |
| `docs/` | Mac | `stress_report_2026-09-08.html` (serving design vs GPU), `serving_config_benchmark_2026-07-16.md` (vLLM/precision/attention sweep), `shim_vs_prod_anchor.diff`. |

## Shim (why "openai")
Stock `vllm serve` / `qwen-asr-serve` only speaks the OpenAI-compatible HTTP API (`/v1/chat/completions`). The shim is a second `backend_decode` that translates the anchor pipeline's decode call into that protocol: context → system message, audio → base64 `audio_url` in the user turn, forced language → assistant prefill `language X<asr_text>` with `continue_final_message`, logprobs → `min_token_confidence`/`confidence`. Nothing goes to OpenAI; it's just the wire format vLLM exposes.

Known gap before prod: the shim does not pass the `RESTRICT_SCRIPTS` token mask that the default `/transcribe` path applies.

## Results in one line
Micro-batcher (`server_merged`) plateaus ~31 rps on a 5090 (≈ 3090 Ti at c4). vLLM continuous batching: 58 rps on 3090 Ti, 242–467 rps on 5090. Anchor logic ports onto serve unchanged via the shim; end-to-end anchor number still needs the guard co-located (pending three-way run, c4/8/16).

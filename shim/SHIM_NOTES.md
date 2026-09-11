# Shim: anchor on stock vLLM serve

Experiment from the 2026-09-08 throughput work on the 5090 box. Not deployed. Code: `shim/anchor_service.py`, diff vs prod: `docs/shim_vs_prod_anchor.diff`.

Stock `vllm serve` / `qwen-asr-serve` only speaks the OpenAI-compatible HTTP API (`/v1/chat/completions`). The shim is a second `backend_decode` that translates the anchor pipeline's decode call into that protocol: context → system message, audio → base64 `audio_url` in the user turn, forced language → assistant prefill `language X<asr_text>` with `continue_final_message`, logprobs → `min_token_confidence`/`confidence`. Nothing goes to OpenAI; it's just the wire format vLLM exposes.

Known gap before prod: the shim does not pass the `RESTRICT_SCRIPTS` token mask that the default `/transcribe` path applies.


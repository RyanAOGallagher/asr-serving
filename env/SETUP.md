# Environment

Frozen package lists from the two live venvs (2026-09-10): `localasr-3090ti.txt` (prod, Python 3.12.3, driver 595.71) and `box-5090.txt` (5090 bench box, Python 3.12.3, driver 580.126). Core pins are identical on both:

```
vllm==0.14.0  qwen-asr==0.0.6  torch==2.9.1  transformers==4.57.6  flashinfer-python==0.5.3  silero-vad==6.2.1
```

Rebuild order (the part that bites, from docs/serving_config_benchmark_2026-07-16.md §7):
1. `python3.12 -m venv .venv && . .venv/bin/activate`
2. `pip install vllm==0.14.0` (pulls the matching torch/cu128).
3. `pip install qwen-asr==0.0.6 --no-deps` — with deps it downgrades torch and orphans nccl. qwen-asr 0.0.6 only works with vLLM 0.14.x (0.19+ removed `vllm.inputs.data`); 0.25.x won't start on Blackwell.
4. Inference deps: `pip install silero-vad soundfile librosa av pydub soxr nagisa soynlp qwen-omni-utils accelerate torchaudio jiwer fastapi uvicorn httpx`
5. Clone upstream `QwenLM/Qwen3-ASR` (localasr is on commit c17a131fe028b2e428b6e80a33d30bb4fa57b8df 2026-01-30, dated ) and drop `prod/` (or `shim/`) files on top; the servers import `qwen_asr` from it.
6. Model: `Qwen/Qwen3-ASR-1.7B` via HF cache. Restart prod with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` or an HF outage takes it down.

Guard (`prod/guard/`) has its own venv on localasr (`~/zipa_test`) — not captured here; needs espeak-ng + onnxruntime-gpu + librosa + epitran + phonemizer (what was installed on the 5090 box for the pending co-located run).

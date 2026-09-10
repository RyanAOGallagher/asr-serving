# Qwen3-ASR benchmark plan (2026-07-15)

Goal: pick the fastest config that keeps accuracy acceptable (esp. Korean / code-switch),
measured cleanly, then confirm on a newer GPU.

## 1. Test set (freeze these)
Latency doesn't care about content; accuracy needs known text. Cover the axes:

| id | clip | lang | length | why |
|----|------|------|--------|-----|
| S-EN | tuesday.wav | English | ~1.2s | short English |
| S-KO | paris.wav | Korean | 3.0s | short Korean (also the force-EN mistranslation case) |
| M-KO | clip6.wav | Korean | ~6s | medium |
| L-KO | long45 (trim to ~15s) | Korean | ~15s | long (shows O(N) growth, batch effects) |
| CS | mix.wav | Korean/EN | ~3s | code-switch-ish |

**Gold transcripts** (for WER): known content —
- S-EN: `Today is Tuesday.`
- S-KO: `파리 가고 싶어요. 파리.`
- M-KO / L-KO / CS: transcribe once with 1.7B fp16 vLLM + hand-verify → freeze as reference.
- **Better**: pull ~30–50 clips with real text from `weavers_tts.tts_generation_log` (our
  text+audio tap) for a proper WER number, not 5 clips. Use the 5 above for quick latency sweeps.

## 2. Conditions
- **Single** — one clip at a time → per-request latency (min + median over 8–15 runs, min = truest).
- **Batch** — batch of 2 / 4 / 8 (same or mixed clips) → throughput (clips/sec) + per-clip latency.
  - Single = what a live caller feels. Batch = server throughput under load.

## 3. Metrics (per config)
- Latency: min / median ms (contention-filtered; note GPU util at test time).
- Throughput: clips/sec at batch 2/4/8.
- Accuracy: WER vs gold (and Korean-specifically — the claims for the QAD model are EN/VI only).
- VRAM used; model load time.

## 4. The matrix (phased — NOT full cross-product)
Hold everything fixed except the axis under test.

**Phase A — Backend** (fix: 1.7B, fp16, 3090Ti)
- vLLM (current) vs transformers-hf (the snippet you sent, `AutoModelForMultimodalLM`, batch list).
- single + batch. Answers "is transformers actually slower, and by how much."

**Phase B — Attention** (fix: 1.7B-hf transformers, fp16)
- SDPA (current default) vs eager vs FlashAttention-2.
- NOTE: flash-attn is NOT installed — needs a ~15min compile (`pip install flash-attn
  --no-build-isolation`, must match torch/CUDA). Decide before this phase.

**Phase C — Precision / quant** (fix: 0.6B, transformers, best attn from B)
- fp32 vs fp16 vs **int8-QAD** (`vrfai/Qwen3-ASR-0.6B-int8-QAD`).
- Accuracy is the key output here (does INT8 hold up on Korean?), plus latency/VRAM.

**Phase D — Model size** (fix: vLLM, fp16, 3090Ti)
- 0.6B vs 1.7B. (Partial data exists in `LATENCY_1.7B_vs_0.6B.md`: 0.6B ~40% faster, worse on
  proper nouns/names — re-confirm with this harness + WER.)

**Phase E — New GPU**
- Take the 1–2 winning configs and re-run the whole single+batch sweep on a newer card
  (dev RunPod 2×RTX 5090 sm_120 @ root@213.173.103.104:34238, or the B200 box).
- Blackwell (5090) needs torch cu128; B200 similar. Expect big throughput gains.

## 5. Harness (build once, reuse)
One config-driven script: `bench.py --backend {vllm,transformers} --model X --dtype {fp16,fp32,int8}
--attn {sdpa,eager,flash2} --batch N`. Loads the model, runs the frozen test set single + batched,
prints a row: `config | load_s | vram | single_min/med | batch_thru | WER`. Append rows to a CSV so
all configs are comparable in one table.

## 6. Practical constraints / order
- **VRAM:** 3090Ti 24GB; the vLLM 1.7B serving instance already uses ~16GB. Transformers models
  load on the same card → do transformers phases when 8006 is stopped, or on a second GPU.
  Only one big model at a time.
- **Downloads needed:** `Qwen3-ASR-1.7B-hf` (~3–4GB, Phase A/B), `Qwen3-ASR-0.6B` fp16 + the QAD
  (Phase C). QAD already downloading.
- **Suggested order:** C (quant, running) → A (backend) → D (size) → B (attn, if we install
  flash-attn) → E (new GPU, winners only).

## 7. Status
- [FAILED] `vrfai/Qwen3-ASR-0.6B-int8-QAD` — loads (~2.2GB) but outputs garbage `'!'` for BOTH
  English (tuesday) and Korean (paris), 6–8s/clip. The community INT8-SmoothQuant checkpoint does
  NOT run via stock `from_pretrained` + transformers 4.57.6 + our qwen_asr — needs the uploader's
  exact quant kernels/env (SmoothQuant scales at inference). DEAD END unless we match their setup
  (rabbit hole) or quantize the official model ourselves with a known-good method.
  → Phase C fallback: compare fp32 / fp16 / (our-own-int8 if we make one), skip this checkpoint.
- flash-attn: NOT installed, install PARKED (user). Phase B = SDPA vs eager only for now.
- Everything else: not started.

# Qwen3-ASR serving benchmark — RTX 5090 (RunPod, Reykjavík)

Ran 2026-07-16. Goal: pick the fastest serving config that keeps accuracy acceptable for
Korean / code-switch / accented speech. Headline metric = **wall-clock time to completion**
(warm, batch=1, min of 8 runs). Every config had to pass a correctness gate
(tuesday → "Today is Tuesday." + paris → Hangul) before any number was recorded.

Hardware: RTX 5090 32GB, **sm_120 / compute cap (12,0)**, driver 580.126.16 (CUDA 13.0).
NOTE: production (localasr) is a **RTX 3090 Ti, sm_86 (Ampere)** — different silicon, see §6.

---

## 1. TL;DR — what to run

**vLLM 0.14.0 + Qwen3-ASR-1.7B + bf16/fp16 + flash attention.**
This is what localasr already runs. **No config change needed.**

The single available improvement: `max_inference_batch_size` **8 → 64** in
`server_merged.py:104` → ~5.7× throughput headroom. Only pays off under concurrent load.

---

## 2. The five axes

### vLLM vs no vLLM (1.7B, best settings each) — biggest lever
| backend | wall-clock | ms/tok | max throughput | WER_en |
|---|---|---|---|---|
| transformers (bf16, sdpa) | 101–624 ms | 20.82 | 11.9 clips/s | 2.05% |
| **vLLM (fp16, flash)** | **20–110 ms** | **3.91** | **472 clips/s** | 2.05% |

→ **5.3× latency, 40× throughput, identical accuracy.**
Mechanism: transformers fit = `24ms + 16.1·tokens`; vLLM fit = `8ms + 3.16·tokens`.
CUDA graphs kill the per-token Python/kernel-launch overhead.

### Model size — answer DEPENDS on backend
| backend | 0.6B | 1.7B | 0.6B gain |
|---|---|---|---|
| transformers | 20.51 | 21.17 | **3%** |
| vLLM | 2.33 | 3.91 | **68%** |

0.6B costs WER_en 2.05% → 4.11% (2×). On transformers the 0.6B is strictly irrational
(3× fewer params, 3% faster). On vLLM it's a real tradeoff (61 vs 108 ms) — but 108 ms is
already imperceptible, so **stay on 1.7B**.

### Quantization — all of it is a trap here
| precision | transformers | vLLM | VRAM (transformers) |
|---|---|---|---|
| fp32 | 22.04 | — | 11,230 MB |
| **fp16** | 21.17 | **3.91** | 5,986 MB |
| bf16 | 20.82 | 4.20 | 5,962 MB |
| fp8 (native Blackwell) | — | 6.61 (**1.7× slower**) | — |
| int8 (bitsandbytes) | 94.56 (**4.5× slower**) | **UNSUPPORTED** | 4,372 MB |

- int8 on vLLM: `Model Qwen3ASRForConditionalGeneration does not support BitsAndBytes
  quantization yet. No 'packed_modules_mapping' found.`
- int8 on transformers: no Blackwell kernel path → slow dequant dominates. Telling detail:
  0.6B int8 (90.79) ≈ 1.7B int8 (94.56) — model size stops mattering, overhead is everything.
- fp8 lost **despite native Blackwell silicon**: at batch=1 the model is overhead/bandwidth-bound,
  not compute-bound, so fp8's arithmetic advantage buys nothing and its scaling costs.
  0.6B fp8 (4.40) is *slower than* 1.7B bf16 (4.20).
- fp32 → fp16 = half the VRAM for free (same speed).

### Attention
| transformers | sdpa | eager | penalty |
|---|---|---|---|
| 1.7B fp16 | 21.17 | 28.13 | **33%** |
| 0.6B fp16 | 20.51 | 27.15 | 32% |

| vLLM backend | ms/tok |
|---|---|
| FLASH_ATTN | **3.96** |
| FLASHINFER | 3.97 |
| auto (default) | 4.20 |
| TRITON_ATTN | 4.48 |

**FlashAttention-2/3 are NOT possible on this hardware**: FA3 is Hopper-only (sm_90a);
flash-attn has no official sm_120 support; pod has no `nvcc` to build it. vLLM bundles its
own flash-attn that DOES work on sm_120 — the vLLM numbers above are already flash numbers.

### vLLM version
| version | works on sm_120? | 1.7B fp16 |
|---|---|---|
| **0.14.0** + qwen_asr shim | **yes** | 3.91 ms/tok, b128 472 clips/s |
| **0.19.0** native | **yes** | 4.64 ms/tok, b128 474 clips/s |
| **0.25.1** native | **NO — engine won't start** | — |

**0.14.0 ≈ 0.19.0.** Throughput identical (472 vs 474). The apparent latency gap is almost
certainly a harness artifact, NOT a regression: the native path makes the model emit
`language X<asr_text>` (~4–5 extra tokens ≈ 13–16 ms at 3.2 ms/tok) which the shim prefills;
that more than covers the observed 20→29 ms difference. Not claimed as a version delta.

---

## 3. vLLM 0.25.1 is BROKEN on Blackwell (open upstream bug)

Chain: forced SM detection works, then FlashInfer **JIT-compiles** its sampler kernels →
needs a CUDA ≥12.9 toolkit → pod has no working nvcc → `ninja` build fails → engine dies.

```
Failed to get device capability: SM 12.x requires CUDA >= 12.9
RuntimeError: FlashInfer requires GPUs with sm75 or higher     # on a 5090!
subprocess.CalledProcessError: ['ninja', '-v', '-C', '.../flashinfer/0.6.13/120f/cached_ops/sampling']
```

Upstream: flashinfer-ai/flashinfer#3493, vllm-project/vllm#44305
("FlashInfer sampler JIT fails on Blackwell SM120f: CUDA compiler and toolkit headers are incompatible").

Nothing routes around it — all fail identically:
`VLLM_ATTENTION_BACKEND=FLASH_ATTN|TRITON_ATTN|FLEX_ATTENTION`, `VLLM_USE_FLASHINFER_SAMPLER=0`;
uninstalling flashinfer → `ModuleNotFoundError` (hard dep).

**Discriminator: the CUDA line.** 0.19.0 pulls torch **cu128** + cuda-bindings 12.9.4 → works.
0.25.1 pulls torch 2.11 **cu130** + cuda-bindings 13.3.1 → broken.

Real fix would be `flashinfer-jit-cache` (precompiled, no JIT) — **not on the tuna mirror**
(published only on flashinfer's own wheel index).

Also: qwen-asr 0.0.6 (latest; no newer exists) only works with vLLM **0.14.x** —
0.19+ removed `vllm.inputs.data` which the shim imports.

---

## 4. Serving parameters (measured against localasr's ACTUAL prod config)

| parameter | prod value | effect on LATENCY | effect on THROUGHPUT |
|---|---|---|---|
| **`max_inference_batch_size`** | **8** | none | **8.5× ceiling — the only real knob** |
| `max_model_len` | 4096 | ~6% (see below) | none |
| `gpu_memory_utilization` | 0.65 | none | none (448 → 457) |
| `max_num_seqs` | unset | none | default already optimal |

All 11 configs landed **113.8–118.9 ms**. These are capacity settings, not performance ones.

**The batch ceiling scales linearly** (throughput flatlines once concurrency > chunk size):
| inf_batch | b128 | vs prod |
|---|---|---|
| 8 (prod) | 52.7 | 1.0× |
| 16 | 97.6 | 1.9× |
| 32 | 171.9 | 3.3× |
| **64** | **298.7** | **5.7×** ← recommended (keeps an OOM bound) |
| -1 | 448.2 | 8.5× |

The source comment says small values "avoid OOM" — the `8` may be a deliberate guard on the
tighter 24GB 3090 Ti (already ~19GB used, shared with the aligner). Hence 64, not -1.

### max_model_len: real but unshippable
Smaller IS faster (monotonic, 12 runs, inf_batch=1): 256 → **111.5 ms**, 512 → 113.1,
1024 → 113.3, 4096 → **118.8**. ~6% gain.

**But it hard-crashes on real audio.** Verified on a real 45s Korean call:
```
mml=256   long45 (45s) -> FAILED: ValueError: decoder prompt (length 603) > max model length 256
mml=4096  long45 (45s) -> OK, 867ms, '여보세요. 안녕하세요... 황민호 고객님 맞으세요.'
mml=100   tuesday (1.2s!) -> FAILED: 101 tokens > 100
```
Token budget ≈ `85 (prompt) + ~13/sec of audio + output`. 4096 ≈ 5 min of audio. **Keep 4096.**

---

## 5. Per-clip wall-clock (vLLM, 5090, warm, min of 8)

| clip | audio | tokens | 1.7B fp16 | 0.6B bf16 |
|---|---|---|---|---|
| tuesday | 1.2s | 4 | 19 ms | 13 ms |
| mix | 3.85s | 4 | 19 ms | 15 ms |
| travel | 6.25s | 8 | 34 ms | 21 ms |
| paris | 3.0s | 11 | 43 ms | 23 ms |
| no_tuesday | 3.36s | 12 | 46 ms | 27 ms |
| greeting | 6.0s | 31 | 106 ms | 61 ms |
| clip6 | 6.0s | 32 | 108 ms | 61 ms |

**Output tokens drive latency, not audio length.** `travel` (6.25s, 8 tok) = 34 ms beats
`clip6` (6.0s, 32 tok) = 108 ms. Language-blind.

**Cold start: exactly one call costs 4×** (1042 ms vs 262 ms warm, calls #2–12 all 262±2).
→ fire a warmup inference at startup. Model load itself = 50–210 s.

---

## 6. Production (localasr) — audited, already optimal

From `/health` + the live process env (NOT source defaults):
| setting | value |
|---|---|
| model | `Qwen/Qwen3-ASR-1.7B` |
| backend | vLLM 0.14.0 via qwen_asr shim |
| dtype | **bfloat16** |
| attention | default → flash |
| aligner | `Qwen3-ForcedAligner-0.6B` (bf16) |
| gpu_memory_utilization | 0.65 (env override; source default 0.45) |
| max_model_len / batch | 4096 / 8 |
| GPU | **RTX 3090 Ti, cap 8.6 (Ampere)** |

Maps exactly to the winning `vllm | 1.7B | bf16 | flash` row.

**Caveat: dtype is bf16 by accident.** `LLM()` passes no `dtype`, so vLLM inherits
`thinker_config.torch_dtype: bfloat16`. Right answer, but nothing pins it — pass it explicitly
if you want it guaranteed.

**5090 numbers do NOT transfer to prod.** Ampere ≠ Blackwell, and prod shares its GPU.
Also **fp8 is impossible on Ampere** (needs sm_89+), and the FlashInfer/Blackwell bug never
affects a 3090 Ti.

### 3090 Ti vs 5090 — MEASURED (2026-07-16), verdict ~1.5–1.67×

Measured directly on prod's 3090 Ti via the `INFER model_ms=` log added to `server_merged.py`
(see §10). Both columns are **pure model inference** (3090 Ti `model_ms` strips HTTP/VAD;
5090 = offline `transcribe()`) → true apples-to-apples. Warm, min of 6–8.

| clip | tokens | 3090 Ti (model_ms) | 5090 (offline) | ratio |
|---|---|---|---|---|
| tuesday | 4 | 29.0 ms | 19 ms | 1.53× |
| mix | 4 | 29.5 ms | 19 ms | 1.55× |
| travel | 8 | 50.4 ms | 34 ms | 1.48× |
| paris | 11 | 65.2 ms | 43 ms | 1.52× |
| no_tuesday | 12 | 70.4 ms | 46 ms | 1.53× |
| greeting | 31 | 167.9 ms | 106 ms | 1.58× |
| clip6 | 32 | 176.7 ms | 108 ms | 1.64× |

Slope fits (fixed term now MATCHES because both are pure inference):
| | fixed overhead | per-token slope |
|---|---|---|
| 3090 Ti | ~8 ms | **5.28 ms/tok** |
| 5090 | ~8 ms | **3.16 ms/tok** |

- **Fixed ~8 ms on BOTH** — CPU-side kernel-launch + Python, GPU-independent. This dilutes the
  ratio on short clips (1.5×) vs long ones (1.64×→approaching the slope ratio).
- **Slope ratio 5.28 / 3.16 = 1.67×** = the actual GPU delta. Matches memory-bandwidth ratio
  (~1792 vs ~1008 GB/s ≈ 1.78×) → confirms **bandwidth-bound, not compute-bound**.

**Verdict: a 5090 buys ~1.6× on real transcription latency** (wordy 6s clip 177→108 ms; short
clips 29→19 ms). NOT the 2×+ the spec sheet implies — the model can't use the 5090's extra
compute, only its extra bandwidth. Same reason the 0.6B, fp8, and quantization all underdelivered.

Method note: the earlier "estimated 1.6×" (old jittery fit `15+0.4·audio_s+5·tokens`, slope 5.0)
predicted this almost exactly; the measured slope is 5.28. A first attempt to bench the 1.7B in a
2nd offline instance failed (HuggingFace 504 outage + no `HF_HUB_OFFLINE` → model wouldn't load,
caused a ~1h prod degradation to ElevenLabs fallback). The working method was far simpler: hit the
already-running `:8006` directly and read `model_ms` from the log. Raw data: `logs/3090ti_*.log`.

---

## 7. Gotchas that cost real time (read before repeating this)

- **PyPI is per-flow shaped to ~200–340 kB/s from this pod** (Reykjavík/Advania → Fastly).
  22 ms RTT, **zero** TCP retransmits, 8 parallel streams → 1.42 MB/s aggregate ⇒ shaping, not loss.
  **Tsinghua mirror = 2.45 MB/s single-stream (15×).** HF CDN = 14 MB/s. GitHub = 52 kB/s.
  → `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` and **NO `PIP_EXTRA_INDEX_URL`**
  (a pypi.org fallback silently re-routes pip back onto the slow path — cost ~1 hr to notice).
- **`--ignore-installed` is a no-arg flag.** `--ignore-installed blinker` makes `blinker` a
  package arg and applies the flag globally → pip reinstalls torch (cu13 over cu128). Nearly
  destroyed the verified stack.
- **Always `--no-deps`** when adding qwen-asr: it will happily downgrade torch and orphan nccl
  (`undefined symbol: ncclDevCommDestroy`) — that broke the 0.25.1 venv beyond repair.
- **vLLM needs `if __name__ == "__main__":`** — its engine core uses spawn; without the guard
  you get a bootstrapping RuntimeError or a silent deadlock.
- **`nohup`/`setsid` remote jobs** — an ssh drop SIGHUPs them and the log just stops mid-load.
- **`pgrep -f "foo"` matches its own command line** → a dead process looks alive forever.
  Monitor log *contents*, not process lists.
- **Blinker is apt-installed with no RECORD** → pip cannot uninstall it. Skip the
  gradio/flask/blinker branch entirely (demo UI only, irrelevant to inference).
- **transformers `attn_implementation` does NOT propagate to `thinker_config`** — the thinker
  (the decoder = hot path) silently stays on sdpa, making sdpa-vs-eager a **fake axis**
  (measured 0.1% difference). Must set `cfg.thinker_config._attn_implementation` explicitly.
  Real penalty once fixed: **~30%**. See `results_all.invalid_prepatch.csv` for the bogus rows.
- **vLLM VRAM readings (~27.7 GB) are KV-cache pool preallocation** (~90% of the card by
  default), NOT model footprint. Not comparable to the transformers column (4.4–11 GB).
- **Restarting :8006 REQUIRES `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.** On startup vLLM does a
  HEAD request to huggingface.co to check the model config; if HF is slow/down (it 504'd for ~1h on
  2026-07-16) the load hangs/dies even though the model is fully cached locally. The prod
  `run_merged.sh` does NOT set offline mode → an HF outage takes prod down on any restart. Clean
  restart: `cd ~/ASR/Qwen3ASR-streaming && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GPU_MEM_UTIL=0.65
  CUDA_VISIBLE_DEVICES=0 setsid nohup ./run_merged.sh > s8006.log 2>&1 < /dev/null &` — then poll
  `/health` for 200 (model load 50–210s, VRAM climbs to ~19GB).
- **Don't stop :8006 to benchmark it.** The model is already loaded and serving. To measure prod:
  redirect the wrapper to ElevenLabs (restart :8002 with `ASR_BACKEND_URL=http://127.0.0.1:9`) so
  real /transcribe traffic fails over, then hit :8006 directly. NEVER stop the model + reload — that
  needs a 12GB-free GPU + HF + 50–210s and is what caused the outage.

---

## 8. Accuracy caveats (be honest about these)

- WER set = **30 TTS-synthesized clips** from `weavers_tts.tts_generation_log` (a proxy for
  real speech), n=15/language, **no code-switch samples** (the clean-CER filter skewed the
  rows monolingual). Production domain is Korean/code-switch/accented — **this set does not
  represent it**. Strengthening it = paginate the log for real code-switch rows.
- 1.7B WER_en is rock stable at **2.05%** across every precision/backend — good evidence the
  harness measures what it claims.
- 0.6B WER wobbles 3.42–4.11% across configs = ±1 word on 15 clips. Small-sample noise, not a
  dtype effect. Don't read into it.
- CER_ko 1.13–1.81%; bf16 looked slightly better than fp16 (1.13 vs 1.58) but n=15 — not claimed.

---

## 9. Files

- `results/results_all.csv` — the 26-config matrix (backend × size × precision × attention)
- `results/results_serve.csv` — 11-config serving-param sweep
- `results/results_019.csv` — vLLM 0.19.0 native rows
- `results/results.csv` — original transformers-only matrix (`-hf` models, transformers 5.13.1)
- `results/results_vllm.csv` — first vLLM pass
- `results/results_all.invalid_prepatch.csv` — **quarantined**: bogus eager rows from before the
  thinker_config attn fix. Do not use.
- `results/wer_manifest.jsonl` — frozen WER set (30 rows)
- `logs/matrix_all.log` — **per-clip wall-clock lives here**, not in the CSVs
- `logs/serve_matrix.log`, `min_matrix.log`, `long_matrix.log`, `bench019.log` — the rest
- `logs/3090ti_prod_bench.log` — failed offline attempt (HF outage; caused prod degradation)
- `logs/3090ti_direct_bench.log` — working direct-to-:8006 run (wall-clock, incl HTTP/VAD)
- `scripts/bench_all.py` — unified harness (both backends, same API via qwen_asr)
- `scripts/bench_serve.py` — serving-param sweep
- `scripts/bench019.py` — vLLM 0.19 native harness (hand-built prompt)
- `scripts/pull_wer_set.py` — Supabase WER set puller
- `data/clips/` (11 latency clips), `data/wer/` (30 WER clips)

Pod (deleted after this): `root@82.221.170.242:28254`, RTX 5090.
Recipe to rebuild: tuna mirror → `vllm==0.14.0` → `qwen-asr==0.0.6 --no-deps` →
inference deps only (nagisa, soynlp, librosa, av, pydub, soxr, qwen-omni-utils, accelerate,
soundfile, torchaudio, jiwer).

---

## 10. Production change made (2026-07-16): per-inference latency logging

Added pure-inference timing to `server_merged.py` `_transcribe_batch` (the short-audio micro-batch
path, inside `_model_lock`). Emits to `logs/requests.log` alongside the existing request lines:
```
INFER model_ms=194.6 batch=1 per_item_ms=194.6                            # normal request
INFER model_ms=75.0  batch=1 per_item_ms=75.0 tokens=12 ms_per_tok=6.25   # if token_logprobs=true
```
- `model_ms` = pure GPU inference (excludes HTTP/VAD/queue). Pair with the `status=200 time=Xs` line
  right below to see the split — overhead is ~15 ms/request (HTTP+VAD).
- `tokens`/`ms_per_tok` only appear when the request sets `token_logprobs=true` (plain path has no
  token count).
- **Only covers the short path** (≤ ~15s audio, the micro-batch queue = ~96% of traffic). The long
  path (`_transcribe_batch` is not used; long audio calls `model.transcribe` separately near the VAD
  chunk loop) is NOT instrumented — add there too if long-call inference timing is wanted.
- Backup: `server_merged.py.bak_latency_YYYYMMDD_HHMMSS` on localasr. Revert = restore backup +
  restart with the HF_HUB_OFFLINE recipe above.
- Confirmed live: first request after restart logged `model_ms=2704.8` (the real cold-start penalty
  from §5, visible in prod → argues for a startup warmup inference).

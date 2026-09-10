# Smart language arbitration + streaming — findings & state (2026-07-14)

Work on the merged Qwen3-ASR stack in `~/ASR/Qwen3ASR-streaming/`.
Backend `server_merged.py` :8006 · wrapper `wrapper_merged.py` :8002 · ZIPA guard `~/zipa_test/guard_service.py` :8010.

## Endpoints built this session (all live)
- **`/stream_full`** (backend + wrapper proxy): streams force-EN partials AND, on `{"type":"end"}`,
  returns BOTH the finalized streaming transcript (`stream_text`) and a fresh full-audio pass
  (`full_text` + `full_pass_ms`). Continuous (stays open). `?skip_final=1` returns last partial
  instead of running the final `finish_streaming_transcribe` decode.
- **`/stream_smart`** (backend + wrapper proxy): early-lock smart streaming. Streams force-EN;
  after ~`arb_after_sec` (default 1.5) of energy-gated speech, hangul-check → passthrough (Korean),
  else batched EN+KO decode + guard argmin → lock; if Korean wins, switch stream to force-KO and
  replay. Emits `{"type":"lock",...}` with `en_text/ko_text/per_en/per_ko/chose`. Continuous.
  Short utterances arbitrated on `end`. `?skip_final=1` supported.
- **`vad_threshold`** query param added to `/stream` (silero VADIterator threshold, default 0.5).

## Config changes (in `run_merged.sh`)
- `ST_UNFIXED_CHUNK_NUM=9999` — streaming now re-decodes fresh every chunk (fully revisable,
  early words self-correct) instead of committing early. Cost: per-chunk latency grows with length
  (regenerates whole transcript each chunk); good for short utterances, laggy for long.
- `qwen_asr/.../qwen3_asr.py:314` SamplingParams got `logprobs=5` (to expose token confidence).
  **PENDING DECISION:** confidence turned out to be the WRONG signal (see below) — consider reverting
  to remove the small per-decode overhead.

## Key findings

### Streaming does NOT cache — it re-infers all audio every chunk
`streaming_transcribe` re-feeds `state.audio_accum` (ALL audio) into `model.generate()` each chunk;
no prefix cache hit. Measured: per-chunk latency grows 1.68x over 45s; total streaming compute =
3.7x a single full pass (O(N^2)). `/stream_full` = 1 pass = 3.7x cheaper AND cleaner transcript.

### Model uses offline vLLM `LLM`, single `_model_lock`
All model calls serialize on one lock (transcribe + streaming share one 1.7B instance; only one fits
on the 24GB 3090Ti). Streaming can't batch (`streaming_transcribe` is one-at-a-time). So two
concurrent STREAMS = ~2x (serialized). But two one-shot decodes submitted together BATCH via the
10ms queue into one forward pass.

### Batched EN+KO cost (contention-filtered min, re-measured)
- 3s clip: +~20ms (1.30x) · 45s clip: +~120ms (1.10x). NOT free, NOT 7ms.
- Peak-time lock contention adds seconds of jitter to median/max (hits single & batched equally).

### Adaptive vs always-batch latency (the tradeoff, simply)
One thing to internalize: **decoding the 2nd language (Korean) costs ~20ms extra on a short (3s) clip**
(~120ms on a 45s clip). It is NOT free. Given that:

- **Current adaptive** = decode EN first; only decode KO if EN came back all-Latin, and it does the two
  **sequentially** (EN finishes → hangul-check → then KO):
  - Korean utterance (force-EN keeps Hangul → passthrough): **~68ms** (1 decode, no KO, no guard).
  - English/all-Latin (needs argmin): **~68 + 68 = ~136ms** + guard (EN then KO, one after the other).
- **Always-batch EN+KO** (both submitted together → one batched forward pass):
  - Any utterance: **~89ms** + guard if needed.

So: Korean case → adaptive wins (68 vs 89). English case → always-batch wins (89 vs 136).
**You can't have both** the hangul-shortcut AND the batching, because the hangul-check needs EN's
result before deciding to run KO — so KO can't ride in EN's batch.
- Korean-dominant traffic → keep adaptive. English/mixed-dominant → always-batch.
(All numbers are contention-filtered min; peak-time lock contention adds seconds of jitter on top.)

### The guard is unreliable — paris case study
Audio `파리 가고 싶어요. 파리.` (Korean). Auto-detect gets it right. `transcribe_smart` forces EN →
mistranslates to `"Paris, I want to go to Paris."` (all-Latin) → argmin.
- ZIPA (`anyspeech/zipa-large-crctc-ns-800k`, fp16) reads the 2s window as `agawspaɪpaɾi` — ~12
  garbled symbols (CTC collapse + weak recognition), giving per_en=per_ko=0.750 → **TIE → default
  English (WRONG)**.
- 3s window: `agawɑzpaɪbɑdi` → 0.846/0.769 → Korean (correct, but thin margin, fragile).
- Xeus (`3060-server-1:8001`, stronger recognizer): `haɾeɡaoɕipaaɾi` → 0.929/0.714 → Korean
  (best margin) — but Xeus BREAKS accented-English (see below), so not a clean swap.

### What does NOT fix the guard
- **Better recognizer (Xeus):** fixes paris but flips genuine accented English to Hangul phonetics —
  because the KO candidate is a phonetic transliteration whose phonemes ARE the accented pronunciation.
  Better audio-IPA amplifies this. Unfixable by ANY acoustic metric (PER/PFER/forced-align).
- **Model confidence:** WRONG signal. Force-EN mistranslation scored HIGHER confidence (mean 0.962)
  than the correct Korean (0.939) — confidence measures text fluency, not audio match.
- **PFER / 3s on Xeus:** Xeus already uses full clip; PFER can't separate phonetically-identical
  candidates (accented English). Would only help genuine phonetic differences (already work).

### The real answer: auto-detect
Plain `/transcribe` (no forced language) = **6/6 correct** on the test clips (paris, travel, travel2,
mix, mix2, + one English). The force-EN+guard machinery MANUFACTURES the mistranslation and can't
reliably clean it up. Auto sidesteps it entirely.

## Changes applied now (stopgap)
- Guard `WIN_SAMPLES` 2s → **3s** (`guard_service.py:48`) — fixes paris, no regression on tested clips.
- **Tie → Korean**: `per_ko < per_en` → `per_ko <= per_en` in `transcribe_smart` (wrapper:454) and
  `stream_smart` (server:793). Korean-dominant domain → ambiguous leans Korean.

## Recommended for LATER
1. **Switch `transcribe_smart`/`stream_smart` to auto-detect** and retire the guard (6/6, simplest).
2. If keeping the guard: **onset→end window** (not fixed 3s — 3s truncates clips >3s) + **margin-gated
   `|per_en-per_ko| < MARGIN → auto` tiebreak** (uses the reliable auto only on the guard's failure zone,
   +1 decode only on near-ties). `MARGIN=0.10` already sits unused in guard_service.
3. **Transliteration-reject** for accented English: if ko_text is just a romanized transliteration of
   en_text, don't flip.
4. Revert `logprobs=5` if confidence diagnostics not needed.

## Clean restart recipe (avoids orphaned VLLM::EngineCore)
```
cd ~/ASR/Qwen3ASR-streaming
PGID=$(ps -o pgid= -p "$(pgrep -f server_merged:app|head -1)"|tr -d ' ')
kill -9 -"$PGID"        # kills uvicorn + EngineCore + resource_tracker as a group (guard :8010 untouched)
sleep 6                 # wait for GPU free >= 12000
nohup setsid ./run_merged.sh >> logs/merged_run.log 2>&1 </dev/null &
```
Guard restart: `~/zipa_test/run_guard.sh`. Wrapper: `./run_wrapper.sh`.

# Constrained-Auto Language Selection (Korean + English)

Notes on whether/how to make Qwen3-ASR auto-detect **only** Korean and English
instead of all 11 supported languages. Written 2026-07-07.

## TL;DR

- There is **no native "auto but restricted to a subset" setting** — not in the
  DashScope API, not in this local model. The `language` param is binary:
  force **one** specific language, or `None` = auto over **all** supported languages.
- It *is* implementable here, because language selection is a decode-time
  prompt-prefix, not a separate classifier. vLLM 0.14 + xgrammar (both installed)
  can mask the language slot to `(Korean|English)`.
- **Latency caveat:** constrained-auto does **not** improve latency. The latency
  win people talk about comes from forcing a *single* language (prefill), which is
  a different thing. See "Latency" below.

## How language selection actually works

The model emits the language label *before* the transcript:

```
language Korean<asr_text>안녕하세요 ...
└─ decoded first in AUTO mode ─┘└─ transcript ─┘
```

Selection is done purely by **prefilling the assistant turn**
(`qwen_asr/core/vllm_backend/qwen3_asr.py:981`, and
`qwen_asr/inference/qwen3_asr.py:454` `_build_text_prompt`):

- `language=None` (AUTO): prompt ends at `assistant\n` → model freely generates
  `language X<asr_text>` and picks any of the 11.
- `language="Korean"` (FORCED): prompt ends at `assistant\nlanguage Korean<asr_text>`
  → language pinned, decoding starts directly at transcript.

Sampling is greedy (`temperature=0.0`, `qwen_asr/inference/qwen3_asr.py:272`).

The model detects language **per chunk** and merges (`merge_languages`), so a long
clip can already come back `"Korean,English"` — but it can't be *told* to only
choose from those two.

## The three ways to constrain to ko+en

1. **Guided decoding on the language slot (true "constrained auto").**
   Prefill `language ` (open slot) and constrain the next tokens with
   `StructuredOutputsParams(regex=r"(Korean|English)<asr_text>[\s\S]*")`.
   xgrammar masks every token except those spelling the two names, then `[\s\S]*`
   lets the transcript decode freely. Single pass. **Recommended.**

2. **Auto + post-filter.** Run auto; if detected language ∉ {ko,en}, clamp or
   re-run forced. Simplest, no internals, but wastes a pass and the first pass may
   already have mis-decoded.

3. **Dual forced pass + pick by logprob.** Run forced-Korean and forced-English,
   keep the higher-scoring transcript. 2× compute, most robust for accented /
   code-switch audio, no reliance on the model's own LID. Good fallback if the
   guided-decoding backend is ever unavailable.

## Environment (verified 2026-07-07 on gpu-225)

- vLLM **0.14.0**
- xgrammar **0.1.29** installed (default guided-decoding backend) ✅
- API note: `GuidedDecodingParams` was **renamed** in 0.14. Correct import is:
  ```python
  from vllm.sampling_params import StructuredOutputsParams
  SamplingParams(
      temperature=0.0, max_tokens=max_new_tokens,
      structured_outputs=StructuredOutputsParams(
          regex=r"(Korean|English)<asr_text>[\s\S]*"),
  )
  ```
  Use `regex`, not `choice` (`choice` constrains the *whole* output to a fixed list;
  the transcript must stay open-ended).

## Files to change (approach #1, file endpoint only, non-streaming)

Core = **2 files**:

- `qwen_asr/inference/qwen3_asr.py`
  - `_build_text_prompt` (:454): add a branch that prefills `language ` (open slot).
  - Sampling params (:272): today `self.sampling_params` is built **once at load**
    and shared. Constrained mode needs a second variant carrying
    `structured_outputs=...`. `_infer_asr_vllm` (~:525) passes `self.sampling_params`
    to the whole batch, so build a constrained variant or select per-request.
  - Thread the mode through `transcribe → _infer_asr → _infer_asr_vllm`.
    `transcribe` runs language through `validate_language` / `normalize_language_name`
    (:362-363), which would reject a `ko+en` sentinel — so a flag must flow down.
- `server_merged.py`
  - Validation at :255 rejects any `language` not in `get_supported_languages()`.
    Let the new `ko+en` value through and pass it down (`/transcribe` endpoint).

Clean version = **3 files**: also add the `ko+en` sentinel to
`qwen_asr/inference/utils.py` so server + model agree in one place.

Everywhere = **4+**: `wrapper_merged.py` (only if reachable through the :8002
wrapper) and the streaming/websocket path (separate per-session sampling params,
meaningfully more work — leave out of a first cut).

**Hard part is not the file count** — it's that `sampling_params` is a single
shared object built at load, so adding a *per-request* regex constraint is the
bit that needs care.

## Latency — read before assuming this helps

The label is emitted before the transcript, so:

| Mode                    | Auto-switch ko↔en? | Latency vs auto              |
|-------------------------|--------------------|------------------------------|
| Full auto (`None`)      | yes, all 11        | baseline                     |
| **Force one** (`Korean`)| no, locked         | **faster** (better TTFP)     |
| Constrained auto (ko+en)| yes, 2 only        | ≈ baseline / slightly slower |

- The latency win comes from **forcing one language**: the `language X<asr_text>`
  prefix becomes prompt prefill (near-free) instead of decoded tokens, so the first
  *decoded* token is already transcript → lower time-to-first-partial.
- **Constrained auto does NOT get this** — the model still decodes the language name
  (choosing between two) and adds xgrammar mask overhead. You can have the latency
  win OR ko↔en flexibility, not both. The win *comes from* not having to pick.
- Magnitudes above are reasoned from the decode mechanism, **not yet measured** on
  this box. To confirm: run the same clips through :8006 in auto / forced-Korean /
  forced-English and compare TTFP + total latency.

## Is it even worth doing?

Only if full-auto is **actually mislabeling** on your data (e.g. a Korean-accented
English clip tagged Japanese/Chinese → garbage transcript). If auto already always
returns ko/en, this is pure complexity — skip it. Decision step: batch your real
accented / code-switch clips through auto and inspect the returned `language` field.
Constrain only if you catch it flipping to a third language.

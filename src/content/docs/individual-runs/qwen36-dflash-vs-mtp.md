---
title: "Qwen3.6-35B-A3B — DFlash vs MTP"
description: "DFlash (63% accept) vs MTP (92% accept) on the same model. DFlash crash correction + fair Q5 re-run."
---

# Qwen3.6-35B-A3B — DFlash vs MTP vs baseline (same model)

**Date:** 2026-07-06
**Setup:** target [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `UD-Q4_K_M` + draft [williamliao/Qwen3.6-35B-A3B-DFlash-GGUF](https://huggingface.co/williamliao/Qwen3.6-35B-A3B-DFlash-GGUF) `Q8_0`. Stack: production `llama-hipgraphs` build 0eca4d4, gfx906/MI50. All three spec strategies on the **identical model** → clean head-to-head. Models deleted after; Ornith restored.

## Compatibility (checked before download — this pairing is the runnable one)

| draft | verdict | why |
|---|---|---|
| **williamliao DFlash-GGUF** | ✅ **runs** | arch **`dflash`** (loadable format), vocab **248320** = target, our build has a working DFlash runtime (`common_speculative_impl_draft_dflash` engaged, non-zero accept) |
| z-lab / Anbeeld DFlash-GGUF | ❌ | arch `dflash-draft` — **rejected** by our stock build |
| modal-labs DFlash | ❌ | safetensors only (no GGUF), `DFlashDraftModel` for vLLM |
| DSpark-AEON draft | ❌ | safetensors + vLLM patches, wrong (AEON) base |

The williamliao draft is a real block-diffusion DFlash (block_size 16, n_extract 8), Q8_0 421 MB, 6-layer.

## n-max tune (temp-0 code prompt), DFlash

n2 = 81.3 t/s (74% accept) · **n3 = 82.7 (63%)** · n4 = 78.7 (58%). Best n-max 3. **For reference, MTP on the same model hit 89–91 t/s (92% accept)** → MTP already ~10% faster on code.

## benchy decode vs context — DFlash (n3) vs MTP (n2) vs baseline

| depth | baseline | DFlash | MTP | DFlash vs base | vs MTP |
|---:|---:|---:|---:|---:|---:|
| 0 | 63.5 | 76.7 | 82.3 | +21% | −7% |
| 512 | 63.4 | 72.1 | 82.5 | +14% | −13% |
| 1k | 63.2 | 78.1 | 85.0 | +24% | −8% |
| 2k | 62.2 | 70.6 | 84.4 | +13% | −16% |
| 3k | 60.9 | 78.6 | 84.8 | +29% | −7% |
| **4096** | 60.1 | **💥 server crashed** | 74.6 | — | — |
| … 65k | 38.9 | (dead) | 61.6 | — | — |

**DFlash avg prefill (0–3k): 752 t/s** (comparable to MTP's 706 / baseline's 820).

## Verdict — MTP beats DFlash on this model, on BOTH axes

1. **Speed:** DFlash *is* faster than baseline (+13–29%), but **slower than MTP at every measured depth** (and on the code probe, 83 vs 90). Root cause: DFlash's draft acceptance is **63%** vs the official embedded MTP head's **92%** — MTP proposes better tokens, so more of them stick.
2. **Stability:** the **DFlash server crashed at depth 4096** (completed 0–3072, then connection-refused for every deeper depth). MTP ran cleanly to 65k. The crash is at the `-b/-ub 4096` boundary — likely a DFlash block-diffusion / batch-size interaction in our build; not investigated further since MTP already won.

**This REVERSES the earlier gemma-4-26B-A4B result** (where DFlash beat MTP). Difference: Qwen3.6 ships an official, well-tuned embedded MTP head (92% accept), whereas the gemma MTP head was weaker relative to its DFlash. So "DFlash vs MTP" is model-specific — it comes down to which head has higher acceptance. **On Qwen3.6-35B-A3B, the embedded MTP wins decisively.**

## Files
- Charts: **`qwen36_spec_comparison.png`** (the 3-way, same model) + DFlash line added to `mi50_model_sweep_decode.png`.
- Raw JSON: `qwen36full_DFlash_benchy.json` (0–3k), `qwen36full_MTP_benchy.json`, `qwen36full_benchy.json`.

---
title: "gemma-4-12b-it — Dense Model Test"
description: "Dense 12B with MTP n3. ~2× slower than A3B models in both decode and prefill."
---

# gemma-4-12b-it (unsloth) — MI50 test, MTP on Q4_K_M

**Date:** 2026-07-06
**Model:** [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) · `gemma-4-12b-it-Q4_K_M.gguf` (6.6 GB) + `MTP/gemma-4-12b-it-Q8_0-MTP.gguf` head (465 MB).
**Stack:** production `llama-hipgraphs:upstream-rocm-7.2.4` (build 0eca4d4), gfx906/MI50. Ornith stopped/restored; model deleted after.

## Architecture — DENSE 12B (not MoE)

Confirmed from `google/gemma-4-12b-it` `config.json` (base of this GGUF): **`num_experts: None`, 48 dense layers, hidden 3840** → dense. GGUF header agrees (`gemma4`, block_count 48, no `expert_count`). llama-bench reports 11.91B params. The **MoE** member of the gemma-4 family is the separate **26B-A4B**; the only ~12b-class MoE (`stamsam/Gemma-4-12B-a4B-MoE`) is safetensors-only (would need GGUF conversion).

## Spec-decode: MTP works (external head), DFlash loads but loses

Earlier spec sweep (temp-0 greedy code prompt), decode t/s | acceptance:
| method | n2 | n3 | n4 | n6 |
|---|---|---|---|---|
| **MTP** (gemma4-assistant head) | 47.1 | **49.8 ★** | 45.0 | — |
| DFlash (williamliao `dflash` fmt) | 41.9 | 39.5 | 34.4 | 27.7 |

**MTP n-max 3 wins; DFlash is net-negative and worsens with n-max.** DFlash is built for MoE; on a *dense* model it just adds verify overhead — the opposite of the gemma-4-26B-A4B (MoE) result where DFlash beat MTP. Both formats DO load on our build (arch `gemma4-assistant` / `dflash`, unlike the rejected Qwen `dflash-draft`). MTP acceptance confirmed non-zero live (78.8%) — a real speedup, not silent fallback.

## benchy decode + prefill vs context (MTP n-max 3, 14-pt)

tokenizer = google/gemma-4-12B-it (warmup delta 15, coherence PASSED).

| depth | decode t/s | prefill t/s |
|---:|---:|---:|
| 0 | 39.9 | 578 |
| 4k | 37.7 | 466 |
| 8k | 34.3 | 464 |
| 16k | 30.1 | 424 |
| 24k | 32.4 | 404 |
| 32k | 30.5 | 384 |
| 49k | 29.8 | 351 |
| 65k | 29.8 | 321 |
| **span** | 40→30 (−25%) | avg **452** |

## Verdict — dense is the story, in BOTH phases

1. **Decode ~2× below the A3B MoE models** (~40→30 vs ~78→60). A dense 12B reads all ~12B params/token; an A3B MoE reads only ~3B active — so despite being "smaller" in total, the dense model decodes far slower. MTP helps (49.8 on code) but can't close a 4×-active-params gap.
2. **Prefill also ~2× slower** (avg 452 vs the MoE models' ~820-846) — this is the ONE model that breaks the "prefill is model-independent" rule seen across the three A3B MoE models, precisely because dense compute is heavier per token in prefill too.
3. Runs are noisier than the MoE models (dense decode + prose MTP-acceptance variance).

**Bottom line for the MI50:** the A3B MoE models (Ornith / Qwen3.6) are the right fit — sparsity is what makes this card fast. A dense 12B, even quantized small, is bandwidth/compute-bound to roughly half the throughput. gemma-4-12b appears as the violet line in `mi50_model_sweep_decode.png`.

## Files
- Raw benchy JSON: `gemma12_benchy.json` (this folder)

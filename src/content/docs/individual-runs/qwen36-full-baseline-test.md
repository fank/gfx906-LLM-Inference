---
title: "Qwen3.6-35B-A3B — MTP OFF vs ON"
description: "Cleanest MTP isolation: same model/vocab/quant, only difference = embedded MTP head. +30–58% gain."
---

# Qwen3.6-35B-A3B (unsloth full) — MI50 test: MTP OFF vs ON

**Date:** 2026-07-06
**TWO GGUFs tested, same base model / vocab / quant (Q4_K_M) — the only difference is the embedded MTP head. This is the cleanest MTP-on-vs-off isolation in the whole sweep (no finetune or vocab confound):**
- **MTP OFF (baseline):** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 20.6 GB — the *plain* quant, block_count 40, **no nextn head**.
- **MTP ON:** [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · same filename `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22.7 GB (bigger = the added MTP head), **block_count 41 with `blk.40.nextn.*`** → `--spec-type draft-mtp`.

**Why two files:** the plain unsloth quant has no MTP path on our stack (no embedded head + external Qwen `dflash-draft` format is rejected by our build), so it ran baseline-only. The **MTP-GGUF is its purpose-built counterpart** — identical model, just with the head baked in — run to measure exactly what MTP buys.

**Stack:** production `llama-hipgraphs:upstream-rocm-7.2.4` (llama.cpp `0eca4d4`), gfx906/MI50. Throwaway `--rm` containers; Ornith stopped for VRAM and restored after. **Both GGUFs deleted after testing** (results retained).

## What it is (GGUF header)

- **arch = `qwen35moe`**, MoE **A3B** (expert_count 256, expert_used 8, ~3B active of 34.66B). Same MoE class as Ornith and the trimmed model.
- **block_count = 40, NO `nextn` tensors → no embedded MTP head** (unlike the Elsephire trimmed GGUF, which had one at block 40).
- Full **248k vocab**.

## MTP-GGUF n-max quick check (temp-0 code prompt)

n1 = 76.9 · **n2 = 89.3** · n3 = 91.0 t/s. n2/n3 near-tie (same as trimmed model); ran the benchy at **n-max 2** to match Ornith and the trimmed model exactly. Live MTP acceptance **92.3%** (highest of any model tested → the head is very well matched to the base).

## Results — decode + prefill vs context (MTP OFF vs ON)

benchy 14-point, runs 2, exact-tg; tokenizer = base Qwen3.6 (warmup delta 9–14, coherence PASSED). Basic llama-bench baseline: pp512 796 / tg128 66.3.

| depth | decode OFF | decode ON (MTP n2) | MTP gain | prefill OFF | prefill ON |
|---:|---:|---:|---:|---:|---:|
| 0 | 63.5 | 82.3 | **+30%** | 719 | 376 |
| 4k | 60.1 | 74.6 | +24% | 898 | 793 |
| 8k | 58.3 | 83.2 | +43% | 925 | 792 |
| 16k | 54.3 | 76.6 | +41% | 853 | 752 |
| 24k | 50.8 | 67.3 | +32% | 795 | 740 |
| 32k | 47.8 | 72.8 | +52% | 746 | 711 |
| 49k | 43.1 | 68.4 | +59% | 664 | 627 |
| 65k | 38.9 | 61.6 | **+58%** | 596 | 566 |
| **decay 0→65k** | **−39%** | **−25%** | — | — | — |
| **avg prefill** | — | — | — | **820** | **706** |

## Verdict — the clean measurement of what MTP buys

Same model, same vocab, same quant — MTP on vs off:

1. **Decode: +30% shallow, growing to +58% at 65k.** MTP's benefit *widens* with context because the no-spec baseline decays hard (−39%) while MTP holds it to −25%. At 65k, 61.6 vs 38.9 t/s.
2. **Confirms Ornith isn't a finetune fluke.** Stock Qwen3.6 + MTP (82→62) lands right on the Ornith curve (78→60) — MTP is the mechanism, not anything special about Ornith's tuning.
3. **The one MTP cost: prefill.** avg prefill drops **820 → 706** (−14%) with MTP on — the extra nextn-head tensors add prompt-processing work. So MTP is a **decode-for-prefill trade**: big decode win, small prefill cost. For single-stream latency-bound serving (n8n), decode dominates → net win; for very prompt-heavy / low-generation workloads the prefill hit matters.

Both still ~2× above the DGX Spark Qwen3.5-122B family (different hardware/size — see the Ornith-vs-DGX report).

## Files
- Raw benchy JSON: `qwen36full_benchy.json` (OFF), `qwen36full_MTP_benchy.json` (ON), both in this folder.
- Charts: dedicated **`qwen36_mtp_onoff.png`** (the on/off isolation); also the orange (OFF) + magenta (ON) lines in `mi50_model_sweep_decode.png`.

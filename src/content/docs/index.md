---
title: MI50 LLM Inference Benchmarks
description: Home — LLM inference benchmarks, tuning results, and production notes for AMD MI50 (gfx906 / Vega 20) on ROCm
---

Welcome to the **MI50 LLM Inference** documentation — a collection of benchmarks, tuning results, and production notes for running LLM inference on an **AMD Instinct MI50 32 GB** (gfx906 / Vega 20) GPU with ROCm.

**Card:** AMD MI50 32 GB (gfx906 / Vega 20 / GCN5, ~1 TB/s HBM2, no matrix cores, passive-cooled)  
**Host:** Nobara/Fedora, Ryzen 7 8845HS (8c/16t), 30 GB RAM  
**Workload:** n8n LLM node — "big data in → small data out" (prefill-dominated), single-user  
**Production stack:** `llama-hipgraphs:upstream-rocm-7.2.4` (llama.cpp `0eca4d4`), model **Ornith-1.0-35B Q4_K_M + embedded MTP**, ~70 t/s

## Key findings — at a glance

- **ROCm/HIP over Vulkan = 8.7× faster decode.** Vulkan was crippling the card (no dp4a/MoE kernels on RADV).
- **MTP is the MI50 win.** On A3B MoE, MTP holds decode across context (−19 to −25%) and lifts throughput +30–58% vs no-spec.
- **Sparsity makes this card fast.** A3B MoE (~3B active) does ~60–80 t/s; dense models are 2–3× slower.
- **MTP > DFlash** on our current stack (92% vs ~60% acceptance), but DFlash is stable with correct batching (`-ub 512`).
- **~70 t/s is the gfx906 A3B ceiling** — well-tuned, not leaving speed on the table.
- **Docker adds 0% overhead** — proven in bare-metal A/B test.

## Reports

| Report | Date | What it covers |
|---|---|---|
| [**Master Report**](/gfx906-LLM-Inference/master-report/) | 2026-07-06 | Full 6-model sweep: MTP on/off, DFlash, trimmed vocab, dense models, Llaminar engine |
| [**Test Log**](/gfx906-LLM-Inference/all-tests-log/) | 2026-07-03 | Complete history from 2026-06-20 → 2026-07-03: hardware validation, ROCm vs Vulkan, quant tests, DFlash/EAGLE3, Docker vs bare-metal |
| [**Individual Runs**](/gfx906-LLM-Inference/individual-runs/) | 2026-07-06 | Per-run deep dives for each model tested in the sweep |
| [**Older Reports**](/gfx906-LLM-Inference/reports/) | 2026-06-20 → 07-03 | Optimization reports, setup guides, and analysis docs |

## Raw data

The `benchy-files/` directory at the [repo root](https://github.com/fank/gfx906-LLM-Inference) contains the raw JSON output from every benchy run, plus the `individual-runs/` markdown files and chart images.

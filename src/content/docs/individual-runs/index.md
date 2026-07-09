---
title: Individual Runs (2026-07-06)
description: Per-model deep-dive reports from the July 6 model sweep
---

Six models/configurations tested on 2026-07-06 using `llama-benchy` v0.4.0 on the same stack.

| Model | Config | Key result |
|---|---|---|
| [Ornith-1.0-35B (prod)](./ornith-benchy-live/) | Q4_K_M + MTP n2 | Flat ~76 t/s 0→16k, shallow −24% decay to 65k |
| [Qwen3.6-35B-A3B vocab-trimmed](./qwen36-trim-vocabulary-test/) | Q4_K_S + MTP n2 | Shallowest decay (−19%), pulls ahead at deep context |
| [Qwen3.6-35B-A3B MTP OFF vs ON](./qwen36-full-baseline-test/) | Q4_K_M | Cleanest isolation: MTP = +58% at 65k |
| [Qwen3.6-35B-A3B DFlash vs MTP](./qwen36-dflash-vs-mtp/) | Q4_K_M | MTP wins (92% accept vs ~60%), DFlash stable with `-ub 512` |
| [gemma-4-12b-it (dense)](./gemma-4-12b-mtp-test/) | Q4_K_M + MTP n3 | Dense penalty: ~2× slower than A3B models |
| [Qwen3.6-27B (dense)](./qwen36-27b-dense-mtp/) | Q4_K_M + MTP n2 | Slowest model (23 t/s); MTP net-neutral on dense |

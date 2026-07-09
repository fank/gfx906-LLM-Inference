---
title: Ornith-IQ4 — IQ4_XS-MTP-graft Benchmark
description: "IQ4_XS body + Q6_K MTP head graft: −18% smaller, faster at every depth, better KLD vs Ornith Q4_K_M"
sidebar:
  order: 1
---

# Ornith-IQ4 — LordNeel IQ4_XS-MTP-graft-headQ6 Benchmark (2026-07-09)

**Model:** `LordNeel/Ornith-1.0-35B-GGUF-llamacpp-tp1:ornith-1.0-35b-IQ4_XS-MTP-graft-headQ6.gguf`
**Base:** `deepreinforce-ai/Ornith-1.0-35B` (Qwen3.5-35B-A3B finetune)
**Format:** GGUFs (IQ4_XS body + Q6_K MTP head graft)
**Arch:** `qwen35moe`, 256 experts (top-8), 1 embedded MTP layer
**Size:** 19.6 GB (vs 24 GB original Q4_K_M-MTP = **−18% smaller**)
**KLD:** 0.073 (vs 0.086 for Q4_K_M — better fidelity per byte)

## Setup

**Stack:** production `llama-hipgraphs:upstream-rocm-7.2.4` (llama.cpp `0eca4d4`), gfx906 / MI50 32 GB.
**Tool:** [eugr/llama-benchy](https://github.com/eugr/llama-benchy) v0.4.0.
**Method:** 14-point context sweep 0→65536, pp512/tg128, runs 2, `--exact-tg`, `--latency-mode generation`, `--concurrency 1`, KV q8_0.
**Tokenizer:** Qwen3.5-35B-A3B snapshot (same as Ornith), warmup delta 14 tok, coherence PASSED.

## n-max sweep

n-max values 1–7 tested with temp-0 code prompt (quicksort, 256 tok generated):

| n-max | Acceptance | Mean len | Notes |
|-------|-----------|---------|-------|
| **1** | 85.0% | 1.83 | Near-perfect position 1 |
| **2** | 80.5% | 2.60 | Best real throughput on MI50 |
| 3 | 70.9% | 3.11 | Diminishing returns |
| 4 | 62.8% | 3.49 | Half of draft wasted |
| 5 | 53.3% | 3.64 | |
| 6 | 48.2% | 3.86 | |
| 7 | 42.3% | 3.92 | Most draft tokens rejected |

**Winner: n-max=2** — 2.60 mean len with 80.5% acceptance, optimal MI50 tradeoff.

## 14-point benchy results (n-max=2)

| Depth | Decode t/s | Prefill t/s | Decay |
|------:|-----------:|------------:|------:|
| 0 | **86.28** | 791 | 0.0% |
| 512 | 86.39 | 786 | −0.1% |
| 1024 | 88.12 | 842 | −2.1% |
| 2048 | 88.52 | 937 | −2.6% |
| 3072 | **91.72** | 1000 | **−6.3%** |
| 4096 | 84.26 | 920 | +2.3% |
| 6144 | 87.86 | 947 | −1.8% |
| 8192 | **82.81** | 931 | **−4.0%** |
| 12288 | 82.16 | 896 | +4.8% |
| 16384 | 82.97 | 877 | +3.8% |
| 24576 | 78.36 | 821 | +9.2% |
| 32768 | 80.68 | 767 | +6.5% |
| 49152 | 71.56 | 680 | +17.1% |
| 65536 | 66.95 | 606 | +22.4% |

**Avg prefill:** 842 tok/s

## Comparison vs Ornith Q4_K_M-MTP (current production)

| Metric | Ornith Q4_K_M | **Ornith-IQ4** | Change |
|--------|:------------:|:--------------:|:------:|
| Model size | 24 GB | **19.6 GB** | **−18%** |
| Decode depth 0 | 78.3 | **86.28** | **+10%** |
| Decode depth 8192 | ~70 | **82.81** | **+18%** |
| Decode depth 49152 | 59.6 | **71.56** | **+20%** |
| Decay 0→49152 | −24% | **−17.1%** | **Better** |
| Avg prefill | 845 | 842 | ~same |
| Context | 262k | **131k** ⚠️ | Half (VRAM) |
| KLD | 0.086 | **0.073** | Better |
| GGUF quant | Q4_K_M | **IQ4_XS + Q6_K head** | Mixed precision |

**Key insight:** Ornith-IQ4 is faster at every depth and decays less. The IQ4_XS body with Q6_K MTP head graft gives better fidelity with less VRAM. The tradeoff is halved context (131k vs 262k) due to VRAM constraints on 32 GB.

## Production deployment

Configured as `Ornith-IQ4` alias on port 8089:

```yaml
- -m /models/lordneel-ornith-mtp-graft/ornith-1.0-35b-IQ4_XS-MTP-graft-headQ6.gguf
- --alias Ornith-IQ4
- --spec-type draft-mtp
- --spec-draft-n-max "2"
- -c "131072"
- --chat-template-file /models/ornith-chat-template.jinja
- --reasoning on
```

VRAM: 24.8/32 GB (7.2 GB free), spec engaged at ~82% acceptance.

**Remote pi-agent model ID:** `Ornith-IQ4` on `mi50-swap` provider.

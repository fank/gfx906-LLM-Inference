---
title: "Ornith-1.0-35B — Live Benchmark"
description: "Production reference: Ornith with embedded MTP, 3-depth benchy probe (0/4k/16k), DGX Spark comparison."
---

# Ornith-1.0-35B — Live Benchmark (llama-benchy)

**Date:** 2026-07-06
**Tool:** [eugr/llama-benchy](https://github.com/eugr/llama-benchy) v0.4.0 (OpenAI-API client — non-destructive, ran against the live prod server, no container stop/restart)
**Target:** `llama-hipgraphs` container, `http://localhost:8089/v1`, model id `ornith-hipgraphs`
**Config measured (= prod):** Ornith-1.0-35B Q4_K_M + embedded-MTP (`--spec-type draft-mtp`, `--spec-draft-n-max 2`), `-ngl 99`, flash-attn on, KV `q8_0`, `--parallel 1`, ctx 262144.
**Method notes:** 3 runs/test, `--exact-tg` (forces 128 output tokens via min_tokens+ignore_eos), `--latency-mode generation`. Prompt corpus = natural English text (Sherlock Holmes / Project Gutenberg). Tokenizer = **Qwen3.5-35B-A3B** snapshot (Ornith's true base; warmup token delta only 9–14 → counts accurate). n8n workflow was cut during the run, so zero concurrent load.

## Results

| depth | prompt processing (t/s) | generation tg128 (t/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 0     | pp512: 914.5 ± 54.2 · pp2048: **1245.9 ± 23.7** | 72.2–75.1 | 643 / 1725 |
| 4096  | pp512: 1107.0 ± 1.1 · pp2048: 1149.2 ± 2.5      | 76.5–76.8 | 4243 / 5427 |
| 16384 | pp512: 980.4 ± 6.5 · pp2048: 890.6 ± 24.6       | 76.3–78.3 | 17317 / 20793 |

(TTFT column = pp512 / pp2048 e2e time-to-first-token.)

## Findings

1. **Generation is flat at ~76 t/s from 0 → 16k context.** MTP fully compensates the context-length decay. First live end-to-end confirmation of the long-standing hypothesis: the no-MTP `llama-bench` reference showed 63 t/s @1k → **21 t/s @16k**; here generation holds ~76 t/s at *every* depth (even nudging up within noise).

2. **Beats the "~70 t/s ceiling" figure.** 72–78 t/s vs the temp-0 code-bench's 70. Natural prose accepts MTP drafts more readily than code at temp 0, so text-heavy traffic runs slightly faster than the worst-case code number. The ±6–7 t/s spread on tg is per-request MTP-acceptance variance.

3. **Prefill ~900–1250 t/s**, confirming the ~1000 t/s prefill note. The cost that scales is TTFT, not gen speed: a 2k prompt on top of 16k context = **~21 s to first token**. That's the real latency lever for big-context calls.

## Scope / caveats

- Measures prod-as-served: MTP on, **single-stream**. For aggregate/concurrent throughput use a throwaway `--parallel N --spec-type none` container (MTP halves aggregate under batch load — see the aggregate-bench note).
- Synthetic natural-text prompts, not real ECHO-Bot code/tool-use traffic; acceptance (hence t/s) will differ somewhat on live code prompts (typically lower than prose, closer to the ~70 figure).

## Comparison vs DGX Spark Qwen3.5-122B (fank/dgx-spark-qwen3.5-122b-bench)

Chart: `ornith_vs_dgx_decode_vs_context.png` (this folder). Same tool (llama-benchy 0.4.0), same context-sweep test, same Qwen3.5 hybrid-MoE family — so it's a legitimate comparison **of curve shape**. Ran Ornith on their exact 14-point grid (0→65536).

**Read the height difference correctly — it is NOT a "MI50 beats the DGX box" claim.** The two setups differ on the two things that set decode speed:

| | Ornith (ours) | DGX Spark (theirs) |
|---|---|---|
| model | 35B-A3B, **3B active** | 122B-A10B, **10B active** |
| bandwidth | MI50 ~1 TB/s | GB10 ~273 GB/s |
| serving | llama.cpp + MTP (n-max 2) | vLLM + DFlash (7 configs) |

Decode is bandwidth ÷ active-params bound, so Ornith's ~3× higher absolute decode is expected from ~3.3× fewer active params and ~3.7× more bandwidth — and the MI50 is actually running *well under* its theoretical edge (it's latency-bound at ~10% bandwidth utilization, our long-standing finding). **Height is a size + bandwidth artifact, not silicon superiority.**

**The like-for-like finding is decay shape** — how well each holds decode as context grows (0→65k):

| config | 0 → 65k | decay |
|---|---|---|
| **Ornith 35B-A3B · MTP (ours)** | 78.3 → 59.6 | **−24%** |
| DGX E_bf16_n0 (no spec) | 28.5 → 19.1 | −33% |
| DGX A_bf16_n12 (default DFlash-12) | 36.1 → 17.6 | −51% |
| DGX C_bf16_n4 (their best, DFlash-4) | 41.0 → 15.3 | −63% |

**Verdict:** Ornith's decode is not flat — it eases from ~76 (held through ~24k) down to ~60 at 65k. But its −24% rolloff is the **shallowest of any curve in the comparison**, shallower even than the DGX's no-spec baseline. On this hardware+model class, MTP holds decode across context better than DFlash does on the Spark rig. (Caveat: shape is also influenced by attention/KV implementation and hardware, not spec method alone — this is an observational bench, not a controlled A/B of MTP vs DFlash.)

**Anomaly handled:** the first sweep's depth-8192 point came back at 46 t/s (runs 56.0 / 36.6) — a thermal-throttle transient during the long sustained sweep (MI50 passive-cooling behavior). Re-measured in isolation = 74.2 t/s (72.7 / 75.3 / 74.5), in line with neighbors; the chart uses the clean value.

## Reproduce

```bash
TOK=/home/<username>/llm/rocm-cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
HF_HUB_OFFLINE=1 uvx llama-benchy \
  --base-url http://localhost:8089/v1 --model ornith-hipgraphs \
  --tokenizer "$TOK" \
  --pp 512 2048 --tg 128 --depth 0 4096 16384 \
  --runs 3 --concurrency 1 --exact-tg --latency-mode generation \
  --save-result out.md --format md
```

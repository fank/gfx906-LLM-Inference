---
title: "Qwen3.6-35B-A3B — Vocabulary-Trimmed"
description: "145k vocab (Latin+Greek only) + embedded MTP, shallowest MoE decay (−19%), n-max sweep."
---

# Qwen3.6-35B-A3B vocabulary-trimmed — MI50 test (llama.cpp+MTP)

**Date:** 2026-07-06
**Model:** [Elsephire/Qwen3.6-35B-A3B-vocabulary-trimming-GGUF](https://huggingface.co/Elsephire/Qwen3.6-35B-A3B-vocabulary-trimming-GGUF) · Q4_K_S, 20.1 GB · file `qwen3.6-35B-vocabulary-trimming-Q4_K_S.gguf`
**Stack:** our production ROCm build `llama-hipgraphs:upstream-rocm-7.2.4` (llama.cpp `0eca4d4`), gfx906/MI50 — same stack that serves Ornith. Throwaway `--rm` containers; Ornith stopped for VRAM and restored after.
**Chart:** `qwen36trim_vs_ornith_decode.png` (this folder).

## What the model is (verified from the GGUF header, not the card)

- **arch = `qwen35moe`** — same family as Ornith (loads on our build, no segfault). Despite the "Qwen3.6" name the GGUF is qwen35moe.
- **vocab = 145,572** (down from Qwen3.6's 248,077, −41%). Latin + Greek scripts only; CJK/Cyrillic/Arabic/etc. removed. `token_embd`/`output` tensors are `2048 × 145572`.
- **Ships its own embedded MTP head** — `blk.40.nextn.eh_proj / enorm / hnorm / shared_head_norm`, `nextn_predict_layers=1`, block_count 41. Same embedded-MTP design as Ornith; the head predicts into the **trimmed** vocab, so it's self-consistent.

## Basic decode/prefill (llama-bench, no spec)

| test | t/s |
|---|---|
| pp512 (prefill) | 1087 |
| **tg128 (decode)** | **71.1** |

## Speculative decoding — what works

**Embedded MTP works natively** (`--spec-type draft-mtp --spec-draft-n-max 2`, Ornith's config). Decode probes vs the 71.1 no-spec baseline:

| prompt type | MTP decode | gain |
|---|---|---|
| prose | 73.6 t/s | +4% |
| factual | 81.6 t/s | +15% |
| code | 89.7 t/s | +26% |

Same content-dependence as Ornith (structured/code accepts MTP drafts best).

**External DFlash / MTP drafters do NOT work — and can't.** Two independent blockers, both confirmed by trying the Anbeeld Qwen3.6 DFlash drafter on disk:
1. **Format:** our stock upstream build rejects it — `unknown model architecture: 'dflash-draft'` (that GGUF is from the PR #22105 fork format, not in our build).
2. **Vocab (the deeper blocker):** the drafter is full-vocab (its `bos_token_id = 248044`) vs the trimmed target's 145,572. Speculative decoding requires drafter and target to share a vocabulary, so **any** full-vocab Qwen3.6 drafter is incompatible with the trimmed model regardless of format.

**Key takeaway — trimming and external speculation are mutually exclusive here.** Vocab-trimming buys a smaller lm_head (+~2 GB VRAM headroom, slightly faster decode) but forecloses every off-the-shelf drafter. The only working accelerator is the model's *own* embedded MTP. (Full, non-trimmed Qwen3.6 + the Anbeeld DFlash is the external-drafter path — but that's a different model and our build would still need the fork format.)

## MTP `--spec-draft-n-max` sweep (temp-0 greedy code prompt = lossless)

| n-max | decode t/s | draft acceptance | mean draft len |
|---:|---:|---:|---:|
| 1 | 77.9 | 84.6% | 1.85 |
| **2** | **87.3** | 80.0% | 2.60 |
| 3 | 87.4 | 69.1% | 3.07 |
| 4 | 79.3 | 58.5% | 3.32 |
| 6 | 72.1 | 44.6% | 3.65 |

**No gain to be had — n-max 2 is optimal** (same as Ornith). 2 and 3 tie at ~87 t/s (87.3 vs 87.4 = noise) and both beat everything else; n-max 2 keeps much higher acceptance (80% vs 69%), which is the robust pick because real traffic runs at temp >0 where acceptance falls and deep drafts get punished harder (n-max 4/6 already net-negative here). The trimmed model shows a slightly wider 2–3 plateau vs Ornith's sharp peak at 2, but the winner is unchanged.

## Prefill speeds (vs Ornith) — trimming does NOT help prefill

Prefill (pp512, t/s) is **essentially identical** to Ornith and decays the same way with depth:

Precise 14-point sweep, pp512 t/s (low-depth points are noisy — 512 tokens is dominated by per-call overhead):

| depth | trimmed Qwen3.6 | Ornith |
|---:|---:|---:|
| 0 | 708 (llama-bench 1087) | 607 |
| 4k | 1107 | 931 |
| 8k | 1011 | 899 |
| 16k | 889 | 958 |
| 32k | 806 | 802 |
| 49k | 723 | 717 |
| 65k | 645 | 630 |

Both peak ~1100 t/s (mid-depth) and converge to ~640 t/s at 65k, tracking each other within noise. **Vocab-trimming leaves prefill unchanged because the lm_head runs once per prompt, not per token** — prefill is MoE-compute/hardware-bound and model-independent here. The trim's benefit shows up only in **decode** (below), where the smaller output projection is paid every step.

## benchy decode-vs-context (MTP-on) — vs Ornith

Precise 14-point grid (0→65536, pp512/tg128, runs 2, exact-tg) — the same test run against Ornith. tokenizer = base Qwen3.6 (warmup delta 9–14 tok, coherence PASSED → trimmed Latin text tokenizes identically, counts valid).

| depth | trimmed Qwen3.6 (MTP) | Ornith (MTP) |
|---:|---:|---:|
| 0 | 78.1 | 78.3 |
| 4k | 77.2 | 75.4 |
| 8k | 75.8 | 74.2 |
| 16k | 75.4 | 74.2 |
| 24k | 73.1 | 72.7 |
| 32k | 67.2 | 67.4 |
| 49k | 71.3 | 64.1 |
| 65k | 63.2 | 59.6 |
| **decay 0→65k** | **−19%** | **−24%** |

*(Trimmed depth-12288 discarded a throttle-transient run [46.9 t/s]; used the clean run [78.1], consistent with neighbors 8k=75.8 / 16k=75.4 — same MI50 passive-throttle artifact handled for Ornith's 8192.)*

**Verdict:** on the benchy **prose** corpus the two are **near-identical up to ~33k** (both ~75→67 t/s), and the trimmed model pulls **ahead only at deep context** (49k: 71.3 vs 64.1; 65k: 63.2 vs 59.6) → a shallower −19% decay vs Ornith's −24%. The trimmed edge is **clearer on code/structured content** — the temp-0 n-max probe hit 87 t/s vs Ornith's ~70. Mechanism: the smaller trimmed lm_head shaves the per-token output projection, and that saving grows in relative terms as the rest of the step slows down at deep context. Caveat: quant differs (Q4_K_S vs Ornith's Q4_K_M) and it's a different base model, so this is not a clean isolation of the trimming effect. Both sit ~3× above the DGX Spark Qwen3.5-122B family (different hardware/size — see the Ornith-vs-DGX report for why height isn't a fair axis).

## Files
- Model kept on disk: `/home/josh/llm/models/qwen36-trim/qwen3.6-35B-vocabulary-trimming-Q4_K_S.gguf`
- Raw benchy JSON: scratchpad `qwen36trim_ctxsweep.json`

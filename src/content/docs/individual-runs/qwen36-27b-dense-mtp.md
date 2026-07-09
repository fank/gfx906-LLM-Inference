---
title: "Qwen3.6-27B — Dense Quick Test"
description: "Slowest model (23 t/s). MTP net-neutral on dense (+3%). DSpark-AEON draft incompatibility."
---

# Qwen3.6-27B (dense) + own MTP — MI50 quick test + DSpark compatibility

**Date:** 2026-07-06
**Requested:** test [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) with the [Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft](https://huggingface.co/Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft) draft.
**Stack:** production `llama-hipgraphs:upstream-rocm-7.2.4` (build 0eca4d4), gfx906/MI50. Model deleted after; Ornith restored.

## DSpark draft — INCOMPATIBLE (checked first, not downloaded)

The DSpark-AEON draft cannot be paired with the 27B on our stack, on three independent grounds (all verified from the repo file list + card):
1. **No GGUF / no llama.cpp.** Repo is `model.safetensors` + `dflash.py` + `markov_head.py` + `vllm_patches/` — a **patched-vLLM-only** artifact ("stock vLLM does not implement the Markov semi-autoregressive drafting path"). We run llama.cpp; gfx906 vLLM is a dead-end anyway.
2. **Wrong base.** Distilled to **AEON logits** (`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored`), not stock Qwen3.6-27B → hidden-state mismatch → degraded/zero acceptance even if it loaded (same base-mismatch rule as the Ornith-DFlash note).
3. **NVIDIA/DGX-Spark target** ("DSpark"), like the ROCmFP4 Deckard model.

**Substitute tested instead:** the model's OWN embedded MTP (`unsloth/Qwen3.6-27B-MTP-GGUF`, Q4_K_M, arch `qwen35` dense, block_count 65 with `blk.64.nextn.*` head) — the working self-speculative path, no external draft needed.

## Findings (basic bench + n-max probe; full benchy skipped by choice — slow non-contender)

- **Dense 27B** (`qwen35`, 27.3B all-active, hidden 5120, hybrid Gated-DeltaNet/Attention). llama-bench Q4_K_M: **pp512 = 244 t/s, tg128 = 23.2 t/s** (no spec).
- **With MTP** (`--spec-type draft-mtp`, embedded head): n-max 2 = **23.9 t/s (+3%)** at 78% draft acceptance.

## Verdict — MTP is ~net-neutral on a dense 27B

Despite 78% acceptance, MTP delivers only **+3%** because on a **dense** model every verify token costs the full 27B forward pass — the verify cost eats almost all the speculation benefit. This is the same dense penalty seen on gemma-4-12b (MTP only +5%) and the mirror of the A3B MoE models where MTP gives +30–58% (verify only activates ~3B there). **Spec-decode's payoff scales with how cheap verify is → big on MoE, marginal on dense.**

Separately, this 27B is **the slowest model in the sweep** (23 t/s decode, 244 t/s prefill) — the dense 27B + hybrid-DeltaNet arch is heavy and appears less optimized in our build than standard attention. Not a serving contender on the MI50 vs the ~60-80 t/s A3B MoE models.

**Bottom line: the DSpark pairing was impossible; the 27B's own MTP works but barely helps because it's dense. The MI50's sweet spot remains A3B MoE + MTP.**

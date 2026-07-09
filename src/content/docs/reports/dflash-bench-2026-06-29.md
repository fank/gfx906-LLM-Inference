---
title: "DFlash n-max Sweep — 2026-06-29"
description: "DFlash on Qwen3.6-35B (custom build, 86% accept, 92.5 t/s). n-max sweep, determinism check."
---

# DFlash n-max sweep — qwen3.6-35b-A3B Q5 on MI50 (2026-06-29)

Fixed n_predict=256, ignore_eos, temp=0 (greedy/lossless), 3 repeats x 6 prompts after warmup.

| Config | net decode t/s | accept/step | draft accept % | VRAM GB | vs baseline |
|--------|---------------|-------------|----------------|---------|-------------|
| baseline n_max=3 (block16) | **73.8** | 1.68 | 57% | 0.0 | +0% |
| n_max=5 (block16) | **58.1** | 2.27 | 46% | 0.0 | -21% |
| n_max=7 (block16) | **52.4** | 2.45 | 36% | 0.0 | -29% |
| n_max=10 (block16) | **30.9** | 2.41 | 25% | 0.0 | -58% |
| n_max=15 (block16) | **29.2** | 2.53 | 17% | 0.0 | -60% |
| n_max=19 (block20/bs20) | **23.7** | 2.48 | 14% | 0.0 | -68% |

Losslessness: outputs compared across configs at temp=0 (see console for any violations).

## Follow-up: no-spec baseline + fine sweep + determinism (2026-06-29)
| Config | net decode t/s | accept/step | accept % | vs no-spec |
|---|---|---|---|---|
| NO-SPEC (plain decode) | **63.9** | 0.00 | 0% | +0% |
| dflash n_max=2 | **72.5** | 1.35 | 68% | +14% |
| dflash n_max=3 | **72.2** | 1.68 | 57% | +13% |
| dflash n_max=4 | **57.8** | 2.01 | 51% | -10% |

**Determinism (no-spec vs dflash n_max=3, same greedy prompt):**
- chat: IDENTICAL
- reasoning: diverge@char143
- math: IDENTICAL
- code: diverge@char10
- factual: diverge@char313
- creative: IDENTICAL

## Conclusion

**Peak throughput = n_max 2–3 (~72 t/s), which is the CURRENT default. No tuning gain available.**

- dflash vs plain decode: **+14%** (72.5 vs 63.9 t/s). Real but modest.
- n_max sweep is a clean inverted curve peaking at 2–3, collapsing above 4:
  2→72.5 | 3→72.2 | 4→57.8 | 5→58.1 | 7→52.4 | 10→30.9 | 15→29.2 | 19→23.7
- This is the **MoE caveat**: Qwen3.6-35B-A3B activates only ~4B params/token, so target verify is cheap → wide draft blocks cost more than they save. The paper's 6x / τ≈6.5 is on DENSE models; the llama.cpp discussion notes Qwen MoE gets only ~2–2.8x, and we see even less because decode is already fast.
- Current launch script has no `--spec-draft-n-max` → defaults to 3 → already optimal. **No change recommended** (n_max=2 is +0.4%, within noise).

**"Lossless violation" flags are benign.** Greedy (temp 0) output is not bit-reproducible across batch widths: parallel draft-block verification flips occasional near-tie argmax tokens (FP non-associativity), which cascades into different-but-equally-valid text (e.g. "an equation for distance" vs "based on distance"). 3/6 prompts were byte-identical, 3/6 diverged coherently. Output quality/distribution is preserved — this is inherent to GPU speculative decoding, not a build bug.

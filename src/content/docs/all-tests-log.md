---
title: "Master Test Log — 2026-06-20 → 2026-07-03"
description: "Complete testing history: hardware validation, ROCm vs Vulkan (8.7×), quant tests, DFlash/EAGLE3, Docker vs bare-metal."
---

# MI50 (gfx906) LLM-Serving — Master Test Log

**Card:** AMD Instinct MI50 32 GB (gfx906 / Vega 20 / GCN5, ~1 TB/s HBM2, **no matrix cores**, passive-cooled) · replaced an Intel Arc B580 (permanent swap)
**Host:** Nobara/Fedora, Ryzen 7 8845HS (8c/16t), 30 GB RAM · SELinux disabled · Docker
**Workload:** n8n LLM node — "big data in → small data out" (prefill-dominated), single-user (`--parallel 1`)
**Compiled:** 2026-07-03 — consolidates every test from 2026-06-20 → 2026-07-03.
**Source reports:** `MI50-report.md`, `MI50-MTP-speculative-decoding-analysis.md`, `MI50-Q5-Q6-quant-test.md`, `DFlash-setup.md`, `dflash-bench-2026-06-29.md`, `ornith-mtp-optimization-report.md`, `ornith-model-docker-logs-2026-07-01.md`, `llama-stats-2026-06-23.md`, memory `mi50-*`.

> **Current production (2026-07-03):** `llama-hipgraphs` container, image `llama-hipgraphs:upstream-rocm-7.2.4` (build commit `0eca4d4`, 2026-06-30), model **Ornith-1.0-35B Q4_K_M + embedded MTP**, `--spec-type draft-mtp --spec-draft-n-max 2`, 262 k ctx, q8_0 KV, `--parallel 1`, port 8089. ~70 t/s.

---

## Timeline at a glance

| Date | Phase | Headline result |
|---|---|---|
| 06-20 | Hardware + Vulkan era | Thermal validated; MoE beats dense; **prefill is the weak spot** |
| 06-20 | **ROCm vs Vulkan** | **8.7× decode** (9.4→82 t/s) — Vulkan was crippling the card |
| 06-20 | MTP economics on MoE | Net-negative on hard content; break-even ≈42 % accept |
| 06-20 | Q5/Q6 quant test | No quality gain over QAT-Q4_0; stay Q4 |
| 06-29 | DFlash on Qwen3.6-35B | Custom build needed; +14–25 %; peak n_max 2–3 |
| 06-30→07-01 | **Ornith-1.0-35B + MTP** (current) | ubatch +46 %, HIP-graphs +12 %, **n-max tuned 6→2 = +19 %** |
| 07-02→03 | Drafter ecosystem scans | Nothing beats stock MTP for Ornith |
| **07-03** | **Today: GLM / DFlash / EAGLE3 / bare-metal** | DFlash>EAGLE3 on Gemma; **Docker = 0 % overhead** |

---

## Phase 0 — Hardware & thermal validation (2026-06-20)

**Thermal test** — 3-min sustained Vulkan load, `rocm-smi -d 0` every 2 s:

| Time | Junction | sclk | State |
|---|---|---|---|
| idle | 34 °C | 1725 | — |
| 40 s | 90 °C | 1725 | full boost |
| 60 s | **100 °C** | 1725 | **throttle onset** |
| 111 s | 100 °C | **930** | hard throttle |
| peak | **102 °C** | — | mem 87 °C (not the limiter — die hotspot is) |
| +4 s idle | 77 °C | — | cools fast |

**Verdict:** passive MI50 throttles at ~100 °C after ~60 s sustained load (1725→~930-1143 MHz). Protects, doesn't damage. **Bursty n8n = fine** (never reaches the 60 s threshold). Longer prefill sustains more → throttles harder.

---

## Phase 1 — Native Vulkan era (2026-06-20) — later fully superseded

Stack: native `llama.cpp` b9565 + Vulkan (RADV), `--device Vulkan1`. Models: Gemma 4 **12B/31B QAT** + **26B-A4B MoE QAT**.
Note: `llama-bench`/`llama-cli` ABI-broken → **all benchmarks via `llama-server` `/completion` timings** (also the only way to read MTP acceptance).

### 1a. MTP draft-depth sweep — Gemma 31B dense, 128k, F16 KV, temp 0
| k (`--spec-draft-n-max`) | gen t/s | MTP accept | peak junction |
|---|---|---|---|
| 1 | 8.36 | 78.5 % | 66 °C |
| 2 | 8.88 | 66.9 % | 75 °C |
| **3** | **9.00 ← peak** | 75.8 % | 80 °C |
| 4 | 8.39 | 36.0 % | 82 °C |
| 5 | 7.86 | 46.3 % | 82 °C |
| 6 | 7.15 | 41.6 % | 87 °C |

**k=3 optimal; acceptance cliff at k≥4** (head predicts ~3 ahead). No thermal confound (<100 °C).

### 1b. Deep-128k prefill (the weak spot) — Vulkan
Prefill **≈165 t/s** at 115k, GPU only ~93 W / 66 °C = **underutilized, backend-limited, not thermal**. `ub=2048` didn't help. Latency: 4k ≈24 s · 16k ≈1.5 min · 60k ≈6 min · 115k ≈11.5 min. Mitigation = KV-prefix reuse (stable bulk first, varying tail last).

### 1c. Dense-vs-MoE three-way — k=3, 128k, F16 KV, temp 0
| Config | arch | gen t/s | prompt t/s | MTP accept | VRAM |
|---|---|---|---|---|---|
| UD-Q4_K_XL 31B + unsloth assistant (267 MB) | dense | 9.0 | ~31 | **75.8 %** | ~18 GB |
| Official Q4_0 31B + official assistant f16 (911 MB) | dense | 3.9 | 32 | ~54 % | ~17 GB |
| ↳ re-run with q8_0 assistant (491 MB) | dense | 5.45 | 32 | 54.5 % | ~17 GB |
| **★ Official Q4_0 26B-A4B + q8_0 assistant (441 MB)** | **MoE ~4B active** | **10.9** | **~120** | ~62 % | ~17 GB |

**MoE wins** — prefills ~4× faster (the metric that matters) + best gen speed. Adopted as endpoint. (f16 head cost ~40 % decode; acceptance is set by base-model *fidelity* + content, not head quant.)

---

## Phase 2 — ★★★ ROCm/HIP vs Vulkan — the decisive finding (2026-06-20)

Same MI50, same Gemma-4-26B Q4_0 raw, Docker `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`:

| metric | Vulkan (RADV) | **ROCm/HIP** | gap |
|---|---|---|---|
| Decode (gen) | 9.4 t/s | **~82 t/s** | **~8.7×** |

**Vulkan was crippling the card the whole time** — gfx906's dp4a INT8 MMQ kernels + efficient MoE expert path exist on ROCm/HIP but **not** on RADV/Vulkan. Every earlier "weak spot" was really the wrong backend. **→ migrated n8n endpoint to ROCm** (reverses the Phase-1 "native Vulkan" decision). This is the single biggest win in the whole log.

---

## Phase 3 — MTP economics on the MoE (2026-06-20)

MTP re-tested on ROCm (draft now fast). **Content-dependent:**

| prompt | accept | MTP gen | raw gen |
|---|---|---|---|
| predictable | 52 % | **94 t/s** | 82 |
| harder | 29 % | 70–74 | 82 |

**Break-even reconstructed:** cycle cost `C_cycle ≈ 1.66× decode` (self-consistent across both runs) → **break-even acceptance ≈ 41–42 %**. 52 %→+15 %, 29 %→−15 %. Root cause = **MoE breaks "free parallel verify"**: verifying K tokens activates the *union* of each token's top-8 experts (sub-linear ~2.5× for 4 tokens, not full K×8). Small ~4B active footprint leaves little for spec-decode to recover.
**→ endpoint ran RAW (82 t/s)** on the MoE — MTP is net-negative on mixed/hard content. Untested lever flagged: confidence gating `--spec-draft-p-min`.

### 3a. Draft-head precision sweep (γ=3, predictable prompt)
**26B MoE** (Q5_K_XL target): raw 69.0 · **Q4_0 head 252 MB → 90.5 t/s / 58 %** · Q8_0 461 MB → 85.1 / 55.7 % · F16 855 MB → 79.4 / 55.7 %. **F16 = Q8_0 byte-identical acceptance** (same argmax) → bigger head only slower. **Use Q4_0 head; F16 is a trap.**
**31B dense**: raw 22.66 · Q4_0 349 MB → 37.57 / 63.6 % · Q8_0 515 MB → 38.01 / 65.2 % · F16 955 MB → 37.34 / 66.9 %. Speed flat, accept rises slightly with precision. **Q8_0 sweet spot.**
**Rule:** MoE → smallest head (verify is cheap, head bytes visible); dense → head is a rounding error. "Smaller head wins" is **MoE-specific**. MTP pays far better on dense (+67 %) than MoE.

---

## Phase 4 — Q5/Q6 quant test — Gemma 26B-A4B MoE (2026-06-20)

Tested unsloth UD-Q5_K_XL (21.2 GB) & UD-Q6_K_XL (23.3 GB) vs QAT-Q4_0, isolated container :8090, prod restored after.

**Quality** (45-prompt greedy eval): Q4_0 / Q5 / Q6 **all 25/25 easy, 18/20 hard, same two failures (gsm3, gsm4), same wrong answer "120"** → no task-correctness difference. (Caveat: short prompts don't probe long-form; PPL discarded — ≈880 artifact.)
**Speed** (q8_0 KV): Q4_0 74.3 · Q5 68.5 (−6 %) · Q6 68.1 (−12 % vs Q4 F16). q8_0 KV itself costs ~7–9 %.
**MTP accept doesn't rise with quant:** Q4 59.6 % > Q5 55.7 % > Q6 51.6 % (trends *down*).
**VRAM** (after reclaim): Q5@48k F16 ~23.7 GB, Q6@48k q8_0 ~25 GB. Context is nearly free (Gemma SWA); model size is the binding constraint.
**Verdict: stay QAT-Q4_0** — Q5/Q6 = all cost, zero measurable gain.
**Side win:** found & killed a **phantom 8.2 GB VRAM baseline** = legacy `prompt.service` (native-Vulkan 12B on :8088, boot-started); `systemctl disable` → full ~32 GB freed.

---

## Phase 5 — DFlash on Qwen3.6-35B-A3B (2026-06-29) — custom build

Drafter `z-lab/Qwen3.6-35B-A3B-DFlash` (0.4B) → converted to GGUF `dflash-draft.gguf` (747 MB F16). **Needed a self-built GPU llama.cpp** (stock gfx906 images lacked `draft-dflash`; min b9831). Trap: a `GGML_HIP=OFF` (CPU-only) build → mount whole `build-rocm/bin` + `LD_LIBRARY_PATH`.

**Verified working:** 30.5 GB VRAM (35B Q5 + draft + 128k q8_0 KV), **86 % accept (193/225), decode 92.5 t/s** (vs ~71–79 plain), prefill 119 t/s.

**Block-size** (`--spec-draft-block-size`, concurrency-1): no-spec ~58 · bs8 70.5/54 % · **bs16 72.7/57 %** · bs20 74.0/58 % · ngram ~66. block-16 best for our concurrency-1; bs8 for high concurrency; **bs8/bs20 deleted**.

**n_max sweep** (Q5, temp 0): no-spec 63.9 · **n2 72.5/68 % (+14 %)** · n3 72.2/57 % · n4 57.8/51 % (−10 %) · n5 58.1/46 % · n7 52.4/36 % · n10 30.9/25 % · n15 29.2/17 % · n19 23.7/14 %. Clean inverted curve, peak n2–3, collapses ≥4. Cause = MoE cheap-verify. Determinism: greedy not bit-reproducible across draft widths (FP non-assoc, 3/6 identical, 3/6 diverge coherently — benign).

---

## Phase 6 — Ornith-1.0-35B + MTP (2026-06-30 → 07-01) — CURRENT PROD

Migrated live model to **Ornith-1.0-35B** (`qwen35moe` arch = Qwen3.5 + Gemma-4 base, hybrid SSM/attention; RL-tuned agentic coding/tool-use). Model `ornith-1.0-35b-Q4_K_M-MTP.gguf` (21 GB) ships MTP head **embedded** (block 40 / `nextn.*`) → no separate draft, just `--spec-type draft-mtp`. `--no-mmap` mandatory (30 GB RAM page-cache thrash).

### 6a. Optimizations
| Change | Effect |
|---|---|
| ubatch 512 → **4096** | prompt +46 % (833→1214 t/s), gen short +50 % (~60→~90); VRAM 78 %→90 % |
| **HIP-graphs build** (`GGML_HIP_GRAPHS=ON`) | +12 % gen (cuts kernel-launch overhead — batch-1 is latency-bound ~10 % BW) |
| ngram vs MTP | MTP 63.6 avg/44 % (stable) > ngram ~61.0/35 % (erratic 14–58 %) — MTP wins |
| f16 K cache | 2.5× *slower* — rejected (no tensor cores) |
| Power-limit / undervolt | skipped — only GPU, not risking it |

### 6b. `--spec-draft-n-max` tuning (2026-07-01) — full sweep, ~4k code prompt, temp 0
| n-max | gen t/s | accept |
|---|---|---|
| 1 | 68.2 | 79 % |
| **2** | **70.2 ★ WINNER** | 64 % |
| 3 | 67.9 | 53 % |
| 4 | 66.2 | 53 % |
| 6 | 59.0 | 41 % |

Was deployed at **n-max 6 (slowest)** → **switching to 2 = +19 %**. Real n8n accept ~40 % (temp 0.6) → pushes sweet spot even lower → 2 is safe.

### 6c. Live n8n request log (2026-07-01, 25 requests)
Gen **40.81–100.28 t/s (avg ~62.5)**, prompt 575–1214 t/s (avg ~931), draft accept 0.24–0.78 (avg ~0.46). Fastest 100.28 t/s (short output, high KV reuse); slowest 40.81 t/s (7.8k prompt). Gen degrades with context length.

### 6d. Hardware ceiling
~70 t/s ≈ all a single MI50 does on an A3B MoE. Independent llama-bench (Qwen3-Coder-30B-A3B, same class): Q4_K_M 73 t/s @128 → 63 @1k → **21 @16k**. RTX 3090 (~same BW, mature CUDA) does Qwen3.6-35B-A3B Q4 ~135 t/s — the ~2× gap is silicon (dp4a-only, no MFMA), not tunable. **vLLM-gfx906 forks = dead** (MoE slow, no MTP, archived Feb 2026).

---

## Phase 7 — Drafter ecosystem scans (2026-07-02 → 07-03)

- **57 Ornith HF repos checked** — every usable MTP GGUF grafts the *same* official head (no gain). A KL-**distilled** MTP head (`shisa-ai/…qwen36-distill`) measured *worse* than the official zero-training head (distilled 67 %/+17 % vs official 67 %/+21.4 %) — a better head was **tried and lost**.
- **No Ornith-native DFlash / EAGLE3 exists** — DFlash 35B GGUFs are all **Qwen3.6** (wrong base; Ornith is `qwen35moe`); confirmed from the running model's GGUF header. A Qwen3.6 drafter on our Qwen3.5-hybrid target = reject-load / ~0 % accept.
- gfx906 upstream: **zero PRs merged after our 06-30 build**; the relevant ones (#21168, #24668) already in. `arte-fact/…turbo` fork trades 3.3× context for −18 % speed (wrong way). Stanford hipkernels = MI350X only. **Future watch:** AMD reportedly re-adding official gfx906 ROCm support next major version.

---

## Phase 8 — Today (2026-07-03)

### 8a. GLM-4.7-Flash test — aborted
`unsloth/GLM-4.7-Flash UD-Q5_K_XL` (21.7 GB, 30B-A3B MoE, deepseek2 arch, 131k ctx). Loaded fine: ~60 t/s, 27.5 GB VRAM, coherent. **MTP impossible** — GGUF has no `nextn` tensors, GLM/deepseek2 MTP unmerged (PR #24868 draft). Reverted to Ornith per user; container removed. *(GLM GGUF still on disk — 21.7 GB, deletable.)*

### 8b. Gemma-4-26B-A4B + DFlash — **DFlash IS net-positive on gfx906** ⭐
Target `unsloth/gemma-4-26B-A4B-it UD-Q4_K_M` (16.95 GB) + `Alittlehammmer/…DFlash-Q8_0` (471 MB, z-lab head). Our build **has the working DFlash runtime** (`common_speculative_impl_draft_dflash … block_size=16`, accept non-zero). @ Q4/16k/temp 0:

| config | gen t/s | accept | vs baseline |
|---|---|---|---|
| baseline (no spec) | 55.9 | — | — |
| DFlash n-max 6 | 64.4 | 36 % | +15 % |
| **DFlash n-max 3** | **75.2** | 54 % | **+35 %** |

Overturns the earlier "DFlash net-neg at Q4 sub-Blackwell" — the difference is a **base-matched** drafter. Gemma+DFlash (75) edges Ornith+MTP (70) on raw t/s, but it's a *different, general* model → Ornith stays prod for quality.

### 8c. Gemma-4-26B-A4B + EAGLE3 — DFlash wins decisively
`williamliao/…speculator.eagle3-Q8_0` (0.99 GB, RedHat's official 0.9B head), `--spec-type draft-eagle3` (runtime works). @ Q4/16k/temp 0 (baseline 55.9):

| method | drafter | gen t/s | accept | vs baseline |
|---|---|---|---|---|
| **DFlash n3** | 0.4B | **75.5** | 54 % | +35 % |
| EAGLE3 n3 | 0.9B | 59.1 | 32 % | +6 % |
| EAGLE3 n5 | 0.9B | 46.9 | 21 % | **−16 %** |

EAGLE3 loses on **both** axes — lower accept (32 vs 54 %) + heavier 0.9B drafter. Caveat: llama.cpp EAGLE3 (PR #18039) appears chain-draft (not tree), so this is EAGLE3-as-it-runs-here, not its vLLM ceiling. **Ranking on gfx906+llama.cpp: DFlash > EAGLE3 > baseline.**

### 8d. Docker vs bare-metal A/B — **container overhead = 0 %**
Extracted the *exact* binary + ROCm libs from the image, ran native with `LD_LIBRARY_PATH` vs in-container, identical flags/model/GPU:

| config | Docker | Bare-metal | Δ |
|---|---|---|---|
| baseline | 55.83 | 55.96 | +0.2 % |
| DFlash n3 | 75.21 | 75.41 | +0.3 % |

Accept byte-identical (54 %, 308/572 both). On native Linux, container GPU work uses the **same host-kernel amdkfd driver** via `/dev/kfd` — no GPU virtualization. (Distinct from the B580 pain, which was Vulkan→CPU *fallback*, a wrong code path, not container tax.) 17 GB extraction deleted after.

### 8e. PCIe link health & stability — clean, at max, not a bottleneck
Prompted by the `arkprojects.space/wiki/AMD_GFX906` scan (whose headline lever is "force PCIe gen4": on MI60 gen1→gen4 = **+57 % prompt**, generation flat). Checked our card (`07:00.0`) three ways:

| Check | Result |
|---|---|
| **Link speed/width** (sysfs) | `max = current = 16.0 GT/s x16` — full **PCIe 4.0 x16**, never downtrained |
| **AER errors** (`aer_dev_correctable/nonfatal/fatal`) | **all 0** — BadTLP 0, BadDLLP 0, RxErr 0, Rollover 0, Timeout 0, TOTAL_ERR_COR/NONFATAL/FATAL all 0, over ~3-day uptime incl. all today's benchmark model-loads |
| **RAS** (`rocm-smi --showrasinfo`) | **0 correctable / 0 uncorrectable** across every block (UMC/GFX/SDMA/MMHUB/PCIE_BIF/…) |
| **Live delta under load** | drove sustained inference (PCIe bw 75–198 MB/s, link held 16 GT/s) → **every AER counter byte-identical before/after (0 new errors)**; prod served the 4096-tok prompt at **72.0 t/s / 65 % accept** |

**Verdict: PCIe is at max speed *and* electrically clean — zero performance lost to the bus.** A flaky link would show BadTLP/BadDLLP (retransmits) + link retrains; there are none. The wiki's gen4 lever is already maxed for us (their MI60 test was x8; ours is x16). If throughput ever feels low the lever is the GPU (the ~70 t/s A3B ceiling), never the link. Also confirmed the used €500 card is RAS-clean.

**Same-day side checks (not PCIe):** `mixa3607/llama.cpp-gfx906` scanned — near-daily tags (newest `b9867`, 07-03) are **automated CI rebuilds of upstream** llama.cpp, not a gfx906 kernel fork; last real source commit 06-28; nothing our self-built image lacks → no switch. Docker cleanup: removed 3 dead LLM containers + 4 images + build cache (~20 GB reclaimed); prod untouched.

### 8f. Aggregate (batched) throughput — 1 MoE card vs their 4-GPU vLLM
Prompted by `arkprojects.space/wiki/…/vllm/benchmark/Gemma4-31B` claiming **210 tok/s**. That number is **aggregate over 16 concurrent requests on TP=4 (four MI50s)**, dense `gemma-4-31B-AWQ-8bit`, no spec decode, 16-tok in → 256-tok out. Per-stream it's only ~13 t/s (210÷16). We reproduced the *shape* (16 concurrent, 256-tok gen) on **one** card, same Ornith Q4_K_M model, `--parallel 16 --cont-batching` throwaway container (stop-test-restore prod):

| config (16 concurrent, 1× MI50) | aggregate | per-stream | batch wall | gate |
|---|---|---|---|---|
| **spec OFF** (fair vs their vLLM) | **166.2 t/s** | 10.9 t/s | 24.6 s | 16/16, 0 err |
| MTP ON (prod spec) | 77.2 t/s | 5.1 t/s | 53.1 s | 16/16, 0 err |
| *their vLLM (reference)* | *210 t/s* | *~13 t/s* | — | *4 GPUs, dense* |

**Validity gate (analog of accept>0):** `batch wall ≈ slowest single req` in both runs (24.6≈24.6, 53.1≈53.1) → the 16 requests genuinely overlapped, not silently serialized; 16/16 completed, 0 errors → aggregate not understated by drops. Slots=16 confirmed.

**Findings:**
- **One MoE card = 166 vs their four dense cards = 210 → ~79 % of a 4-GPU rig on a single GPU.** Per-card: 166 vs 52.5 = **~3.2× per card**. This is the **MoE-sparsity** story (Ornith ~3B active/token vs dense 31B), *not* a raw-silicon claim. Their only edge is in-batch per-stream latency (~13 vs 10.9), bought purely by spreading 16 users over 4 cards.
- **MTP *halves* aggregate under batch load (166 → 77):** speculative decode spends compute the saturated batch already needs → pure waste when 16 streams compete. Clean regime split:
  - **Single-stream (prod / n8n):** MTP **on**, ~70 t/s (vs ~50 baseline). Latency-bound, spare compute → drafting pays.
  - **Batched serving:** spec **off**, 166 t/s aggregate. Compute-bound → drafting hurts.
- **Prod unchanged:** n8n hits us one request at a time, so single-stream latency (70) gates responses, not aggregate. This test answered "can one card rival their 4-GPU headline?" (yes, 79 %) — not "should we switch prod" (no).

---

## Standing conclusions (what to actually do)

1. **Prod = Ornith-1.0-35B Q4_K_M + embedded MTP, n-max 2, ROCm/HIP Docker.** Don't touch.
2. **~70 t/s is the gfx906 A3B ceiling** — 2× needs new silicon, not tweaks. Well-tuned, not leaving speed on the table.
3. **Backend: ROCm/HIP only** (8.7× over Vulkan). **Docker adds nothing** — proven bare-metal.
4. **Quant: Q4 (QAT/UD)** — higher quant = cost, no measurable gain. Q4 is also *fastest* (bandwidth-bound).
5. **Speculator by regime:** embedded MTP for Ornith (only base-matched option); for a *base-matched* Gemma, **DFlash > EAGLE3**. External drafters need a base-model match — arch mismatch = dead.
6. **Head/drafter: smaller wins on MoE** (cheap verify); low n-max wins everywhere on this card (accept collapses with depth).
7. **Spec decode is single-stream-only:** MTP helps at concurrency 1 (~70 vs ~50) but *halves* aggregate under batch load (166→77). If we ever serve concurrent traffic, run `--parallel N` with **spec off** (166 t/s aggregate on one card) — don't stack drafting on a saturated batch.

## Disk cleanup (done 07-03)
**Deleted:** GLM-4.7-Flash GGUF (21.7 GB), `models/dflash-draft/` (~1.5 GB), all Gemma-4 test files (target+DFlash+EAGLE3, ~18.5 GB), bare-metal ROCm extraction (17 GB), stale compose files. → `/home` from ~245 GB used down to **187 GB used / 268 GB free**.
**Remaining non-prod alternates (keep only if wanted):** `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` (26.6 GB), `ornith-1.0-35b-Q5_K_M.gguf` (24.7 GB).
**Prod files (keep):** `ornith-1.0-35b-Q4_K_M-MTP.gguf` (21.7 GB), `mmproj-F16.gguf` (0.9 GB), `docker-compose-hipgraphs.yml`, image `llama-hipgraphs:upstream-rocm-7.2.4` (+ `hipgraphs-builder` for rebuilds).

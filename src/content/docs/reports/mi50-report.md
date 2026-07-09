---
title: "MI50 Report — Initial Setup & Decisions"
description: "Hardware validation, thermal tests, Vulkan vs ROCm migration, MTP economics, Gemma-4 MoE adoption."
---

# AMD Instinct MI50 — LLM Inference Setup, Tests & Decisions

**Machine:** Nobara (Fedora-based) Linux dev sandbox · **Card:** AMD Instinct MI50 32 GB (gfx906 / Vega 20)
**Workload:** AI/LLM inference for an n8n node — "big data in → small data out" (prefill-dominated). Not gaming.
**Last updated:** 2026-06-20 (living document — appended as tests run)
**Current state (TL;DR):** endpoint = **ROCm/HIP Docker** (`mi50-llm`, port 8089), Gemma-4-26B MoE Q4_0, 128k, **raw (no MTP)**, **~82 tok/s** (8.7× the old Vulkan), boot-persistent, n8n at `http://172.20.0.1:8089/v1`. Key findings: §7.6 (ROCm vs Vulkan = 8.7×), §7.7/§7.7b (MTP off — MoE breaks speculative decoding, externally confirmed).

> Purpose of this file: keep a running record of *what* we did, *what we measured*, and *why* — so decisions are traceable and reproducible.

---

## 1. Goal & context

Run modern models (Gemma 4 **12B / 31B QAT** with **MTP / speculative decoding**) on the MI50 with **≥ 128k context**, optimizing **prompt + generation tokens/second**. The card was bought used (€500, Germany) and **physically swapped in for the previous Intel Arc B580** (same slot, permanent — the B580 is gone).

Hard requirements from the user:
- **≥ 128k context** (non-negotiable).
- **Native, cleanly-supported setup** (not a fragile stack).
- **Decisions backed by measurement, not speculation.**

---

## 2. Hardware facts

| Property | Value |
|---|---|
| GPU | MI50 32 GB, gfx906 / Vega 20, GCN5 |
| Memory | 32 GB HBM2, ~1 TB/s bandwidth |
| Matrix cores | **None** (GCN5 — relies on dp4a/INT8 for speedups) |
| Cooling | **Passive** (datacenter card, no fan) — intentional, accepted |
| Vulkan id | **Vulkan1** = `RADV VEGA20` (Vulkan0 = 780M iGPU) |
| rocm-smi id | **GPU0** = MI50 (has junction + memory sensors); GPU1 = iGPU |

**Why this matters:** ~1 TB/s HBM2 is the MI50's strength — generation (decode) is memory-bandwidth-bound, so the big VRAM + bandwidth suit large models and large inputs. No matrix cores means it leans on INT8/dp4a paths; for our purposes (Vulkan llama.cpp) it's competitive for the prefill-heavy n8n workload.

---

## 3. Thermal validation (TESTED)

The card is passively cooled, so we measured behavior under sustained load before committing.

**Method:** 3-minute sustained Vulkan inference load, junction/edge/memory sampled every 2 s via `rocm-smi -d 0`.

| Time | Junction | sclk | State |
|---|---|---|---|
| idle | 34 °C | 1725 | — |
| 40 s | 90 °C | 1725 | full boost |
| 60 s | **100 °C** | 1725 | **throttle onset** |
| 111 s | 100 °C | **930** | hard throttle |
| peak | **102 °C** | — | mem peaked 87 °C |
| +4 s idle | 77 °C | — | cools fast |

**Verdict:** The MI50 **thermal-throttles under sustained load** — it hits ~100–102 °C at ~60 s and clocks down (1725 → ~1143 MHz, ~15–30% slower). It protects itself; it does not damage itself. HBM2 memory (87 °C) is **not** the limiter — the die hotspot is.

**Why we proceed anyway:** The n8n workload is **bursty** (big input ingest + short output). Short bursts never reach the 60 s throttle threshold → they run at full 1725 MHz. Only sustained, minutes-long generation throttles. This matches the user's stance ("passively cooled, don't worry about fans"). **Implication:** the longer a model's prefill, the more it sustains into throttle, so the 31B throttles harder on huge inputs than the 12B.

---

## 4. Software stack — why NATIVE Vulkan, not Docker/ROCm

**Decision: native llama.cpp + Vulkan (RADV). Docker/ROCm path abandoned.**

- The earlier plan built a Docker/ROCm (HIP) stack (mixa3607 gfx906 images). We **dropped it**: the native Vulkan server already runs Gemma 4 + MTP at high acceptance, so Docker added complexity for no benefit.
- **Build in use:** `/home/josh/llama.cpp/build/bin/llama-server` — **b9565 (2026-06-08)**. Gemma 4 MTP needs ≥ b9553 ✅.
- **`--device Vulkan1`** targets the MI50.

**Tooling gotcha (TESTED):** `llama-bench` and `llama-cli` in that build are **ABI-broken** — `undefined symbol: cpu_get_num_math` (Apr-19 binaries vs Jun-08 shared libs). Rebuilding `llama-bench` fails (`vulkan-shaders-gen` won't compile). **→ All benchmarking is done via `llama-server`'s `/completion` timings JSON**, which is actually better here: it's the real production path *and* it reports MTP acceptance (which llama-bench cannot).

---

## 5. Models & context limits (TESTED via GGUF metadata + load logs)

Downloaded to `/home/josh/llm/models/`:

| Model | File | Size | MTP head | head `n_ctx_train` |
|---|---|---|---|---|
| Gemma 4 12B QAT | `gemma-4-12b-it-qat-q4_0.gguf` | 6.5 GB | `gemma-4-12b-qat-it-assistant-Q8_0.gguf` | 262144 |
| Gemma 4 12B QAT (canonical) | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | 6.3 GB | `mtp-gemma-4-12B-it.gguf` | 262144 |
| Gemma 4 31B QAT | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` | 17 GB | `mtp-gemma-4-31B-it.gguf` | **131072** |

**Key facts:**
- **12B is natively 256k** (model + MTP head) → 128k with full headroom, **no RoPE scaling**.
- **31B's MTP head caps at exactly 131072 = 128k** → 128k is met but with **zero headroom above it**. If you ever need > 128k, only the 12B can do it.
- MTP heads are arch `gemma4-assistant`, 4 blocks, `nextn_predict_layers=4` (predict up to ~4 tokens ahead — relevant to the k-sweep below).
- QAT quality lives at ~4-bit, so Q4_0 / UD-Q4_K_XL ≈ higher-bit quality while staying small + MTP-capable.

---

## 6. Benchmark methodology

- **Server-based** (the broken llama-bench forced this, and it's the faithful path). A Python harness starts `llama-server`, sends controlled requests, parses `timings` JSON for `prompt_per_second` (prefill) and `predicted_per_second` (decode), and reads MTP acceptance from the server log.
- **Realistic input:** ~120k tokens of *varied real source code* (not repeated filler — repetition fakes high MTP acceptance) + a *real task* prompt.
- **Two thermal regimes:** cool-burst (peak, matches bursty n8n) and full-128k prefill (throttled).
- **temp 0** for determinism (also maximizes MTP acceptance and makes runs comparable — identical output across runs).

---

## 7. Results

### 7.1 — 128k fit gate (TESTED) ✅
**31B + F16 KV @ 128k loads with ~16–18 GiB / 32 GiB used — no OOM, comfortable headroom.** Gemma 4's sliding-window attention (SWA) keeps the 128k KV small (only the few global layers store full context). **F16 KV is viable at 128k** (so we keep it — quantized KV would hurt MTP acceptance). This was the riskiest assumption; it passed.

### 7.2 — MTP draft-depth sweep `--spec-draft-n-max` k = 1…6 (TESTED)
31B QAT, 128k ctx, F16 KV, temp 0, thinking off, identical 250-word task each run:

| k | gen tok/s | MTP accept | peak junction |
|---|---|---|---|
| 1 | 8.36 | 78.5% | 66 °C |
| 2 | 8.88 | 66.9% | 75 °C |
| **3** | **9.00 ← peak** | **75.8%** | 80 °C |
| 4 | 8.39 | 36.0% | 82 °C |
| 5 | 7.86 | 46.3% | 82 °C |
| 6 | 7.15 | 41.6% | 87 °C |

**Verdict: k = 3 is optimal (9.0 tok/s).** Acceptance **falls off a cliff at k = 4** (36%) because the head reliably predicts ~3 tokens ahead; forcing a 4th+ wastes draft compute. The spread (k1→k3) is ~7% — a fine-tune, not a game-changer — but k ≥ 4 is actively harmful. All runs stayed < 100 °C, so no thermal confound. **→ Endpoint set to k = 3.**

### 7.3 — Deep-128k prefill throughput (TESTED) ⚠️ the weak spot
Measured the real cost of ingesting a large input (the n8n "big in" case): sent ~115k real tokens to the live MoE and timed prefill via the server's own `prompt processing … tokens per second` log.

- **Prefill ≈ 165 tok/s** at depth, and the GPU sat at only **~93 W / 66 °C** — far below its ~200 W+ capacity. So prefill is **NOT compute-bound** (and does **not** thermally throttle — it never gets hot enough).
- Raising the micro-batch to **`ub=2048` did not help** (~140 t/s first batch, ~76 W). The bottleneck is a **Vulkan / MoE-prefill inefficiency on gfx906**, not a tunable batch size. (The dense 31B is no better — its deep-prefill probe timed out at >600 s for 115k, i.e. <190 t/s.)

**Implication for n8n — prefill latency scales with input size at ~165 tok/s:**

| input tokens | ingest time |
|---|---|
| 4k | ~24 s |
| 16k | ~1.5 min |
| 60k | ~6 min |
| 115k | ~11.5 min |

**Honest assessment:** moderate inputs (≤ ~16k tokens) are usable; very large inputs (50k+) are slow to ingest. This is the **one real weak spot** for the "big data in → small data out" goal — the card stays cool, but **raw prefill is the limiter** and it underutilizes the GPU (software/backend issue, not silicon).

**Mitigation — KV-prefix reuse (the highest-impact lever, and free):** the latency above is only paid for *unique* input. llama-server keeps each slot's KV and **reuses the longest common prefix** with the next prompt (the prompt cache is enabled — "prompt cache … size limit: 8192 MiB" appears in the load log). So if your n8n pattern is *same big document / system-prompt, different question*, **put the stable bulk first and the varying part last, and reuse the same slot** → only the changed tail is prefilled, collapsing minutes to seconds. `--cache-reuse N` extends this to partial/non-contiguous reuse; `--slot-save-path` + the `/slots` endpoint persist KV to disk across restarts. **Action: characterize your prefix overlap** — high overlap = a 10–100× win on this exact metric; fully-unique inputs = caching only saves the system-prompt prefix.

**FlashAttention is already on** (`--flash-attn on`) — it trims 128k KV memory and speeds attention; an A/B with/without on the 115k probe is a remaining tuning knob.

**Reframe — decode is *also* backend-limited, not just prefill.** ~9–11 tok/s is well below the HBM2 memory-bound ceiling (a dense 31B-Q4 reads ~17 GB/token → ~55 t/s ceiling at ~1 TB/s; the MoE should be higher still). So the **Vulkan/gfx906 backend caps decode too** — "decode healthy" is more honestly "decode less-bad." The same future lever (**ROCm/HIP**, kernel tuning) that could fix prefill would also lift decode. *(KV-reuse, FA, and this decode-ceiling reframe were raised by an external review — good catches, folded in here.)*

### 7.4 — Official Google QAT Q4_0 + fitting assistant vs unsloth UD-Q4_K_XL (TESTED)
Switched targets to the **official `google/gemma-4-qat-q4-0` collection** on the (sound) principle that Q4_0 is the QAT-native format. What the measurements actually showed:

**Collection & memory.** Official model `gemma-4-31B_q4_0-it.gguf` = **17.65 GB** vs unsloth UD-Q4_K_XL = **17.29 GB** → the official Q4_0 is *slightly bigger*, not smaller. So the "strange quant" was never a memory problem; both sit ~17–18 GB at 128k. The official **assistant** ships only as safetensors (no GGUF in the repo).

**Converting the official assistant works (answers the literal question).** `convert_hf_to_gguf.py` handles arch `Gemma4AssistantForCausalLM` through the normal path (the Qwen-only `--mtp` flag is irrelevant — it's for *extracting* a head from a full model; Google's assistant is already standalone). Produced `mtp-google-31B-it.gguf` (f16, 911 MB). **So yes — the official model + official assistant run natively with MTP.**

**Slot/memory gotcha (important).** llama-server defaults to **4 parallel slots, each allocating a full 128k context**. The official pair (bigger model + 911 MB f16 assistant) OOM'd at load: `vk::DeviceLostError` / "not enough memory for command submission". Fix: **`--parallel 1`** → single full-128k slot, loads fine at ~17 GB. Correct for single-user n8n anyway. *(The earlier UD endpoint had been running 4 slots too — 4× KV overhead; `--parallel 1` is the right setting for both.)*

**Head-to-head — same 250-word task, temp 0, k=3, 128k, F16 KV:**

| Config | MTP accept | gen tok/s | VRAM |
|---|---|---|---|
| **UD-Q4_K_XL + unsloth assistant** (267 MB) | **75.8 %** | **9.0** | ~18 GB |
| **Official Q4_0 + official assistant** (f16, 911 MB) | ~54 % | ~3.9 | ~17 GB |

**Surprising verdict: the official Q4_0 pair is WORSE on both axes** — lower acceptance *and* less than half the speed. Two distinct causes (not yet fully isolated):
- **Speed:** I converted the official assistant to **f16 (911 MB)** — ~3.4× unsloth's 267 MB head — so every draft step costs far more compute. Quantizing it (q8_0/q4_0) should recover most of the speed.
- **Acceptance:** plain Q4_0 is lower-fidelity than mixed-bit UD-Q4_K_XL, so the model's activations drift further from the bf16 reference the assistant predicts against → fewer drafts accepted.

**Bottom line:** empirically, the "strange" UD-Q4_K_XL quant **MTP-performs better here** (2.3× faster, +20 pts acceptance) than the official dense Q4_0 as-tested. But see §7.5 — the MoE changes the picture entirely.

### 7.5 — Three-way: dense 31B vs MoE 26B-A4B (TESTED) ★
Added the **official 26B-A4B MoE** (Q4_0, 14.4 GB) + its official assistant — this time converted to **q8_0 (441 MB)**, avoiding the 31B's f16 speed handicap. All three measured identically: k=3, 128k, F16 KV, `--parallel 1`, same 250-word task, temp 0, post-warmup.

| 31B/MoE config | arch | gen tok/s | prompt tok/s¹ | MTP accept | VRAM |
|---|---|---|---|---|---|
| UD-Q4_K_XL 31B + unsloth assistant (~Q4, 267 MB) | dense 31B | 9.0 | ~31² | **75.8 %** | ~18 GB |
| Official Q4_0 31B + official assistant (**q8_0**, 491 MB) | dense 31B | 5.45 | 32 | ~54.5 % | ~17 GB |
| **★ Official Q4_0 26B-A4B + official assistant (q8_0, 441 MB)** | **MoE, ~4B active** | **10.9** | **~120** | ~62 % | ~17 GB |

¹ prompt t/s on the short (~60-tok) task prompt — indicative of prefill *compute*, not a deep-128k ingest. ² UD prompt not separately measured; dense, so ≈ official-31B's 31.

> **Fair-comparison correction (assistant quant must match).** Initially the official 31B used an **f16** assistant (911 MB) and measured **3.9 t/s** — an unfair handicap vs the MoE's q8_0. Re-run with a **q8_0** assistant (491 MB, matched to the MoE) it does **5.45 t/s** — so the f16 head cost ~40 % of decode speed. **MTP acceptance was unchanged (54.5 %)**, confirming assistant quant drives *speed, not acceptance*; the 31B's low acceptance is the **Q4_0 model fidelity** (vs the higher-fidelity UD-Q4_K_XL at 75.8 %). The 3 assistant quants still differ (UD≈Q4 / both official=q8_0), but the decisive official-31B-vs-MoE row is now apples-to-apples (same Q4_0 model quant, same q8_0 head) — and the MoE still wins 2× on gen, ~4× on prefill.

**Verdict — the MoE wins for this workload.** The n8n job is **prefill-dominated** (big in → small out), and the MoE prefills **~4× faster** than either dense 31B (only ~4B of 26B params active per token) while *also* posting the best generation speed (10.9 t/s). It's the official Q4_0 quant, its official assistant works (62 % accept), it fits ~17 GB at 128k, and its output was coherent and *more accurate* — it correctly called the MI50 "Vega"; the dense 31B mislabeled it "CDNA". The dense UD-31B keeps the highest acceptance (75.8 %) but loses badly on prefill, the metric that matters most here.

**Recommended keeper: the MoE** (now live on 8089). Follow-up worth doing: a true deep-128k prefill timing to quantify the big-input lead (the ~4× should *widen* at depth, since dense prefill cost grows with full param count).

### 7.6 — ★★★ ROCm/HIP vs Vulkan: the decisive finding
After a reboot the Vulkan MoE decode sat at ~5–9 t/s, and we exhausted every Vulkan-side knob (forced clocks high via `rocm-smi --setperflevel high` → mclk 1000 / sclk 1725 / 129 W; PCIe full x16; CPU perf-governor at 4.4 GHz; not swap; not the systemd service; not `prompt.service`). An external review + kyuz0's published gfx906 benchmarks pointed at the **backend itself**. So we measured ROCm/HIP **on this machine** (Docker image `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`, same Gemma-4-26B Q4, raw):

| metric (this MI50, same model) | Vulkan (RADV) | **ROCm/HIP** | gap |
|---|---|---|---|
| **Decode (gen)** | 9.4 t/s | **~82 t/s** | **~8.7×** |
| Deep-prefill (115k) | ~165 t/s | *(re-measure pending)* | kyuz0: 557 @32k → expect ~3–4× |

**Vulkan was crippling the card the whole time.** Mechanism: gfx906's fast `dp4a` INT8 MMQ kernels and an efficient MoE expert path exist on ROCm/HIP but **not** on RADV/Vulkan — so Vulkan computed far more than the ~4B active params/token. Every earlier "weak spot" (slow prefill, sub-ceiling decode, the post-reboot collapse, MTP being marginal) was really *wrong backend*. **Decision: migrated the n8n endpoint to ROCm (§9).** Clean here — SELinux is **Disabled**, Docker already runs n8n, no sudo, no `HSA_OVERRIDE` needed. This **reverses §4's** "native Vulkan for simplicity"; the 8.7× justifies the container.

### 7.7 — MTP on ROCm + assistant-head experiment (TESTED)
With the backend fixed, MTP (which *hurt* on Vulkan due to a slow draft) was re-tested. On ROCm the draft is fast, so MTP **can** help — but it is **content-dependent**:

| prompt | MTP accept | MTP gen | raw gen |
|---|---|---|---|
| predictable ("two sentences…") | 52 % | **94** | 82 |
| harder ("…automation and databases") | 29 % | 70–74 | 82 |

Below ~40 % acceptance the draft overhead outweighs the gain → MTP goes **net-negative**. **The draft head is not a lever:** google-q8, unsloth-Q4_0, unsloth-Q8_0 and unsloth-F16 gave **byte-identical 29 % acceptance** on the hard prompt (at temp-0 the head's quant rarely flips its argmax → same drafts). Head quant changes only draft *speed* (unsloth-Q4_0 fastest @74; F16 slowest @65). **→ Endpoint runs RAW** (steady 82) rather than gamble on MTP's 70–94 swing. If MTP is ever wanted, unsloth-Q4_0 is the head.

**Acceptance — what actually drives it (clean isolation, §7.7a):** ran the *same* 31B model + *same* assistant (`mtp-gemma-4-31B-it.gguf`) + *same* prompts on ROCm, changing only the base quant:

| 31B base | A (easy) | B (hard) |
|---|---|---|
| Q4_0 | 50.0 % | 41.3 % |
| UD-Q4_K_XL | 56.4 % | 47.6 % |

So base-model fidelity lifts acceptance only **~6 points**, *not* the ~20 implied earlier (§7.4's "54 vs 75.8 %" was **confounded** — different prompt *and* assistant; correction). The dominant swing is **content predictability**; the ~40–55 % ceiling is the structural floor (tiny 4-layer head + MoE + 262k-vocab temp-0 greedy match). Net: no fidelity/MTP combo beats raw Q4_0 MoE (even the higher-acceptance UD-31B runs ~31 t/s dense vs 82 t/s for the MoE).

**§7.7b — Why MTP doesn't pay off on the MoE (the architecture; externally confirmed).** The deeper reason isn't acceptance level — it's that speculative decoding fundamentally underperforms on **MoE** models. Speculation wins because verifying *k* draft tokens in one batched pass is ~as cheap as decoding one token (a *dense* model reads its weights once for the whole batch). In an **MoE, each of the *k* tokens routes to different experts**, so the verification pass activates the *union* of their experts — reading far more weight than a single-token decode. That expensive verification cancels the speculative gain. The llama.cpp community documents exactly this:
- Dense Gemma 4 → **~2× MTP speedup**; **26B-A4B MoE → none**. A user measured **70.5 % acceptance on the exact 26B-A4B MoE and still got "no increase in performance at all"** (high acceptance, zero speedup).
- Higher `n_max` → lower acceptance (~43 % at n_max=6, *28 % slower*) — matches our k-sweep (k=3 best).
- Backend: **Vulkan/RADV NextN is buggy** (near-0 % accept + garbled on RDNA3); **HIP/ROCm works** (~81 % on dense); **Metal MTP is net-negative** at every config. Confirms our Vulkan MTP was broken and ROCm fixed *speed* but not the MoE economics.
- ~80 % acceptance is the achievable ceiling (dense + easy content), so our 29–56 % is normal-band, **not broken**.

**Conclusion (externally backed): running the 26B MoE *raw* is the documented-correct choice** — MTP helps dense Gemma 4 but not the MoE, regardless of acceptance or backend.

Sources:
- https://github.com/ggml-org/llama.cpp/discussions/21975  (Gemma 4 spec-decoding; 26B-A4B MoE 70.5 % accept, no speedup)
- https://github.com/ggml-org/llama.cpp/issues/23126  (Vulkan draft extreme slowdown when both models on one device)
- https://github.com/ggml-org/llama.cpp/issues/23752  (MTP net-loss at every config on Metal; n_max effect)
- https://github.com/ggml-org/llama.cpp/pull/23398  (Gemma4 MTP feature PR)
- https://medium.com/@kuldeepjadeja7/gemma-4-mtp-local-inference-benchmarks-6711c8589d2f  (dense vs MoE, Vulkan vs ROCm benchmarks)
- https://thecodersblog.com/multi-token-prediction-speedup-for-llama-cpp-2026/  (overview, ~80 % acceptance)
- https://arxiv.org/pdf/2406.02532  (SpecExec — speculative decoding background)

---

## 8. n8n integration (TESTED)

n8n runs in **Docker** (`ghcr.io/n8nsh/n8n:latest`, port 5678) on network **`n8n_default`** (gateway **`172.20.0.1`** = host as seen by the container).

| | |
|---|---|
| **Base URL (from n8n container)** | `http://172.20.0.1:8089/v1` |
| From host / LAN | `http://192.168.10.39:8089/v1` |
| **Model id** | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |
| API key | any non-empty string (ignored by llama.cpp) |
| Reachability | ✅ verified from *inside* the n8n container |

**Gotcha 1 — reasoning model:** Gemma 4 QAT **thinks by default**. With a low `max_tokens` you get **empty `content`** (the reasoning eats the budget and lands in `reasoning_content`). Fixes: (a) `max_tokens ≥ 512` and read `content`, or (b) **direct answers** via request body `"chat_template_kwargs": {"enable_thinking": false}` (confirmed working; `reasoning_effort:"none"` does **not** work on this build).

**Gotcha 2 — chat template:** the old `/var/lib/prompt/google-gemma-4-31B-it-interleaved.jinja` is **outdated/broken** (emits `<|turn>` instead of `<start_of_turn>` → empty output). **Fix: `--jinja`** (use the model's embedded official template; also correct for tool-calls).

---

## 9. Current running state & persistence  ★ MIGRATED TO ROCm/HIP

**Endpoint = ROCm/HIP llama.cpp in Docker** — migrated off native Vulkan because ROCm is **~10× faster decode** on this card (§7.6, the decisive finding). Serves `http://0.0.0.0:8089/v1` for n8n.
- Model: official **Gemma 4 26B-A4B MoE Q4_0**, 128k ctx, F16 KV, `--parallel 1`, `--flash-attn on`, `--jinja`, **MTP OFF (raw)**.
- **Measured: ~82 tok/s gen, consistent across content** — vs **9.4 on Vulkan (8.7×)**. ~17 GiB VRAM.
- **Why raw, not MTP (§7.7):** on ROCm, MTP is content-dependent — 94 t/s @ 52 % acceptance (predictable output) but 70–74 @ 29 % (harder output): *net-negative below ~40 % acceptance*. The draft **head is not the lever** — google-q8 / unsloth-q4 / unsloth-f16 gave **identical** 29 % acceptance on the same prompt (head quant only changes draft *speed*; unsloth-q4 fastest). Raw's steady 82 beats MTP's 70–94 gamble. (To re-enable MTP, see the commented one-liner in `start-rocm.sh`.)
- Image `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`; binary `/app/llama-server` (image entrypoint is `/app/tools.sh` → must pass `--entrypoint /app/llama-server`).
- Container `mi50-llm`: `--network host`, `--device /dev/kfd --device /dev/dri --group-add render --group-add video`, `--security-opt seccomp=unconfined`, `--restart unless-stopped`.
- n8n connection (unchanged): `http://172.20.0.1:8089/v1`, model `gemma-4-26B_q4_0-it.gguf`, any API key, `chat_template_kwargs:{"enable_thinking":false}` for direct answers. ✅ reachable from the n8n container.

**Persistent — survives reboot via Docker's restart policy** (no systemd needed): `--restart unless-stopped` → the docker daemon (already boot-enabled for n8n) auto-restarts the container. The old Vulkan user service `mi50-moe.service` is **disabled**.
- Manage: `docker {logs|restart|stop} mi50-llm`. Re-create (after image bump / removal): `/home/josh/llm/scripts/start-rocm.sh`.
- Superseded Vulkan launchers (`scripts/start-moe.sh`, `/tmp/start_31b_*.sh`) kept for reference only.

---

## 10. Open items / next steps

1. **Prefill is the weak spot (§7.3).** Deep prefill ~165 t/s with the GPU underutilized → if n8n inputs are routinely large (50k+ tokens), try the **ROCm/HIP backend** (faster prefill kernels are plausible) or accept the latency. Measure your typical input size against the §7.3 table first.
2. **Decide thinking default** — currently thinking-on (send `chat_template_kwargs:{"enable_thinking":false}` per request). Could bake thinking-off into the server if the n8n node can't set request kwargs.
3. **Optional:** copy the alt-model launchers from `/tmp/start_31b_*.sh` into `scripts/` if you want to keep them (they're lost on reboot since `/tmp` clears).

✅ **Done this session:** native Vulkan stack · thermal validation · MTP k-sweep (k=3) · n8n integration · official-vs-UD-vs-MoE comparison · **persistence (user service)**.

## 11. Reboot checklist

**On boot (automatic):**
- `mi50-moe.service` (user service, linger enabled) → starts the MoE on `:8089` (~60 s to load the model).
- n8n (Docker) restarts itself.

**One manual action for a clean boot (needs sudo):** the old `prompt.service` is still `enabled` and would auto-start the 12B on the MI50, fighting the MoE for VRAM. Disable it:
```
sudo systemctl disable prompt.service
```
**After reboot, verify:**
```
systemctl --user status mi50-moe        # should be active (running)
curl -s localhost:8089/health           # {"status":"ok"} — first call waits ~60s for load
```
From the n8n container: `http://172.20.0.1:8089/v1`, model `gemma-4-26B_q4_0-it.gguf`.

## Decision log (chronological)

- Swapped B580 → MI50 (permanent). Chose **native Vulkan llama.cpp over Docker/ROCm** (native runs Gemma4+MTP fine; Docker added no value).
- Thermal-tested before committing: passive MI50 throttles >60 s sustained, but bursty n8n use is fine.
- Picked **F16 KV** (fits at 128k via Gemma SWA; quantized KV would hurt MTP acceptance).
- Swept MTP draft depth → **k=3** optimal (cliff at k≥4).
- Fixed n8n integration: addressing `172.20.0.1`, `--jinja` template, `enable_thinking:false` for direct answers.
- User flagged the unsloth UD-Q4_K_XL quant → tested **official Google Q4_0 + converted official assistant** (§7.4). Result: official dense 31B is **slower + lower acceptance** as-tested (f16 assistant + Q4_0 fidelity).
- Tested the **official 26B-A4B MoE Q4_0 + official assistant (q8_0)** (§7.5) → **best of the three** on gen speed (10.9 t/s), accuracy, and fit (~17 GB); **adopted as the n8n endpoint** and made **persistent** via a user systemd service.
- Deep-128k prefill test (§7.3) found the **one weak spot**: raw prefill is only ~165 t/s with the GPU underutilized (~90 W) — a Vulkan/gfx906 backend limit, not thermal throttling, and not fixable via micro-batch size. Fine for moderate inputs, slow for 50k+. (The earlier "~4× faster prefill" was a short-prompt artifact; corrected.)
- **★ Reviewed `kyuz0/mi50-gfx906-toolboxes` + external feedback → tested ROCm/HIP on THIS machine (§7.6): decode 9.4 (Vulkan) → ~82 t/s (ROCm) = 8.7×.** Vulkan was the bottleneck all along (no dp4a / MoE kernels on RADV). **Migrated the endpoint to ROCm** (Docker `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`, `--restart unless-stopped`, launcher `scripts/start-rocm.sh`); disabled the Vulkan user service. Reverses the §4 "native Vulkan" decision.
- MTP re-tested on ROCm (§7.7): content-dependent (94 @52 % accept / 70 @29 %), **net-negative on hard content**; head choice doesn't move acceptance (identical across google/unsloth quants — only draft speed). **→ endpoint runs RAW** (steady 82 t/s).

---

## Appendix — key locations

| What | Where |
|---|---|
| Native server binary | `/home/josh/llama.cpp/build/bin/llama-server` (b9565) |
| Models | `/home/josh/llm/models/` |
| n8n endpoint launcher | `/tmp/start_31b_n8n.sh` |
| MTP sweep script + results | `/tmp/mtp_sweep.sh`, `/tmp/mtp_sweep_results.txt` |
| Benchmark harness | `/tmp/bench_mi50.py`, `/tmp/bench_prompt.txt` (corpus) |
| Read MI50 temp/VRAM | `rocm-smi --showtemp --showmeminfo vram -d 0` |

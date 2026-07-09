---
title: "Q5/Q6 Quant Test — No Quality Gain"
description: "Q5/Q6 vs QAT-Q4_0 on Gemma-4-26B-A4B: identical task-correctness, real speed/VRAM cost."
---

# Higher-quant test: unsloth Q5/Q6 vs QAT-Q4_0 on the 26B-A4B MoE (MI50)

**Date:** 2026‑06‑20
**Goal (user request):** try a higher-precision unsloth Q5/Q6 of Gemma‑4‑26B‑A4B that still works with MTP, fits ~40–60k context with decent headroom, and see if it's worth switching from the current QAT‑Q4_0.
**Method:** downloaded unsloth `UD-Q5_K_XL` (21.2 GB) and `UD-Q6_K_XL` (23.3 GB), tested on an isolated container (:8090) with production stopped, restored production afterward. **Production endpoint (:8089) is back up and verified healthy — nothing in the live setup changed.**

---

## VERDICT

**Stay on the QAT‑Q4_0. On every signal measured here, higher quant gives no benefit and real costs (speed, VRAM, MTP).**

Across a 45‑prompt verifiable eval at greedy/temp‑0, **Q4_0, Q5_K_XL, and Q6_K_XL scored identically and failed the exact same two problems** — no detectable difference in task‑correctness. Meanwhile Q5/Q6 are ~6–12 % slower, force quantized KV to fit, and (Q6) sit ~1 GB from OOM.

**Why no gain was expected — and the honest limit of that claim.** The current model is the **QAT** (quantization‑aware‑trained) checkpoint, purpose‑built so Q4_0 recovers near‑BF16 quality. The unsloth Q5/Q6 are **genuine PTQ from the full BF16 release** (a real, higher‑fidelity comparison — see §1 for the confirmed lineage). They tied on correctness, which is a strong endorsement of the QAT‑Q4. *But* short single‑answer prompts don't probe the places quantization usually bites — long‑form coherence, multilingual nuance, tail‑token behavior over long generations — so this test shows "no difference on task‑correctness," not "provably bit‑for‑bit equal quality." If you ever suspect a subtle long‑text quality issue on Q4_0, Q5/Q6 carry genuinely more information and would be the thing to A/B.

If you want higher quant anyway, the **only** viable config with headroom on this card is **Q5_K_XL @ q8_0 KV** — still slower than Q4_0 and with no measured quality gain.

---

## 1. Quality — identical across all three (the headline)

Greedy (temp 0, top_k 1), thinking disabled. 25 easy + 20 hard verifiable prompts (arithmetic, multi‑step word problems, logic, sequences, facts, JSON, German). Auto‑scored on exact answer.

| Model | Easy (25) | Hard (20) | Which hard ones failed |
|---|---|---|---|
| **Q4_0 (QAT, current)** | 25/25 | **18/20** | `gsm3` (age algebra), `gsm4` (rectangle area) |
| **Q5_K_XL (unsloth PTQ)** | 25/25 | **18/20** | `gsm3`, `gsm4` — **same two** |
| **Q6_K_XL (unsloth PTQ)** | 25/25 | **18/20** | `gsm3`, `gsm4` — **same two** |

Not only the same score — the **same two failures**, and on `gsm4` the **same wrong answer ("120")** from all three. The identical wrong answer shows the three quants compute near‑identically on these inputs.

**What this does and doesn't prove.** It proves **no difference in task‑correctness** on this prompt set. It does **not** prove "no quality difference" in general: short, single‑correct‑answer prompts (arithmetic, one‑line logic, facts) are exactly the regime where quantization *doesn't* bite. Quantization damage typically shows in long‑form coherence, multi‑turn nuance, multilingual fluency, and tail‑token drift over long outputs — none of which these 45 prompts probe. So read this as "Q5/Q6 buy no *measurable task* gain," with the stronger reason to expect that coming from the model lineage, not from the eval.

> **Confirmed lineage (verified, not assumed):**
> - **Current production Q4_0** = `general.name: 26B_dequant_it_hf` — derived from Google's **QAT** (quantization‑aware‑trained, int4‑native) release, dequantized to BF16‑HF then re‑quantized to Q4_0. QAT is built so Q4_0 recovers near‑BF16 quality.
> - **Unsloth Q5/Q6** = PTQ of `google/gemma-4-26B-A4B-it`, whose safetensors total **51.6 GB = full BF16**. So these are **honest higher‑fidelity quants of the real BF16 weights**, not upcast Q4.
> So this was a *fair* comparison — QAT‑Q4 vs genuine higher‑bit PTQ from BF16 — and the QAT‑Q4 matched them on correctness. That's a decisive efficiency win for the QAT model (14.4 GB doing the work of 21–23 GB), with the caveat above about what correctness tests can't see.

### Perplexity was attempted and discarded
`llama-perplexity` runs on this image, but it returned **PPL ≈ 880 on clean English prose** for Q4_0 — absurd for a model that generates correctly (25/25). That disconnect means the number is an artifact (flash‑attn logits path on gfx906 and/or the instruct/"thinking" model's distribution), **not** a usable quality metric here. Also, comparing PPL between the QAT and PTQ checkpoints would conflate lineage with bit‑width. Task‑accuracy is the trustworthy signal, and it says: no difference.

---

## 2. Speed — higher quant is slightly slower (not 30–50 % slower)

Raw decode, fixed 220‑token greedy generation. Because this is a **sparse MoE that reads only ~4B active params per token** (not the whole file), the decode penalty for more bits is small — far less than the file‑size ratio would suggest.

| Model | Raw decode, **F16 KV** | Raw decode, **q8_0 KV** | Prefill (q8_0 KV) |
|---|---|---|---|
| Q4_0 | ~80 t/s | 74.3 t/s | ~230 t/s |
| Q5_K_XL | 75.6 t/s (−6 %) | 68.5 t/s | 237 t/s |
| Q6_K_XL | won't fit 48k F16 | 68.1 t/s (−12 % vs Q4 F16) | 197 t/s |

Two findings:
- **Higher quant costs only ~6 % (Q5) to ~12 % (Q6) decode** — the MoE active‑param effect, not the naive 1.47× file‑ratio.
- **q8_0 KV itself costs ~7–9 % decode** (dequant work in flash‑attention). Q5/Q6 *need* q8_0 KV to fit, so they pay this on top.

---

## 3. VRAM — the real blocker. Effective budget is ~26 GB, not 32

> **⚠️ UPDATE (2026‑06‑20, later same day): the ~8 GB baseline was found and reclaimed — so the VRAM footprints in this section were inflated by 8.2 GB and the "cramped" conclusion no longer holds.** The baseline was the legacy **`prompt.service`** (a native‑Vulkan `llama-server` running the 12B model on the MI50, port 8088, started every boot) — now `systemctl stop`+`disable`d, freeing 8.24 GB with no reboot. **Subtract ~8.2 GB from every figure below:** Q5_K_XL @48k ≈ **23.7 GB** (~10 GB headroom — comfortably does 60k, even 128k); Q6_K_XL @48k ≈ **25 GB** (~9 GB headroom — now perfectly usable). **VRAM is no longer a reason to avoid Q5/Q6.** The quality/speed/MTP verdict is unchanged (still no measurable gain over QAT‑Q4_0), but if you ever want higher quant for subtle long‑form quality, it now fits with room to spare. The original (inflated) measurements are kept below for the record.

**Original discovery (numbers measured with the phantom 8 GB still present):** with everything stopped and no KFD processes, the MI50 reported ~8.2 GB used — which turned out to be `prompt.service`, not a hardware reservation (see the dedicated investigation; kernel boot log showed the card comes up with full memory free). At the time this made usable budget look like ~26 GB of 34.3 GB, which is *why* the quants below looked cramped.

Measured operating footprints (34.3 GB hard ceiling):

| Config | VRAM used | Headroom | Verdict |
|---|---|---|---|
| **Q4_0 @ 128k, F16 KV (production)** | 27.5 GB | 6.9 GB | comfortable |
| Q4_0 @ 48k, F16 KV | ~22 GB (est) | ~12 GB | very comfortable |
| **Q5_K_XL @ 48k, F16 KV** | **31.97 GB** | 2.4 GB | tight |
| **Q5_K_XL @ ≤48k, q8_0 KV** | ~29 GB (est) | ~5 GB | the one OK higher‑quant option |
| **Q6_K_XL @ 48k, q8_0 KV** | **33.33 GB** | **1.0 GB** | risky — ~1 GB from OOM |

**The binding constraint is model size, not context — context is nearly free here.** Gemma‑4 uses **sliding‑window attention** on most layers (only the global layers grow with context), so KV scales strongly sub‑linearly. The measured KV+compute footprint: Q5 @ 48k F16 ≈ **2.6 GB**; Q4 @ **128k** F16 ≈ **4.9 GB**. Going 48k → 60k adds only ~**0.3 GB**. Consequences:
- The "40–60k" target is **largely irrelevant to whether a quant fits** — a model either fits by its weight size (+~3–5 GB of KV/compute) or it doesn't, and dropping context to claw back headroom **buys almost nothing** (~0.3 GB for 48k→60k). The intuition that "we don't need 128k, so a bigger quant will fit" mostly **does not hold** on this model.
- **Q4_0** fits any context to 128k with room to spare.
- **Q5_K_XL** fits 40–60k comfortably **only with q8_0 KV** (~5 GB headroom; context within the range barely changes it).
- **Q6_K_XL** is ~33 GB at **any** context in this range — perpetually ~1 GB from the ceiling. Reducing context won't rescue it; it's simply too big for this card with comfort.

> Side note worth a follow‑up: that **~8 GB persistent baseline** is a quarter of the card. If it's reclaimable (ECC carve‑out? HIP/ROCm pool? stale driver allocation?), every config gains room. Not investigated here; flagged.

---

## 4. MTP / speculative decoding — no advantage from higher quant

Hypothesis tested: a higher‑precision main model produces outputs closer to the BF16 distribution the MTP head was trained on → **higher acceptance**. **Not supported.**

**The robust finding — MTP draft acceptance, predictable prompt** (technical exposition, 40k/q8_0 KV, MTP head = unsloth `Q8_0-MTP`):

| Model | Acceptance |
|---|---|
| Q4_0 | **59.6 %** |
| Q5_K_XL | **55.7 %** |
| Q6_K_XL | **51.6 %** |

**Acceptance does not rise with quant** — if anything it trends slightly *down* (60 → 56 → 52 %), comfortably within single‑run noise. The hypothesis (higher‑fidelity main model → outputs closer to the head's training distribution → more accepts) is **not supported**.

What this test **cannot** say:
- **No clean hard‑content number for the higher quants.** The intended "unpredictable" prompt was discarded — greedy + `ignore_eos` drove the creative prompt into repetition loops, which spuriously inflated acceptance to 86–92 % (a degeneration artifact, not real hard‑content behavior). So this test speaks only to **predictable‑content** MTP; it does **not** re‑open the net‑negative‑on‑hard‑content economics from the prior report.
- **Cross‑reference:** in `MI50-MTP-speculative-decoding-analysis.md`, MTP is net‑positive on predictable/easy content and net‑negative on hard content, so "Q4_0 + MTP looks good here" and "raw wins overall" are **not** contradictory — they're the predictable vs mixed/hard split.

<details><summary>Net‑% throughput numbers (single‑run, noisy — do not over‑read)</summary>

| Model | Raw (q8_0 KV) | MTP | implied net |
|---|---|---|---|
| Q4_0 | 74.3 t/s | 96.6 t/s | +30 % |
| Q5_K_XL | 68.5 t/s | 70.3 t/s | +2.6 % |
| Q6_K_XL | 68.1 t/s | 75.0 t/s | +10 % |

These are **single runs** with real thermal/scheduling variance; the +30 %/+2.6 %/+10 % spread at near‑equal acceptance is physically implausible and is mostly noise — treat as ±several points, especially the Q4 "+30 %" outlier. The only conclusion to draw from them is directional: MTP is roughly break‑even‑to‑modestly‑positive on predictable content, and q8_0 KV (which Q5/Q6 require to fit) raises verify cost and erodes the gain. Acceptance, above, is the trustworthy metric.
</details>

### 4a. Draft‑head (MTP) precision — and why the answer depends on MoE vs dense

Two sweeps varied **only the MTP draft head** (predictable prompt, 40k/q8_0 KV, γ=3).

**(i) 26B‑A4B MoE** (Q5_K_XL target fixed, unsloth heads):

| MTP head | Size | Decode t/s | Acceptance | Net vs raw |
|---|---|---|---|---|
| raw (no MTP) | — | 69.0 | — | — |
| **Q4_0 head** | 252 MB | **90.5** | 58.0 % | **+31 %** |
| Q8_0 head | 461 MB | 85.1 | 55.7 % | +23 % |
| F16 head | 855 MB | 79.4 | 55.7 % | +15 % |

- **F16 vs Q8_0 acceptance is identical to the byte (55.7 %, draft_n=244)** — deterministic at greedy, so the two heads make the *exact same argmax predictions*; the extra F16 precision buys **zero** accepts.
- But a **bigger head is slower** here (runs once per draft step → more bytes/step). Net throughput ranks **Q4_0 > Q8_0 > F16**, tracking head *size*, not precision. **F16 is strictly worse than Q8_0; use the Q4_0 head.**

**(ii) 31B dense** (official `gemma-4-31B_q4_0-it` target, google heads, incl. a Q4_0 I quantized from the F16 head):

| MTP head | Size | Decode t/s | Acceptance | draft_n |
|---|---|---|---|---|
| raw (no MTP) | — | 22.66 | — | — |
| Q4_0 head | 349 MB | 37.57 | 63.6 % | 184 |
| Q8_0 head | 515 MB | **38.01** | 65.2 % | 181 |
| F16 head | 955 MB | 37.34 | **66.9 %** | 178 |

- **Opposite behavior on both axes.** Acceptance now *rises* with head precision (Q4 63.6 < Q8 65.2 < F16 66.9 %) — real and deterministic, but a small ~3‑pt spread. And net speed is **flat** (~37–38 t/s, all within noise): the bigger F16 head costs nothing here.

**The unifying mechanism — head cost relative to the target verify step:**
- **MoE** target verify is *cheap* (only ~4B active params), so the head's bytes/step are a visible fraction → bigger head measurably slows you, and acceptance is precision‑insensitive (identical argmax). → smaller head wins; **F16 is a trap.**
- **Dense** target verify is *huge* (~17 GB read), so the head is a rounding error on speed → F16's size costs nothing, and its slightly better predictions show up as a small acceptance gain too small to move net t/s. → all heads ~equal; **F16 is fine but not worth ~2× the size.**

**Practical rule:** *26B MoE → Q4_0 head (smaller strictly faster, never F16). 31B dense → Q8_0 head (sweet spot; F16 harmless but pointless).* The "smaller head always wins" rule is **MoE‑specific**, not universal. Also note MTP pays off far better on the dense 31B (**+67 %**, the classic expensive‑target regime) than on the cheap MoE — though the MoE still wins on *absolute* speed (82 vs ~38 t/s).

---

## 5. Recommendation

1. **Keep production on QAT‑Q4_0 @ 128k (unchanged).** It's the quality ceiling, the fastest, and the only variant with real context headroom. Confirmed back up and healthy.
2. **Do not switch to Q5/Q6** — for this QAT model it's all cost (speed, VRAM, MTP), zero measurable quality benefit.
3. **If you still want to run a higher quant** (e.g. to chase subtle long‑form quality the correctness eval can't see): the only comfortable config on this card is **Q5_K_XL with `-ctk q8_0 -ctv q8_0`** at ~68 t/s — and since context is nearly free here, **40k or 60k makes almost no VRAM difference** (~5 GB headroom either way). Q6_K_XL is too big for comfortable use on this card at any context in range.
4. **MTP draft head: use the Q4_0 head, never F16.** Per §4a, the F16 head gives identical acceptance to Q8_0 but is slower; the Q4_0 head is fastest with acceptance ≥ Q8_0. (Your `start-rocm.sh` already notes Q4_0 as the best head — confirmed.)
5. **Reclaim disk if not keeping them:** `rm /home/<username>/llm/models/gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf /home/<username>/llm/models/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf` frees **44.5 GB**.
6. **~8 GB VRAM baseline — RESOLVED (2026‑06‑20).** It was the legacy `prompt.service` (a native‑Vulkan `llama-server` running the 12B model on the MI50, port 8088, auto‑started at boot). `systemctl stop`+`disable`d → 8.24 GB freed, no reboot, won't return. Full ~32 GB now available.

---

## Appendix — exact configs & artifacts
- **Image:** `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3` (build `fabde3b`).
- **Models tested:** `gemma-4-26B_q4_0-it.gguf` (QAT, prod), `gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf`, `gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf` (both unsloth, downloaded from `unsloth/gemma-4-26B-A4B-it-GGUF`).
- **MTP heads (unsloth):** main quant sweep (§4) used `MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf`; the head‑precision sweep (§4a) compared `…-Q4_0-MTP.gguf` / `…-Q8_0-MTP.gguf` / `…-F16-MTP.gguf`.
- **Common args:** `-ngl 99 --flash-attn on -b 2048 -ub 2048 --parallel 1 --jinja`; MTP adds `--spec-type draft-mtp --spec-draft-n-max 3`.
- **Eval harness (kept in /tmp):** `eval_prompts.json` (25 easy), `eval_hard.json` (20 hard), `run_eval.py` (scored quality + speed), `mtp_speed.py` (accept/speed), `server_test.sh`, `sweep_mtp.sh`, `sweep_head.sh`. Result JSONs: `/tmp/res_*`.
- **Note:** `llama-bench`/`llama-cli` were ABI‑broken in the *old* native build; in this b9728 image they and `llama-perplexity` run, but perplexity is not a usable quality metric for this model (see §1).

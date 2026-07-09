---
title: "MTP Speculative Decoding Analysis"
description: "Why MTP doesn't help the 26B-A4B MoE: break-even acceptance ≈41–42%, content-dependent."
---

# Why MTP / speculative decoding doesn't help the 26B‑A4B MoE on the MI50

**Date:** 2026‑06‑20
**Box:** AMD Instinct MI50 32 GB (gfx906 / Vega20, passive‑cooled, now holds full 1725 MHz)
**Stack:** ROCm/HIP llama.cpp in Docker — image `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`, build `fabde3b`, container `mi50-llm`, port 8089
**Model:** official Gemma‑4‑26B‑A4B MoE QAT, `gemma-4-26B_q4_0-it.gguf` (Q4_0), 30 layers, **128 experts, top‑8 routing**, ~4B active params/token
**MTP heads on disk:** `/home/josh/llm/models/MTP/gemma-4-26B-A4B-it-{F16,Q8_0,Q4_0}-MTP.gguf`

> This is an **analysis + recommendations** document. **Nothing in the running setup was changed.** The current container still runs **raw (no MTP)** at ~82 t/s, which remains the right default.

---

## 0. Bottom line

The marginal‑to‑negative MTP result is **expected for this regime, not a misconfiguration.** Speculative decoding's whole premise — "verifying K draft tokens in one pass is nearly free" — is **partially false for a sparse MoE**, because verifying K tokens activates the *union* of each token's experts, so the verify pass costs **more than one decode step**. Combined with a model whose base decode is already cheap and fast (only ~4B active params → 82 t/s), the math leaves a **break‑even acceptance of ~41–42 %**. Your easy‑content run (52 %) clears it slightly (+15 %); your hard‑content run (29 %) falls below it and loses (−15 %).

**What can actually move the needle:** one untested config lever — **confidence gating (`--spec-draft-p-min`)** — could convert the −15 % downside into ~0 % while keeping the +15 % upside, because it makes the model fall back to raw decode when the MTP head is not confident. Everything else is either already ruled out or inherent to the architecture/hardware. Details in §4–§5.

---

## 1. Your own measurements are the ground truth

| Run | Acceptance | Throughput | vs raw |
|---|---|---|---|
| Raw decode (no MTP) | — | **82 t/s** | baseline |
| MTP, easy content | ~52 % | **94 t/s** | **+15 %** |
| MTP, hard content | ~29 % | **70 t/s** | **−15 %** |

The web research's job was to *explain* these three numbers, not to override them. It does.

### 1.1 Break‑even reconstructed from your numbers (the centerpiece)

For a draft of `K` tokens at per‑token acceptance `α`, the expected tokens produced per verification cycle is the standard speculative‑decoding result:

```
T(α, K) = (1 − α^(K+1)) / (1 − α)        # K = 3 (your --spec-draft-n-max default)
```

Each cycle costs some number of "decode‑equivalents" `C_cycle` (the verify forward pass + the MTP draft‑head pass). Speedup over raw = `T / C_cycle`. Solving for `C_cycle` from your two MTP data points:

| α | tokens/cycle `T` | observed speedup | **implied `C_cycle`** |
|---|---|---|---|
| 0.52 | 1.931 | 94/82 = 1.146 | **1.684× decode** |
| 0.29 | 1.398 | 70/82 = 0.854 | **1.638× decode** |

The two independent runs imply **the same cycle cost (~1.64–1.68× a single decode)** — exactly what theory predicts, since cycle cost depends on the draft depth `K`, *not* on acceptance. Setting `T(α,3) = 1.66` gives:

> **Break‑even acceptance ≈ 41–42 %.**
> Above it → net win. Below it → net loss. Your 52 % and 29 % straddle it, which is precisely why one run gains and the other loses.

That a single self‑consistent constant (`C_cycle ≈ 1.66`) reproduces both measurements is strong evidence the mechanism below is the real one — and that **the lever that matters is acceptance staying above ~42 %**, nothing else.

This ~1.66× cycle cost also matches the literature anchor: confirmed sources put **MoE verify at ~1.3–1.7× a single decode for K=3–4** [S1]; the remaining ~0.0–0.3× is the MTP draft‑head pass.

---

## 2. Root causes — organized list

### A. Inherent to the regime (cannot be fixed by configuration)

**A1 — MoE breaks "parallel verification is nearly free." (dominant cause)**
In a *dense* model, verifying K draft tokens in one batch touches the same weights as one token → bandwidth‑bound, nearly free. In your **128‑expert / top‑8** MoE, each of the K verified positions routes to its *own* top‑8 experts, so the verify pass must load the **union** of experts across all K positions. A single decode loads 8 experts/layer; verifying K=3 loads materially more, so the verify pass costs **>1× a decode** instead of ~1×. This is the well‑supported, dominant mechanism [S1, confirmed; also confirmed: "larger draft trees activate more experts and incur higher cost," "expected number of activated experts grows with the number of verified tokens"].
*Refinement (important, keeps us honest):* the union is **sub‑linear**, not the full `K×8`. Routing is **temporally correlated** across adjacent tokens, so verifying 4 tokens activates roughly ~2.5× the top‑k experts, not the ~3.2–3.6× that independent routing would predict [S15]. So verify cost is elevated but bounded — consistent with the measured ~1.66× cycle cost, **not** the catastrophic blow‑up a naive "full union" estimate would give. (The strong "full union / net‑negative even at 100 % acceptance" framing was **explicitly refuted** in verification — see §6.)

**A2 — Small active‑parameter footprint → little headroom for spec decoding to recover.**
Speculative decoding pays off most when the target model is *expensive* per token. Your model activates only ~4B params, so raw decode is already fast (82 t/s) and light on bandwidth. The fixed overheads (draft‑head pass, verify batch setup, sampling/rejection) are a **large fraction of a small per‑token cost**, so they easily swamp the savings. Confirmed framing: "speculation utility is the ratio of target‑model time saved to draft+verify overhead"; "when speculative‑decoding gains do not offset the added overhead, throughput drops" [S5, S6, confirmed].

**A3 — Acceptance, not cleverness, is the only free variable — and hard content can't clear the bar.**
Because `C_cycle ≈ 1.66` is fixed by `K`, the *only* thing that determines win/loss is acceptance vs the ~42 % break‑even. Hard/unpredictable output structurally yields low acceptance (your 29 %), and **no setting can raise the acceptance of genuinely unpredictable text** — the MTP head simply can't guess it. This is why the loss on hard content is not a bug to fix but a property of the content.

### B. Contributing factors (hardware / build — secondary, plausible)

**B1 — gfx906 small‑batch GEMM is an awkward middle.**
The fast ROCm path on gfx906 is the dp4a INT8 **MMQ** kernel, tuned for **batch‑1 decode** and **large‑batch prefill**. The K=3–4 verify batch sits between those two optimized regimes, so the verify pass may not hit peak efficiency, nudging `C_cycle` upward [S16, S17, S18 — gfx906 baseline + this fork's kernel notes; treat as plausible contributor, not proven for the mtp path specifically].

**B2 — Memory‑bound → compute‑bound crossover.**
The "nearly free verify" assumption holds only while decode is memory‑bandwidth‑bound. The extra work of drafting + verifying pushes toward compute‑bound, where the extra FLOPs are no longer free [S14]. *(This specific framing was a borderline 1–2 refute in verification — included as a contributing factor, not load‑bearing.)*

---

## 3. What the research did NOT support (kept honest)

Adversarial verification (3 votes/claim, needs 2/3 to kill) **killed several plausible‑sounding but overstated claims.** They are *not* part of the explanation above:

- ❌ "Speculative decoding is impractical/unprofitable in MoE models in general; the result is inherent to *all* MoE" — **refuted 0–3.** SD *does* win on MoE in other regimes (larger active params, batched serving). Your result is regime‑specific, not a universal MoE verdict.
- ❌ "Expert‑saturation threshold `T_thres = log_(1−ρ)(0.05) ≈ 94 tokens`; net‑negative even at 100 % acceptance" — **refuted 0–3.** The specific formula/number is unsupported, and your own +15 % easy‑content win disproves "net‑negative even at 100 %."
- ❌ "Active footprint sets break‑even: A10B gains +15–119 %, A3B always loses" — **refuted 0–3.** Direction is plausible; the specific numbers are not substantiated.
- ❌ "MoE verifying K tokens activates the *full* union, contradicting any benefit" — **refuted 0–3.** Overstated; temporal correlation compresses the union (see A1 refinement).
- ❌ "llama.cpp docs say MoE requires long ngram drafts" — **refuted 0–3.** The doc citation could not be verified; do **not** treat the ngram suggestion below as doc‑backed.

This is why §4's ngram idea is flagged "test, don't assume."

---

## 4. Config levers — three buckets

### ✅ Bucket 1: Will likely help (worth testing)

**L1 — Confidence gating via `--spec-draft-p-min` (the single best lever; never tested).**
Your build **has** this flag (verified in `--help`): `--spec-draft-p-min P` (default **0.00** = *always* draft). Every MTP test you've run drafted unconditionally, so on hard content it always paid the verify penalty → the −15 %. With a threshold (e.g. `0.6`–`0.8`), the model only speculates when the MTP head is confident; when it isn't, it **falls back to plain decode (82 t/s)** instead of the penalized 70. Expected effect: **keep the +15 % on easy/confident content, turn the −15 % on hard content into ~0 %.** This directly attacks the only failure case.
*Caveat:* the flag exists and is wired for the generic draft path; its exact behavior on the **`draft-mtp`** path in build `fabde3b` is **unverified** — confirm empirically before trusting it.
*How to try (non‑destructive, separate port):*
```
--spec-draft-model /models/MTP/gemma-4-26B-A4B-it-Q4_0-MTP.gguf \
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7
```
Sweep `p-min ∈ {0.5, 0.6, 0.7, 0.8}` and measure t/s on a hard prompt; pick the lowest value that removes the regression.

**L2 — Draft depth `--spec-draft-n-max` / `--spec-draft-n-min`.**
Lower `K` shrinks the verify‑batch expert union → lowers `C_cycle` → lowers the break‑even acceptance, at the cost of fewer tokens/cycle. On marginal content this can flip a loss to a wash. Sweep `n-max ∈ {2, 3, 4}`. (Your model's MTP head is single‑depth‑trained, so very large K likely won't help acceptance.)

### 🧪 Bucket 2: Worth testing but unproven (don't assume)

**L3 — Draft‑free ngram speculation for structured/repetitive output (`--spec-type ngram-mod` or `ngram-cache`).**
Your build supports `ngram-simple / ngram-map-k / ngram-map-k4v / ngram-mod / ngram-cache`. These draft from the **prompt/context n‑grams at zero draft‑compute cost** (no draft model forward pass), which removes the draft‑head term from `C_cycle`. For the **prefill‑heavy n8n workload with repetitive/JSON‑ish output**, accepted ngram tokens can be genuinely cheap. **BUT** ngram still pays the **MoE union verify cost** (A1), so it only nets a win when acceptance on the repetitive segments is high enough to clear break‑even — and the "MoE needs long ngram drafts" guidance was **refuted**, so treat this as a hypothesis to measure, not a known win. Quick test on a representative n8n JSON‑output prompt:
```
--spec-type ngram-mod --spec-ngram-mod-n-max 4 --spec-ngram-mod-n-match 24
```

### ❌ Bucket 3: Tested and ruled out / inherent (don't bother)

- **Draft‑head quantization (F16 / Q8_0 / Q4_0).** Swept directly (2026‑06‑20, predictable prompt, γ=3) — and the answer **depends on MoE vs dense**:
  - **26B MoE (Q5_K_XL target):** acceptance is precision‑insensitive — **Q8_0 and F16 heads gave byte‑for‑byte identical acceptance (55.7%, draft_n=244)**, Q4_0 ≈58%. But a bigger head is slower (runs per draft step): net **Q4_0 90.5 t/s > Q8_0 85.1 > F16 79.4** (raw 69.0). → **Use the Q4_0 head; F16 is strictly worse.**
  - **31B dense (official Q4_0 target, google heads):** the opposite — acceptance *rises* slightly with precision (Q4_0 63.6% < Q8_0 65.2% < F16 66.9%, small/real), and net speed is **flat** (~37–38 t/s; raw 22.66). → **Q8_0 head is the sweet spot; F16 is harmless but pointless.**
  - **Mechanism (unifies both):** the head's cost matters only *relative to the target verify step*. MoE verify is cheap (≈4B active) → head bytes are visible → smaller head wins, argmax is quant‑insensitive. Dense verify is huge (~17 GB) → head is a rounding error on speed → F16's size is free but its tiny acceptance edge can't move net t/s. The "smaller head always wins" rule is **MoE‑specific**. (Aside: MTP nets **+67%** on the dense 31B — the classic expensive‑target regime — vs marginal on the cheap MoE; see [[Q5‑Q6 quant‑test §4a]].)
- **Ungated MTP at defaults.** This *is* the current net‑negative configuration. Don't ship it as the default; raw 82 t/s wins on mixed/hard content.
- **The MoE union verify cost (A1), small‑footprint headroom (A2), and gfx906 small‑batch path (B1).** Architectural/hardware — no flag removes them.

---

## 5. Recommended configuration

**Default endpoint: keep raw (no MTP).** It already is. Raw ~82 t/s is the most predictable choice across mixed n8n content and never regresses.

```bash
# current mi50-llm (unchanged) — keep as the production default
-m /models/gemma-4-26B_q4_0-it.gguf -ngl 99 --flash-attn on \
-c 131072 -b 2048 -ub 2048 --parallel 1 --jinja --host 0.0.0.0 --port 8089
```

**If you want to chase the upside,** stand up a *second* container on a different port (e.g. 8090) so production is untouched, and run this A/B test plan:

1. **Gated MTP (L1):** add `--spec-draft-model .../Q4_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7`. Measure t/s on (a) an easy prompt and (b) a hard prompt. **Success = hard‑prompt t/s ≈ 82 (no regression) AND easy‑prompt t/s > 82.**
2. **Depth sweep (L2):** repeat with `n-max ∈ {2,3,4}` at the best `p-min`.
3. **ngram (L3):** on a real n8n JSON‑output prompt, try `--spec-type ngram-mod`; keep only if it beats raw on *that* workload.
4. Promote a setting to the default **only** if it is net‑positive or break‑even across *both* easy and hard prompts. Otherwise raw stays.

Benchmark via `llama-server /completion` timings JSON (note: `llama-bench`/`llama-cli` are ABI‑broken in this image). Use a fixed prompt + `ignore_eos` + `n_predict:512` for repeatable numbers, and watch `rocm-smi --showtemp -d 0` (post‑cooling‑upgrade the card holds 1725 MHz, so thermal throttling is no longer a confound).

---

## 6. Sources & confidence

Confirmed = survived 3‑vote adversarial verification (≥2/3 defend). Refuted = killed (≥2/3 refute).

**Core mechanism (confirmed, high confidence)**
- **[S1]** Cascade/MoE speculative‑decoding analysis — arXiv 2506.20675 — *MoE verify activates the union of experts; the verify/decode ratio (not acceptance alone) governs speedup; K=3–4 verify ≈ 1.3–1.7× decode.* **Primary.** ⭐ load‑bearing.
- **[S5]** inference.net — speculation utility = time‑saved ÷ overhead ratio. Blog, confirmed.
- **[S6]** bentoml.com — overhead can exceed gains → throughput drops. Secondary, confirmed.
- **[S14]** arXiv 2508.08192 — memory‑bound→compute‑bound crossover removes the "free verify." Primary; *borderline (1–2)* — contributing factor only.
- **[S15]** Cohere — *temporal routing correlation compresses the activated‑expert union (~2.5× for 4 tokens, not full independence).* Primary; the "this contradicts the union premise" framing was **refuted**, but the underlying compression fact is sound and is used in A1's refinement.

**MoE + small/cheap‑model regime (confirmed)**
- arXiv 2505.19645 (MoE SD batch‑size dependence), 2605.00342 (cascade verify), fergusfinn.com (economics of SD), dev.to/defilan ("why it didn't help on my home cluster").

**llama.cpp specifics**
- llama.cpp GitHub issues #23752, #23203 (spec‑decoding real‑world reports). Forum.
- llama.cpp `docs/speculative.md` — *the "MoE needs long ngram drafts" claim from here was **refuted**; do not rely on it.*

**gfx906 / MI50 hardware (secondary, plausible)**
- skyne98 gfx906 architecture baseline; iacopPBK/llama.cpp‑gfx906 kernel fork; hackmd MI50 perf notes.

**Refuted (NOT used in the explanation):** "SD universally impractical on MoE" (0–3); "T_thres≈94 tokens / net‑negative even at 100 % accept" (0–3); "active‑footprint fixed break‑even numbers" (0–3); "full‑union activation" (0–3); "docs mandate long MoE ngram drafts" (0–3). See §3.

**Research run stats:** 5 search angles → 18 sources fetched → 85 claims extracted → 25 verified → 18 confirmed, 7 killed.

---

## 7. Honest limitations of this analysis

- The **break‑even ≈ 42 %** is reconstructed from *your three measurements* + a literature anchor, not measured by an isolated micro‑benchmark of verify cost. It is self‑consistent (both runs imply the same `C_cycle`), which is strong but not a controlled proof.
- The confirmed "1.3–1.7× verify" figure comes largely from **K=7 Mixtral on batched datacenter GPUs**, not gfx906 single‑stream — directionally right, magnitude approximate for your box.
- **`--spec-draft-p-min` on the `draft-mtp` path is unverified for build `fabde3b`.** L1 is the highest‑value recommendation *and* the one most needing an empirical check before you trust it.
- gfx906 small‑batch GEMM (B1) is a plausible contributor, not independently profiled here.

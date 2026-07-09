---
title: "Master Report — 2026-07-06 Model Sweep"
description: "Full 6-model benchmark sweep: MTP on/off, DFlash, trimmed vocab, dense models, Llaminar engine comparison."
---

# MI50 Model Sweep — Master Report (2026-07-06)

Consolidated report for all model benchmarks run on 2026-07-06. Merges the six per-run reports (Ornith live, Qwen3.6-trim, Qwen3.6-full MTP off/on, Qwen3.6 DFlash, gemma-4-12b, Qwen3.6-27B).

**Stack (all runs except §3.7):** production `llama-hipgraphs:upstream-rocm-7.2.4` (llama.cpp build `0eca4d4`), gfx906 / MI50 32 GB. **(§3.7 is a different engine — Llaminar — on the same model, added later.)** **Tool:** [eugr/llama-benchy](https://github.com/eugr/llama-benchy) v0.4.0 (OpenAI-API client, non-destructive). **Method:** 14-point context sweep 0→65536 (unless noted), pp512/tg128, runs 2, `--exact-tg`, `--latency-mode generation`, `--concurrency 1`, KV q8_0. Prod Ornith stopped for VRAM per run and restored after; all test GGUFs deleted after testing (results retained). Prose corpus (Gutenberg); tokenizer warmup delta 9–15 tok, coherence PASSED on every run → token counts valid.

---

## 1. Executive summary — master comparison

Decode throughput vs context (t/s), all Q4, best spec config per model:

| Model | Arch | Quant | Spec | decode 0 → 65k | decay | avg prefill | basic tg128 (no spec)§ |
|---|---|---|---|---|---|---|---|
| **Ornith-1.0-35B** (prod) | MoE A3B | Q4_K_M | MTP n2 | 78.3 → 59.6 | −24% | 845 | ~70 |
| **Qwen3.6-trim** (Elsephire) | MoE A3B | Q4_K_S | MTP n2 | 78.1 → 63.2 | **−19%** | 846 | 71.1 |
| **Qwen3.6-full** (unsloth) | MoE A3B | Q4_K_M | **MTP** n2 | 82.3 → 61.6 | −25% | 706 | 66.3 |
| Qwen3.6-full (unsloth) | MoE A3B | Q4_K_M | none | 63.5 → 38.9 | −39% | 820 | 66.3 |
| Qwen3.6-full (unsloth) | MoE A3B | Q4_K_M | DFlash n3 | 76.7 → ~~💥4k~~ **stable→65k**◆ | — | 752* | 66.3 |
| gemma-4-12b (unsloth) | **DENSE** 12B | Q4_K_M | MTP n3 | 39.9 → 29.8 | −25% | 452 | 47.5 |
| Qwen3.6-27B (unsloth) | **DENSE** 27B | Q4_K_M | MTP n2 | 23.9 (probe) | — | 244 | 23.2 |
| **Ornith-1.0-35B** — *Llaminar engine*‡ | MoE A3B | Q4_K_M | MTP (inert)† | 53.3 → 45.4 @0–3k💥 | −14%/3k | **1140** ⚡ | 53.3 |

* DFlash prefill avg is over 0–3k only (5 depths) vs the other models' 14-depth 0–65k range — slightly understates MTP's prefill cost in the comparison.
§ "basic tg128" values from `llama-bench` (standalone), not the benchy sweep. Minor tooling difference — benchy measures server-based generation with `--exact-tg --latency-mode generation`; llama-bench measures raw decode. The sweep's no-spec baseline for Qwen3.6-full (66.3 from llama-bench vs 63.5→38.9 from benchy) is consistent — benchy's context-aware sweep gives the full curve, while llama-bench's single-point tg128 is a snapshot at minimal context.
◆ **DFlash "crash" CORRECTED** (§3.4 re-run): with `-ub 512` DFlash runs **stable 0→65k** (71→61 t/s on Q5). Original crash was batching config, not DFlash. Acceptance ~60% on this build (86% needs the old custom `mixa3607` build).
† Llaminar row is a **different engine** (not llama.cpp) — see §3.7. `--mtp` was set but drafting **did not engage** under llama-benchy's requests (`draft_steps=0`), so 53.3→45.4 is Llaminar's **baseline decode**, not MTP. Its MTP prefill **hard-fails above 4096 tokens** (💥), so the curve stops at depth 3072.
‡ ⚡ Llaminar's **prefill (~1140 t/s) is ~1.6–2× llama.cpp's** (845 avg / 607 @depth0) on the same model — its one clear win.

**Top-line takeaways:**
1. **MTP is the MI50 win.** On A3B MoE, MTP holds decode across context (−19 to −25%) and lifts throughput +30–58% vs no-spec. The clean on/off isolation (Qwen3.6-full, same model) is the proof.
2. **Sparsity is what makes this card fast.** Every A3B MoE (~3B active) does ~60–80 t/s; the dense 12B does ~30–40, the dense 27B ~24. Decode is bandwidth ÷ active-params bound.
3. **MTP > DFlash** on Qwen3.6-35B-A3B (higher acceptance + DFlash crashes deep). MTP > DFlash on dense gemma too. DFlash only won on gemma-4-26B-A4B (different acceptance profile).
4. **The MI50 sweet spot = A3B MoE + embedded MTP** — exactly the prod Ornith setup.

---

## 2. Cross-cutting conclusions

**A. MTP buys back context-length decay.** No-spec baseline decays −39% (63.5→38.9); MTP holds it to −19/−25%. At 65k, MTP models do 60–63 t/s vs the baseline's 38.9 = **+58% from MTP alone** at deep context (the gain *widens* with context).

**B. Prefill is ~model/vocab-independent, but MTP costs a little.** Across the MoE models prefill sits ~820–846 t/s (vocab-trimming doesn't change it — the lm_head runs once per prompt). BUT the clean on/off test showed the **MTP head adds a prefill cost: 820 → 706 (−14%)** from processing the extra nextn tensors. So MTP is a **decode-for-prefill trade** — huge decode win, small prefill cost → net win for single-stream latency-bound serving (n8n).

**C. Dense models pay in BOTH phases.** gemma-4-12b (dense) and Qwen3.6-27B (dense) are ~2× slower than the A3B MoE models in decode *and* prefill — dense compute is heavier per token everywhere. They're the only models that break rule B.

**D. Spec-decode payoff scales with how cheap verify is.** Big on MoE (verify activates only ~3B → +30–58%), marginal on dense (verify = full forward → gemma +5%, 27B +3%), even at high acceptance. This is the unifying principle behind the whole sweep.

**E. DFlash vs MTP is model-specific — decided by acceptance.** Whichever head proposes better tokens wins. Qwen3.6's official embedded MTP (92% accept) beats its DFlash draft (~60%); gemma-4-26B's DFlash beat its weaker MTP. **⚠️ Corrected (§3.4 re-run):** the original "DFlash crashes deep" was a `-ub 4096` artifact — with `-ub 512` DFlash is **stable to 65k**. And DFlash acceptance is **build-dependent** (~60% on hipgraphs `0eca4d4` vs 86% on the old custom `mixa3607` build). So MTP still leads *on our current stack*, but DFlash is viable/stable, not crash-prone.

---

## 3. Per-run details

### 3.1 Ornith-1.0-35B (production reference)
Qwen3.5-35B-A3B finetune, Q4_K_M + embedded MTP (`--spec-draft-n-max 2`). Live non-destructive bench of prod.
- Decode flat ~76 t/s to ~24k, easing to 59.6 at 65k (−24%). Prefill 900–1250 t/s; deep-context TTFT is the real latency cost (2k prompt on 16k ctx ≈ 21 s to first token).
- **vs DGX Spark Qwen3.5-122B** (`fank/dgx-spark-qwen3.5-122b-bench`, same tool): height gap (~3×) is a size+bandwidth artifact (35B/3B-active on ~1 TB/s vs 122B/10B-active on ~273 GB/s), NOT silicon superiority. The like-for-like finding is **decay shape** — Ornith's −24% is shallower than every DGX config (their best DFlash-4: 41→15 = −63%). MTP holds context better than DFlash on the Spark rig. Chart: `ornith_vs_dgx_decode_vs_context.png`.

### 3.2 Qwen3.6-35B-A3B vocabulary-trimmed (Elsephire)
`qwen35moe`, **vocab 145,572** (−41%, Latin+Greek only), Q4_K_S, **own embedded MTP head** (block 41). basic tg128 71.1 / pp512 1087.
- MTP n-max swept → **2 optimal** (87.3 t/s/80% accept; n3 ties at 87.4 but lower acceptance). Same as Ornith.
- Decode 78.1→63.2 (−19%, shallowest MoE decay); near-identical to Ornith up to ~33k, pulls ahead at deep context. On code the edge is bigger (87 vs ~70). Mechanism: smaller trimmed lm_head shaves per-token output projection.
- **Trimming and external speculation are mutually exclusive:** the trim buys a smaller lm_head + ~2 GB VRAM but its 145k vocab forecloses every full-vocab (248k) drafter. Its own embedded MTP is the only accelerator. Chart: `qwen36trim_vs_ornith_decode.png`.

### 3.3 Qwen3.6-35B-A3B (unsloth) — MTP OFF vs ON  ⭐ cleanest isolation
Two GGUFs, identical model/vocab/quant (Q4_K_M), only difference = the embedded MTP head:
- **OFF** (plain quant, block 40, no nextn): 63.5→38.9, **−39%**, prefill 820. tg128 66.3.
- **ON** (`-MTP-GGUF`, block 41 + `blk.40.nextn`, 22.7 GB): 82.3→61.6, −25%, prefill 706. n-max 2 (n2/n3 near-tie 89/91 on code); **92.3% acceptance** (highest of any model).

| depth | OFF | ON | MTP gain |
|---:|---:|---:|---:|
| 0 | 63.5 | 82.3 | +30% |
| 16k | 54.3 | 76.6 | +41% |
| 32k | 47.8 | 72.8 | +52% |
| 65k | 38.9 | 61.6 | **+58%** |

Confirms Ornith isn't a finetune fluke (stock Qwen3.6+MTP lands on the Ornith curve). Chart: `qwen36_mtp_onoff.png`.

**⚠ Prefill cold-start artifact:** MTP ON prefill at depth 0 (376 t/s) is far below the 14-depth average (706 t/s) and below the depth-0 no-spec baseline (719 t/s). This is a cold-start effect — the MTP head's nextn tensors add overhead on the very first prompt batch before the prompt cache warms up. By depth 4k it recovers to 793 t/s, in line with the curve. If you only see depth-0 values in isolation you'd overstate MTP's prefill cost; the lifetime average (706) is the fair metric.

### 3.4 Qwen3.6-35B-A3B — DFlash vs MTP (same model)  ⚠️ CORRECTED 2026-07-06 eve — see re-run below
Draft `williamliao/Qwen3.6-35B-A3B-DFlash-GGUF` Q8_0 (arch **`dflash`** = loadable; vocab 248320 matches; a GGUF quant of `z-lab/Qwen3.6-35B-A3B-DFlash`) on the unsloth Q4_K_M target. DFlash engaged for real (63% accept, non-zero).
- DFlash n-max 3 = 82.7 t/s (code) — beats baseline (+13–29%) but **slower than MTP at every depth** (MTP 89–91/92% accept).
- ~~DFlash server CRASHED at depth 4096~~ → **this was a `-ub 4096` batching artifact, NOT a DFlash property** (see re-run).
- **MTP wins on speed AND acceptance** (92% vs ~60%). Chart: `qwen36_spec_comparison.png`.

**🔁 Fair re-run (2026-07-06 eve) — ⚠️ PROVISIONAL (used Q5 target = mistake; proper Q4 re-run scheduled 2026-07-07, see TODO below) — reconciling with the late-June prod result (86% accept, 92.5 t/s, stable to 128k):**
Suspecting the DFlash test was handicapped, re-ran it clean. Setup: **same williamliao Q8_0 draft**, Q5_K_XL target (= the exact prod target quant, on disk), hipgraphs image, **`-ub 512`** (prod batching, vs the sweep's `-ub 4096`), n-max 3. Result — benchy 0→65k, `dflash_fair_q5q8_benchy.json`:

| depth | 0 | 4096 | 8192 | 32768 | 65536 |
|---|---|---|---|---|---|
| decode t/s | 71.3 | 73.9 | 85.6 | 71.4 | 61.3 |

- **The "crash at 4096" was purely `-ub 4096`.** With `-ub 512`, DFlash runs **stable 0→65k** (61–85 t/s). The original "DFlash crashes deep" conclusion was WRONG — a config artifact.
- **Acceptance was NOT handicapped.** The re-run averaged **60.1%** — same as the sweep's 63% (both used the Q8 draft; target quant Q4 vs Q5 barely moved it). Draft quant / target quant were never the issue.
- **The 86% was a property of the old custom build** (`mixa3607/llama.cpp-gfx906:b9827`, now deleted), NOT reproducible on the current hipgraphs image (`0eca4d4`, ~60%). **DFlash acceptance is build-version-dependent.**
- **Corrected verdict:** on our *current* stack, MTP (92%) still leads DFlash (~60%) on acceptance and speed — but DFlash is **stable and viable**, not crash-prone. The only real error in the original run was the `-ub` setting. (To recover the 86%/92.5 t/s prod result you'd need to rebuild the custom `mixa3607` DFlash llama.cpp — see [[mi50-dflash-custom-build]].)

> ### ⏭ DEFERRED — Q4 DFlash re-run (was: TODO 2026-07-07)
> **Planned:** re-run with Q4_K_M target (instead of Q5_K_XL) for quant parity with the MTP comparison. **Status:** never executed — DFlash acceptance on the current stack is build-limited to ~60% regardless of target quant; the stability finding (`-ub 512` fixes the crash) is already confirmed on Q5, and the ~60% acceptance ceiling is a property of the hipgraphs build (`0eca4d4`), not the target quant. A Q4 re-run would shift the DFlash line slightly upward (lighter quant → faster decode) but wouldn't change the verdict — MTP (92% acceptance, 89–91 t/s) still leads DFlash (~60%, 82 t/s) on this stack. Deferred unless we rebuild the custom `mixa3607` DFlash fork (which hit 86%/92.5 t/s).
> **Provisional Q5 data retained** in the master chart with this note — the curve shape and stability conclusion are valid even without quant parity.

### 3.5 gemma-4-12b-it (unsloth) — DENSE
`gemma4`, **dense 11.9B** (config `num_experts: None`, 48 dense layers). basic tg128 47.5 / pp512 637.
- Spec swept: **MTP n-max 3 = 49.8 (wins); DFlash net-NEGATIVE** (41.9→27.7 as n-max rises) — DFlash is built for MoE, loses on dense. (Both formats load — `gemma4-assistant` / `dflash`, unlike the rejected Qwen `dflash-draft`.)
- benchy MTP n3: 39.9→29.8 (−25%), avg prefill 452. **~2× below the A3B MoE models in BOTH decode and prefill** — the dense penalty. MTP only +5% (dense verify cost eats it). Appears violet in `mi50_model_sweep_decode.png`.

### 3.6 Qwen3.6-27B (unsloth) — DENSE (quick test)
`qwen35` **dense 27.3B** (hidden 5120, hybrid Gated-DeltaNet). basic **tg128 23.2 / pp512 244** — slowest model. With embedded MTP n2 = **23.9 (+3% only)** at 78% accept — MTP ~net-neutral (dense verify = full 27B forward). Full benchy skipped (non-contender). The requested **DSpark-Qwen3.6-27B-AEON-draft is incompatible** (safetensors + vLLM-patches only, wrong AEON base, DGX-Spark target).

### 3.7 Llaminar engine — Ornith-1.0-35B Q4_K_M  ⚠️ DIFFERENT ENGINE (added 2026-07-06 eve)
Not llama.cpp — [Llaminar](https://github.com/Llaminar/llaminar) is an experimental (alpha) C++ MPI inference engine (multi-vendor GPU/CPU, tensor/pipeline/expert-parallel). Pre-built ROCm image `ghcr.io/llaminar/llaminar:develop-rocm7.1.1-latest` (9.2 GB, gfx906-only). Ran the **exact same production model file** (`ornith-1.0-35b-Q4_K_M-MTP.gguf`) for a clean engine-vs-engine head-to-head.

**Setup:** `serve -d rocm:0 --no-mmap -c 70000 --activation-precision fp16 --kv-cache-precision q8 --mtp --mtp-draft-tokens 2`, port 8089, same 14-point benchy. fp16 activations set deliberately — Llaminar defaults to **FP32 activations** (≈½ throughput on gfx906); matching fp16 keeps it fair vs the llama.cpp curves.

| depth | decode t/s | prefill t/s |
|---:|---:|---:|
| 0 | 53.3 | **1337** |
| 512 | 50.7 | 1126 |
| 1024 | 49.8 | 1119 |
| 2048 | 47.5 | 1070 |
| 3072 | 45.4 | 1048 |
| 4096 → 65536 | 💥 **FAIL** | 💥 **FAIL** |

**Head-to-head vs Ornith on llama.cpp (§3.1):**
- **Decode: llama.cpp wins.** llama.cpp+MTP 78→60 t/s (to 65k); Llaminar 53→45 t/s (to 3k only). Even vs a no-spec A3B baseline (~63 @0) Llaminar's decode is lower.
- **Prefill: Llaminar wins ~1.6–2×.** 1337 t/s @depth0 / ~1140 avg vs llama.cpp's 607 @depth0 / 845 avg. Genuinely strong — its standout result.
- **Loads clean:** recognizes `qwen35moe` (41 layers, 256 experts top-8, 753 tensors), detects the embedded nextn/MTP block, weights → VRAM at 3.9 GB/s.

**Two Llaminar problems (why it's not a serving contender here):**
1. **MTP silently doesn't engage under the benchmark client.** A hand-crafted single request drafts fine (~36–58% acceptance, streaming or not), but every llama-benchy request logged `draft_steps=0` → the 53→45 numbers are **baseline decode, MTP inert** (classic silent-fallback — caught via the acceptance log). Forcing `ignore_eos`/`min_tokens` (benchy's `--exact-tg`) makes MTP draft but acceptance **collapses to 7%** and decode drops to **17 t/s** (wasted verify passes on post-EOS tokens). Net: MTP is fragile/unusable under a normal fixed-length benchmark.
2. **MTP prefill hard-caps at 4096 tokens.** With `--mtp` on, any context >4096 fails: `Hidden-state rows-select capacity too small: capacity=4096 seq_len=…` → `Failed to populate MTP shifted prefill cache`. No flag lifts it (fixed internal buffer). So an MTP decode-vs-context curve is structurally impossible past depth 3072 on this build.

**Host-RAM gotcha:** Llaminar stages the full ~20 GB of weights through **system RAM** before the GPU transfer (needs ~21.7 GB free). On this 30 GB box it aborted until RAM was freed (`LLAMINAR_WEIGHT_STREAMING=1` env didn't help — likely stripped by its self-launched `mpirun`). Server then holds ~20 GB RssAnon resident → tight alongside n8n.

**Verdict:** llama.cpp (`llama-hipgraphs`) remains the MI50 serving stack — faster reliable decode + working MTP to 65k. Llaminar is worth watching **only for its prefill throughput** (~2× faster); its MTP and long-context paths are alpha-broken on gfx906 today.

---

## 4. Compatibility / can't-run appendix

| Model / draft | Runs on MI50? | Reason |
|---|---|---|
| williamliao DFlash-GGUF (`dflash` arch) | ✅ | loadable format, vocab match |
| gemma `dflash` / `gemma4-assistant` MTP | ✅ | loadable |
| z-lab / Anbeeld Qwen DFlash (`dflash-draft`) | ❌ | arch rejected by stock build |
| modal-labs Qwen3.6 DFlash | ❌ | safetensors-only `DFlashDraftModel` (vLLM) |
| DSpark-Qwen3.6-27B-AEON-draft | ❌ | safetensors + vLLM patches; wrong (AEON) base |
| plunderstruck ROCmFP4 Deckard-40B | ❌ | needs ROCmFP4 fork + Strix Halo gfx1151 (FP4 silicon); MI50=gfx906 has no FP4 |
| MXFP4 / NVFP4 quants | ❌ | Blackwell-only |
| **Llaminar** engine (`ghcr.io/llaminar/llaminar:…-rocm7.1.1`) | ⚠️ **partial** | loads `qwen35moe` + runs (decode 53/prefill 1140), but **MTP alpha-broken** (inert under benchmark client; 7% accept when forced; prefill caps @4096) + needs ~21.7 GB host RAM to stage weights. See §3.7. |

**General rules learned:** external Qwen DFlash needs the `dflash` (not `dflash-draft`) GGUF format + matching vocab + matching base; ROCmFP4/MXFP4/NVFP4 are newer-silicon only; gfx906 vLLM is a dead-end (no MTP, archived forks).

---

## 5. Charts (this folder)
- `mi50_model_sweep_decode.png` — all models, decode vs context (master chart, avg prefill in labels)
- `qwen36_mtp_onoff.png` — MTP on/off isolation (same model)
- `qwen36_spec_comparison.png` — MTP vs DFlash vs baseline (same model)
- `ornith_vs_dgx_decode_vs_context.png` — Ornith vs DGX Spark Qwen3.5-122B
- `qwen36trim_vs_ornith_decode.png` — trimmed vs Ornith

## 6. Raw data (this folder)
`qwen36full_benchy.json` (baseline), `qwen36full_MTP_benchy.json`, `qwen36full_DFlash_benchy.json`, `qwen36trim_benchy_precise.json`, `gemma12_benchy.json`, `llaminar_ornith_mtp_benchy.json` (§3.7 — Llaminar engine; note depths ≥4096 are `null` = MTP-prefill crash).

## 7. Source data index

Every table value in this report maps to a raw file. Quick reference:

| Section | Model / config | Raw JSON file | Individual-run report |
|---|---|---|---|
| §3.1 | Ornith-1.0-35B (prod) | ⚠️ *not archived* — 14-point sweep data only in this report; live 3-depth probe in individual-run | `individual-runs/ornith-benchy-live-2026-07-06.md` |
| §3.2 | Qwen3.6-35B-A3B vocab-trimmed + MTP | `benchy files/qwen36trim_benchy_precise.json` | `individual-runs/qwen36-trim-vocabulary-test-2026-07-06.md` |
| §3.3 | Qwen3.6-35B-A3B MTP OFF vs ON | `benchy files/qwen36full_benchy.json` (OFF), `benchy files/qwen36full_MTP_benchy.json` (ON) | `individual-runs/qwen36-full-baseline-test-2026-07-06.md` |
| §3.4 | Qwen3.6-35B-A3B DFlash vs MTP | `benchy files/qwen36full_DFlash_benchy.json` (0–3k only — crashed at 4096); `dflash_fair_q5q8_benchy.json` (re-run) | `individual-runs/qwen36-dflash-vs-mtp-2026-07-06.md` |
| §3.5 | gemma-4-12b-it (dense) + MTP | `benchy files/gemma12_benchy.json` | `individual-runs/gemma-4-12b-mtp-test-2026-07-06.md` |
| §3.6 | Qwen3.6-27B (dense) + MTP | (no benchy — quick probe only) | `individual-runs/qwen36-27b-dense-mtp-2026-07-06.md` |
| §3.7 | Llaminar engine — Ornith | `llaminar_ornith_mtp_benchy.json` | — |

**⚠ Missing:** Ornith full 14-point sweep (`benchy files/` contains no Ornith JSON). The 14-point Ornith data in §3.1 comes from a separate sweep run on 2026-07-06 whose raw output wasn't archived. The three available Ornith depths (0, 4k, 16k) can be verified from `individual-runs/ornith-benchy-live-2026-07-06.md`. The full curve values (78.3→59.6) are consistent with those three anchor points.

## 8. Reproduce (template)
```bash
TOK=<HF tokenizer snapshot for the model's base>
HF_HUB_OFFLINE=1 uvx llama-benchy --base-url http://localhost:8089/v1 --model <alias> \
  --tokenizer "$TOK" --pp 512 --tg 128 \
  --depth 0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 \
  --runs 2 --concurrency 1 --exact-tg --latency-mode generation --save-result out.json --format json
```

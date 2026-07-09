---
title: "RUNBOOK — MI50 Model Testing"
description: "Step-by-step guide for benchmarking new models: compatibility checks, VRAM management, benchy procedure."
---

# RUNBOOK — MI50 model testing (llama.cpp + MTP/DFlash)

Scan-and-go guide for benchmarking a new model on the MI50. Written 2026-07-06 from the day's sweep (Ornith, Qwen3.6-trim, Qwen3.6-full off/on/DFlash, gemma-4-12b, Qwen3.6-27B). Follow top-to-bottom.

> **⚠ GOLDEN RULE: Ornith is production. It must be serving on port 8089 when you finish.**
> Testing needs Ornith's VRAM, so you stop it, test, and **always restore + verify it afterward.** Never leave a session with Ornith down. The restore/verify step is Step 8 — do not skip it.

---

## 0. Baseline software stack (don't change without reason)

| Thing | Value |
|---|---|
| GPU | AMD MI50 32 GB, **gfx906 / Vega20** (no FP4, no bf16-tensor; has dp4a/INT8) |
| Container image | **`llama-hipgraphs:upstream-rocm-7.2.4`** (local build, llama.cpp **`0eca4d4`**) — best gfx906 stack available (HIP-graphs + working MTP/DFlash/EAGLE3 runtimes) |
| Prod model | Ornith-1.0-35B Q4_K_M + embedded MTP, container `llama-hipgraphs`, compose `/home/<username>/llm/docker-compose-hipgraphs.yml`, **port 8089**, model id `ornith-hipgraphs` |
| Bench tool | **`uvx llama-benchy`** (eugr/llama-benchy 0.4.0) — OpenAI-API client, non-destructive |
| Models dir | `/home/<username>/llm/models` (mounted as `/models` in containers) |
| Reports | `~/Dokumente/LLM Tests/` (note: system is **German locale**, so `Dokumente` not `Documents`) |
| Scratch | session scratchpad for transient scripts/JSON |

**VRAM reality:** ~20 GB model + KV fills the 32 GB card → **only one model at a time.** Always stop Ornith before launching a test server.

---

## 1. Compatibility check FIRST (before any download)

Cheap HF-API checks save a 20 GB download of something that can't run. A model/draft runs on our stack only if **all** of these hold:

- **It's a GGUF.** `safetensors`-only = vLLM/PyTorch → **won't run** (gfx906 vLLM is a dead-end). Check the repo file list.
- **Draft arch is loadable:** `dflash` ✅ / `gemma4-assistant` ✅ / embedded `nextn` (MTP) ✅. **`dflash-draft` ❌ REJECTED** by our stock build (z-lab/Anbeeld format, PR#22105 fork).
- **Draft base == target base** (a draft distilled to a *different* finetune's logits → ~0% acceptance).
- **Not ROCmFP4 / MXFP4 / NVFP4** — those need Blackwell/Strix Halo FP4 silicon; gfx906 can't load them.

Quick check:
```bash
curl -sL "https://huggingface.co/api/models/<repo>" | python3 -c "import json,sys;d=json.load(sys.stdin);sib=[f['rfilename'] for f in d['siblings']];print('gguf:',[f for f in sib if f.endswith('.gguf')]);print('safetensors:',[f for f in sib if f.endswith('.safetensors')])"
```

---

## 2. Download (HF token + Xet OFF — both matter)

```bash
export HF_TOKEN=<token>            # REQUIRED — anon downloads get throttled to ~0 after a few GB
export HF_HUB_DISABLE_XET=1        # REQUIRED — Xet backend STALLS at 0 B/s mid-download; plain HTTPS is reliable (100+ MB/s)
uvx --from huggingface_hub hf download <repo> <file.gguf> --local-dir /home/<username>/llm/models/<dir>
```
Run big downloads in the background. Rotate the HF token after a session if it was pasted into chat.

---

## 3. Inspect the GGUF header (arch / dense-vs-MoE / MTP)

```bash
uvx --from gguf gguf-dump --no-tensors <file>.gguf | grep -iE 'general.architecture|block_count|expert_count|nextn_predict'
uvx --from gguf gguf-dump <file>.gguf | grep -i nextn      # embedded MTP head = blk.N.nextn.* tensors
```
- `expert_count` present → **MoE** (fast on this card); absent → **dense** (~2× slower, spec barely helps).
- `nextn_predict_layers` + `blk.N.nextn.*` → has embedded MTP (`block_count` = real layers + 1).

---

## 4. Stop Ornith + load gate (llama-bench)

```bash
docker stop llama-hipgraphs
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 -v /home/<username>/llm/models:/models \
  --entrypoint /app/llama-bench llama-hipgraphs:upstream-rocm-7.2.4 \
  -m /models/<dir>/<file>.gguf -ngl 99 -fa 1 -p 512 -n 128
```
This is the **load gate** — if it segfaults, the model doesn't run on our build → restore Ornith and stop. If it prints pp512 + tg128, note them (that's the no-spec baseline).

---

## 5. Launch the test server (spec on)

```bash
docker run -d --rm --name test-server --network host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 -v /home/<username>/llm/models:/models \
  --entrypoint /app/llama-server llama-hipgraphs:upstream-rocm-7.2.4 \
  -m /models/<dir>/<target>.gguf --alias test \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8089 -ngl 99 --no-mmap --flash-attn on \
  -c 262144 -b 4096 -ub 4096 -ctk q8_0 -ctv q8_0 --parallel 1 --jinja
# external draft (DFlash) instead of embedded MTP:
#   --spec-type draft-dflash -md /models/<dir>/<draft>.gguf --spec-draft-ngl 99 --spec-draft-n-max 3
# wait: curl -s http://localhost:8089/health  → {"status":"ok"}
```
`--no-mmap` is mandatory (30 GB-RAM box thrashes otherwise).

**⚠ Verify spec ACTUALLY engaged (silent-fallback trap):** run one generation, then
```bash
docker logs test-server 2>&1 | grep -i 'draft acceptance'   # must be NON-ZERO
```
If acceptance is 0.000, spec silently fell back to baseline — the number is a lie. (DFlash also logs `common_speculative_impl_draft_dflash: adding ...` on engage; the `dflash requires ctx_other` line is a normal mem-fit probe, not an error.)

---

## 6. Tune n-max + run the benchy

**n-max sweep** (temp-0 code prompt = lossless → pure throughput). Restart the server per n-max value, probe decode + acceptance. Typical winner: **n-max 2** for A3B MoE MTP (n2/n3 often tie; pick 2 for higher acceptance → robust at real temp>0). Deeper n-max collapses acceptance.

**14-point benchy** (run in background — a 65k sweep is ~35 min; the Bash tool caps at 2 min):
```bash
TOK=<tokenizer snapshot for the model's BASE>   # see table below
HF_HUB_OFFLINE=1 uvx llama-benchy --base-url http://localhost:8089/v1 --model test --tokenizer "$TOK" \
  --pp 512 --tg 128 --depth 0 512 1024 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536 \
  --runs 2 --concurrency 1 --exact-tg --latency-mode generation --save-result out.json --format json
```
| model family | tokenizer snapshot |
|---|---|
| Ornith / Qwen3.5-A3B | `/home/<username>/llm/rocm-cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754` |
| Qwen3.6 (any) | `/home/<username>/llm/rocm-cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0` |
| gemma-4-12b | `/home/<username>/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/e18f459f54832f4ae2ab6686b935a2268668a9e9` |

Sanity: benchy prints **warmup delta** (should be **9–15 tok**) + **coherence PASSED**. Big delta → wrong tokenizer (depth axis unreliable; decode t/s still valid, it uses server counts).

---

## 7. What to track (record these per model)

| Field | From | Why |
|---|---|---|
| arch, dense/MoE, vocab, embedded-MTP? | GGUF header (Step 3) | predicts speed + compatibility |
| basic **pp512 / tg128** (no spec) | llama-bench (Step 4) | baseline |
| spec type + **n-max sweep** (decode + acceptance) | Step 6 | best config |
| **draft acceptance %** | server logs | MUST be non-zero; explains the speedup size |
| benchy **decode 0→65k + decay %** | benchy JSON | the headline curve |
| **avg prefill** | benchy JSON | MTP costs ~14% here; dense ~2× worse |
| anomalies | — | thermal transient etc. (see gotchas) |

---

## 8. RESTORE ORNITH (never skip) + finalize

```bash
docker stop test-server 2>/dev/null; docker rm -f test-server 2>/dev/null
cd /home/<username>/llm && docker compose -f docker-compose-hipgraphs.yml up -d
# VERIFY:
curl -s http://localhost:8089/health                    # {"status":"ok"}
curl -s http://localhost:8089/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])"   # ornith-hipgraphs
```
Then: **delete the test GGUF** (`rm -rf /home/<username>/llm/models/<dir>`), write the report to `~/Dokumente/LLM Tests/`, add its line to the master report + sweep chart, update memory.

---

## 9. Gotchas learned (the ones that bit us)

- **HF Xet stalls at 0 B/s** → always `HF_HUB_DISABLE_XET=1`. Anon throttle → always `HF_TOKEN`.
- **Silent-fallback trap** → always confirm `draft acceptance` is non-zero before trusting a spec number.
- **Thermal transient:** one run mid-sweep can come back way low (MI50 passive-throttles under sustained load). If a point is far off its neighbors (e.g. 46 t/s between two 75s), re-measure it in isolation and use the clean value; note it.
- **DFlash "deep crash" was a `-ub` artifact, not a DFlash limit** (confirmed 2026-07-06 re-run). The sweep died at depth 4096 with `-b/-ub 4096`; re-running with **`-ub 512`** (prod batching) → DFlash **stable 0→65k**. Always run DFlash with `-ub 512` (like prod), not the `-ub 4096` used for the MoE sweeps. Also: **DFlash acceptance is build-dependent** — ~60% on hipgraphs `0eca4d4`, but 86% on the old custom `mixa3607/…:b9827` build ([[mi50-dflash-custom-build]]); if DFlash acceptance looks low, suspect the build, not the draft/target quant (Q4 vs Q8 draft barely moved it).
- **MTP is a decode-for-prefill trade:** ~+30–58% decode but **−14% prefill** (extra nextn tensors). Net win for single-stream latency-bound serving (n8n).
- **Dense penalty:** dense models (~2× slower decode AND prefill vs A3B MoE) barely benefit from spec (verify = full forward). Not MI50 contenders.
- **DFlash-vs-MTP is model-specific** — whoever's head accepts more wins. Official embedded MTP usually wins (Qwen3.6: 92% vs DFlash 63%).
- **Bash tool 2-min cap** → run downloads and benchy in the background.
- Charts: dataviz-skill palette, clean colored lines + right-side key with **proper model names + quant + spec**. Validate palette before shipping.

---

## 10. Where everything lives
- Prod compose: `/home/<username>/llm/docker-compose-hipgraphs.yml` · models: `/home/<username>/llm/models/`
- Reports + charts: `~/Dokumente/LLM Tests/` (master: `MI50-model-sweep-MASTER-2026-07-06.md`; per-run backups in `individual-runs/`)
- Memory notes: `mi50-model-sweep-2026-07-06`, `mi50-ornith-mtp-serving`, `mi50-qwen36-trim-test`, `llama-benchy`, `mi50-nomap-ram-thrash`

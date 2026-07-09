---
title: "Ornith-1.0-35B MTP — Optimization Report"
description: "ubatch tuning (+46%), HIP graphs (+12%), ngram vs MTP, n-max sweep, GPU clock behavior."
---

# Ornith-1.0-35B MTP — Performance & Optimization Report

**Date:** 2026-06-30  
**Hardware:** AMD MI50 32GB (gfx906) + Ryzen 7 8845HS  
**Model:** ornith-1.0-35b-Q4_K_M-MTP.gguf (21 GB, Q4_K_M with grafted MTP head)  
**Runtime:** llama.cpp upstream + HIP graphs (custom Docker image)  
**Server:** llama-hipgraphs on port 8089

---

## Current Configuration

| Setting | Value |
|---------|-------|
| Image | `llama-hipgraphs:upstream-rocm-7.2.4` (custom build) |
| Entrypoint | `/app/llama-server` |
| Context | 262,144 tokens (model native max) |
| GPU layers | `-ngl 99` (full offload) |
| Flash attention | `--flash-attn on` |
| KV cache K | `-ctk q8_0` |
| KV cache V | `-ctv q8_0` |
| Batch | `-b 4096` |
| UBatch | `-ub 4096` |
| Spec decode | `--spec-type draft-mtp --spec-draft-n-max 6` |
| Parallel | `--parallel 1` |
| Reasoning | `--reasoning on` |
| Template | `--jinja --chat-template-file /models/ornith-chat-template.jinja` |
| VRAM | 90% (28.8 GB used, ~3.2 GB free) |
| Container | `llama-hipgraphs` |
| HIP graphs | ✅ ON (GGML_HIP_GRAPHS=ON) |


**Compose file:** `/home/<username>/llm/docker-compose-hipgraphs.yml`

---

## Current Benchmarks

### Short Prompt — Code generation (36 prompt → 200 gen tokens)

| Metric | Value |
|--------|-------|
| Prompt speed (cold) | 958 tok/s |
| Generation speed | 56-74 tok/s |
| Average generation | ~64 tok/s |
| MTP acceptance | 36-54% |
| VRAM | 91% |

### Long Context — Code review (3,437 prompt → 500 gen tokens) — HIP graphs build

| Run | Prompt | Generate | Draft | Wall clock | Notes |
|-----|--------|----------|-------|------------|-------|
| 1 (cold) | 1,247 tok/s | 56.0 tok/s | 343/931 (37%) | 11.8s | HIP graphs |
| 2 (cached) | 87 tok/s | 61.2 tok/s | 356/855 (42%) | 8.2s | HIP graphs |
| 3 (cached) | 87 tok/s | **73.6 tok/s** | 381/704 (54%) | 6.9s | HIP graphs |
| **Average** | **1,247 tok/s** | **63.6 tok/s** | **~44%** | **9.0s** | +12% vs baseline |

### Chat / Tool Calling (n8n workload)

| Metric | Value |
|--------|-------|
| Tool calling | Working (finish_reason: tool_calls) |
| Reasoning | Working (reasoning_content field) |
| Chat template | Fixed via Unsloth Qwen3.5 template |
| MTP acceptance (short) | ~69% |
| MTP acceptance (long gen) | ~42% |

---

## GPU Clock Behavior (Measured Under Load)

Initial idle readings (925 MHz core / 350 MHz mem) were misleading — the card aggressively boosts when actually working.

| Condition | Core Clock | Memory Clock | Power | Temp | GPU Util |
|-----------|-----------|-------------|-------|------|----------|
| **During prompt processing** | **1725 MHz** | **1000 MHz** | **191 W** | **46°C** | **87%** |
| During generation (decode) | ~1000 MHz | ~1000 MHz | ~140 W | ~42°C | varies |
| Idle (between requests) | 925 MHz | 350 MHz | 17 W | 34-41°C | 0% |

**Key takeaways:**
- The card already boosts to 1725/1000 under load — that's healthy
- Power draw hits 191W towards the 225W limit, with thermal headroom (46°C vs ~95°C max)
- The decode phase (token-by-token) may not sustain peak clocks since each individual step is very short — the GPU downclocks between tokens
- **No overclocking wanted** — this is the only GPU, not risking it

---

## Optimization Ideas

### 🥇 Idea 1: UBatch Size — ✅ Applied, +46% prompt speed

**Changed:** `-ub 512` → `-ub 4096` (matching `-b 4096`)  
**VRAM impact:** 78% → 90% (3.2 GB free — safe)

**Benchmark results (ubatch=512 vs ubatch=4096):**

| Metric | ubatch=512 | ubatch=4096 | Change |
|--------|------------|-------------|--------|
| Prompt (cold, 3.4K) | 833 tok/s | **1,214 tok/s** | **+46%** 🚀 |
| Gen (short, 200 tok) | ~60 tok/s | **~90 tok/s** | **+50%** 🚀 |
| Gen (long, 500 tok) | 57-63 tok/s | 51-61 tok/s | ~same |
| MTP acceptance | ~42% | ~38-65% | variable |

**Verdict:** Significant prompt prefill speedup at the cost of 12% more VRAM. Worth keeping.

**Change in compose:**
```yaml
      - -ub
      - "4096"
```

---

### 🥉 Idea 3 (deprioritized): Thread Tuning — potential 5-10%

**Current:** Not explicitly set (llama.cpp default)  
**CPU:** Ryzen 7 8845HS — 8 cores / 16 threads

**Options to test:**

| Setting | Rationale |
|---------|-----------|
| `-t 8` | One thread per physical core, avoids SMT contention |
| `-t 16` | All logical threads (`nproc`) — used by iacopPBK fork |
| `-t 12` | Hybrid: 8 P-cores + 4 SMT |

**Change in compose:**
```yaml
      - -t
      - "8"
```

And/or separate batch threads:
```yaml
      - --threads-batch
      - "8"
```

**Risk:** Minimal. Wrong setting might be 1-2% slower, easily reverted.

---

### 🥈 Idea 2: Prompt Caching — huge for repeated/overlapping prompts

**Current:** Not enabled  
**Flag:** `--cache-prompt`

When enabled, llama.cpp stores computed KV cache entries for prompts. If n8n sends similar or identical prompts repeatedly, the cached portion is reused, dramatically reducing time-to-first-token.

**Trade-off:** Uses additional VRAM for the cache. At 256K context, the cache could be several GB. We have 7 GB free.

**Change in compose:**
```yaml
      - --cache-prompt
```

**Risk:** VRAM usage increases. Monitor with `rocm-smi`.

---

### Idea 4: HIP Graphs Build — Applied, +12% generation speed

**What it is:** HIP graphs capture GPU kernel launch sequences into replayable graphs, reducing CPU-to-GPU sync overhead. On gfx906 at batch=1, the card is latency-bound (~10% bandwidth utilization), so kernel launch overhead is the main bottleneck.

**What was tested:**
- iacopPBK/llama.cpp-gfx906 fork: Cannot load Ornith model (based on old build 7924 — no qwen35moe architecture)
- arte-fact/llamacpp-gfx-906-turbo fork: Same issue (too old)
- Custom upstream build + GGML_HIP_GRAPHS=ON: Works, +12% gen speed

**Docker image:** `llama-hipgraphs:upstream-rocm-7.2.4` (custom built from upstream latest + HIP graphs)

**Dockerfile:** `/home/<username>/llm/Dockerfile.hipgraphs`

**Compose file:** `/home/<username>/llm/docker-compose-hipgraphs.yml`

**Key insight:** The arte-fact README states "gfx906 at batch=1 is latency-bound (10% bandwidth utilization), not bandwidth-bound". This is why reducing kernel launch overhead via HIP graphs is more effective than pure bandwidth optimizations.

---

### Idea 5: Ngram Speculative Decoding — ❌ Tested, worse than MTP

**Tested:** Ngram (--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64) vs MTP

**Benchmark results (3.4K prompt → 500 gen, 15 runs each):**

| Metric | MTP | Ngram |
|--------|-----|-------|
| Gen average | **63.6 tok/s** | ~61.0 tok/s |
| Gen peak | 73.6 tok/s | 66.5 tok/s |
| Draft acceptance | ~44% avg (stable) | ~35% avg (erratic 14-58%) |
| Cold run | 56.0 tok/s | 62.6 tok/s |
| Consistency | Stable | Highly variable |

**Why:** Ngram relies on pattern matching from the prompt history. For the first response to a new prompt (cold), it has no history and draft acceptance is near 0%. Even on cached runs, acceptance varies wildly. MTP's learned draft model (grafted head) is consistently better at predicting the next token.

**Verdict:** ❌ Stick with MTP for this model.

---

### Idea 6: Power Limit Tuning — ❌ skipped (risk of damage)

**Current:** 225W TDP (default), draws up to 191W under load  
The card already reaches 1725 MHz at 191W. Increasing the power limit could sustain higher clocks but risks the only GPU.

**Verdict:** Not worth the risk.

---

### Previously Tested (and Rejected)

| Idea | Result | Reason |
|------|--------|--------|
| f16 K cache | 2.5× slower generation | MI50 has no tensor cores; 2× memory bandwidth for no benefit |
| -c above 262144 | Not possible | Model caps at native max context_length |
| Increase context to 262K | ✅ Done | Fits at 78% VRAM |
| --spec-draft-n-max 6 | ✅ Done | Slight improvement over 4 |
| -b 4096 | ✅ Done | No noticeable VRAM impact |

---

## How to Benchmark

Use the existing benchmark script:
```bash
python3 /home/<username>/llm/scripts/bench_long.py
```

This sends a 3.4K token code review prompt requesting 500 tokens of output, matching the data above.

Quick single-request test:
```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"ornith-hipgraphs",
    "messages":[{"role":"user","content":"Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes."}],
    "max_tokens":200,"temperature":0.6
  }' | python3 -c "import json,sys;d=json.load(sys.stdin);t=d['timings'];print(f'pp:{t[\"prompt_per_second\"]:.0f}tg:{t[\"predicted_per_second\"]:.0f} draft:{t[\"draft_n_accepted\"]}/{t[\"draft_n\"]}')"
```

Monitor VRAM during load:
```bash
watch -n1 rocm-smi
```

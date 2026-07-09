---
title: "DFlash Speculative Decoding — Setup Guide"
description: "DFlash GGUF conversion, build requirements, server setup, block-size tuning."
---

# DFlash Speculative Decoding Setup

## Goal
Enable DFlash speculative decoding for Qwen3.6-35B on MI50 GPU (llama-swap, ROCm), using the 0.4B draft model `z-lab/Qwen3.6-35B-A3B-DFlash`.

## Components

| Component | File |
|---|---|
| GGUF draft model | `/home/<username>/llm/models/dflash-draft/dflash-draft.gguf` (747 MB, F16) |
| Run script (bs16 default) | `/home/<username>/llm/models/run-qwen-dflash.sh` |
| Run script (bs8) | `/home/<username>/llm/models/run-qwen-dflash-bs8.sh` |
| Run script (bs20) | `/home/<username>/llm/models/run-qwen-dflash-bs20.sh` |
| Config | `/home/<username>/llm/config/mi50-swap.yaml` |
| Server binary | b9831 (built from source, in container at `/app/llama-server`) |

## Conversion (not needed again — for reference)

The GGUF draft was converted using llama.cpp's `convert_hf_to_gguf.py` (b9831):

```bash
# Inside Docker container mi50-swap
cd /tmp/llama.cpp
python3 convert_hf_to_gguf.py /models/dflash-draft/ \
  --outfile /models/dflash-draft/dflash-draft.gguf \
  --outtype f16 \
  --target-model-dir /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<hash>/
```

`--target-model-dir` is **required** — it points the converter to the target model's HF cache so it can copy the tokenizer (vocab + 247k BPE merges).

**Key issue fixed**: The Anbeeld HF repo (`Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF`) had the wrong architecture name `"dflash-draft"` and wrong tensor names (`dflash_fc.weight`, `dflash_hidden_norm.weight`). Converting with the b9831 converter produces the correct `"dflash"` architecture and `fc.weight` / `enc.output_norm.weight` tensor names.

## Server Setup

The b9831 `/app/llama-server` was built from source with ROCm/HIP for gfx906. Build:
```bash
cd /home/<username>/llm/llama.cpp-b9831
mkdir build && cd build
cmake .. -DLLAMA_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CUDA=OFF -DAMDGPU_TARGETS=gfx906
cmake --build . --target llama-server -- -j$(nproc)
```

The binary was copied to `/app/llama-server` inside the container so all run scripts automatically use the DFlash-capable version.

## Usage via llama-swap

Available model IDs for inference:

- `qwen3.6-35b-dflash` — DFlash with block_size=16 (default)
- `qwen3.6-35b-dflash-bs8` — DFlash with block_size=8
- `qwen3.6-35b-dflash-bs20` — DFlash with block_size=20

API endpoint: `http://localhost:8089/v1/chat/completions`

```bash
curl http://localhost:8089/v1/chat/completions \
  -d '{"model":"qwen3.6-35b-dflash","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

## Performance

| Configuration | Gen Speed | Draft/Acc | Accept Rate |
|---|---|---|---|
| No speculation | ~58 tok/s | — | — |
| DFlash bs=8 | 70.5 tok/s | 169/92 | 54% |
| DFlash bs=16 | 72.7 tok/s | 163/94 | 57% |
| DFlash bs=20 | 74.0 tok/s | 160/94 | 58% |
| n-gram spec | ~66 tok/s | — | — |

**DFlash gives ~20-25% speedup** over no speculation (~70 vs ~58 tok/s). Block_size makes little difference in the 8–20 range.

## Architecture

The DFlash draft runs in the **same llama-server process** as the target model:

```
llama-swap (port 8089)
  └── llama-server (port XXXX)
       ├── Target model: Qwen3.6-35B (35B params, Q5_K_XL)
       ├── Draft model: DFlash (0.4B, F16)
       └── --spec-type draft-dflash
```

Both models share VRAM via `-ngl 99`. The draft model uses ~0.7 GB, negligible on the 16 GB MI50.

## Notes

- The old b9728 llama-server did NOT support DFlash — b9831 was the minimum.
- `--spec-draft-model` specifies the draft GGUF path.
- `--spec-type draft-dflash` enables DFlash (other options: `draft-eagle3`, `draft-mtp`, `ngram-simple`, etc.).
- DFlash uses `mask_token_id=248077` and `block_size=16` (trained value — defines max 15 draft tokens/step).
- Target model also needs the Qwen3.6-35B-A3B model files in HF cache for tokenizer sharing.

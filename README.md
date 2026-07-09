# MI50 (gfx906) LLM Inference Benchmarks

> LLM inference benchmarks, tuning results, and production notes for AMD MI50 (gfx906 / Vega 20) on ROCm — covering Ornith-1.0-35B, Gemma-4, Qwen3.6, MTP, DFlash, EAGLE3, Llaminar, and Docker vs bare-metal comparisons.

📖 **Read the full docs:** [fank.github.io/gfx906-LLM-Inference](https://fank.github.io/gfx906-LLM-Inference/)

**Card:** AMD Instinct MI50 32 GB (gfx906 / Vega 20 / GCN5, ~1 TB/s HBM2)  
**Host:** Nobara/Fedora, Ryzen 7 8845HS (8c/16t), 30 GB RAM  
**Production stack:** `llama-hipgraphs` (llama.cpp `0eca4d4`), Ornith-1.0-35B Q4_K_M + embedded MTP, ~70 t/s

## Repository structure

```
├── src/content/docs/          # 📝 All documentation (Markdown)
│   ├── index.md               #   Home page
│   ├── master-report.md       #   Master Model Sweep (2026-07-06)
│   ├── all-tests-log.md       #   Complete Test Log (2026-06-20 → 07-03)
│   ├── individual-runs/       #   Per-model deep dives
│   └── reports/               #   Older analysis and setup docs
├── public/images/             # 📊 Charts and diagrams
├── benchy-files/              # 📈 Raw benchy JSON data
├── astro.config.mjs           # ⚙️ Starlight/Astro config
├── .github/workflows/         # 🔄 Auto-deploy to GitHub Pages
└── README.md                  # You are here
```

## Key results — at a glance

- **ROCm/HIP over Vulkan = 8.7× faster decode** — the single biggest win
- **A3B MoE + MTP is the MI50 sweet spot** — Ornith does ~70 t/s, MTP holds context decay to −19/−25%
- **MTP > DFlash on current stack** (92% vs ~60% acceptance), but DFlash is stable with `-ub 512`
- **Dense models pay double** in both decode and prefill — avoid on gfx906
- **Docker adds 0% overhead** — bare-metal A/B confirmed

## Raw data

All benchmark data is available as raw JSON in [`benchy-files/`](./benchy-files/). Each file maps to a section in the docs — see the Source Data Index in the master report for the full mapping.

## Building the site locally

```bash
npm install
npm run build    # builds to dist/
npm run dev      # dev server at localhost:4321
```

The site auto-deploys to GitHub Pages on every push to `main` via GitHub Actions.

## License

Everything here is CC0 / public domain — use it however you like, no attribution needed.

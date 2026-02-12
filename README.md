# Core-WR-NLP (code-only, data-free)

This repository accompanies an arXiv preprint that introduces a transparent, unsupervised,
model-agnostic pipeline for linking **structured planning artifacts** (e.g., CoRe tables)
to **unstructured written reflections** at the sentence level using embedding-based semantic similarity.

## What this repository provides

- End-to-end pipeline:
  1. Load a CoRe-like planning table (CSV) and a reflection (TXT)
  2. Embed planning units and reflection sentences with an arbitrary embedding model
  3. Compute exhaustive pairwise cosine similarities
  4. Apply deterministic thresholding to derive inspectable binary references
- A structural mapping visualization (Figure 1-style)

## What this repository intentionally does *not* provide

- No empirical data (bring your own data).
- No facet aggregation / Figure 2 implementation.
  Mapping guiding questions to "knowledge facets" can be study-specific and may introduce interpretive assumptions.
  The arXiv paper discusses such aggregation as a *derived* view; users can implement their own mapping if desired.

## Installation

Python 3.10+ recommended.

### Option A: Ollama backend (default)

1. Install and run Ollama locally.
2. Install this package:

```bash
pip install -e .
```

### Option B: sentence-transformers backend (optional)

```bash
pip install -e .[st]
```

## Quickstart

### 1) Run the pipeline

```bash
python scripts/run_pipeline.py \
  --core examples/example_inputs/core_template.csv \
  --reflection examples/example_inputs/reflection_template.txt \
  --backend ollama --model mxbai-embed-large --threshold 0.7 \
  --out outputs/demo
```

Outputs are written to `outputs/demo/` (see `docs/OUTPUT_FORMAT.md`).

### 2) Create Figure 1-style mapping

```bash
python scripts/make_figure1.py \
  --core examples/example_inputs/core_template.csv \
  --sentences outputs/demo/sentences.txt \
  --references outputs/demo/references.csv \
  --out outputs/demo/figure1_mapping.png
```

## Robustness checks (optional)

If you run the same input pair with multiple embedding models, you can compute robustness metrics.

Create a manifest JSON (example):

```json
{
  "pairs": [
    {"pair_id": "pair01", "model": "nomic-embed-text", "out_dir": "outputs/pair01_nomic"},
    {"pair_id": "pair01", "model": "mxbai-embed-large", "out_dir": "outputs/pair01_mxbai"},
    {"pair_id": "pair01", "model": "bge-m3", "out_dir": "outputs/pair01_bge"}
  ]
}
```

Run:

```bash
python scripts/robustness_metrics.py --manifest manifest.json --tau 0.7 --delta 0.05 --outprefix robustness
```

## Input format

See `docs/INPUT_FORMAT.md`.

## License

MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`.

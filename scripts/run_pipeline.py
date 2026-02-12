"""CLI to run the Core-WR-NLP pipeline.

Example:
  python scripts/run_pipeline.py \
    --core examples/example_inputs/core_template.csv \
    --reflection examples/example_inputs/reflection_template.txt \
    --backend ollama --model mxbai-embed-large --threshold 0.7 \
    --out outputs/demo

This repository intentionally ships *no data*. Replace the example templates with your own inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core_wr_nlp import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--core", required=True, help="Path to a wide CSV representing the planning table (CoRe-like). ")
    p.add_argument("--reflection", required=True, help="Path to a reflection text file.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--backend", default="ollama", choices=["ollama", "sentence-transformers"], help="Embedding backend.")
    p.add_argument("--model", default="mxbai-embed-large", help="Model identifier for the backend.")
    p.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (tau) for references.")
    p.add_argument("--exclude-empty", action="store_true", help="Exclude empty planning cells from similarity computation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(embedding_backend=args.backend, model=args.model, threshold=args.threshold)
    summary = run_pipeline(
        core_csv_path=args.core,
        reflection_path=args.reflection,
        out_dir=args.out,
        config=cfg,
        include_empty_core_cells=(not args.exclude_empty),
    )
    print("Pipeline completed.")
    print(summary)


if __name__ == "__main__":
    main()

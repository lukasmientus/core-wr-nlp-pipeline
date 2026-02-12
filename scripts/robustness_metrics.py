"""Compute robustness checks across multiple embedding model runs.

This is an adapted and repo-ready version of the author's analysis script.
It assumes you have run the pipeline multiple times (same pair, different models) and
stored outputs in a directory structure like:

outputs/
  pair01_modelA/
    similarity.npy
    meta.json
  pair01_modelB/
    similarity.npy
    meta.json
  pair02_modelA/
  ...

You can choose any naming scheme as long as you provide a manifest JSON that maps
(pair_id, model) -> output_dir.

No data is shipped with this repository. You generate the outputs locally.

Example manifest (manifest.json):
{
  "pairs": [
    {"pair_id": "pair01", "model": "nomic-embed-text", "out_dir": "outputs/pair01_nomic"},
    {"pair_id": "pair01", "model": "mxbai-embed-large", "out_dir": "outputs/pair01_mxbai"}
  ]
}

Usage:
  python scripts/robustness_metrics.py --manifest manifest.json --tau 0.7 --delta 0.05 --outprefix robustness
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def load_sim(out_dir: str) -> np.ndarray:
    return np.load(Path(out_dir) / "similarity.npy")


def spearman_matrix_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between flattened matrices (no scipy dependency)."""
    xa = a.ravel()
    xb = b.ravel()
    # rank
    ra = xa.argsort().argsort().astype(np.float64)
    rb = xb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())) + 1e-12
    return float((ra * rb).sum() / denom)


def reference_set(sim: np.ndarray, tau: float) -> set:
    idx = np.argwhere(sim >= tau)
    return set(map(tuple, idx.tolist()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sentence_coverage(sim: np.ndarray, tau: float) -> float:
    """Fraction of reflection sentences that link to at least one planning unit."""
    if sim.size == 0:
        return 0.0
    covered = (sim >= tau).any(axis=0)
    return float(covered.mean())


def count_references(sim: np.ndarray, tau: float) -> int:
    return int((sim >= tau).sum())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest JSON mapping (pair_id, model) to output dirs.")
    p.add_argument("--tau", type=float, default=0.7, help="Default threshold.")
    p.add_argument("--delta", type=float, default=0.05, help="Threshold variation (+/- delta).")
    p.add_argument("--outprefix", default="robustness", help="Output prefix for CSV files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest.get("pairs", [])
    if not rows:
        raise ValueError("Manifest must contain a non-empty 'pairs' list.")

    df = pd.DataFrame(rows)
    required = {"pair_id", "model", "out_dir"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest rows must have keys: {sorted(required)}")

    per_rows: List[Dict[str, Any]] = []

    # Group by pair_id
    for pair_id, g in df.groupby("pair_id"):
        models = list(g["model"].values)
        out_dirs = {row.model: row.out_dir for row in g.itertuples(index=False)}

        # Similarity structure consistency (Spearman) + Jaccard on references
        for m1, m2 in combinations(models, 2):
            s1 = load_sim(out_dirs[m1])
            s2 = load_sim(out_dirs[m2])
            rho = spearman_matrix_corr(s1, s2)
            jac = jaccard(reference_set(s1, args.tau), reference_set(s2, args.tau))
            per_rows.append(
                {
                    "pair_id": pair_id,
                    "model_a": m1,
                    "model_b": m2,
                    "spearman_rho": rho,
                    "jaccard_refs": jac,
                }
            )

        # Threshold sensitivity and coverage per model
        for m in models:
            s = load_sim(out_dirs[m])
            n0 = count_references(s, args.tau)
            n_lo = count_references(s, args.tau - args.delta)
            n_hi = count_references(s, args.tau + args.delta)
            cov = sentence_coverage(s, args.tau)
            per_rows.append(
                {
                    "pair_id": pair_id,
                    "model_a": m,
                    "model_b": "",
                    "spearman_rho": np.nan,
                    "jaccard_refs": np.nan,
                    "n_refs_tau": n0,
                    "n_refs_tau_minus": n_lo,
                    "n_refs_tau_plus": n_hi,
                    "coverage_tau": cov,
                }
            )

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(f"{args.outprefix}_per_pair.csv", index=False)

    # Summary stats (min/median/max)
    summary_rows: List[Dict[str, Any]] = []
    for col in ["spearman_rho", "jaccard_refs", "n_refs_tau", "coverage_tau"]:
        if col in per_df.columns:
            vals = per_df[col].dropna()
            if len(vals) == 0:
                continue
            summary_rows.append(
                {
                    "metric": col,
                    "min": float(vals.min()),
                    "median": float(vals.median()),
                    "max": float(vals.max()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(f"{args.outprefix}_summary.csv", index=False)

    print(f"Wrote: {args.outprefix}_per_pair.csv and {args.outprefix}_summary.csv")


if __name__ == "__main__":
    main()

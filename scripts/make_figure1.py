"""Create the Figure 1-style mapping visualization from pipeline outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from core_wr_nlp.io import load_core_wide_csv
from core_wr_nlp.visualize import plot_figure1_mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--core", required=True, help="Path to the same core CSV used for the pipeline.")
    p.add_argument("--sentences", required=True, help="Path to outputs/sentences.txt")
    p.add_argument("--references", required=True, help="Path to outputs/references.csv")
    p.add_argument("--out", required=True, help="Output image path (png/pdf). ")
    p.add_argument("--title", default=None, help="Optional title.")
    return p.parse_args()


def read_sentences(path: str) -> List[str]:
    txt = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in txt.splitlines() if line.strip()]


def read_references(path: str) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("unit_id"):
                continue
            refs.append(
                {
                    "unit_id": row["unit_id"],
                    "row_index": int(row["row_index"]),
                    "col_index": int(row["col_index"]),
                    "sent_id": int(row["sent_id"]),
                    "similarity": float(row["similarity"]),
                }
            )
    return refs


def main() -> None:
    args = parse_args()
    core = load_core_wide_csv(args.core)
    sentences = read_sentences(args.sentences)
    refs = read_references(args.references)

    plot_figure1_mapping(core=core, sentences=sentences, references=refs, out_path=args.out, title=args.title)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()

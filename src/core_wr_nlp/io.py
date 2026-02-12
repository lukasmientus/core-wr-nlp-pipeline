"""Input/output helpers and schema validation.

This repository is data-free by design. Users provide their own planning artifacts and
reflections, following the formats documented in docs/INPUT_FORMAT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import csv
import json
from pathlib import Path


@dataclass(frozen=True)
class CoreGrid:
    """A grid representation of a structured planning artifact (e.g., a CoRe table)."""
    row_labels: List[str]
    col_labels: List[str]
    unit_ids: List[List[str]]          # shape: (n_rows, n_cols)
    texts: List[List[str]]             # shape: (n_rows, n_cols)

    def shape(self) -> Tuple[int, int]:
        return (len(self.row_labels), len(self.col_labels))


def load_reflection_text(path: Union[str, Path]) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8")


def load_core_wide_csv(path: Union[str, Path]) -> CoreGrid:
    """Load a CoRe-like table from a *wide* CSV.

    Expected format:
    - First row: header with an empty first cell, then column labels (e.g., Big Ideas)
    - First column: row labels (e.g., Guiding Questions)
    - Remaining cells: text entries (may be empty)

    Returns
    -------
    CoreGrid
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(
            "Wide CSV must have at least 2 rows and 2 columns (including headers). "
            "See docs/INPUT_FORMAT.md."
        )

    col_labels = [c.strip() for c in rows[0][1:]]
    row_labels: List[str] = []
    texts: List[List[str]] = []
    unit_ids: List[List[str]] = []

    for r_i, row in enumerate(rows[1:]):
        if len(row) < 1:
            continue
        row_label = (row[0] or "").strip()
        row_labels.append(row_label)

        row_texts: List[str] = []
        row_ids: List[str] = []
        # Pad short rows
        cells = row[1:] + [""] * max(0, len(col_labels) - len(row[1:]))
        for c_i, cell in enumerate(cells[: len(col_labels)]):
            txt = (cell or "").strip()
            row_texts.append(txt)
            row_ids.append(f"r{r_i}_c{c_i}")
        texts.append(row_texts)
        unit_ids.append(row_ids)

    return CoreGrid(row_labels=row_labels, col_labels=col_labels, unit_ids=unit_ids, texts=texts)


def coregrid_to_units(core: CoreGrid, include_empty: bool = True) -> List[Dict[str, Any]]:
    """Flatten a CoreGrid into a list of unit dicts.

    Each unit dict contains:
    - unit_id, row_index, col_index, row_label, col_label, text, is_empty
    """
    units: List[Dict[str, Any]] = []
    n_rows, n_cols = core.shape()
    for r in range(n_rows):
        for c in range(n_cols):
            text = core.texts[r][c]
            is_empty = (text.strip() == "")
            if (not include_empty) and is_empty:
                continue
            units.append(
                {
                    "unit_id": core.unit_ids[r][c],
                    "row_index": r,
                    "col_index": c,
                    "row_label": core.row_labels[r],
                    "col_label": core.col_labels[c],
                    "text": text,
                    "is_empty": is_empty,
                }
            )
    return units


def save_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Union[str, Path], rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

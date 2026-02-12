"""Reference identification via deterministic thresholding."""

from __future__ import annotations
from typing import Dict, Any, List, Sequence, Tuple
import numpy as np


def threshold_references(
    sim: np.ndarray,
    core_units: Sequence[Dict[str, Any]],
    sentences: Sequence[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Convert a similarity matrix into a list of binary references.

    A reference is created for each pair (core_unit, sentence) with sim >= threshold.

    Returns a list of dicts with:
    - unit_id, sent_id, similarity, is_reference
    - plus row/col positions for plotting convenience
    """
    if sim.shape != (len(core_units), len(sentences)):
        raise ValueError("Similarity matrix shape does not match units/sentences.")

    refs: List[Dict[str, Any]] = []
    for i, u in enumerate(core_units):
        for j in range(len(sentences)):
            s = float(sim[i, j])
            if s >= threshold:
                refs.append(
                    {
                        "unit_id": u["unit_id"],
                        "row_index": int(u["row_index"]),
                        "col_index": int(u["col_index"]),
                        "sent_id": int(j),
                        "similarity": s,
                    }
                )
    return refs

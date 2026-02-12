"""Visualization utilities.

Implements a structural mapping plot similar to the manuscript's Figure 1.

The plot shows:
- Top: the original planning grid (e.g., CoRe table) as dots (filled=non-empty)
- Bottom: reflection sentences as dots
- Edges: thresholded references, with opacity/linewidth proportional to similarity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from .io import CoreGrid


def plot_figure1_mapping(
    core: CoreGrid,
    sentences: Sequence[str],
    references: Sequence[Dict[str, Any]],
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Create the mapping visualization and optionally save it to disk."""
    n_rows, n_cols = core.shape()
    n_sent = len(sentences)

    # Layout coordinates
    # CoRe grid occupies x in [0, n_cols-1], y in [0, n_rows-1]
    # Sentence row is below the grid at y = -gap
    gap = max(3.0, n_rows * 0.6)
    y_sent = -gap

    # Sentence x positions scaled to grid width
    if n_sent <= 1:
        sent_x = np.array([0.5 * (n_cols - 1)], dtype=float) if n_sent == 1 else np.array([], dtype=float)
    else:
        sent_x = np.linspace(0, n_cols - 1, n_sent)

    fig_w = max(8.0, n_cols * 0.8)
    fig_h = max(6.0, n_rows * 0.6 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Plot CoRe dots
    for r in range(n_rows):
        for c in range(n_cols):
            text = core.texts[r][c].strip()
            filled = bool(text)
            ax.scatter(
                [c],
                [n_rows - 1 - r],  # invert y so first row is top
                s=90,
                marker="o",
                facecolors="black" if filled else "white",
                edgecolors="black",
                linewidths=1.2,
                zorder=3,
            )
            # Small unit label (optional, minimal)
            ax.text(c, n_rows - 1 - r + 0.15, str(core.unit_ids[r][c]), fontsize=7, ha="center", va="bottom")

    # Plot sentence dots
    for j in range(n_sent):
        ax.scatter(
            [sent_x[j]],
            [y_sent],
            s=70,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )
        ax.text(sent_x[j], y_sent - 0.35, str(j + 1), fontsize=7, ha="center", va="top")

    # Draw reference edges
    if references:
        sims = [float(r["similarity"]) for r in references]
        smin, smax = min(sims), max(sims)
        denom = max(1e-9, smax - smin)

        for ref in references:
            r = int(ref["row_index"])
            c = int(ref["col_index"])
            j = int(ref["sent_id"])
            sim = float(ref["similarity"])

            x0, y0 = c, n_rows - 1 - r
            x1, y1 = float(sent_x[j]), y_sent

            # Map similarity to alpha/linewidth (kept transparent and inspectable)
            t = (sim - smin) / denom
            alpha = 0.15 + 0.75 * t
            lw = 0.5 + 2.5 * t

            ax.plot([x0, x1], [y0, y1], linewidth=lw, alpha=alpha, zorder=1, color="black")

    # Cosmetics
    ax.set_xlim(-0.7, n_cols - 1 + 0.7)
    ax.set_ylim(y_sent - 1.0, n_rows - 1 + 1.0)
    ax.axis("off")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

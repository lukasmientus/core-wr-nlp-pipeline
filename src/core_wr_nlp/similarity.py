"""Similarity computation."""

from __future__ import annotations
import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarities for all pairs between rows of `a` and rows of `b`.

    Parameters
    ----------
    a:
        Shape (n_a, d)
    b:
        Shape (n_b, d)

    Returns
    -------
    np.ndarray
        Shape (n_a, n_b) with cosine similarities.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Embedding dims differ: {a.shape[1]} vs {b.shape[1]}")
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_norm @ b_norm.T).astype(np.float32)

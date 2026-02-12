"""Embedding backends.

Design goal: model-agnostic pipeline.
This module provides thin wrappers so users can swap embedding models without changing
pipeline logic.

Backends:
- OllamaEmbeddingBackend: calls a local Ollama server (http://localhost:11434 by default)
- SentenceTransformersEmbeddingBackend: optional local backend (requires sentence-transformers)
- CustomEmbeddingBackend: user passes a callable(texts)->np.ndarray
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence
import json
import time
import numpy as np
import urllib.request


class EmbeddingBackend:
    """Abstract interface for text embedding backends."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class OllamaEmbeddingBackend(EmbeddingBackend):
    """Embeddings via a running Ollama server.

    Notes
    -----
    - Requires Ollama to be installed and running.
    - Uses the /api/embeddings endpoint.
    """
    model: str
    base_url: str = "http://localhost:11434"
    request_timeout_s: int = 120
    batch_size: int = 32
    sleep_s: float = 0.0

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for t in batch:
                out = self._post_json(url, {"model": self.model, "prompt": t})
                if "embedding" not in out:
                    raise RuntimeError(f"Unexpected Ollama response keys: {list(out.keys())}")
                vectors.append(np.asarray(out["embedding"], dtype=np.float32))
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors)


@dataclass
class SentenceTransformersEmbeddingBackend(EmbeddingBackend):
    """Local embeddings via sentence-transformers (optional dependency)."""
    model: str
    batch_size: int = 32
    normalize: bool = True

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install .[st]"
            ) from e
        self._st = SentenceTransformer(self.model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._st.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize)
        return np.asarray(emb, dtype=np.float32)


@dataclass
class CustomEmbeddingBackend(EmbeddingBackend):
    """A backend that wraps a user-supplied callable."""
    fn: Callable[[Sequence[str]], np.ndarray]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = self.fn(texts)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Custom embedding function must return a 2D array.")
        if arr.shape[0] != len(texts):
            raise ValueError("Custom embedding function must return one vector per text.")
        return arr


def make_backend(backend: str, model: str) -> EmbeddingBackend:
    backend = backend.lower().strip()
    if backend == "ollama":
        return OllamaEmbeddingBackend(model=model)
    if backend in {"sentence-transformers", "st"}:
        return SentenceTransformersEmbeddingBackend(model=model)
    raise ValueError(f"Unknown embedding backend: {backend}")

"""Configuration objects for the pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal


EmbeddingBackendName = Literal["ollama", "sentence-transformers", "custom"]


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration.

    Attributes
    ----------
    embedding_backend:
        Which embedding backend to use.
    model:
        Model identifier for the selected backend (e.g., an Ollama model name or a
        SentenceTransformers model name).
    threshold:
        Similarity threshold (tau) for turning similarities into binary references.
    language:
        Optional language hint (currently unused, reserved for future extensions).
    """
    embedding_backend: EmbeddingBackendName = "ollama"
    model: str = "mxbai-embed-large"
    threshold: float = 0.7
    language: Optional[str] = None

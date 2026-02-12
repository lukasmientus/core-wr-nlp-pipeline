"""Core-WR-NLP: A transparent pipeline to link structured planning artifacts to written reflections.

This package implements:
- Sentence-level embedding of planning units and reflection segments
- Exhaustive pairwise cosine similarity computation
- Deterministic thresholding to derive inspectable references
- A structural visualization (Figure 1-style mapping)

NOTE: Facet aggregation / Figure 2 is intentionally excluded from this repository because
facet mappings from guiding questions to "knowledge facets" are study-specific and can
carry interpretive assumptions.
"""

from .config import PipelineConfig
from .pipeline import run_pipeline

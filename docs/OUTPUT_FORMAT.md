# Output formats

Running the pipeline writes inspectable intermediate and final outputs:

- `meta.json`: configuration and basic counts
- `core_units.csv`: flattened planning units (one row per cell)
- `sentences.txt`: one segmented sentence per line
- `embeddings_core.npy`: planning unit embeddings (NumPy array)
- `embeddings_reflection.npy`: sentence embeddings (NumPy array)
- `similarity.npy`: cosine similarity matrix (shape: n_units x n_sentences)
- `references.csv`: thresholded references (only pairs with similarity >= tau)

The repository intentionally does not compute facet-level aggregations (Figure 2) because
facet mappings are study-specific.

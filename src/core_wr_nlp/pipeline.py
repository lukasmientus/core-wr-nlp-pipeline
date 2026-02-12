"""End-to-end pipeline runner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .config import PipelineConfig
from .embed import make_backend
from .io import CoreGrid, coregrid_to_units, load_core_wide_csv, load_reflection_text, save_csv, save_json
from .preprocess import split_reflection_into_sentences
from .reference import threshold_references
from .similarity import cosine_similarity_matrix


def run_pipeline(
    core_csv_path: str,
    reflection_path: str,
    out_dir: str,
    config: PipelineConfig,
    include_empty_core_cells: bool = True,
) -> Dict[str, Any]:
    """Run the complete pipeline and write inspectable outputs.

    Outputs written to `out_dir`:
    - meta.json
    - core_units.csv
    - sentences.txt
    - embeddings_core.npy, embeddings_reflection.npy
    - similarity.npy
    - references.csv

    Returns
    -------
    dict
        A small summary with output paths and counts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    core = load_core_wide_csv(core_csv_path)
    core_units = coregrid_to_units(core, include_empty=include_empty_core_cells)

    reflection_text = load_reflection_text(reflection_path)
    sentences = split_reflection_into_sentences(reflection_text)

    backend = make_backend(config.embedding_backend, config.model)

    # Embed (empty cells get an embedding too if include_empty=True; that's inspectable by design)
    core_texts = [u["text"] for u in core_units]
    sent_texts = list(sentences)

    emb_core = backend.embed(core_texts)
    emb_ref = backend.embed(sent_texts)

    sim = cosine_similarity_matrix(emb_core, emb_ref)
    references = threshold_references(sim, core_units, sentences, threshold=config.threshold)

    # Save outputs
    np.save(out / "embeddings_core.npy", emb_core)
    np.save(out / "embeddings_reflection.npy", emb_ref)
    np.save(out / "similarity.npy", sim)

    save_json(out / "meta.json", {"config": asdict(config), "n_core_units": len(core_units), "n_sentences": len(sentences)})
    save_csv(out / "core_units.csv", core_units)
    (out / "sentences.txt").write_text("\n".join(sentences), encoding="utf-8")
    if references:
        save_csv(out / "references.csv", references)
    else:
        # still create an empty file with header for reproducibility
        save_csv(out / "references.csv", [{"unit_id": "", "row_index": 0, "col_index": 0, "sent_id": 0, "similarity": 0.0}],
                 fieldnames=["unit_id", "row_index", "col_index", "sent_id", "similarity"])
        # then truncate to header only
        (out / "references.csv").write_text("unit_id,row_index,col_index,sent_id,similarity\n", encoding="utf-8")

    return {
        "out_dir": str(out),
        "n_core_units": len(core_units),
        "n_sentences": len(sentences),
        "n_references": len(references),
        "paths": {
            "meta": str(out / "meta.json"),
            "similarity": str(out / "similarity.npy"),
            "references": str(out / "references.csv"),
        },
    }

"""
similarity.py
=============
Wardrobe Circle Phase 2 — Visual similarity search.

Loads the reference embedding index built by build_embeddings.py and provides
fast cosine-similarity search for a query image.

The index is loaded once (cached) and kept in memory.
Cosine similarity is computed via NumPy dot product (embeddings are L2-normalised,
so cosine similarity = dot product).

Usage
-----
    from similarity import SimilarityIndex

    index = SimilarityIndex("data/embeddings/index.npz")
    results = index.search(query_embedding, top_k=5)
    # [
    #   {"image_path": "...", "category": "Tops", "similarity": 0.91},
    #   ...
    # ]
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class SimilarityIndex:
    """
    In-memory cosine similarity index over the reference embedding set.

    Parameters
    ----------
    index_path : path to index.npz (output of build_embeddings.py)

    Notes
    -----
    - Embeddings are expected to be L2-normalised (from EmbeddingExtractor).
    - Cosine similarity = dot product when both vectors are unit-normalised.
    - All operations are CPU / NumPy — no GPU needed for retrieval.
    """

    def __init__(self, index_path: str) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Embedding index not found: {index_path}\n"
                "Run: python src/build_embeddings.py"
            )
        data = np.load(index_path, allow_pickle=True)
        self.embeddings:   np.ndarray = data["embeddings"].astype(np.float32)
        self.image_paths:  list[str]  = data["image_paths"].tolist()
        self.categories:   list[str]  = data["categories"].tolist()
        print(
            f"[SimilarityIndex] Loaded {len(self.image_paths):,} embeddings "
            f"(dim={self.embeddings.shape[1]}) from {index_path}"
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        exclude_path: str | None = None,
    ) -> list[dict]:
        """
        Find the top-K most similar images to a query embedding.

        Parameters
        ----------
        query        : np.ndarray  shape (D,)  L2-normalised query embedding
        top_k        : number of results to return
        exclude_path : optional path to exclude from results (e.g., the query itself)

        Returns
        -------
        list of dicts:
            {
                "image_path": str,
                "category":   str,
                "similarity": float   (cosine similarity, 0–1)
            }
        """
        query = query.astype(np.float32)

        # Re-normalise the query in case it came in un-normalised
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # Cosine similarity via dot product (embeddings are already unit-normed)
        similarities = self.embeddings @ query  # (N,)

        # Get top-K indices (sorted descending)
        # Use argpartition for efficiency then sort the top-K slice
        k = min(top_k + 1, len(similarities))  # +1 to handle self-exclusion
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            path = self.image_paths[idx]
            # Optionally skip the query image itself
            if exclude_path and os.path.abspath(path) == os.path.abspath(exclude_path):
                continue
            sim = float(similarities[idx])
            # Clamp to [0, 1] — numerical noise can produce tiny negatives
            sim = max(0.0, min(1.0, sim))
            results.append({
                "image_path": path,
                "category":   self.categories[idx],
                "similarity": round(sim, 4),
            })
            if len(results) >= top_k:
                break

        return results

    def __len__(self) -> int:
        return len(self.image_paths)

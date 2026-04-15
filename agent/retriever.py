from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from library.tool_library import ToolLibrary

log = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the embedding model (~80 MB, 384-dim)."""
    log.info("Loading sentence-transformers model: %s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


def embed(text: str) -> np.ndarray:
    """Embed a string as a normalized float32 384-dim vector."""
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


class ToolRetriever:
    def __init__(self, library: ToolLibrary, threshold: float = 0.75) -> None:
        self.library = library
        self.threshold = threshold

    def retrieve(self, task: str, top_k: int = 3) -> list[dict[str, Any]] | None:
        """Return top-k tools above threshold, or None if nothing qualifies.

        Returns None to signal the orchestrator to create a new tool.
        """
        query_emb = embed(task)
        results = self.library.search(
            query_embedding=query_emb,
            top_k=top_k,
            threshold=self.threshold,
        )
        if not results:
            log.debug(
                "Retriever: no tools above threshold %.2f for '%s'",
                self.threshold,
                task[:60],
            )
            return None
        log.debug("Retriever: found %d tools for '%s'", len(results), task[:60])
        return results

"""
memory/retriever.py
Embedding-based retrieval using all-MiniLM-L6-v2 (local, CPU, ~80MB).
Embeds reflections at creation time. At query time, embeds the new
broadcast and returns top-K most similar reflection IDs.

No API calls, no rate limits.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.src.memory.store import MemoryStore

_model = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence transformer model (downloads once, ~80MB)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Memory] Loading embedding model '{model_name}' (first use may download ~80MB)...")
            _model = SentenceTransformer(model_name)
            print("[Memory] Embedding model ready.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for memory retrieval.\n"
                "Install it with:  pip install sentence-transformers"
            )
    return _model


def embed(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """Compute embedding for a single text string."""
    model = _get_model(model_name)
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def embed_batch(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """Compute embeddings for a list of strings (more efficient than one-by-one)."""
    if not texts:
        return []
    model = _get_model(model_name)
    vecs = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_top_k(
    query_text: str,
    store: "MemoryStore",
    top_k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Retrieve the top-K reflections most semantically similar to query_text.
    Returns list of reflection dicts (with id, content, similarity score).
    Returns [] if no reflections exist yet.
    """
    index = store.load_index()
    if not index:
        return []

    query_vec = embed(query_text, model_name)

    scored = []
    for entry in index:
        emb = entry.get("embedding")
        if not emb:
            continue
        sim = cosine_similarity(query_vec, emb)
        scored.append({
            "id":         entry["id"],
            "content":    entry["content"],
            "similarity": round(sim, 4),
            "round":      entry.get("round"),
            "speakers":   entry.get("speakers", []),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]

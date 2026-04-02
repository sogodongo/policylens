from sentence_transformers import CrossEncoder

# Lazy-load the model on first use — avoids slowing down imports
# across the whole codebase just because this module is imported
_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("[reranker] Loading cross-encoder model...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-scores candidate chunks by reading the query and each chunk together.

    Expects candidates in the format returned by hybrid_search — dicts
    with at least a 'text' field. Returns top_k results sorted by
    cross-encoder score descending.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

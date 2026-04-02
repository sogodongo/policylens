from retrieval.embedder import embed_chunks
from retrieval.pinecone_store import query_index
from retrieval.bm25_store import search_bm25


# k=60 is the standard RRF constant — higher k reduces the impact of
# top-ranked results, lower k amplifies them. 60 is well-validated empirically.
RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _embed_query(query_text: str) -> list[float]:
    # Reuse embed_chunks with a minimal fake chunk — avoids duplicating
    # the OpenAI client setup just for single-query embedding
    result = embed_chunks([{
        "text": query_text,
        "heading": "", "page": 0, "doc_title": "",
        "source_url": "", "doc_type": "", "jurisdiction": "",
    }])
    return result[0]["embedding"]


def hybrid_search(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
    dense_candidates: int = 8,
    sparse_candidates: int = 8,
) -> list[dict]:
    """
    Queries both Pinecone (dense) and Elasticsearch (sparse) then fuses
    results using Reciprocal Rank Fusion.

    Returns top_k chunks ranked by combined RRF score. Each result
    includes both its original scores and the final rrf_score for debugging.
    """
    query_vector = _embed_query(query)

    dense_results  = query_index(query_vector, namespace=namespace, top_k=dense_candidates)
    sparse_results = search_bm25(query, namespace=namespace, top_k=sparse_candidates)

    # Build a unified score map keyed by chunk text
    # Using text as key because vector IDs and ES doc IDs are different formats
    scores: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        key = result["text"]
        if key not in scores:
            scores[key] = result.copy()
            scores[key]["rrf_score"] = 0.0
            scores[key]["pinecone_rank"] = rank + 1
            scores[key]["bm25_rank"] = None
        scores[key]["rrf_score"] += _rrf_score(rank + 1)

    for rank, result in enumerate(sparse_results):
        key = result["text"]
        if key not in scores:
            scores[key] = result.copy()
            scores[key]["rrf_score"] = 0.0
            scores[key]["pinecone_rank"] = None
        scores[key]["rrf_score"] += _rrf_score(rank + 1)
        scores[key]["bm25_rank"] = rank + 1

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]

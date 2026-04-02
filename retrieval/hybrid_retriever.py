from retrieval.embedder import embed_chunks
from retrieval.pinecone_store import query_index
from retrieval.bm25_store import search_bm25

RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _embed_query(query_text: str) -> list[float]:
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
    use_hyde: bool = False,
) -> list[dict]:
    """
    Queries Pinecone (dense) and Elasticsearch (sparse) then fuses
    results with RRF.

    When use_hyde=True, generates hypothetical document embeddings
    to improve dense retrieval recall on short or vague queries.
    The BM25 search always uses the original query — HyDE only
    helps the vector search side.
    """
    if use_hyde:
        from retrieval.hyde import expand_query_hyde
        hypotheticals = expand_query_hyde(query, n_hypothetical=2)
        print(f"[hyde] Generated {len(hypotheticals)} hypothetical excerpts")

        # Embed the original query + both hypotheticals and merge candidates
        all_query_texts = [query] + hypotheticals
        dense_results = []
        seen_texts = set()

        for text in all_query_texts:
            vec = _embed_query(text)
            results = query_index(vec, namespace=namespace, top_k=dense_candidates)
            for r in results:
                if r["text"] not in seen_texts:
                    dense_results.append(r)
                    seen_texts.add(r["text"])
    else:
        query_vector = _embed_query(query)
        dense_results = query_index(query_vector, namespace=namespace, top_k=dense_candidates)

    sparse_results = search_bm25(query, namespace=namespace, top_k=sparse_candidates)

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

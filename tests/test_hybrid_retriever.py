import sys
sys.path.insert(0, ".")

from retrieval.hybrid_retriever import hybrid_search

query = "euro-currency market borrowers independent growth"
results = hybrid_search(query, namespace="cbk", top_k=5)

print(f"Hybrid results for: '{query}'\n")
for r in results:
    print(f"RRF score     : {r['rrf_score']:.4f}")
    print(f"Pinecone rank : {r.get('pinecone_rank')}")
    print(f"BM25 rank     : {r.get('bm25_rank')}")
    print(f"Heading       : {r['heading']}")
    print(f"Text          : {r['text'][:200]}")
    print("-" * 60)

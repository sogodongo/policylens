import sys
sys.path.insert(0, ".")

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank

query = "How did the euro-currency market grow independently?"

# Get hybrid candidates first
candidates = hybrid_search(query, namespace="cbk", top_k=8)

print(f"Before re-ranking (RRF order):\n")
for i, r in enumerate(candidates):
    print(f"  {i+1}. RRF={r['rrf_score']:.4f} | {r['text'][:100]}")

# Re-rank with cross-encoder
reranked = rerank(query, candidates, top_k=5)

print(f"\nAfter re-ranking (cross-encoder order):\n")
for i, r in enumerate(reranked):
    print(f"  {i+1}. score={r['rerank_score']:.4f} | {r['text'][:100]}")

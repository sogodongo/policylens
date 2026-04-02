import sys
sys.path.insert(0, ".")

from retrieval.hyde import expand_query_hyde
from retrieval.hybrid_retriever import hybrid_search

query = "What controls exist on euro-currency market growth?"

print(f"Query: {query}\n")

# Show what HyDE generates
hypotheticals = expand_query_hyde(query, n_hypothetical=2)
print("HyDE hypothetical excerpts:")
for i, h in enumerate(hypotheticals):
    print(f"\n  [{i+1}] {h[:300]}")

# Compare results with and without HyDE
print(f"\n\n--- Without HyDE ---\n")
results_plain = hybrid_search(query, namespace="cbk", top_k=3, use_hyde=False)
for r in results_plain:
    print(f"  RRF={r['rrf_score']:.4f} | {r['text'][:120]}")

print(f"\n--- With HyDE ---\n")
results_hyde = hybrid_search(query, namespace="cbk", top_k=3, use_hyde=True)
for r in results_hyde:
    print(f"  RRF={r['rrf_score']:.4f} | {r['text'][:120]}")

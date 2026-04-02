import sys
import json
sys.path.insert(0, ".")

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank


def evaluate_retrieval(golden_path: str = "evaluation/golden_dataset.json", top_k: int = 5):
    with open(golden_path) as f:
        golden = json.load(f)

    hits = 0
    reciprocal_ranks = []
    results_log = []

    print(f"Running retrieval eval on {len(golden)} queries...\n")

    for item in golden:
        query    = item["query"]
        fragment = item["expected_fragment"].lower()
        namespace = item["namespace"]

        candidates = hybrid_search(query, namespace=namespace, top_k=top_k * 2)
        reranked   = rerank(query, candidates, top_k=top_k)

        hit = False
        rank_of_hit = None

        for rank, chunk in enumerate(reranked):
            if fragment in chunk["text"].lower():
                hit = True
                rank_of_hit = rank + 1
                break

        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank_of_hit)
        else:
            reciprocal_ranks.append(0.0)

        results_log.append({
            "query":       query,
            "hit":         hit,
            "rank":        rank_of_hit,
            "fragment":    item["expected_fragment"],
        })

    hit_rate = hits / len(golden)
    mrr      = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"{'Query':<55} {'Hit':<6} {'Rank'}")
    print("-" * 70)
    for r in results_log:
        rank_str = str(r["rank"]) if r["rank"] else "miss"
        print(f"{r['query'][:54]:<55} {str(r['hit']):<6} {rank_str}")

    print(f"\n{'='*70}")
    print(f"Hit Rate @ {top_k} : {hit_rate:.2f}  ({hits}/{len(golden)})")
    print(f"MRR          : {mrr:.2f}")
    print(f"{'='*70}")

    return {"hit_rate": hit_rate, "mrr": mrr}


if __name__ == "__main__":
    evaluate_retrieval()

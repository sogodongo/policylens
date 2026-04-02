import sys
import json
sys.path.insert(0, ".")

from generation.chain import run_rag_query

query = "How did the euro-currency market grow independently from domestic banking?"

result = run_rag_query(query, namespace="cbk", top_k=5)

print("\n" + "=" * 70)
print("POLICYLENS ANSWER")
print("=" * 70)
print(f"\nAnswer:\n{result['answer']}")
print(f"\nConfidence : {result['confidence_score']}")
print(f"Jurisdiction: {result.get('jurisdiction', 'N/A')}")
print(f"Legal review required: {result.get('requires_legal_review')}")
print(f"\nCitations ({len(result.get('citations', []))}):")
for c in result.get("citations", []):
    print(f"  [{c['source_id']}] {c['doc_title']} | {c['heading'][:50]}")
    print(f"         Relevance: {c.get('relevance', '')[:100]}")
if result.get("gaps"):
    print(f"\nGaps:")
    for g in result["gaps"]:
        print(f"  - {g}")
print("=" * 70)

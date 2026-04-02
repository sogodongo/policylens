import sys
sys.path.insert(0, ".")

from generation.chain import run_rag_query

query = "How did the euro-currency market grow independently from domestic banking?"
result = run_rag_query(query, namespace="cbk", top_k=5)

print("\n" + "=" * 70)
print("POLICYLENS ANSWER")
print("=" * 70)
print(f"\nAnswer     : {result.answer}")
print(f"Confidence : {result.confidence_score}")
print(f"Jurisdiction: {result.jurisdiction}")
print(f"Legal review: {result.requires_legal_review}")
print(f"\nCitations ({len(result.citations)}):")
for c in result.citations:
    print(f"  [{c.source_id}] {c.doc_title} | p.{c.page}")
    print(f"         {c.relevance[:100]}")
if result.gaps:
    print(f"\nGaps:")
    for g in result.gaps:
        print(f"  - {g}")

# Confirm it's a proper Pydantic model
print(f"\nResponse type : {type(result).__name__}")
print(f"JSON output   : {result.model_dump_json()[:200]}...")
print("=" * 70)

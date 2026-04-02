import sys
sys.path.insert(0, ".")

from generation.chain import run_rag_query

query = "How did the euro-currency market grow independently from domestic banking?"
output = run_rag_query(query, namespace="cbk", top_k=5)

result = output["response"]

print("\n" + "=" * 70)
print("POLICYLENS ANSWER")
print("=" * 70)
print(f"\nAnswer     : {result.answer}")
print(f"Confidence : {result.confidence_score}")
print(f"Jurisdiction: {result.jurisdiction}")
print(f"Legal review: {result.requires_legal_review}")

print(f"\n{output['citation_block']}")

print(f"\nCitation validation:")
for v in output["citation_validation"]:
    status = "OK" if v["valid"] else "HALLUCINATED"
    print(f"  {v['source_id']}: {status}")

if result.gaps:
    print(f"\nGaps:")
    for g in result.gaps:
        print(f"  - {g}")
print("=" * 70)

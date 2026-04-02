import sys
sys.path.insert(0, ".")

from generation.chain import run_rag_query

query = "How did the euro-currency market grow independently from domestic banking?"
output = run_rag_query(query, namespace="cbk", top_k=5, run_faithfulness=True)

result = output["response"]

print("\n" + "=" * 70)
print("POLICYLENS ANSWER")
print("=" * 70)
print(f"\nAnswer     : {result.answer}")
print(f"Confidence : {result.confidence_score}")

print(f"\n{output['citation_block']}")

print(f"\nCitation validation:")
for v in output["citation_validation"]:
    status = "OK" if v["valid"] else "HALLUCINATED"
    print(f"  {v['source_id']}: {status}")

if output["faithfulness"]:
    f = output["faithfulness"]
    print(f"\nFaithfulness check:")
    print(f"  Score  : {f['faithfulness_score']}")
    print(f"  Verdict: {f['verdict']}")
    if f["unsupported_claims"]:
        print(f"  Unsupported claims:")
        for u in f["unsupported_claims"]:
            print(f"    - {u['claim'][:100]} (score={u['entailment_score']})")
    else:
        print(f"  All claims supported by context.")
print("=" * 70)

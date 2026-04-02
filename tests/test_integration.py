import sys
import json
sys.path.insert(0, ".")

from ingestion.pipeline import ingest_document
from generation.chain import run_rag_query

PDF_PATH    = "data/cbk_test.pdf"
DOC_TITLE   = "CBK Test Document"
NAMESPACE   = "cbk"
JURISDICTION = "Kenya"

TEST_QUERIES = [
    {
        "query":    "How did the euro-currency market grow independently?",
        "min_confidence": 0.6,
    },
    {
        "query":    "What happened to oil prices and the euro-currency market?",
        "min_confidence": 0.5,
    },
    {
        "query":    "What is quantum computing?",
        "min_confidence": 0.0,
        "expect_low_confidence": True,
    },
]


def run_integration_test():
    print("\n" + "=" * 70)
    print("POLICYLENS INTEGRATION TEST")
    print("=" * 70)

    # Step 1: Ingestion
    print("\n[Step 1] Ingesting document...")
    summary = ingest_document(
        pdf_path=PDF_PATH,
        doc_title=DOC_TITLE,
        doc_type="circular",
        jurisdiction=JURISDICTION,
        namespace=NAMESPACE,
    )
    print(f"  Status  : {summary['status']}")
    if summary["status"] == "ingested":
        print(f"  Chunks  : {summary['chunks']}")
        print(f"  Vectors : {summary['vectors_upserted']}")

    # Step 2: Query each test case
    print(f"\n[Step 2] Running {len(TEST_QUERIES)} test queries...\n")

    results = []
    passed  = 0
    failed  = 0

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        print(f"  Query {i}: {query}")

        output = run_rag_query(
            query=query,
            namespace=NAMESPACE,
            top_k=5,
            run_faithfulness=True,
        )

        response    = output["response"]
        faithfulness = output["faithfulness"]
        validation  = output["citation_validation"]

        hallucinated = [v for v in validation if not v["valid"]]
        all_citations_valid = len(hallucinated) == 0

        confidence_ok = (
            response.confidence_score >= test["min_confidence"]
            if not test.get("expect_low_confidence")
            else True
        )

        test_passed = confidence_ok and all_citations_valid

        status = "PASS" if test_passed else "FAIL"
        if test_passed:
            passed += 1
        else:
            failed += 1

        result = {
            "query":              query,
            "status":             status,
            "confidence":         response.confidence_score,
            "citations":          len(response.citations),
            "hallucinated":       len(hallucinated),
            "faithfulness_score": faithfulness["faithfulness_score"] if faithfulness else "N/A",
            "faithfulness_verdict": faithfulness["verdict"] if faithfulness else "N/A",
        }
        results.append(result)

        print(f"    Status      : {status}")
        print(f"    Confidence  : {response.confidence_score}")
        print(f"    Citations   : {len(response.citations)} "
              f"({'all valid' if all_citations_valid else f'{len(hallucinated)} hallucinated'})")
        if faithfulness:
            print(f"    Faithfulness: {faithfulness['verdict']} "
                  f"(score={faithfulness['faithfulness_score']})")
        print()

    # Step 3: Summary
    print("=" * 70)
    print(f"INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    print(f"\n{'Query':<50} {'Status':<6} {'Conf':<6} {'Faith'}")
    print("-" * 75)
    for r in results:
        print(f"{r['query'][:49]:<50} {r['status']:<6} "
              f"{r['confidence']:<6} {r['faithfulness_verdict']}")

    print(f"\n{'OVERALL: PASS' if failed == 0 else 'OVERALL: FAIL'}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)

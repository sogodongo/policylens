import sys
sys.path.insert(0, ".")

from ingestion.pipeline import ingest_document

# First run — should ingest
result = ingest_document(
    pdf_path="data/cbk_test.pdf",
    doc_title="CBK Test Document",
    doc_type="circular",
    jurisdiction="Kenya",
    namespace="cbk",
)
print(f"\nFirst run  : {result}")

# Second run — should skip because hash hasn't changed
result2 = ingest_document(
    pdf_path="data/cbk_test.pdf",
    doc_title="CBK Test Document",
    doc_type="circular",
    jurisdiction="Kenya",
    namespace="cbk",
)
print(f"Second run : {result2}")

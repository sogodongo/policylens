import sys
import json
sys.path.insert(0, ".")

from ingestion.parsers.pdf_parser import parse_pdf

blocks = parse_pdf(
    pdf_path="data/cbk_test.pdf",
    doc_title="CBK Test Document",
    doc_type="circular",
    jurisdiction="Kenya"
)

with open("data/blocks_cache.json", "w") as f:
    json.dump(blocks, f, indent=2)

print(f"Cached {len(blocks)} blocks to data/blocks_cache.json")

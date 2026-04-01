import sys
sys.path.insert(0, ".")

from ingestion.parsers.pdf_parser import parse_pdf

blocks = parse_pdf(
    pdf_path="data/cbk_test.pdf",
    doc_title="CBK Test Document",
    doc_type="circular",
    jurisdiction="Kenya"
)

print(f"Total blocks: {len(blocks)}")

headed = [b for b in blocks if b["heading"]]
print(f"Blocks with headings: {len(headed)}")
print("\n--- First 3 blocks with headings ---\n")
for b in headed[:3]:
    print(f"Heading : {b['heading']}")
    print(f"Page    : {b['page']}")
    print(f"Text    : {b['text'][:200]}")
    print("-" * 60)

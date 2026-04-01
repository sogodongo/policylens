from docling.document_converter import DocumentConverter
from pathlib import Path


def parse_pdf(pdf_path: str, doc_title: str, doc_type: str, jurisdiction: str, source_url: str = "") -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[pdf_parser] Parsing {path.name} ...")
    converter = DocumentConverter()
    doc = converter.convert(str(path)).document

    blocks = []
    current_heading = ""

    # Docling uses 'section_header' not 'heading' — keeping this explicit
    # rather than a fuzzy match so we don't accidentally catch other labels
    HEADING_LABELS = {"section_header", "heading"}

    for element, _level in doc.iterate_items():
        label = str(getattr(element, "label", "")).lower()

        if "picture" in label:
            continue

        if hasattr(element, "text"):
            text = element.text.strip()
        elif hasattr(element, "export_to_markdown"):
            try:
                text = element.export_to_markdown(doc=doc).strip()
            except TypeError:
                text = element.export_to_markdown().strip()
        else:
            continue

        if not text:
            continue

        if label in HEADING_LABELS:
            current_heading = text
            continue

        blocks.append({
            "text": text,
            "heading": current_heading,
            "page": getattr(element, "page_no", 0),
            "doc_title": doc_title,
            "source_url": source_url,
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
        })

    print(f"[pdf_parser] {len(blocks)} blocks extracted")
    return blocks

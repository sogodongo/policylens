import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.chunker import chunk_blocks
from retrieval.embedder import embed_chunks
from retrieval.pinecone_store import upsert_chunks

load_dotenv()

REGISTRY_PATH = "data/ingestion_registry.json"


def _load_registry() -> dict:
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _file_hash(path: str) -> str:
    # MD5 is fine here — we're detecting changes, not securing anything
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def ingest_document(
    pdf_path: str,
    doc_title: str,
    doc_type: str,
    jurisdiction: str,
    namespace: str,
    source_url: str = "",
    force: bool = False,
) -> dict:
    """
    Full ingestion pipeline: PDF -> blocks -> chunks -> embeddings -> Pinecone.

    Skips re-ingestion if the document hash hasn't changed unless force=True.
    Returns a summary dict for logging/auditing.
    """
    registry = _load_registry()
    file_hash = _file_hash(pdf_path)
    doc_key = f"{namespace}::{doc_title}"

    if not force and registry.get(doc_key) == file_hash:
        print(f"[pipeline] Skipping '{doc_title}' — already ingested and unchanged.")
        return {"status": "skipped", "doc_title": doc_title}

    print(f"[pipeline] Starting ingestion: {doc_title}")

    blocks = parse_pdf(
        pdf_path=pdf_path,
        doc_title=doc_title,
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        source_url=source_url,
    )

    if not blocks:
        print(f"[pipeline] No blocks extracted from {pdf_path} — skipping.")
        return {"status": "empty", "doc_title": doc_title}

    chunks = chunk_blocks(blocks)
    print(f"[pipeline] {len(blocks)} blocks -> {len(chunks)} chunks")

    embedded = embed_chunks(chunks)
    upsert_chunks(embedded, namespace=namespace)

    # Record the hash so we don't re-ingest unchanged documents
    registry[doc_key] = file_hash
    _save_registry(registry)

    summary = {
        "status": "ingested",
        "doc_title": doc_title,
        "namespace": namespace,
        "blocks": len(blocks),
        "chunks": len(chunks),
        "vectors_upserted": len(embedded),
    }

    print(f"[pipeline] Done: {summary}")
    return summary

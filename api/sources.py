import json
from pathlib import Path

REGISTRY_PATH = "data/ingestion_registry.json"


def get_ingested_sources() -> list[dict]:
    """
    Reads the ingestion registry and returns a list of ingested documents.
    The registry key format is 'namespace::doc_title'.
    """
    if not Path(REGISTRY_PATH).exists():
        return []

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    sources = []
    for key, file_hash in registry.items():
        parts = key.split("::", 1)
        namespace = parts[0] if len(parts) == 2 else "default"
        doc_title = parts[1] if len(parts) == 2 else parts[0]
        sources.append({
            "namespace": namespace,
            "doc_title": doc_title,
            "file_hash": file_hash[:8] + "...",
        })

    return sources

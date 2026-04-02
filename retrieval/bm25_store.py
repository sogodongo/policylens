from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv()

_es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "policylens_bm25"


def _create_index_if_missing():
    existing = _es.cat.indices(format="json")
    index_names = [idx["index"] for idx in existing]

    if INDEX_NAME in index_names:
        return

    _es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "text":         {"type": "text", "analyzer": "english"},
                "heading":      {"type": "text", "analyzer": "english"},
                "doc_title":    {"type": "keyword"},
                "jurisdiction": {"type": "keyword"},
                "doc_type":     {"type": "keyword"},
                "page":         {"type": "integer"},
                "source_url":   {"type": "keyword"},
                "chunk_index":  {"type": "integer"},
                "namespace":    {"type": "keyword"},
            }
        }
    )
    print(f"[bm25] Created index '{INDEX_NAME}'")


def index_chunks(chunks: list[dict], namespace: str = "default") -> int:
    _create_index_if_missing()

    actions = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{chunk['doc_title'].replace(' ', '_')[:40]}_{namespace}_{i}"
        actions.append({
            "_index": INDEX_NAME,
            "_id": doc_id,
            "_source": {
                "text":         chunk["text"],
                "heading":      chunk["heading"],
                "doc_title":    chunk["doc_title"],
                "jurisdiction": chunk["jurisdiction"],
                "doc_type":     chunk["doc_type"],
                "page":         chunk["page"],
                "source_url":   chunk["source_url"],
                "chunk_index":  chunk.get("chunk_index", 0),
                "namespace":    namespace,
            },
        })

    success, failed = helpers.bulk(_es, actions, raise_on_error=False)
    print(f"[bm25] Indexed {success} chunks (namespace={namespace}, failed={len(failed)})")
    return success


def search_bm25(query: str, namespace: str = "default", top_k: int = 8) -> list[dict]:
    _create_index_if_missing()

    response = _es.search(
        index=INDEX_NAME,
        size=top_k,
        query={
            "bool": {
                "must":   {"match": {"text": {"query": query, "operator": "or"}}},
                "filter": {"term": {"namespace": namespace}},
            }
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        result = hit["_source"].copy()
        result["bm25_score"] = hit["_score"]
        results.append(result)

    return results

import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "policylens")

# 3072 matches text-embedding-3-large output dimensions
EMBEDDING_DIM = 3072
UPSERT_BATCH_SIZE = 100


def get_or_create_index():
    """
    Creates the Pinecone index if it doesn't exist yet.
    Serverless on AWS us-east-1 is the free tier option.
    """
    existing = [idx.name for idx in _pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"[pinecone] Creating index '{INDEX_NAME}'...")
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Index takes a few seconds to become ready
        while not _pc.describe_index(INDEX_NAME).status["ready"]:
            print("[pinecone] Waiting for index to be ready...")
            time.sleep(2)
        print(f"[pinecone] Index '{INDEX_NAME}' ready.")
    else:
        print(f"[pinecone] Index '{INDEX_NAME}' already exists.")

    return _pc.Index(INDEX_NAME)


def upsert_chunks(chunks: list[dict], namespace: str = "default") -> int:
    """
    Upserts embedded chunks into Pinecone.

    Each vector needs a unique ID — we build it from doc_title + chunk
    position so re-ingesting the same document overwrites cleanly rather
    than creating duplicates.
    """
    index = get_or_create_index()
    vectors = []

    for i, chunk in enumerate(chunks):
        if "embedding" not in chunk:
            raise ValueError(f"Chunk {i} has no embedding. Run embed_chunks first.")

        # Sanitize the ID — Pinecone IDs can't have spaces or special chars
        doc_id = chunk["doc_title"].replace(" ", "_").replace("/", "-")[:40]
        vector_id = f"{doc_id}_{namespace}_{i}"

        # Store everything except the embedding itself as metadata
        # so we can retrieve it alongside the vector match
        metadata = {
            "text": chunk["text"],
            "heading": chunk["heading"],
            "page": chunk["page"],
            "doc_title": chunk["doc_title"],
            "source_url": chunk["source_url"],
            "doc_type": chunk["doc_type"],
            "jurisdiction": chunk["jurisdiction"],
        }

        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": metadata,
        })

    # Upsert in batches
    upserted = 0
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i: i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        upserted += len(batch)
        print(f"[pinecone] Upserted {upserted}/{len(vectors)} vectors...")

    print(f"[pinecone] Done. {upserted} vectors stored in namespace '{namespace}'.")
    return upserted


def query_index(query_embedding: list[float], namespace: str = "default", top_k: int = 5) -> list[dict]:
    """
    Finds the top_k most semantically similar chunks to a query vector.
    Returns a list of metadata dicts with a 'score' field attached.
    """
    index = get_or_create_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    matches = []
    for match in results.matches:
        result = match.metadata.copy()
        result["score"] = match.score
        result["vector_id"] = match.id
        matches.append(result)

    return matches

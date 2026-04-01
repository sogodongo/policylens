import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Batching keeps API calls efficient — embedding one chunk at a time
# would hit rate limits fast on large document sets
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-large"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an 'embedding' field to each chunk by calling the OpenAI
    embeddings API in batches.

    Returns the same list with embeddings attached in-place.
    Rate limit handling is intentionally simple — exponential backoff
    is overkill for the batch sizes we're dealing with here.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    embedded = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_chunks = chunks[i: i + BATCH_SIZE]

        print(f"[embedder] Embedding batch {i // BATCH_SIZE + 1} "
              f"({len(batch_texts)} chunks)...")

        try:
            response = _client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
            )
        except Exception as e:
            # On rate limit, wait and retry once before giving up
            if "rate_limit" in str(e).lower():
                print(f"[embedder] Rate limited, waiting 20s...")
                time.sleep(20)
                response = _client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts,
                )
            else:
                raise

        for chunk, embedding_obj in zip(batch_chunks, response.data):
            chunk["embedding"] = embedding_obj.embedding
            embedded.append(chunk)

    print(f"[embedder] Done. {len(embedded)} chunks embedded.")
    return embedded

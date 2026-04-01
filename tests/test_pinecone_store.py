import sys
import json
sys.path.insert(0, ".")

from ingestion.chunker import chunk_blocks
from retrieval.embedder import embed_chunks
from retrieval.pinecone_store import upsert_chunks, query_index

with open("data/blocks_cache.json") as f:
    blocks = json.load(f)

chunks = chunk_blocks(blocks)

# Embed and upsert first 10 chunks
sample = embed_chunks(chunks[:10])
upsert_chunks(sample, namespace="cbk")

# Wait a moment for Pinecone to index the vectors
import time
print("\n[test] Waiting 5s for Pinecone to index...")
time.sleep(5)

# Now query with a sample question
from retrieval.embedder import embed_chunks as get_embedding
query_text = "What is the euro-currency market?"
query_embedded = embed_chunks([{"text": query_text, "heading": "", "page": 0,
                                "doc_title": "", "source_url": "",
                                "doc_type": "", "jurisdiction": ""}])
query_vector = query_embedded[0]["embedding"]

results = query_index(query_vector, namespace="cbk", top_k=3)

print(f"\n--- Top 3 results for: '{query_text}' ---\n")
for r in results:
    print(f"Score   : {r['score']:.4f}")
    print(f"Heading : {r['heading']}")
    print(f"Text    : {r['text'][:200]}")
    print("-" * 60)

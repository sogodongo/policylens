import sys
import json
sys.path.insert(0, ".")

from ingestion.chunker import chunk_blocks
from retrieval.bm25_store import index_chunks, search_bm25

with open("data/blocks_cache.json") as f:
    blocks = json.load(f)

chunks = chunk_blocks(blocks)
index_chunks(chunks, namespace="cbk")

results = search_bm25("euro-currency market borrowers", namespace="cbk", top_k=3)

print(f"\nBM25 results for 'euro-currency market borrowers':\n")
for r in results:
    print(f"Score   : {r['bm25_score']:.4f}")
    print(f"Heading : {r['heading']}")
    print(f"Text    : {r['text'][:200]}")
    print("-" * 60)

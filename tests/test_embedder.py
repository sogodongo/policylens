import sys
import json
sys.path.insert(0, ".")

from ingestion.chunker import chunk_blocks
from retrieval.embedder import embed_chunks

with open("data/blocks_cache.json") as f:
    blocks = json.load(f)

chunks = chunk_blocks(blocks)

# Only embed first 5 to keep test cost minimal
sample = chunks[:5]
embedded = embed_chunks(sample)

print(f"\nEmbedded {len(embedded)} chunks")
print(f"Embedding dimensions: {len(embedded[0]['embedding'])}")
print(f"\n--- Sample ---")
print(f"Text      : {embedded[0]['text'][:100]}")
print(f"Embedding : [{embedded[0]['embedding'][0]:.6f}, "
      f"{embedded[0]['embedding'][1]:.6f}, ... "
      f"{embedded[0]['embedding'][-1]:.6f}]")

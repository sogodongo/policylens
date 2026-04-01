import sys
import json
sys.path.insert(0, ".")

from ingestion.chunker import chunk_blocks
import tiktoken

with open("data/blocks_cache.json") as f:
    blocks = json.load(f)

chunks = chunk_blocks(blocks)

tokenizer = tiktoken.get_encoding("cl100k_base")
token_counts = [len(tokenizer.encode(c["text"])) for c in chunks]

print(f"Blocks in  : {len(blocks)}")
print(f"Chunks out : {len(chunks)}")
print(f"\nToken size stats:")
print(f"  Min     : {min(token_counts)}")
print(f"  Max     : {max(token_counts)}")
print(f"  Avg     : {sum(token_counts) // len(token_counts)}")
print(f"  Over 512: {sum(1 for t in token_counts if t > 512)}")
print(f"\n--- First 3 chunks ---\n")
for c in chunks[:3]:
    print(f"Heading     : {c['heading']}")
    print(f"Chunk index : {c['chunk_index']}")
    print(f"Text        : {c['text'][:200]}")
    print("-" * 60)

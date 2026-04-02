import sys
import json
sys.path.insert(0, ".")

from ingestion.chunker import chunk_blocks
from retrieval.embedder import embed_chunks
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from retrieval.assembler import deduplicate_mmr, assemble_context

# Load cached blocks
with open("data/blocks_cache.json") as f:
    blocks = json.load(f)

chunks = chunk_blocks(blocks)

query = "How did the euro-currency market grow independently?"

# Full retrieval pipeline
candidates = hybrid_search(query, namespace="cbk", top_k=8)
reranked   = rerank(query, candidates, top_k=5)

# Embed the reranked chunks so MMR can compare them
# In production these embeddings come from Pinecone metadata
# but for chunks from BM25-only results we need to embed them here
texts_to_embed = [c for c in reranked if "embedding" not in c]
if texts_to_embed:
    embedded = embed_chunks(texts_to_embed)
    for chunk, emb in zip(texts_to_embed, embedded):
        chunk["embedding"] = emb["embedding"]

deduped   = deduplicate_mmr(reranked)
assembled = assemble_context(deduped)

print(f"\nChunks after re-rank  : {len(reranked)}")
print(f"Chunks after MMR dedup: {len(deduped)}")
print(f"Context tokens        : {assembled['token_count']}")
print(f"\nCitation map:")
for sid, meta in assembled["citation_map"].items():
    print(f"  {sid}: {meta['doc_title']} | {meta['heading'][:60]}")
print(f"\n--- Context block preview ---\n")
print(assembled["context"][:800])

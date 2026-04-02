import sys
import json
sys.path.insert(0, ".")

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from retrieval.assembler import deduplicate_mmr, assemble_context
from generation.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT

query = "How did the euro-currency market grow independently?"

candidates = hybrid_search(query, namespace="cbk", top_k=8)
reranked   = rerank(query, candidates, top_k=5)
deduped    = deduplicate_mmr(reranked)
assembled  = assemble_context(deduped)

# Render the prompt template with real context
rendered_prompt = RAG_PROMPT_TEMPLATE.format(
    context=assembled["context"],
    query=query,
)

print(f"System prompt chars  : {len(SYSTEM_PROMPT)}")
print(f"Rendered prompt chars: {len(rendered_prompt)}")
print(f"Context chunks used  : {len(assembled['citation_map'])}")
print(f"Context tokens       : {assembled['token_count']}")
print(f"\n--- Rendered prompt preview (first 600 chars) ---\n")
print(rendered_prompt[:600])

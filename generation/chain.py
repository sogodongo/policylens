import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from retrieval.assembler import deduplicate_mmr, assemble_context
from generation.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

load_dotenv()

_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def _parse_json_response(raw: str) -> dict:
    # GPT-4o occasionally wraps JSON in markdown code fences
    # Strip them before parsing
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def run_rag_query(
    query: str,
    namespace: str = "default",
    use_hyde: bool = False,
    top_k: int = 5,
) -> dict:
    """
    Full RAG pipeline: retrieval → context assembly → generation → parsed response.

    Returns a structured dict with answer, citations, confidence score, and gaps.
    Raises on JSON parse failure rather than returning a partial result —
    callers should handle this and retry or surface the error to the user.
    """
    print(f"\n[chain] Query: {query}")

    # Retrieval
    candidates = hybrid_search(
        query, namespace=namespace,
        top_k=top_k * 2, use_hyde=use_hyde
    )
    reranked  = rerank(query, candidates, top_k=top_k)
    deduped   = deduplicate_mmr(reranked)
    assembled = assemble_context(deduped)

    if not assembled["citation_map"]:
        return {
            "answer": "No relevant context found in the knowledge base.",
            "confidence_score": 0.0,
            "citations": [],
            "gaps": ["No documents matched this query."],
            "jurisdiction": "Unknown",
            "requires_legal_review": True,
        }

    # Fill prompt template with retrieved context
    user_message = RAG_PROMPT_TEMPLATE.format(
        context=assembled["context"],
        query=query,
    )

    print(f"[chain] Calling GPT-4o with {assembled['token_count']} context tokens...")

    response = _llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw_output = response.content
    print(f"[chain] Response received ({len(raw_output)} chars)")

    result = _parse_json_response(raw_output)

    # Enrich citations with full metadata from the citation map
    for citation in result.get("citations", []):
        sid = citation.get("source_id")
        if sid and sid in assembled["citation_map"]:
            meta = assembled["citation_map"][sid]
            citation.setdefault("doc_title",    meta.get("doc_title", ""))
            citation.setdefault("heading",      meta.get("heading", ""))
            citation.setdefault("page",         meta.get("page", 0))
            citation.setdefault("source_url",   meta.get("source_url", ""))
            citation.setdefault("jurisdiction", meta.get("jurisdiction", ""))

    return result

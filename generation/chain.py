import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from retrieval.assembler import deduplicate_mmr, assemble_context
from generation.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from generation.output_parser import parse_llm_output, PolicyLensResponse
from generation.citation_builder import validate_citations_against_context, build_citation_block
from generation.faithfulness import check_faithfulness

load_dotenv()

_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON or did not match the required schema. "
    "Respond ONLY with a valid JSON object. No markdown, no prose, no code fences. "
    "Start your response with { and end with }."
)


def _call_llm(system: str, user: str) -> str:
    response = _llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


def run_rag_query(
    query: str,
    namespace: str = "default",
    use_hyde: bool = False,
    top_k: int = 5,
    run_faithfulness: bool = True,
) -> dict:
    """
    Full RAG pipeline: retrieval → assembly → generation → validation → faithfulness.

    run_faithfulness=False skips the NLI check — useful in development
    when you want faster iteration and don't need the faithfulness score.
    """
    print(f"\n[chain] Query: {query}")

    candidates = hybrid_search(
        query, namespace=namespace,
        top_k=top_k * 2, use_hyde=use_hyde,
    )
    reranked  = rerank(query, candidates, top_k=top_k)
    deduped   = deduplicate_mmr(reranked)
    assembled = assemble_context(deduped)

    if not assembled["citation_map"]:
        return {
            "response": PolicyLensResponse(
                answer="No relevant context found in the knowledge base.",
                confidence_score=0.0,
                gaps=["No documents matched this query."],
                requires_legal_review=True,
            ),
            "citation_block":      "No citations available.",
            "citation_validation": [],
            "faithfulness":        None,
        }

    user_message = RAG_PROMPT_TEMPLATE.format(
        context=assembled["context"],
        query=query,
    )

    print(f"[chain] Calling GPT-4o ({assembled['token_count']} context tokens)...")
    raw = _call_llm(SYSTEM_PROMPT, user_message)

    try:
        result = parse_llm_output(raw)
    except (ValueError, Exception) as e:
        print(f"[chain] First attempt failed: {e}. Retrying...")
        raw = _call_llm(SYSTEM_PROMPT, user_message + _RETRY_SUFFIX)
        result = parse_llm_output(raw)

    for citation in result.citations:
        sid = citation.source_id
        if sid in assembled["citation_map"]:
            meta = assembled["citation_map"][sid]
            citation.doc_title    = citation.doc_title    or meta.get("doc_title", "")
            citation.heading      = citation.heading      or meta.get("heading", "")
            citation.page         = citation.page         or meta.get("page", 0)
            citation.source_url   = citation.source_url   or meta.get("source_url", "")
            citation.jurisdiction = citation.jurisdiction or meta.get("jurisdiction", "")

    citation_validation = validate_citations_against_context(
        result.citations, assembled["citation_map"]
    )

    hallucinated = [v for v in citation_validation if not v["valid"]]
    if hallucinated:
        print(f"[chain] WARNING: {len(hallucinated)} hallucinated citation(s) detected")

    citation_block = build_citation_block(result.citations)

    faithfulness_result = None
    if run_faithfulness:
        faithfulness_result = check_faithfulness(
            answer=result.answer,
            context=assembled["context"],
        )
        print(f"[chain] Faithfulness: {faithfulness_result['verdict']} "
              f"(score={faithfulness_result['faithfulness_score']})")

    print(f"[chain] Done. Confidence: {result.confidence_score}")
    return {
        "response":            result,
        "citation_block":      citation_block,
        "citation_validation": citation_validation,
        "faithfulness":        faithfulness_result,
    }

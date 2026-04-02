from generation.output_parser import Citation


def format_citation(citation: Citation) -> str:
    """
    Produces a human-readable citation string for display in the UI
    or API response.

    Handles missing page numbers and URLs gracefully — a citation with
    incomplete metadata is still better than no citation at all.
    """
    parts = [citation.doc_title or "Unknown document"]

    if citation.heading:
        # Truncate long headings — some regulatory sections have very long titles
        heading = citation.heading[:80]
        if len(citation.heading) > 80:
            heading += "..."
        parts.append(f"Section: {heading}")

    if citation.page and citation.page > 0:
        parts.append(f"Page {citation.page}")

    if citation.jurisdiction and citation.jurisdiction != "Unknown":
        parts.append(f"[{citation.jurisdiction}]")

    formatted = " — ".join(parts)

    if citation.source_url:
        formatted += f"\n  Source: {citation.source_url}"

    return formatted


def build_citation_block(citations: list[Citation]) -> str:
    """
    Builds a full citation block for inclusion in API responses
    or document exports.

    Format mirrors academic citation style so compliance teams can
    paste it directly into audit reports.
    """
    if not citations:
        return "No citations available."

    lines = ["References:"]
    for i, citation in enumerate(citations, 1):
        lines.append(f"\n[{citation.source_id}] {format_citation(citation)}")
        if citation.relevance:
            lines.append(f"     Relevance: {citation.relevance}")

    return "\n".join(lines)


def validate_citations_against_context(
    citations: list[Citation],
    citation_map: dict,
) -> list[dict]:
    """
    Cross-checks the LLM's cited SOURCE_IDs against the actual context
    that was provided. Flags any SOURCE_ID the LLM invented that wasn't
    in the retrieved context — a hallucinated citation is worse than
    no citation.

    Returns a list of validation results, one per citation.
    """
    results = []
    for citation in citations:
        sid = citation.source_id
        in_context = sid in citation_map
        results.append({
            "source_id": sid,
            "valid":     in_context,
            "issue":     None if in_context else f"{sid} was not in the retrieved context",
        })
    return results

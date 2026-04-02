import numpy as np
import tiktoken

_tokenizer = tiktoken.get_encoding("cl100k_base")

# Hard limit on context tokens — leaves room for system prompt + response
# inside GPT-4o's 128K context window
MAX_CONTEXT_TOKENS = 6000
MMR_SIMILARITY_THRESHOLD = 0.97


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def deduplicate_mmr(chunks: list[dict]) -> list[dict]:
    """
    Removes near-duplicate chunks using cosine similarity on embeddings.
    Chunks without embeddings are passed through unchanged — this handles
    cases where chunks came from BM25 only and were never embedded.
    """
    if not chunks:
        return []

    selected = []
    for candidate in chunks:
        candidate_embedding = candidate.get("embedding")

        if candidate_embedding is None:
            selected.append(candidate)
            continue

        is_duplicate = False
        for kept in selected:
            kept_embedding = kept.get("embedding")
            if kept_embedding is None:
                continue
            sim = _cosine_similarity(candidate_embedding, kept_embedding)
            if sim >= MMR_SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(candidate)

    removed = len(chunks) - len(selected)
    if removed:
        print(f"[assembler] MMR removed {removed} near-duplicate chunks")

    return selected


def assemble_context(chunks: list[dict]) -> dict:
    """
    Packages chunks into a structured context block for the LLM.

    Each chunk gets a SOURCE_ID tag ([S1], [S2]...) that the LLM uses
    when citing its answer. The assembler enforces a token budget —
    chunks are added in order until the budget is exhausted.

    Returns a dict with the formatted context string and a citation map
    that links SOURCE_IDs back to document metadata.
    """
    context_parts = []
    citation_map = {}
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        source_id = f"S{i + 1}"
        block = (
            f"[{source_id}]\n"
            f"Document : {chunk.get('doc_title', 'Unknown')}\n"
            f"Section  : {chunk.get('heading', '')}\n"
            f"Page     : {chunk.get('page', 'N/A')}\n"
            f"---\n"
            f"{chunk['text']}\n"
        )

        block_tokens = _token_count(block)
        if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
            print(f"[assembler] Token budget reached at chunk {i + 1}, stopping.")
            break

        context_parts.append(block)
        citation_map[source_id] = {
            "doc_title":  chunk.get("doc_title", ""),
            "heading":    chunk.get("heading", ""),
            "page":       chunk.get("page", 0),
            "source_url": chunk.get("source_url", ""),
            "jurisdiction": chunk.get("jurisdiction", ""),
        }
        total_tokens += block_tokens

    context_str = "\n".join(context_parts)
    print(f"[assembler] Context assembled: {len(citation_map)} chunks, "
          f"{total_tokens} tokens")

    return {
        "context":      context_str,
        "citation_map": citation_map,
        "token_count":  total_tokens,
    }

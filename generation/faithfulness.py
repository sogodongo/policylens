from sentence_transformers import CrossEncoder

_nli_model = None


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print("[faithfulness] Loading NLI model...")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _extract_plain_text(context: str) -> list[str]:
    """
    Strips SOURCE_ID headers and formatting from the assembled context block
    and returns a list of plain text chunks.

    The NLI model expects clean premise text — the formatting tags we add
    for the LLM confuse the entailment scorer.
    """
    chunks = []
    current_lines = []

    for line in context.split("\n"):
        stripped = line.strip()
        # Skip the metadata header lines
        if (stripped.startswith("[S")
                or stripped.startswith("Document :")
                or stripped.startswith("Section  :")
                or stripped.startswith("Page     :")
                or stripped == "---"):
            if current_lines:
                text = " ".join(current_lines).strip()
                if len(text) > 20:
                    chunks.append(text)
                current_lines = []
            continue
        if stripped:
            current_lines.append(stripped)

    if current_lines:
        text = " ".join(current_lines).strip()
        if len(text) > 20:
            chunks.append(text)

    return chunks


def check_faithfulness(answer: str, context: str) -> dict:
    """
    Checks whether the answer is faithful to the provided context using NLI.

    Each answer sentence is checked against each context chunk individually.
    A sentence is considered supported if ANY context chunk entails it above
    the threshold — this is more accurate than checking against the full
    concatenated context block.
    """
    model = _get_nli_model()

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
    context_chunks = _extract_plain_text(context)

    if not sentences or not context_chunks:
        return {
            "faithfulness_score": 1.0,
            "unsupported_claims": [],
            "verdict": "PASS",
        }

    # NLI labels: 0=contradiction, 1=entailment, 2=neutral
    ENTAILMENT_IDX = 1
    unsupported = []
    sentence_scores = []

    for sentence in sentences:
        # Check this sentence against every context chunk
        pairs = [(chunk, sentence) for chunk in context_chunks]
        scores = model.predict(pairs, apply_softmax=True)

        # Take the max entailment score across all chunks —
        # if any chunk supports the claim, it's considered faithful
        max_entailment = max(float(s[ENTAILMENT_IDX]) for s in scores)
        sentence_scores.append(max_entailment)

        if max_entailment < 0.4:
            unsupported.append({
                "claim":            sentence,
                "entailment_score": round(max_entailment, 3),
            })

    faithfulness_score = (
        sum(sentence_scores) / len(sentence_scores)
        if sentence_scores else 1.0
    )

    verdict = "PASS" if faithfulness_score >= 0.5 else "FAIL"

    if unsupported:
        print(f"[faithfulness] {len(unsupported)} unsupported claim(s) detected")

    return {
        "faithfulness_score": round(faithfulness_score, 3),
        "unsupported_claims": unsupported,
        "verdict":            verdict,
    }

import sys
import json
sys.path.insert(0, ".")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from retrieval.assembler import deduplicate_mmr, assemble_context
from generation.chain import run_rag_query

# RAGAS needs LLM and embedding wrappers
_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
))

_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
))

GOLDEN_PATH = "evaluation/golden_dataset.json"


def build_ragas_dataset(golden: list[dict], namespace: str = "cbk") -> dict:
    """
    Runs each golden query through the retrieval pipeline and collects
    the inputs RAGAS needs: question, answer, contexts, ground_truth.
    """
    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for item in golden:
        query = item["query"]
        print(f"[ragas] Processing: {query[:60]}...")

        # Get retrieved context
        candidates = hybrid_search(query, namespace=namespace, top_k=8)
        reranked   = rerank(query, candidates, top_k=5)
        deduped    = deduplicate_mmr(reranked)
        assembled  = assemble_context(deduped)

        # Get generated answer
        output   = run_rag_query(
            query=query,
            namespace=namespace,
            top_k=5,
            run_faithfulness=False,
        )
        answer = output["response"].answer

        # Contexts are the plain text of each retrieved chunk
        chunk_texts = [c["text"] for c in deduped if "text" in c]

        questions.append(query)
        answers.append(answer)
        contexts_list.append(chunk_texts)
        # Use the expected fragment as a simple ground truth
        ground_truths.append(item.get("expected_fragment", ""))

    return {
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
        "ground_truth": ground_truths,
    }


def run_ragas_eval(namespace: str = "cbk", sample_size: int = 5) -> dict:
    """
    Runs RAGAS evaluation on a sample of the golden dataset.

    Uses sample_size=5 by default to keep API costs low during
    development — run with the full dataset before deployment.
    """
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    # Sample to control cost — full eval on all 10 queries costs ~$0.10
    sample = golden[:sample_size]
    print(f"[ragas] Running eval on {len(sample)} queries...")

    data   = build_ragas_dataset(sample, namespace=namespace)
    dataset = Dataset.from_dict(data)

    print("[ragas] Scoring with RAGAS metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=_llm,
        embeddings=_embeddings,
    )

    # RAGAS returns per-sample lists in newer versions — take the mean
    import numpy as np

    def _mean(val):
        if isinstance(val, list):
            valid = [v for v in val if v is not None]
            return round(float(np.mean(valid)), 3) if valid else 0.0
        return round(float(val), 3)

    scores = {
        "faithfulness":     _mean(results["faithfulness"]),
        "answer_relevancy": _mean(results["answer_relevancy"]),
        "context_recall":   _mean(results["context_recall"]),
    }

    print(f"\n{'='*50}")
    print(f"RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Faithfulness     : {scores['faithfulness']}")
    print(f"Answer relevancy : {scores['answer_relevancy']}")
    print(f"Context recall   : {scores['context_recall']}")
    print(f"{'='*50}")

    return scores


if __name__ == "__main__":
    run_ragas_eval()

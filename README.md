# PolicyLens — Regulatory Intelligence RAG System

A production-grade Retrieval-Augmented Generation (RAG) system that lets compliance teams query regulatory documents and get grounded, citation-backed answers.

Built for fintechs, insurers, and legal teams that need to search across hundreds of pages of regulatory PDFs without reading every document manually.

---

## What it does

A compliance officer asks: *"Does our loan product violate CBK's 2024 Consumer Credit Circular?"*

PolicyLens:
1. Searches a vector index of ingested regulatory documents by meaning, not keywords
2. Retrieves the most relevant clauses with their source section and page number
3. Generates a structured answer with citations traceable to the exact paragraph
4. Validates every claim against the source using NLI faithfulness checking
5. Flags low-confidence answers and identifies gaps in the knowledge base

---

## Architecture
```
PDF / DOCX / Web → Ingestion Pipeline → Pinecone + Elasticsearch
                                                ↓
User Query → HyDE Expansion → Hybrid Retrieval (Dense + BM25)
          → RRF Fusion → Cross-encoder Re-rank → MMR Dedup
          → Context Assembly → GPT-4o → Pydantic Validation
          → Citation Validation → NLI Faithfulness Check
          → Cited JSON Answer
```

---

## Sample output
```json
{
  "answer": "The euro-currency market grew independently due to the absence of reserve requirements giving Euro-banking a competitive edge [S1]. The credit-creating capacity was further boosted by large-scale central-bank depositing of reserves [S3].",
  "confidence_score": 0.9,
  "citations": [
    {
      "source_id": "S1",
      "doc_title": "CBK Test Document",
      "heading": "II. The hypothesis of independent growth",
      "page": 12,
      "relevance": "Explains the competitive edge of Euro-banking"
    }
  ],
  "gaps": [],
  "jurisdiction": "Kenya",
  "requires_legal_review": false
}
```

---

## Retrieval performance

Evaluated on a 10-query golden dataset:

| Metric | Score |
|--------|-------|
| Hit Rate @ 5 | 0.90 |
| MRR | 0.82 |

---

## Stack

- **Python 3.11**
- **Docling** — PDF parsing with table and heading structure preservation
- **LangChain** — RAG chain orchestration
- **OpenAI** — `text-embedding-3-large` for embeddings, `GPT-4o` for generation
- **Pinecone** — serverless vector database with namespace-scoped retrieval
- **Elasticsearch** — BM25 keyword index for hybrid retrieval
- **sentence-transformers** — cross-encoder re-ranking + NLI faithfulness checking
- **Pydantic** — structured output validation with retry
- **FastAPI** — REST API layer
- **RAGAS + LangSmith** — evaluation and observability

---

## Project structure
```
policylens/
├── ingestion/
│   ├── pipeline.py          # Orchestrates parse → chunk → embed → upsert
│   ├── chunker.py           # 512-token recursive chunker with heading prefix
│   └── parsers/
│       └── pdf_parser.py    # Docling-based PDF extraction
├── retrieval/
│   ├── embedder.py          # OpenAI batch embedder
│   ├── pinecone_store.py    # Upsert, namespace queries, ANN search
│   ├── bm25_store.py        # Elasticsearch BM25 index
│   ├── hybrid_retriever.py  # RRF fusion + optional HyDE expansion
│   ├── reranker.py          # Cross-encoder re-ranking
│   ├── assembler.py         # MMR dedup + context assembly
│   └── hyde.py              # HyDE query expansion
├── generation/
│   ├── prompts.py           # System, RAG, and faithfulness prompts
│   ├── chain.py             # Full RAG chain with retry logic
│   ├── output_parser.py     # Pydantic structured output validation
│   ├── citation_builder.py  # Citation formatting and hallucination detection
│   └── faithfulness.py      # NLI faithfulness checker
├── api/
│   └── main.py              # FastAPI endpoints
├── evaluation/
│   ├── golden_dataset.json  # Labeled queries for retrieval eval
│   └── retrieval_eval.py    # Hit Rate + MRR evaluation runner
└── dashboard/
    └── app.py               # Streamlit confidence + citation dashboard
```

---

## Quick start
```bash
git clone https://github.com/sogodongo/policylens
cd policylens
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
docker compose up -d
```

---

## Ingesting a document
```python
from ingestion.pipeline import ingest_document

ingest_document(
    pdf_path="data/cbk_circular_2024.pdf",
    doc_title="CBK Consumer Credit Circular No. 3 2024",
    doc_type="circular",
    jurisdiction="Kenya",
    namespace="cbk",
    source_url="https://centralbank.go.ke/..."
)
```

---

## Querying
```python
from generation.chain import run_rag_query

output = run_rag_query(
    query="Does our loan product require APR disclosure?",
    namespace="cbk",
    top_k=5,
    run_faithfulness=True,
)

print(output["response"].answer)
print(output["citation_block"])
print(output["faithfulness"])
```

---

## Running evaluations
```bash
# Retrieval evaluation
python3 evaluation/retrieval_eval.py

# Integration test
python3 tests/test_integration.py
```

---

## Status

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Ingestion pipeline | Done |
| Week 2 | Hybrid retrieval | Done |
| Week 3 | Generation + citations | Done |
| Week 4 | API + evaluation | In progress |

---

## Engineering decisions

- **512-token chunks with 15% overlap** — preserves clause integrity in dense regulatory text
- **Heading prefix injection** — every chunk carries its section context
- **HyDE query expansion** — embeds hypothetical answer documents for better recall on short queries
- **RRF fusion** — combines dense and sparse rankings by position, not score
- **Cross-encoder re-ranking** — reads query + chunk together for precision scoring
- **MMR deduplication** — removes near-duplicate chunks before context assembly
- **Pydantic output validation with retry** — enforces output contract, handles malformed LLM responses
- **Per-chunk NLI faithfulness** — checks each answer sentence against each context chunk individually
- **Citation hallucination detection** — validates every SOURCE_ID against the actual retrieved context
- **Document registry with MD5 hashing** — prevents re-ingestion of unchanged documents

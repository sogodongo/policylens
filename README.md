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
4. Flags low-confidence answers and identifies gaps in the knowledge base

---

## Architecture
```
PDF / DOCX / Web → Ingestion Pipeline → Pinecone Vector Index
                                                ↓
User Query → HyDE Expansion → Hybrid Retrieval (Dense + BM25)
          → RRF Fusion → Cross-encoder Re-rank → MMR Dedup
          → Context Assembly → GPT-4o → Cited JSON Answer
```

---

## Retrieval performance

Evaluated on a 10-query golden dataset against a real regulatory document:

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
- **sentence-transformers** — cross-encoder re-ranking
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
│   ├── hybrid_retriever.py  # RRF fusion over dense + sparse results
│   ├── reranker.py          # Cross-encoder re-ranking
│   ├── assembler.py         # MMR dedup + context assembly
│   └── hyde.py              # HyDE query expansion
├── generation/
│   ├── prompts.py           # System, HyDE, and faithfulness prompts
│   ├── chain.py             # LangChain RAG chain
│   └── output_parser.py     # Pydantic structured output + citation map
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
docker compose up -d  # starts Elasticsearch
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

## Running retrieval evaluation
```bash
python3 evaluation/retrieval_eval.py
```

---

## Status

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Ingestion pipeline | Done |
| Week 2 | Hybrid retrieval | Done |
| Week 3 | Generation + citations | In progress |
| Week 4 | API + evaluation | Upcoming |

---

## Engineering decisions

- **512-token chunks with 15% overlap** — preserves clause integrity in dense regulatory text
- **Heading prefix injection** — every chunk carries its section context
- **HyDE query expansion** — embeds hypothetical answer documents instead of raw queries, improving recall on short regulatory queries
- **RRF fusion** — combines dense and sparse rankings by position, not score, making it scale-agnostic
- **Cross-encoder re-ranking** — reads query + chunk together for precision scoring after initial retrieval
- **MMR deduplication** — removes near-duplicate chunks before context assembly to avoid wasting token budget
- **Document registry with MD5 hashing** — prevents re-ingestion of unchanged documents

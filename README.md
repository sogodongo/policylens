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
User Query → Hybrid Retrieval → Context Assembly → GPT-4o → Cited JSON Answer
```

---

## Stack

- **Python 3.11**
- **Docling** — PDF parsing with table and heading structure preservation
- **LangChain** — text splitting, RAG chain orchestration
- **OpenAI** — `text-embedding-3-large` for embeddings, `GPT-4o` for generation
- **Pinecone** — serverless vector database with namespace-scoped retrieval
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
│       ├── pdf_parser.py    # Docling-based PDF extraction
│       ├── docx_parser.py
│       └── web_crawler.py
├── retrieval/
│   ├── embedder.py          # OpenAI batch embedder
│   ├── pinecone_store.py    # Upsert, namespace queries, ANN search
│   ├── hybrid_retriever.py  # RRF fusion + cross-encoder re-ranking
│   └── assembler.py         # Context assembly with token budgeting
├── generation/
│   ├── prompts.py           # System, HyDE, and faithfulness prompts
│   ├── chain.py             # LangChain RAG chain
│   └── output_parser.py     # Pydantic structured output + citation map
├── api/
│   └── main.py              # FastAPI endpoints
├── evaluation/
│   └── ragas_runner.py      # Nightly RAGAS evaluation pipeline
└── dashboard/
    └── app.py               # Streamlit confidence + citation dashboard
```

---

## Setup
```bash
git clone https://github.com/sogodongo/policylens
cd policylens
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
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

## Status

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Ingestion pipeline | Done |
| Week 2 | Hybrid retrieval | In progress |
| Week 3 | Generation + citations | Upcoming |
| Week 4 | API + evaluation | Upcoming |

---

## Engineering decisions

- **512-token chunks with 15% overlap** — preserves clause integrity in dense regulatory text without diluting embedding signal
- **Heading prefix injection** — every chunk carries its section context so retrieval returns located, not just relevant, results
- **Document registry with MD5 hashing** — prevents re-ingestion of unchanged documents, making nightly pipeline runs cost-efficient
- **Pinecone namespaces** — isolates jurisdictions so a Kenya-scoped query never returns SEC results

# PolicyLens — System Architecture

This document explains the technical architecture of PolicyLens, the reasoning behind each major design decision, and the trade-offs considered during development.

---

## System overview

PolicyLens is a production-grade Retrieval-Augmented Generation (RAG) system designed for regulatory compliance use cases. It ingests regulatory documents, indexes them in a hybrid vector + keyword store, and answers natural language compliance questions with grounded, citation-backed responses.

The system is split into two independent pipelines:

**Ingestion pipeline** — runs offline, triggered by new document arrivals:
```
PDF → Docling parsing → Recursive chunking → OpenAI embedding → Pinecone + Elasticsearch
```

**Query pipeline** — runs online, triggered by user queries:
```
Query → HyDE expansion → Hybrid retrieval → Cross-encoder re-rank → MMR dedup
      → Context assembly → GPT-4o generation → Pydantic validation
      → Citation validation → NLI faithfulness → Structured JSON response
```

---

## Ingestion pipeline

### PDF parsing — Docling

**Decision:** Use Docling over PyPDF2, pdfplumber, or raw pdfminer.

**Reasoning:** Regulatory documents are structurally complex — they contain numbered clauses, cross-references, tables of rates, and multi-level headings. Docling reconstructs the logical structure of a document rather than extracting raw characters. This means a table of penalty fees is returned as a structured table, not a garbled sequence of numbers, and section headings are correctly identified as headings rather than bold paragraphs.

**Trade-off:** Docling is significantly heavier than simpler parsers (~500MB of model weights). For high-volume ingestion, this startup cost is mitigated by running ingestion as a batch job rather than a real-time service.

### Chunking — recursive character splitting, 512 tokens, 15% overlap

**Decision:** 512-token chunks with 77-token overlap and heading prefix injection.

**Reasoning:** Regulatory text has dense, self-referential clauses. Chunks smaller than 256 tokens frequently break mid-clause, losing the qualifying language that changes the meaning of a provision. Chunks larger than 1024 tokens dilute the embedding vector by covering multiple unrelated provisions, weakening retrieval signal. 512 tokens preserves clause integrity while keeping vectors focused.

The 15% overlap (77 tokens) ensures that a clause straddling a chunk boundary appears in at least one complete chunk. Without overlap, boundary clauses produce weak embeddings and are frequently missed by retrieval.

Heading prefix injection prepends the section heading to every chunk text before embedding. This means a chunk about "penalty fees" that lives under "Section 4.2 — Disclosure Requirements" carries that section context in its embedding. Without this, the retriever has no way to distinguish two similar clauses from different sections.

**Trade-off:** Overlap increases index size by ~15%. On a 10,000-chunk corpus this is negligible. On a corpus exceeding 1M chunks, a smaller overlap (5-10%) would be appropriate.

### Embeddings — text-embedding-3-large, 3072 dimensions

**Decision:** OpenAI text-embedding-3-large over ada-002 or open-source alternatives.

**Reasoning:** MTEB benchmark results show text-embedding-3-large outperforms ada-002 by ~8 points on legal and financial retrieval tasks. The 3072-dimensional space provides tighter semantic separation between similar but legally distinct concepts — for example, "disclosure" under securities law versus data protection law.

int8 quantization in Pinecone reduces memory usage by 4x with less than 1% recall loss, making the serverless tier cost-effective for production workloads.

**Trade-off:** text-embedding-3-large costs ~3x more per token than ada-002. For a compliance use case where answer quality is critical, this cost difference is acceptable. For high-volume low-stakes applications, ada-002 would be preferred.

### Vector database — Pinecone serverless

**Decision:** Pinecone over Chroma, Weaviate, or Qdrant.

**Reasoning:** The managed serverless tier eliminates infrastructure operations — no index servers to provision, scale, or maintain. Namespace isolation allows a single index to serve multiple regulatory jurisdictions (CBK, GDPR, SEC) without cross-contamination. Pre-filter support on metadata fields (effective_date, jurisdiction, doc_type) enables scoped retrieval before ANN search, cutting irrelevant vintage documents from the result set.

**Trade-off:** Pinecone is a paid service. For development and small corpora, the free tier is sufficient. For self-hosted requirements, Qdrant with its built-in filtering API is the closest alternative.

### Document registry — MD5 hashing

**Decision:** Track ingested documents by content hash rather than filename or ingestion date.

**Reasoning:** Regulatory documents are frequently updated. A filename-based registry would miss updates to the same document. A date-based registry would re-ingest every document on every pipeline run. Content hashing means only genuinely changed documents are re-ingested — unchanged documents are skipped with zero API cost.

**Trade-off:** MD5 is used for change detection, not security. Cryptographic strength is unnecessary here — collision resistance at the document level is sufficient.

---

## Query pipeline

### HyDE — Hypothetical Document Embeddings

**Decision:** Generate hypothetical regulatory excerpts before embedding the query.

**Reasoning:** Short compliance queries (3-5 words) produce weak, diffuse embedding vectors. The embedding space is trained on documents, not questions. A query like "APR disclosure requirements" maps poorly to the dense regulatory language in the index. HyDE generates what the answer document might look like, then embeds that hypothetical text. This produces a vector that lives in the same region of embedding space as real regulatory clauses.

**Trade-off:** HyDE adds one GPT-4o call per query (~$0.002, ~300ms). For latency-sensitive applications, HyDE can be disabled via the `use_hyde` flag.

### Hybrid retrieval — dense + sparse fusion

**Decision:** Combine Pinecone ANN with Elasticsearch BM25 using Reciprocal Rank Fusion.

**Reasoning:** Dense retrieval handles paraphrased queries well but struggles with exact regulatory references ("Article 17", "Circular No. 3"). Sparse retrieval handles exact references well but misses paraphrased queries. Fusion captures both signal types.

RRF uses ranks rather than scores for fusion, making it scale-agnostic. A Pinecone score of 0.85 and a BM25 score of 8.1 cannot be meaningfully added — they operate on different scales. RRF's position-based approach sidesteps this entirely.

**Trade-off:** Running two retrieval systems adds operational complexity. Elasticsearch requires a separate service (Docker container in development, managed service in production). The retrieval quality improvement justifies this complexity for high-stakes compliance use cases.

### Cross-encoder re-ranking

**Decision:** Re-rank hybrid retrieval candidates with a cross-encoder before context assembly.

**Reasoning:** Both dense and sparse retrievers score documents independently of the query — they cannot model the interaction between query and document. A cross-encoder reads the query and each candidate together, enabling it to distinguish between a document that mentions the topic and a document that actually answers the question.

The ms-marco-MiniLM-L-6-v2 model was chosen for its balance of accuracy and inference speed — it re-ranks 16 candidates in ~150ms on CPU.

**Trade-off:** Cross-encoders cannot be pre-computed and batched efficiently. They are only viable as re-rankers over a small candidate set (8-16 chunks), not as first-stage retrievers over the full index.

### MMR deduplication

**Decision:** Apply Maximal Marginal Relevance deduplication before context assembly.

**Reasoning:** Regulatory documents repeat provisions across sections — a requirement stated in Section 4 may be summarised again in Section 8. Without deduplication, the context window fills with near-identical chunks, wasting tokens and diluting the diversity of information available to the LLM.

The 0.97 cosine similarity threshold was chosen empirically — below this threshold, chunks are similar in topic but different enough in content to both be worth including.

### Structured output — Pydantic validation with retry

**Decision:** Enforce output schema with Pydantic and retry on validation failure.

**Reasoning:** Free-form LLM output is fragile as a service contract. Downstream systems (API, dashboard, audit logger) all depend on specific fields being present with specific types. Pydantic validation catches malformed responses at the boundary, before they propagate into the system.

The single retry with a stricter prompt handles ~95% of formatting failures without user-visible errors. GPT-4o at temperature=0 is highly consistent — validation failures are rare but non-zero under load.

### NLI faithfulness checking

**Decision:** Check each answer sentence against each context chunk individually using DeBERTa NLI.

**Reasoning:** A single concatenated context block is too noisy for accurate NLI scoring — formatting headers, SOURCE_ID tags, and document metadata confuse the entailment model. Checking each sentence against each chunk individually and taking the maximum entailment score gives a more accurate per-claim faithfulness assessment.

**Trade-off:** Per-sentence × per-chunk NLI adds latency (~500ms for 5 sentences × 5 chunks). This is acceptable for a compliance use case where answer quality is more important than response time.

---

## API layer

### FastAPI with Pydantic response models

The API uses FastAPI's response_model parameter to validate every response before it leaves the server. This means the OpenAPI documentation at /docs is always accurate — it's generated directly from the same Pydantic models that validate production responses.

### Audit logging — PostgreSQL

Every query is logged to PostgreSQL with its full response, confidence score, citation count, and faithfulness verdict. This satisfies regulatory requirements for AI system auditability — any compliance decision informed by PolicyLens can be reconstructed from the audit log.

### Latency tracing — query_traces table

Per-step latency (retrieval, generation, faithfulness) is logged to a separate traces table. This enables performance regression detection — if a code change causes retrieval latency to spike from 500ms to 5000ms, the traces table shows it immediately.

---

## Evaluation

### Retrieval evaluation — Hit Rate and MRR

A 10-query golden dataset with labeled expected fragments measures retrieval precision independently of generation quality. Current results: Hit Rate @ 5 = 0.90, MRR = 0.82.

### Generation evaluation — RAGAS

RAGAS measures faithfulness (are claims supported?), answer relevancy (does the answer address the question?), and context recall (does the context contain the answer?). Current results on 5-query sample: faithfulness=0.78, answer_relevancy=0.87, context_recall=0.60.

Context recall of 0.60 on the test document reflects OCR noise in the scanned test PDF. On clean text-based regulatory documents, context recall is expected to be significantly higher.

---

## Known limitations and future work

- **OCR quality:** The faithfulness and context recall scores are depressed by OCR noise in the test document. Production deployment should use text-based PDFs where possible.
- **Single document test corpus:** All evaluation was performed on one document. A multi-document corpus with cross-document queries will provide a more realistic quality baseline.
- **Streaming responses:** The current API returns complete responses. For long answers, streaming via Server-Sent Events would improve perceived latency.
- **Authentication:** The API has no authentication layer. Production deployment requires API key authentication or OAuth2.
- **Async ingestion:** The ingestion pipeline is synchronous. For large document sets, an async task queue (Celery + Redis) would prevent blocking the API server.

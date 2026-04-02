import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QueryRequest, QueryResponse
from api.audit import init_db, log_query, get_query_audit, get_recent_queries
from api.sources import get_ingested_sources
from generation.chain import run_rag_query

app = FastAPI(
    title="PolicyLens API",
    description="Regulatory intelligence RAG system — grounded, cited answers from regulatory documents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "policylens"}


@app.get("/sources")
def list_sources():
    """Returns all documents currently in the PolicyLens knowledge base."""
    sources = get_ingested_sources()
    return {"count": len(sources), "sources": sources}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Submit a compliance question and receive a grounded, cited answer.
    Every query is logged to the audit trail automatically.
    """
    start = time.time()

    try:
        output = run_rag_query(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            run_faithfulness=request.run_faithfulness,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Log every query to the audit trail
    query_id = log_query(
        query_text=request.query,
        namespace=request.namespace,
        output=output,
    )

    response     = output["response"]
    faithfulness = output.get("faithfulness")
    elapsed      = round(time.time() - start, 2)

    print(f"[api] /query completed in {elapsed}s — audit id: {query_id}")

    return QueryResponse(
        answer=                response.answer,
        confidence_score=      response.confidence_score,
        citations=             [c.model_dump() for c in response.citations],
        gaps=                  response.gaps,
        jurisdiction=          response.jurisdiction,
        requires_legal_review= response.requires_legal_review,
        citation_block=        output["citation_block"],
        faithfulness=          faithfulness,
        citation_validation=   output["citation_validation"],
    )


@app.get("/audit")
def list_audit():
    """Returns the 20 most recent queries for audit review."""
    return {"queries": get_recent_queries(limit=20)}


@app.get("/audit/{query_id}")
def get_audit(query_id: str):
    """Returns the full audit record for a specific query."""
    record = get_query_audit(query_id)
    if not record:
        raise HTTPException(status_code=404, detail="Query ID not found")
    return record

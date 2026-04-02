import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QueryRequest, QueryResponse
from generation.chain import run_rag_query

app = FastAPI(
    title="PolicyLens API",
    description="Regulatory intelligence RAG system — grounded, cited answers from regulatory documents.",
    version="0.1.0",
)

# Allow browser clients to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "policylens"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Submit a compliance question and receive a grounded, cited answer.

    The response includes:
    - answer with inline SOURCE_ID citations
    - confidence score (0.0 = no relevant context, 1.0 = fully supported)
    - full citation list with document metadata
    - faithfulness score from NLI verification
    - gaps listing any sub-questions the context couldn't answer
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
        # Don't leak internal stack traces to API clients
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    response    = output["response"]
    faithfulness = output.get("faithfulness")

    elapsed = round(time.time() - start, 2)
    print(f"[api] /query completed in {elapsed}s")

    return QueryResponse(
        answer=               response.answer,
        confidence_score=     response.confidence_score,
        citations=            [c.model_dump() for c in response.citations],
        gaps=                 response.gaps,
        jurisdiction=         response.jurisdiction,
        requires_legal_review=response.requires_legal_review,
        citation_block=       output["citation_block"],
        faithfulness=         faithfulness,
        citation_validation=  output["citation_validation"],
    )

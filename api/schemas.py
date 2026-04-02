from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query:            str   = Field(..., min_length=5, max_length=1000)
    namespace:        str   = Field(default="default")
    top_k:            int   = Field(default=5, ge=1, le=20)
    use_hyde:         bool  = Field(default=False)
    run_faithfulness: bool  = Field(default=True)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query":     "Does our loan product require APR disclosure?",
                "namespace": "cbk",
                "top_k":     5,
                "use_hyde":  False,
                "run_faithfulness": True,
            }
        }
    }


class CitationOut(BaseModel):
    source_id:    str
    doc_title:    str
    heading:      str
    page:         int
    source_url:   str
    jurisdiction: str
    relevance:    str


class FaithfulnessOut(BaseModel):
    faithfulness_score: float
    verdict:            str
    unsupported_claims: list[dict]


class QueryResponse(BaseModel):
    answer:                str
    confidence_score:      float
    citations:             list[CitationOut]
    gaps:                  list[str]
    jurisdiction:          str
    requires_legal_review: bool
    citation_block:        str
    faithfulness:          FaithfulnessOut | None
    citation_validation:   list[dict]

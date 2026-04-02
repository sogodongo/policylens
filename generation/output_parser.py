import json
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    source_id:    str
    doc_title:    str = ""
    heading:      str = ""
    page:         int = 0
    source_url:   str = ""
    jurisdiction: str = ""
    relevance:    str = ""

    @field_validator("source_id")
    @classmethod
    def source_id_must_be_valid(cls, v):
        if not v.startswith("S"):
            raise ValueError(f"source_id must start with 'S', got: {v}")
        return v


class PolicyLensResponse(BaseModel):
    answer:               str
    confidence_score:     float = Field(ge=0.0, le=1.0)
    citations:            list[Citation] = []
    gaps:                 list[str] = []
    jurisdiction:         str = "Unknown"
    requires_legal_review: bool = False

    @field_validator("confidence_score", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        # GPT-4o sometimes returns confidence as a string
        return float(v)

    @field_validator("citations", mode="before")
    @classmethod
    def ensure_citations_list(cls, v):
        if v is None:
            return []
        return v


def parse_llm_output(raw: str) -> PolicyLensResponse:
    """
    Parses and validates the raw LLM output string into a PolicyLensResponse.

    Strips markdown code fences if present, then validates against the
    Pydantic schema. Raises ValueError with a clear message on failure
    so the chain can decide whether to retry.
    """
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        # parts[1] is the content between the first pair of fences
        cleaned = parts[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}\nRaw: {raw[:200]}")

    return PolicyLensResponse(**data)

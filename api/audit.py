import os
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://policylens:policylens@localhost:5432/policylens"
)

_engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db():
    """Creates the audit table if it doesn't exist."""
    with _engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_audit (
                query_id          TEXT PRIMARY KEY,
                query_text        TEXT NOT NULL,
                namespace         TEXT NOT NULL,
                answer            TEXT,
                confidence_score  FLOAT,
                citation_count    INTEGER,
                faithfulness_score FLOAT,
                faithfulness_verdict TEXT,
                requires_legal_review BOOLEAN,
                gaps              TEXT,
                full_response     TEXT,
                created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        conn.commit()
    print("[audit] Database initialized.")


def log_query(
    query_text: str,
    namespace: str,
    output: dict,
) -> str:
    """
    Writes a query and its full response to the audit log.
    Returns the query_id for retrieval later.
    """
    query_id = str(uuid.uuid4())
    response = output["response"]
    faithfulness = output.get("faithfulness") or {}

    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO query_audit (
                query_id, query_text, namespace, answer,
                confidence_score, citation_count,
                faithfulness_score, faithfulness_verdict,
                requires_legal_review, gaps, full_response
            ) VALUES (
                :query_id, :query_text, :namespace, :answer,
                :confidence_score, :citation_count,
                :faithfulness_score, :faithfulness_verdict,
                :requires_legal_review, :gaps, :full_response
            )
        """), {
            "query_id":             query_id,
            "query_text":           query_text,
            "namespace":            namespace,
            "answer":               response.answer,
            "confidence_score":     response.confidence_score,
            "citation_count":       len(response.citations),
            "faithfulness_score":   faithfulness.get("faithfulness_score"),
            "faithfulness_verdict": faithfulness.get("verdict"),
            "requires_legal_review": response.requires_legal_review,
            "gaps":                 json.dumps(response.gaps),
            "full_response":        json.dumps(output["citation_validation"]),
        })
        conn.commit()

    return query_id


def get_query_audit(query_id: str) -> dict | None:
    """Retrieves a single query audit record by ID."""
    with _engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM query_audit WHERE query_id = :qid"),
            {"qid": query_id}
        ).fetchone()

    if not row:
        return None

    return dict(row._mapping)


def get_recent_queries(limit: int = 20) -> list[dict]:
    """Returns the most recent queries for the audit dashboard."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT query_id, query_text, namespace, confidence_score,
                   faithfulness_verdict, requires_legal_review, created_at
            FROM query_audit
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()

    return [dict(r._mapping) for r in rows]

import time
from api.audit import _engine
from sqlalchemy import text


def init_traces_table():
    with _engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_traces (
                trace_id         TEXT PRIMARY KEY,
                query_text       TEXT,
                namespace        TEXT,
                retrieval_ms     INTEGER,
                generation_ms    INTEGER,
                faithfulness_ms  INTEGER,
                total_ms         INTEGER,
                chunks_retrieved INTEGER,
                context_tokens   INTEGER,
                confidence       FLOAT,
                created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        conn.commit()


def log_trace(trace_id: str, query_text: str, namespace: str, timings: dict, metrics: dict):
    with _engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO query_traces (
                trace_id, query_text, namespace,
                retrieval_ms, generation_ms, faithfulness_ms, total_ms,
                chunks_retrieved, context_tokens, confidence
            ) VALUES (
                :trace_id, :query_text, :namespace,
                :retrieval_ms, :generation_ms, :faithfulness_ms, :total_ms,
                :chunks_retrieved, :context_tokens, :confidence
            )
        """), {
            "trace_id":        trace_id,
            "query_text":      query_text,
            "namespace":       namespace,
            "retrieval_ms":    timings.get("retrieval_ms", 0),
            "generation_ms":   timings.get("generation_ms", 0),
            "faithfulness_ms": timings.get("faithfulness_ms", 0),
            "total_ms":        timings.get("total_ms", 0),
            "chunks_retrieved": metrics.get("chunks_retrieved", 0),
            "context_tokens":  metrics.get("context_tokens", 0),
            "confidence":      metrics.get("confidence", 0.0),
        })
        conn.commit()

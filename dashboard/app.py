import sys
sys.path.insert(0, ".")

import streamlit as st
from generation.chain import run_rag_query
from api.sources import get_ingested_sources

st.set_page_config(
    page_title="PolicyLens",
    page_icon="",
    layout="wide",
)

st.title("PolicyLens")
st.caption("Regulatory Intelligence — grounded, cited answers from your document knowledge base")

# Sidebar — knowledge base status
with st.sidebar:
    st.header("Knowledge base")
    sources = get_ingested_sources()
    if sources:
        for s in sources:
            st.markdown(f"**{s['doc_title']}**")
            st.caption(f"Namespace: {s['namespace']} | Hash: {s['file_hash']}")
    else:
        st.warning("No documents ingested yet.")

    st.divider()
    st.header("Query settings")
    namespace  = st.selectbox("Jurisdiction namespace", ["cbk", "gdpr", "sec", "default"])
    top_k      = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    use_hyde   = st.toggle("HyDE query expansion", value=False)
    run_faith  = st.toggle("Faithfulness check", value=True)

# Main query interface
query = st.text_area(
    "Compliance question",
    placeholder="e.g. Does our loan product require APR disclosure under CBK regulations?",
    height=100,
)

search_clicked = st.button("Search", type="primary")

if search_clicked and not query.strip():
    st.warning("Please enter a question.")

if search_clicked and query.strip():
    with st.spinner("Searching knowledge base..."):
        output = run_rag_query(
            query=query,
            namespace=namespace,
            top_k=top_k,
            use_hyde=use_hyde,
            run_faithfulness=run_faith,
        )

    response     = output["response"]
    faithfulness = output.get("faithfulness")
    validation   = output["citation_validation"]

    # Confidence indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        score = response.confidence_score
        color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
        st.metric("Confidence", f"{score:.0%}")

    with col2:
        if faithfulness:
            fcolor = "green" if faithfulness["verdict"] == "PASS" else "red"
            st.metric("Faithfulness", faithfulness["faithfulness_score"])
        else:
            st.metric("Faithfulness", "Not checked")

    with col3:
        hallucinated = [v for v in validation if not v["valid"]]
        st.metric(
            "Citation integrity",
            "All valid" if not hallucinated else f"{len(hallucinated)} flagged"
        )

    st.divider()

    # Answer
    st.subheader("Answer")
    if response.requires_legal_review:
        st.warning("Legal review recommended before acting on this answer.")
    st.markdown(response.answer)

    # Gaps
    if response.gaps:
        with st.expander("Unanswered gaps"):
            for g in response.gaps:
                st.markdown(f"- {g}")

    st.divider()

    # Citations
    st.subheader("Sources")
    if response.citations:
        for c in response.citations:
            with st.expander(f"[{c.source_id}] {c.doc_title} — {c.heading[:60]}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Document:** {c.doc_title}")
                    st.markdown(f"**Section:** {c.heading}")
                    st.markdown(f"**Page:** {c.page if c.page > 0 else 'N/A'}")
                with col_b:
                    st.markdown(f"**Jurisdiction:** {c.jurisdiction}")
                    if c.source_url:
                        st.markdown(f"**Source:** [{c.source_url}]({c.source_url})")
                st.markdown(f"**Relevance:** {c.relevance}")
    else:
        st.info("No citations — answer may be based on insufficient context.")

    # Faithfulness detail
    if faithfulness and faithfulness.get("unsupported_claims"):
        st.divider()
        st.subheader("Faithfulness warnings")
        for u in faithfulness["unsupported_claims"]:
            st.warning(f"Low support (score={u['entailment_score']}): {u['claim'][:150]}")

    # Raw citation block for audit copy-paste
    with st.expander("Full citation block (for audit reports)"):
        st.code(output["citation_block"], language=None)

    st.caption(f"Trace ID: {output.get('trace_id', 'N/A')}")



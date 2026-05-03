import inspect
import pathlib
import time

import streamlit as st
from dotenv import load_dotenv

from db.connection import db_conn
from rag.generation import generate_answer, suggest_questions
from rag.ingest import chunk_text, embed_chunks, parse_document, store_document
from rag.retrieval import hybrid_search, reciprocal_rank_fusion

load_dotenv()

st.set_page_config(page_title="DocSage", page_icon="📚", layout="wide")

with st.sidebar:
    st.header("📚 DocSage")
    st.markdown("""
**Stack:**
- 🧠 `all-MiniLM-L6-v2` — 384-dim local embeddings
- 🐘 Neon PostgreSQL + pgvector + tsvector
- 🔀 Hybrid BM25 + cosine → RRF
- 🤖 Claude `claude-haiku-4-5-20251001`
""")
    st.divider()
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                doc_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cur.fetchone()[0]
        st.metric("Documents", doc_count)
        st.metric("Chunks indexed", chunk_count)
        st.success("Neon connected ✓")
    except Exception as e:
        st.error(f"DB error: {e}")

st.title("📚 DocSage")
st.caption("Hybrid RAG pipeline demo — pgvector cosine + PostgreSQL BM25 + Reciprocal Rank Fusion")

tab_ingest, tab_query = st.tabs(["📥 Ingest", "🔍 Query"])

_HERE = pathlib.Path(__file__).parent

SAMPLE_DOCS = {
    "rag_guide.md":            _HERE / "sample_docs" / "rag_guide.md",
    "system_design.txt":       _HERE / "sample_docs" / "system_design.txt",
    "distributed_systems.txt": _HERE / "sample_docs" / "distributed_systems.txt",
}


def run_ingest(file_bytes: bytes, filename: str):
    with st.spinner("Parsing and chunking..."):
        t0 = time.time()
        text = parse_document(file_bytes, filename)
        chunks = chunk_text(text)
        parse_time = time.time() - t0

    with st.spinner("Generating embeddings..."):
        t1 = time.time()
        embeddings = embed_chunks(chunks)
        embed_time = time.time() - t1

    with st.spinner("Storing in Neon..."):
        t2 = time.time()
        with db_conn() as conn:
            store_document(conn, filename, filename.rsplit(".", 1)[-1], chunks, embeddings)
        insert_time = time.time() - t2

    st.success(f"✅ Ingested **{filename}** — {len(chunks)} chunks in {parse_time + embed_time + insert_time:.2f}s")

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks", len(chunks))
    col2.metric("Embed time", f"{embed_time:.2f}s")
    col3.metric("DB insert", f"{insert_time:.2f}s")

    with st.spinner("Generating suggested questions..."):
        chunk_dicts = [{"content": c} for c in chunks]
        questions = suggest_questions(filename, chunk_dicts)
    if questions:
        st.markdown("**💡 Questions you can ask about this document:**")
        for q in questions:
            st.markdown(f"- {q}")

    with st.expander("⚙️ How ingestion works"):
        st.subheader("1. Chunking — fixed-size sliding window")
        st.markdown(
            "**512-char windows, 50-char overlap.** Overlap ensures a sentence split at a boundary "
            "still appears intact in the adjacent chunk, preventing retrieval misses on boundary-straddling content."
        )
        st.code(inspect.getsource(chunk_text), language="python")

        st.subheader("2. Embedding — all-MiniLM-L6-v2")
        st.markdown(
            f"**384-dimensional dense vectors**, ~90MB model, runs on CPU in <1s/batch. "
            f"No API key required. Each chunk becomes a point in ℝ³⁸⁴; "
            f"cosine similarity ≈ semantic relevance. This run: **{len(chunks)} chunks**, batch size **32**."
        )

        st.subheader("3. Storage — pgvector + tsvector")
        st.code("""
CREATE TABLE chunks (
  id          UUID PRIMARY KEY,
  doc_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
  content     TEXT,
  embedding   vector(384),          -- pgvector: ANN via ivfflat index
  chunk_index INT,
  ts_content  TSVECTOR GENERATED    -- auto-populated for BM25 search
    ALWAYS AS (to_tsvector('english', content)) STORED
);
-- Two indexes, one query planner, no separate vector DB
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING GIN (ts_content);
""", language="sql")


with tab_ingest:
    st.header("Knowledge Base")

    st.subheader("Upload a document")
    uploaded = st.file_uploader("PDF, TXT, MD, or DOCX", type=["pdf", "txt", "md", "docx"])
    if uploaded:
        try:
            run_ingest(uploaded.read(), uploaded.name)
        except Exception as e:
            st.error(f"Ingest failed: {e}")

    st.subheader("Or load a sample")
    cols = st.columns(3)
    for i, (name, path) in enumerate(SAMPLE_DOCS.items()):
        if cols[i].button(f"📄 {name}"):
            try:
                with open(path, "rb") as f:
                    run_ingest(f.read(), name)
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.subheader("Indexed documents")
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, file_type, chunk_count, uploaded_at FROM documents ORDER BY uploaded_at DESC"
                )
                rows = cur.fetchall()
        if rows:
            st.dataframe(
                [{"File": r[0], "Type": r[1], "Chunks": r[2], "Uploaded": r[3]} for r in rows],
                use_container_width=True,
            )
        else:
            st.info("No documents ingested yet. Upload a file or load a sample above.")
    except Exception as e:
        st.error(f"Could not fetch documents: {e}")

with tab_query:
    st.header("Ask a Question")
    st.caption(
        "Try: *'What is retrieval-augmented generation?'* · "
        "*'Explain the CAP theorem'* · "
        "*'How does Raft consensus work?'*"
    )

    question = st.text_input("Question", placeholder="e.g. What is retrieval-augmented generation?")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving relevant chunks..."):
            with db_conn() as conn:
                chunks, debug = hybrid_search(conn, question, top_k=5)

        if not chunks:
            st.warning("No relevant chunks found. Try ingesting some documents first.")
        else:
            with st.spinner("Generating answer with Claude..."):
                answer, meta = generate_answer(question, chunks)

            st.subheader("Answer")
            st.info(answer)

            col1, col2, col3 = st.columns(3)
            col1.metric("Input tokens", meta["input_tokens"])
            col2.metric("Output tokens", meta["output_tokens"])
            col3.metric("Latency", f"{meta['latency_ms']}ms")

            with st.expander("⚙️ How retrieval works"):
                st.subheader("1. Vector search — cosine similarity via ivfflat")
                st.markdown(
                    f"Embeds the query with the same `all-MiniLM-L6-v2` model, "
                    f"then finds the **{debug['vector_results_count']} nearest chunks** by cosine distance. "
                    f"ivfflat is an approximate index (O(√n)) — faster than exact, "
                    f"~5% recall loss at 100 lists, acceptable for demo scale."
                )
                st.code(debug["vector_search_sql"].strip(), language="sql")

                st.subheader("2. BM25 — PostgreSQL full-text search")
                st.markdown(
                    f"Runs a `tsvector`/`GIN` query against the same `chunks` table. "
                    f"Found **{debug['bm25_results_count']} matching chunks**. "
                    f"Catches cases vector similarity misses: acronyms (*'CAP'*), proper nouns, "
                    f"exact technical strings that aren't semantically close in embedding space."
                )
                st.code(debug["bm25_search_sql"].strip(), language="sql")

                st.subheader("3. Reciprocal Rank Fusion")
                st.markdown(r"""
Merges both result lists. For each chunk: **RRF score = Σ 1 / (60 + rankᵢ)** across every list it appears in.
k=60 is the empirically established constant from Cormack et al. (2009) — it dampens top-rank dominance
without discarding lower-ranked results. A chunk appearing in both lists scores ~2× a chunk in only one.
""")
                st.code(inspect.getsource(reciprocal_rank_fusion), language="python")

                st.subheader("4. Prompt sent to Claude")
                st.code(meta["prompt"], language="text")

            st.subheader("Retrieved chunks")
            st.dataframe(
                [
                    {
                        "Source": c["filename"],
                        "Chunk #": c["chunk_index"],
                        "Vector score": round(c.get("vector_score") or 0, 4),
                        "BM25 rank": c.get("bm25_rank") or "—",
                        "RRF score": round(c["rrf_score"], 6),
                        "Content (truncated)": c["content"][:120] + ("..." if len(c["content"]) > 120 else ""),
                    }
                    for c in chunks
                ],
                use_container_width=True,
            )

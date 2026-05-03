# DocSage RAG Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DocSage, a portfolio RAG demo app that ingests documents and answers questions using hybrid search (pgvector cosine + PostgreSQL BM25 + Reciprocal Rank Fusion), with Streamlit UI panels that expose the actual code and SQL at every step for a technical audience.

**Architecture:** Ingestion pipeline parses/chunks/embeds files into Neon PostgreSQL (pgvector + tsvector); queries combine cosine ANN and BM25 full-text search via RRF before sending top-5 chunks to Claude; each pipeline step exposes its actual source code and SQL in Streamlit expander panels.

**Tech Stack:** Python 3.11+, Streamlit, Anthropic SDK (`claude-haiku-4-5-20251001`), sentence-transformers (`all-MiniLM-L6-v2`), Neon PostgreSQL, psycopg2, pgvector, PyMuPDF, python-docx, pytest, pytest-mock

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt` | Python dependencies |
| `.env.example` | Env var template |
| `db/schema.sql` | PostgreSQL DDL — run once |
| `db/__init__.py` | Package marker |
| `db/connection.py` | Neon psycopg2 pool + `db_conn()` context manager |
| `rag/__init__.py` | Package marker |
| `rag/ingest.py` | `parse_document`, `chunk_text`, `embed_chunks`, `store_document` |
| `rag/retrieval.py` | `vector_search`, `bm25_search`, `reciprocal_rank_fusion`, `hybrid_search` |
| `rag/generation.py` | `build_prompt`, `generate_answer` |
| `sample_docs/rag_guide.md` | Sample: RAG concepts (~800 words, meta demo) |
| `sample_docs/system_design.txt` | Sample: CAP theorem, consistent hashing, bloom filters |
| `sample_docs/distributed_systems.txt` | Sample: Raft, CRDTs, LSM trees, write amplification |
| `tests/__init__.py` | Package marker |
| `tests/test_connection.py` | Integration test: DB connectivity |
| `tests/test_ingest.py` | Unit + integration tests for ingest pipeline |
| `tests/test_retrieval.py` | Unit tests for RRF + integration tests for search |
| `tests/test_generation.py` | Unit tests for prompt builder + mocked Claude |
| `app.py` | Streamlit entry point: sidebar + two tabs + technical explanation panels |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `db/__init__.py`
- Create: `rag/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.35
anthropic>=0.28
sentence-transformers>=3.0
psycopg2-binary>=2.9
pgvector>=0.3
PyMuPDF>=1.24
python-docx>=1.1
python-dotenv>=1.0
numpy>=1.26
pytest>=8.0
pytest-mock>=3.0
```

- [ ] **Step 2: Create .env.example**

```
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 3: Create empty package init files**

Create three empty files: `db/__init__.py`, `rag/__init__.py`, `tests/__init__.py`

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without errors. `sentence-transformers` pulls PyTorch — first install takes ~2 minutes.

- [ ] **Step 5: Copy .env.example and fill in credentials**

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL (from Neon dashboard) and ANTHROPIC_API_KEY
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example db/__init__.py rag/__init__.py tests/__init__.py
git commit -m "feat: project setup and dependencies"
```

---

## Task 2: Database Schema and Connection

**Files:**
- Create: `db/schema.sql`
- Create: `db/connection.py`
- Create: `tests/test_connection.py`

- [ ] **Step 1: Write db/schema.sql**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,
  file_type   TEXT NOT NULL,
  uploaded_at TIMESTAMPTZ DEFAULT NOW(),
  chunk_count INT
);

CREATE TABLE IF NOT EXISTS chunks (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,
  embedding   vector(384),
  chunk_index INT,
  metadata    JSONB DEFAULT '{}',
  ts_content  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_ts_idx
  ON chunks USING GIN (ts_content);
```

- [ ] **Step 2: Apply schema to Neon**

```bash
psql $DATABASE_URL -f db/schema.sql
```

Expected output:
```
CREATE EXTENSION
CREATE TABLE
CREATE TABLE
CREATE INDEX
CREATE INDEX
```

- [ ] **Step 3: Write failing test**

Create `tests/test_connection.py`:

```python
import os
import pytest
from db.connection import get_conn, return_conn

def test_connection_executes_query():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS val")
            row = cur.fetchone()
        assert row[0] == 1
    finally:
        return_conn(conn)
```

- [ ] **Step 4: Run test — verify it fails**

```bash
pytest tests/test_connection.py -v
```

Expected: `ModuleNotFoundError: No module named 'db.connection'`

- [ ] **Step 5: Write db/connection.py**

```python
import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 5, os.environ["DATABASE_URL"])
    return _pool


def get_conn():
    return _get_pool().getconn()


def return_conn(conn):
    _get_pool().putconn(conn)


@contextmanager
def db_conn():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)
```

- [ ] **Step 6: Run test — verify it passes**

```bash
pytest tests/test_connection.py -v
```

Expected: `PASSED`

- [ ] **Step 7: Commit**

```bash
git add db/schema.sql db/connection.py tests/test_connection.py
git commit -m "feat: db schema and connection pool"
```

---

## Task 3: Document Parsing and Chunking

**Files:**
- Create: `rag/ingest.py` (parse_document, chunk_text)
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest.py`:

```python
import pytest
from rag.ingest import parse_document, chunk_text


def test_chunk_text_single_chunk_under_limit():
    chunks = chunk_text("hello world", chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_splits_at_chunk_size():
    text = "a" * 600
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) == 2
    assert len(chunks[0]) == 512
    # second chunk starts at 512-50=462
    assert chunks[1] == text[462:]


def test_chunk_text_overlap_carries_content():
    text = "a" * 100 + "b" * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # chunk[1] starts at offset 80, so first 20 chars are still "a"
    assert chunks[1][:20] == "a" * 20


def test_parse_document_plain_text():
    result = parse_document(b"Hello, world!", "test.txt")
    assert result == "Hello, world!"


def test_parse_document_markdown():
    result = parse_document(b"# Title\n\nSome content here.", "test.md")
    assert "Title" in result
    assert "Some content here" in result
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: `ImportError: cannot import name 'parse_document' from 'rag.ingest'`

- [ ] **Step 3: Write rag/ingest.py with parse_document and chunk_text**

```python
import io
import uuid
import numpy as np
from typing import Optional

import fitz  # PyMuPDF
import docx
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def parse_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    else:  # txt, md
        return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list[str]) -> np.ndarray:
    return _get_model().encode(chunks, batch_size=32, show_progress_bar=False)


def store_document(
    conn, filename: str, file_type: str, chunks: list[str], embeddings: np.ndarray
) -> uuid.UUID:
    doc_id = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, filename, file_type, chunk_count) VALUES (%s, %s, %s, %s)",
            (str(doc_id), filename, file_type, len(chunks)),
        )
        rows = [
            (str(uuid.uuid4()), str(doc_id), chunk, embeddings[i].tolist(), i, "{}")
            for i, chunk in enumerate(chunks)
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (id, doc_id, content, embedding, chunk_index, metadata) VALUES %s",
            rows,
        )
    return doc_id
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: All 5 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingest.py
git commit -m "feat: document parsing and chunking"
```

---

## Task 4: Embeddings and Storage Tests

**Files:**
- Modify: `tests/test_ingest.py` (add embedding + storage tests)

- [ ] **Step 1: Add failing tests for embed_chunks and store_document**

Append to `tests/test_ingest.py`:

```python
import numpy as np
import os


def test_embed_chunks_returns_correct_shape():
    from rag.ingest import embed_chunks
    embeddings = embed_chunks(["hello world", "foo bar baz"])
    assert embeddings.shape == (2, 384)


def test_embed_chunks_single():
    from rag.ingest import embed_chunks
    embeddings = embed_chunks(["single sentence"])
    assert embeddings.shape == (1, 384)


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="requires DATABASE_URL")
def test_store_document_inserts_rows():
    from rag.ingest import chunk_text, embed_chunks, store_document
    from db.connection import db_conn

    text = "This is a test document about retrieval augmented generation."
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    embeddings = embed_chunks(chunks)

    with db_conn() as conn:
        doc_id = store_document(conn, "test_doc.txt", "txt", chunks, embeddings)

        with conn.cursor() as cur:
            cur.execute("SELECT chunk_count FROM documents WHERE id = %s", (str(doc_id),))
            assert cur.fetchone()[0] == len(chunks)

            cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (str(doc_id),))
            assert cur.fetchone()[0] == len(chunks)

            # cleanup
            cur.execute("DELETE FROM documents WHERE id = %s", (str(doc_id),))
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_ingest.py -v
```

Note: First run of `test_embed_chunks_*` downloads the ~90MB model. Subsequent runs are instant.
Expected: All tests `PASSED` (integration test skipped if no DATABASE_URL, or `PASSED` if connected)

- [ ] **Step 3: Commit**

```bash
git add tests/test_ingest.py
git commit -m "test: embedding and storage integration tests"
```

---

## Task 5: Hybrid Search (Vector + BM25 + RRF)

**Files:**
- Create: `rag/retrieval.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval.py`:

```python
import os
import pytest
from rag.retrieval import reciprocal_rank_fusion


def test_rrf_combines_results_from_both_lists():
    vector_results = [
        {"id": "a", "content": "alpha", "vector_score": 0.9},
        {"id": "b", "content": "beta", "vector_score": 0.8},
        {"id": "c", "content": "gamma", "vector_score": 0.7},
    ]
    bm25_results = [
        {"id": "b", "content": "beta", "bm25_score": 0.95},
        {"id": "a", "content": "alpha", "bm25_score": 0.85},
        {"id": "d", "content": "delta", "bm25_score": 0.75},
    ]
    results = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    ids = [r["id"] for r in results]
    assert "a" in ids
    assert "b" in ids
    assert len(results) == 4  # a, b, c, d


def test_rrf_deduplicates_same_chunk():
    result = {"id": "x", "content": "foo", "vector_score": 0.9}
    merged = reciprocal_rank_fusion([result], [result], k=60)
    assert len([r for r in merged if r["id"] == "x"]) == 1


def test_rrf_returns_sorted_by_score_descending():
    vector_results = [{"id": str(i), "content": f"c{i}", "vector_score": 0.9 - i * 0.1} for i in range(3)]
    bm25_results = [{"id": str(i), "content": f"c{i}", "bm25_score": 0.9 - i * 0.1} for i in range(3)]
    results = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_chunk_appearing_in_both_lists_scores_higher():
    shared = {"id": "shared", "content": "shared", "vector_score": 0.5}
    unique = {"id": "unique", "content": "unique", "vector_score": 0.99}
    results = reciprocal_rank_fusion([shared], [shared, unique], k=60)
    shared_score = next(r["rrf_score"] for r in results if r["id"] == "shared")
    unique_score = next(r["rrf_score"] for r in results if r["id"] == "unique")
    assert shared_score > unique_score


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="requires DATABASE_URL")
def test_hybrid_search_returns_results_with_debug():
    from rag.ingest import chunk_text, embed_chunks, store_document
    from rag.retrieval import hybrid_search
    from db.connection import db_conn

    text = "Retrieval augmented generation combines a retriever and a language model."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    embeddings = embed_chunks(chunks)

    with db_conn() as conn:
        doc_id = store_document(conn, "test_retrieval.txt", "txt", chunks, embeddings)
        results, debug = hybrid_search(conn, "retrieval augmented generation", top_k=3)

        assert len(results) > 0
        assert "rrf_score" in results[0]
        assert "content" in results[0]
        assert "vector_search_sql" in debug
        assert "bm25_search_sql" in debug

        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (str(doc_id),))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_retrieval.py -v -k "not integration and not DATABASE"
```

Expected: `ImportError: cannot import name 'reciprocal_rank_fusion'`

- [ ] **Step 3: Write rag/retrieval.py**

```python
import numpy as np
from pgvector.psycopg2 import register_vector

VECTOR_SEARCH_SQL = """
    SELECT c.id, c.content, c.chunk_index, d.filename,
           1 - (c.embedding <=> %s::vector) AS vector_score
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
"""

BM25_SEARCH_SQL = """
    SELECT c.id, c.content, c.chunk_index, d.filename,
           ts_rank(c.ts_content, plainto_tsquery('english', %s)) AS bm25_score
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    WHERE c.ts_content @@ plainto_tsquery('english', %s)
    ORDER BY bm25_score DESC
    LIMIT %s
"""


def vector_search(conn, query_embedding: np.ndarray, k: int = 20) -> list[dict]:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(VECTOR_SEARCH_SQL, (query_embedding.tolist(), query_embedding.tolist(), k))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def bm25_search(conn, query_text: str, k: int = 20) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(BM25_SEARCH_SQL, (query_text, query_text, k))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result["id"]
        if chunk_id not in scores:
            scores[chunk_id] = {**result, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None, "bm25_score": None}
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[chunk_id]["vector_rank"] = rank + 1
        scores[chunk_id]["vector_score"] = result.get("vector_score", 0.0)

    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        if chunk_id not in scores:
            scores[chunk_id] = {**result, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None, "vector_score": None}
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[chunk_id]["bm25_rank"] = rank + 1
        scores[chunk_id]["bm25_score"] = result.get("bm25_score", 0.0)

    return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(conn, query_text: str, top_k: int = 5) -> tuple[list[dict], dict]:
    from rag.ingest import embed_chunks
    query_embedding = embed_chunks([query_text])[0]

    v_results = vector_search(conn, query_embedding, k=20)
    b_results = bm25_search(conn, query_text, k=20)
    merged = reciprocal_rank_fusion(v_results, b_results)[:top_k]

    debug = {
        "vector_search_sql": VECTOR_SEARCH_SQL,
        "bm25_search_sql": BM25_SEARCH_SQL,
        "query_text": query_text,
        "vector_results_count": len(v_results),
        "bm25_results_count": len(b_results),
        "query_embedding_dim": len(query_embedding),
    }
    return merged, debug
```

- [ ] **Step 4: Run all retrieval tests**

```bash
pytest tests/test_retrieval.py -v
```

Expected: All unit tests `PASSED`; integration test `PASSED` or `SKIPPED`

- [ ] **Step 5: Commit**

```bash
git add rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: hybrid search with RRF"
```

---

## Task 6: Generation (Claude)

**Files:**
- Create: `rag/generation.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generation.py`:

```python
from rag.generation import build_prompt


def test_build_prompt_includes_question():
    chunks = [{"content": "RAG combines retrieval and generation.", "filename": "doc.txt", "chunk_index": 0}]
    prompt = build_prompt("What is RAG?", chunks)
    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation" in prompt


def test_build_prompt_numbers_chunks():
    chunks = [
        {"content": "First chunk.", "filename": "a.txt", "chunk_index": 0},
        {"content": "Second chunk.", "filename": "b.txt", "chunk_index": 1},
    ]
    prompt = build_prompt("test", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_build_prompt_no_chunks():
    prompt = build_prompt("Any question?", [])
    assert "Any question?" in prompt
    assert "No context" in prompt


def test_generate_answer_calls_claude(mocker):
    from rag.generation import generate_answer
    mock_anthropic = mocker.patch("rag.generation.anthropic.Anthropic")
    mock_response = mocker.MagicMock()
    mock_response.content = [mocker.MagicMock(text="The answer is 42.")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_anthropic.return_value.messages.create.return_value = mock_response

    chunks = [{"content": "The answer is 42.", "filename": "test.txt", "chunk_index": 0}]
    answer, meta = generate_answer("What is the answer?", chunks)

    assert answer == "The answer is 42."
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 10
    assert "latency_ms" in meta
    assert "prompt" in meta
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_generation.py -v
```

Expected: `ImportError: cannot import name 'build_prompt'`

- [ ] **Step 3: Write rag/generation.py**

```python
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using ONLY the provided "
    "context chunks. If the answer is not in the context, say "
    "'I don't have enough context to answer that.' Be concise and technical."
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "No context available."
    else:
        context = "\n\n".join(
            f"[{i + 1}] (source: {c['filename']}, chunk {c['chunk_index']})\n{c['content']}"
            for i, c in enumerate(chunks)
        )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, dict]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_prompt(question, chunks)

    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - start) * 1000)

    return response.content[0].text, {
        "model": MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms,
        "prompt": prompt,
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_generation.py -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add rag/generation.py tests/test_generation.py
git commit -m "feat: Claude generation with prompt builder"
```

---

## Task 7: Sample Documents

**Files:**
- Create: `sample_docs/rag_guide.md`
- Create: `sample_docs/system_design.txt`
- Create: `sample_docs/distributed_systems.txt`

- [ ] **Step 1: Create sample_docs/rag_guide.md**

```markdown
# Retrieval-Augmented Generation: A Technical Guide

Retrieval-Augmented Generation (RAG) is an architecture pattern that enhances large language models
by grounding their responses in retrieved documents rather than relying solely on parametric memory.
LLMs encode knowledge in weights during training, but weights are frozen at a knowledge cutoff date
and cannot be updated cheaply. RAG sidesteps this by keeping knowledge in an external store that is
queried at inference time.

## The Two-Stage Architecture

A RAG system has two stages: a retriever and a generator. The retriever finds document chunks
relevant to the user query. The generator — a language model — reads those chunks as context and
produces an answer. The separation is deliberate: retrieval can be updated independently of the
model, and retrieval failures are auditable in a way that weight-encoded knowledge is not.

## Why RAG Over Fine-tuning

Fine-tuning a model on proprietary data is expensive, produces a frozen snapshot, and makes it hard
to audit where an answer came from. RAG addresses all three: you update the document store without
retraining, each answer cites retrievable source chunks, and the pipeline cost is dominated by
inference rather than training. For dynamic or proprietary knowledge bases, RAG is almost always
the right architectural choice.

## Chunking

Documents must be split into chunks before embedding. Fixed-size chunking (e.g., 512 characters
with 50-character overlap) is simple and predictable. The overlap ensures a sentence that straddles
a boundary appears intact in at least one chunk, preventing retrieval misses on boundary-split
content. Semantic chunking splits on paragraph or sentence boundaries, producing variable-length
chunks that may better preserve meaning but require more complex tooling. For most production systems,
fixed-size chunking with overlap is the pragmatic default.

## Embeddings and Vector Similarity

Each chunk is converted to a dense vector by an embedding model. Models like `all-MiniLM-L6-v2`
map text to 384-dimensional space such that semantically similar text lands nearby. At query time,
the question is embedded with the same model, and the nearest chunks by cosine similarity are
retrieved. Cosine similarity measures the angle between vectors, not magnitude, which makes it
robust to variable-length text.

## Why Pure Vector Search Falls Short

Cosine similarity captures semantic proximity but fails on exact-term matching. A query for "CAP"
(the distributed systems theorem) may not rank chunks mentioning "Consistency, Availability, and
Partition tolerance" highly because the acronym and its expansion are not close in embedding space.
BM25, a probabilistic term-frequency ranking function, handles this well. Hybrid search combines
both signals.

## Reciprocal Rank Fusion

RRF merges two ranked lists without requiring calibrated scores from each system. For each chunk,
`RRF score = sum(1 / (k + rank_i))` across all lists it appears in. The constant k=60 was
established empirically by Cormack, Clarke, and Buettcher (2009) as a value that dampens top-rank
dominance while still rewarding high-ranked results. A chunk appearing in both the vector and BM25
lists scores roughly twice a chunk appearing in only one.

## Limitations

RAG is not a silver bullet. Context windows cap how many chunks can fit in a single prompt.
Retrieval misses — when the relevant chunk scores too low to appear in top-k — produce silently
wrong answers. The generator can still hallucinate details not present in retrieved context.
Chunk quality matters: a chunk that contains the answer but is poorly written scores lower than
a fluent but irrelevant chunk. Monitoring retrieval quality (e.g., checking whether the expected
source appears in retrieved chunks) is essential for production systems.
```

- [ ] **Step 2: Create sample_docs/system_design.txt**

```
System Design Concepts: A Technical Reference

CAP THEOREM

The CAP theorem, formulated by Eric Brewer in 2000 and proved by Gilbert and Lynch in 2002,
states that a distributed system can satisfy at most two of three properties: Consistency (every
read receives the most recent write), Availability (every request receives a non-error response),
and Partition tolerance (the system continues operating despite network partitions). Since network
partitions are unavoidable in distributed systems, practical systems choose between CP (consistent
and partition-tolerant) and AP (available and partition-tolerant). CP systems like HBase and
Zookeeper block or return errors during a partition to preserve consistency. AP systems like
Cassandra and DynamoDB continue serving requests during partitions, accepting that different nodes
may return stale data.

CONSISTENT HASHING

Consistent hashing solves the data redistribution problem in distributed caches and databases.
In a naive hash ring, adding or removing a node requires remapping roughly 1/N of all keys. In
consistent hashing, both nodes and keys are mapped to positions on a circular ring using the same
hash function. Each key is owned by the nearest node clockwise on the ring. When a node is added,
only the keys between the new node and its predecessor move — approximately 1/N of the total.
Virtual nodes (vnodes) improve load distribution by assigning each physical node multiple positions
on the ring. Cassandra uses 256 vnodes per node by default.

BLOOM FILTERS

A Bloom filter is a probabilistic data structure that tests set membership in O(1) time and O(m)
space, where m is the bit array size. It admits false positives (reporting an element is in the
set when it is not) but never false negatives (never misses an actual member). The false positive
rate is approximately (1 - e^(-kn/m))^k, where k is the number of hash functions and n is the
number of inserted elements. LSM-tree databases like RocksDB and Cassandra use Bloom filters on
SSTable files to avoid disk reads for keys that do not exist in a given file, dramatically reducing
read amplification.

LSM TREES

Log-Structured Merge trees optimize for write throughput. Writes are buffered in an in-memory
structure (memtable), typically a red-black tree or skip list. When the memtable reaches its size
threshold, it is flushed to disk as a sorted, immutable SSTable file. Reads must check the memtable
and all on-disk SSTables, consulting the most recent level first. To bound read costs, LSM trees
periodically merge and compact SSTables, discarding obsolete versions of overwritten keys.
Compaction strategies — leveled, tiered, and FIFO — trade read amplification against write
amplification. Leveled compaction (used by RocksDB by default) keeps SSTables non-overlapping
within a level, guaranteeing at most one SSTable per level contains a given key, which improves
read performance at the cost of more compaction I/O.
```

- [ ] **Step 3: Create sample_docs/distributed_systems.txt**

```
Distributed Systems Internals: A Technical Reference

RAFT CONSENSUS

Raft is a consensus algorithm designed to be more understandable than Paxos. A Raft cluster elects
a leader using randomized election timeouts (typically 150-300ms). Each server starts as a follower.
If a follower receives no heartbeat within its timeout, it becomes a candidate, increments its term,
votes for itself, and sends RequestVote RPCs to peers. A candidate wins if it receives votes from a
majority. The leader then handles all client requests, appending entries to its log and replicating
them to followers via AppendEntries RPCs. An entry is committed once a majority of nodes have
written it to their logs. The log matching property guarantees that if two logs have the same entry
at the same index, all preceding entries are also identical. Leader crashes trigger a new election;
the new leader must have the most up-to-date log among the quorum that elects it.

VECTOR CLOCKS

Vector clocks track causal relationships in distributed systems without synchronized physical clocks.
Each node maintains a vector of counters, one per node. On a local event, a node increments its own
counter. On a send event, the current vector is attached to the message. On receive, each element is
set to the maximum of the local and received vectors, then the local counter is incremented. Two
events A and B have a happened-before relationship (A → B) if A's vector is component-wise less than
or equal to B's vector with at least one strictly less component. If neither A → B nor B → A, the
events are concurrent and represent a potential conflict. Vector clocks detect concurrency but do not
resolve it — conflict resolution is application-specific (e.g., last-write-wins or merge functions).

CRDTS

Conflict-free Replicated Data Types (CRDTs) are data structures that guarantee convergence: all
replicas that receive the same set of updates, in any order, reach the same state. A G-Counter
(grow-only counter) is the simplest example: each node maintains its own counter, increments only
its own slot, and the global count is the sum of all slots. Merging two G-Counters takes the
component-wise maximum. A LWW-Register (last-write-wins) attaches a timestamp to each write;
merge picks the entry with the higher timestamp. The convergence guarantee holds because all CRDT
merge operations are commutative, associative, and idempotent — properties that make network
reorder and retry safe. CRDTs are used in systems like Redis (HyperLogLog), Riak, and collaborative
editors (CRDT-based text types underpin Figma's multiplayer).

WRITE AMPLIFICATION IN LSM TREES

Write amplification is the ratio of bytes written to disk to bytes written by the application. In
LSM trees, the same data is written multiple times: once to the WAL, once to the memtable, once to
L0 on flush, and then again on each compaction pass as it moves through levels. In leveled
compaction, a byte written at L0 may be rewritten at L1 (10× L0 size), then L2 (10× L1), and so on.
Total write amplification can reach 10-30× in the worst case. Tiered compaction reduces write
amplification at the cost of higher read amplification (more SSTables to check per read) and space
amplification (obsolete data stays around longer). RocksDB exposes compaction_style, max_bytes_for_
level_base, and level_multiplier to tune the tradeoff. Write amplification directly impacts SSD
endurance, since flash cells have a finite write cycle count.
```

- [ ] **Step 4: Verify word counts**

```bash
wc -w sample_docs/rag_guide.md sample_docs/system_design.txt sample_docs/distributed_systems.txt
```

Expected: Each file > 600 words.

- [ ] **Step 5: Commit**

```bash
git add sample_docs/
git commit -m "feat: add sample documents for demo"
```

---

## Task 8: Streamlit App (Both Tabs)

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

```python
import inspect
import os
import time

import streamlit as st
from dotenv import load_dotenv

from db.connection import db_conn
from rag.generation import generate_answer
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

SAMPLE_DOCS = {
    "rag_guide.md": "sample_docs/rag_guide.md",
    "system_design.txt": "sample_docs/system_design.txt",
    "distributed_systems.txt": "sample_docs/distributed_systems.txt",
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
        run_ingest(uploaded.read(), uploaded.name)

    st.subheader("Or load a sample")
    cols = st.columns(3)
    for i, (name, path) in enumerate(SAMPLE_DOCS.items()):
        if cols[i].button(f"📄 {name}"):
            with open(path, "rb") as f:
                run_ingest(f.read(), name)

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
                        "Content (truncated)": c["content"][:120] + "...",
                    }
                    for c in chunks
                ],
                use_container_width=True,
            )
```

- [ ] **Step 2: Run the app and verify Tab 1**

```bash
streamlit run app.py
```

Open http://localhost:8501. Test:
- Sidebar shows "Neon connected ✓" with doc/chunk counts
- Click "📄 rag_guide.md" → progress spinners → success message with metrics
- "⚙️ How ingestion works" expander opens, shows chunking source code and SQL schema
- Document appears in the "Indexed documents" table

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit app with ingest and query tabs"
```

---

## Task 9: End-to-End Verification

**Files:** no changes — this is a manual test pass

- [ ] **Step 1: Reset DB and reload all samples**

```bash
psql $DATABASE_URL -c "TRUNCATE documents CASCADE;"
streamlit run app.py
```

- [ ] **Step 2: Ingest all three sample docs**

In the Ingest tab, click all three sample buttons. Verify each shows a success message and chunk count.

- [ ] **Step 3: Query verification checklist**

Switch to Query tab and run each question. Verify the answer is factually grounded in the sample content:

| Question | Expected source |
|---|---|
| "What is retrieval-augmented generation?" | rag_guide.md |
| "Explain the CAP theorem" | system_design.txt |
| "How does Raft consensus work?" | distributed_systems.txt |
| "What causes write amplification?" | distributed_systems.txt |
| "What is the capital of France?" | Answer: "I don't have enough context..." |
| "CAP" (single acronym only) | BM25 rank should be populated in chunks table |

- [ ] **Step 4: Verify hybrid search advantage**

For the query "CAP": check the retrieved chunks table in the UI. The `BM25 rank` column should show a number (not "—"), proving BM25 caught the exact-string match that pure vector search would miss.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All unit tests `PASSED`; integration tests `PASSED` or `SKIPPED`.

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: DocSage complete — hybrid RAG portfolio app"
```

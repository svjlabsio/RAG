# DocSage — Architecture & Design Guide

A technical deep-dive into every design decision in the DocSage hybrid RAG pipeline. Written for software engineers who want to understand not just *what* the system does but *why* each component was built the way it was.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Database: Why PostgreSQL Instead of a Vector DB](#2-database-why-postgresql-instead-of-a-vector-db)
3. [Schema Design](#3-schema-design)
4. [Connection Pool](#4-connection-pool)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [Hybrid Search](#6-hybrid-search)
7. [Reciprocal Rank Fusion](#7-reciprocal-rank-fusion)
8. [Generation](#8-generation)
9. [Thread Safety Patterns](#9-thread-safety-patterns)
10. [Tradeoffs and Known Limitations](#10-tradeoffs-and-known-limitations)

---

## 1. System Overview

DocSage is a **hybrid search RAG pipeline** backed by a single PostgreSQL database. The high-level data flow:

```
User uploads file
       │
       ▼
  parse_document()        ← PyMuPDF / python-docx / UTF-8 decode
       │
       ▼
   chunk_text()           ← fixed-size sliding window, 512 chars / 50 overlap
       │
       ▼
  embed_chunks()          ← all-MiniLM-L6-v2, 384-dim, local CPU inference
       │
       ▼
 store_document()         ← bulk INSERT into Neon PostgreSQL
       │
  [chunks table]
  ├── embedding vector(384)       ─── pgvector ivfflat index
  └── ts_content TSVECTOR         ─── GIN index (auto-generated)

User asks a question
       │
       ▼
  embed_chunks([query])   ← same model, same space
       │
    ┌──┴──────────────────┐
    ▼                     ▼
vector_search()      bm25_search()
(cosine ANN)         (tsvector GIN)
    │                     │
    └──────────┬───────────┘
               ▼
  reciprocal_rank_fusion()   ← merges, deduplicates, re-ranks
               │
               ▼ top-5 chunks
        build_prompt()
               │
               ▼
    Claude claude-haiku-4-5-20251001
               │
               ▼
           answer + metadata
```

**Key architectural constraint:** everything lives in one PostgreSQL database. No Pinecone, no Weaviate, no Elasticsearch — just Neon with two extensions.

---

## 2. Database: Why PostgreSQL Instead of a Vector DB

The obvious question: why not use a purpose-built vector database like Pinecone, Weaviate, or Qdrant?

**Operational simplicity.** A dedicated vector DB is a second stateful service to provision, monitor, back up, and keep in sync with your relational data. For a production system this cost is often worth it. For a portfolio demo — and for the large class of real applications with moderate data volumes — it is pure overhead.

**BM25 is already in PostgreSQL.** The `tsvector`/`GIN` full-text search stack has been in PostgreSQL since version 8.3. `ts_rank` implements a variant of BM25 scoring. Getting both vector similarity and keyword search from the same database means a single transaction, a single connection pool, and no cross-service consistency issues.

**pgvector is production-grade.** The `vector` extension adds approximate nearest-neighbor search (via `ivfflat` or `hnsw`) and exact kNN directly to PostgreSQL. Cosine, L2, and inner-product distance operators are first-class. Neon, the managed PostgreSQL provider used here, ships pgvector pre-installed.

**The tradeoff you accept:** pgvector's ANN performance tops out around tens of millions of vectors at acceptable recall. Pinecone or Weaviate scale to billions. If you're building the next large-scale semantic search engine, reach for a dedicated vector DB. If you're building anything with fewer than ~10M chunks, PostgreSQL + pgvector is the right default.

---

## 3. Schema Design

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,
  file_type   TEXT NOT NULL,
  uploaded_at TIMESTAMPTZ DEFAULT NOW(),
  chunk_count INT
);

CREATE TABLE chunks (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,
  embedding   vector(384),
  chunk_index INT,
  metadata    JSONB DEFAULT '{}',
  ts_content  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunks USING GIN (ts_content);
```

**`ts_content` is a generated column, not application-maintained.** This is the key schema decision. A `GENERATED ALWAYS AS ... STORED` column means PostgreSQL writes the tsvector automatically on every INSERT or UPDATE — the application never has to remember to populate it, and it cannot get out of sync with `content`. The GIN index on the generated column is then kept current by the same mechanism.

**`ON DELETE CASCADE` on `doc_id`.** Deleting a document row removes all its chunks atomically. No orphan cleanup job needed.

**`ivfflat` with `lists = 100`.** IVFFlat (Inverted File Flat) partitions the vector space into `lists` Voronoi cells at index build time. At query time it searches the `probes` nearest cells (default: 1). This makes it approximate — you trade a small recall loss (~5% at `probes=1`) for O(√n) query time instead of O(n). The `lists = 100` value is appropriate for up to ~1M vectors; the pgvector docs recommend `lists = rows / 1000` for larger tables.

**Important:** ivfflat requires at least `lists * 3 = 300` rows before the query planner uses the index. Below that threshold PostgreSQL correctly falls back to a sequential scan. This is expected behavior, not a bug.

**`metadata JSONB`.** An escape hatch for future per-chunk metadata (page number, section header, confidence score from a classifier). Currently stored as `{}` but the column is indexed by default via PostgreSQL's JSONB operators.

---

## 4. Connection Pool

```python
_pool = None
_pool_lock = threading.Lock()

def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pool.ThreadedConnectionPool(1, 5, os.environ["DATABASE_URL"])
    return _pool
```

**Double-checked locking.** The outer `if _pool is None` is the fast path — no lock acquired on the common case where the pool is already initialized. The inner `if _pool is None` inside the lock is the safety check: two threads could both pass the outer check before either acquires the lock, so the second check prevents creating two pools.

**`ThreadedConnectionPool(min=1, max=5)`.** Streamlit re-renders the page on every interaction and may do so from multiple threads. `ThreadedConnectionPool` is thread-safe (backed by a `threading.Lock` internally). `min=1` keeps one connection warm. `max=5` is a conservative cap appropriate for a demo app on Neon's free tier, which limits concurrent connections.

**`db_conn()` context manager.** The caller gets a connection, commits on clean exit, rolls back on exception, and always returns the connection to the pool — even if `rollback()` itself throws (a dead connection). The broken-connection case:

```python
finally:
    if conn.closed:
        _get_pool().putconn(conn, close=True)  # discard, don't recycle
    else:
        return_conn(conn)
```

`putconn(conn, close=True)` tells the pool to drop this slot rather than recycle a dead connection back to waiting callers.

---

## 5. Ingestion Pipeline

### Parsing

```python
def parse_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    elif ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in {"txt", "md"}:
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(...)
```

All parsing operates on `bytes` in memory — no temp files. PyMuPDF (`fitz`) opens a PDF from a stream directly; `python-docx` wraps a `BytesIO`. This keeps the pipeline stateless and makes it trivially testable.

### Chunking

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap   # step = 462
    return chunks
```

**Character-based fixed-size sliding window.** Step size is `chunk_size - overlap = 462` characters. The 50-character overlap ensures a sentence straddling a chunk boundary appears intact in the adjacent chunk, preventing retrieval misses on boundary-split content.

**Why not semantic chunking?** Semantic chunking (splitting on paragraph or sentence boundaries) produces higher retrieval quality in theory: each chunk contains a complete thought, improving cosine similarity scores. The practical reasons to stay with fixed-size for this project:

- Semantic chunking requires a sentence tokenizer (spaCy, NLTK). That is a multi-hundred-MB dependency for a marginal gain on the dense technical prose in our sample docs.
- Fixed-size chunks have predictable embedding latency — you always know how many chunks a document will produce.
- The 50-char overlap already catches the most common failure mode of semantic chunking (the boundary miss).
- The upgrade path is clear: swap `chunk_text` for a recursive character splitter (split on `\n\n` → `\n` → `. ` → character count) if retrieval quality becomes the bottleneck.

### Embedding

```python
def embed_chunks(chunks: list[str]) -> np.ndarray:
    return _get_model().encode(chunks, batch_size=32, show_progress_bar=False)
```

**`all-MiniLM-L6-v2`** is a 22M-parameter model producing 384-dimensional embeddings. It is not the most capable embedding model available — `text-embedding-3-large` from OpenAI produces 3072 dimensions with meaningfully better semantic understanding. The tradeoffs that make MiniLM the right choice here:

| | all-MiniLM-L6-v2 | text-embedding-3-large |
|---|---|---|
| Cost | Free, local CPU | ~$0.13 / 1M tokens |
| Latency | ~50ms/batch on CPU | ~200ms + network |
| Quality | Good | Excellent |
| Dimensions | 384 | 3072 |
| Dependency | PyTorch (~500MB) | Anthropic/OpenAI API key |

For a portfolio demo, zero API cost and offline operation are decisive. The model is lazy-loaded and cached in a module-level singleton (see §9).

### Storage

`store_document` uses `psycopg2.extras.execute_values` for bulk chunk insertion — a single multi-row `INSERT` instead of N round-trips. The caller owns the transaction boundary: `store_document` does not commit; the `db_conn()` context manager does. This allows callers to compose multiple operations into one atomic transaction if needed.

---

## 6. Hybrid Search

A single retrieval strategy has a fundamental weakness:

- **Pure vector search** captures semantic similarity but fails on exact-term queries. A search for `"CAP"` may not retrieve chunks about the CAP theorem because the acronym and its expansion ("Consistency, Availability, Partition tolerance") are not geometrically close in 384-dimensional embedding space.
- **Pure BM25** is excellent at exact and near-exact term matching but blind to paraphrase. A search for `"how do distributed systems handle network splits"` will miss chunks that say `"partition tolerance"` but never use the word "splits".

Hybrid search runs both in parallel and merges the results.

### Vector search

```sql
WITH q AS (SELECT %s::vector AS vec)
SELECT c.id, c.content, c.chunk_index, d.filename,
       1 - (c.embedding <=> q.vec) AS vector_score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
CROSS JOIN q
ORDER BY c.embedding <=> q.vec
LIMIT %s
```

**`<=>` is the cosine distance operator** added by pgvector. `1 - distance = cosine similarity`. The `ORDER BY` on the raw distance (not `1 - distance`) lets the ivfflat index serve the query; ordering by the computed similarity column would force a full scan.

**The CTE is a deliberate optimization.** Binding the 384-element query vector (a `float32[384]` serialized to ~6KB of wire data) as a single parameter inside a CTE means it is serialized and sent to PostgreSQL exactly once per query. The naive approach — passing it twice as separate `%s` parameters — doubles the wire overhead on every search.

**`register_vector(conn)` is called at most once per connection** via a `weakref.WeakSet`. The `WeakSet` holds live connection objects directly (not their memory addresses via `id()`), so entries are automatically removed when a connection is garbage-collected by the pool. This avoids both a memory leak and the id-reuse hazard that would occur if a closed connection's address were reassigned to a new one.

### BM25 search

```sql
SELECT c.id, c.content, c.chunk_index, d.filename,
       ts_rank(c.ts_content, plainto_tsquery('english', %s)) AS bm25_score
FROM chunks c
JOIN documents d ON d.id = c.doc_id
WHERE c.ts_content @@ plainto_tsquery('english', %s)
ORDER BY bm25_score DESC
LIMIT %s
```

`plainto_tsquery` parses the query string into a tsquery, normalising each word to its stem (`running` → `run`, `CAP` → `cap`). `ts_rank` scores each matching chunk using a BM25-variant formula that accounts for term frequency and inverse document frequency. The `@@` operator uses the GIN index — it does not scan `ts_content` values; it walks the GIN posting lists.

---

## 7. Reciprocal Rank Fusion

RRF merges two ranked lists without requiring their scores to be on the same scale or even comparable. This is the right choice here because cosine similarity (0 to 1) and `ts_rank` (arbitrary positive float) are incommensurable — normalizing and weighting them directly would require careful calibration that breaks as the corpus grows.

```python
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, result in enumerate(vector_results):
        chunk_id = result["id"]
        if chunk_id not in scores:
            scores[chunk_id] = {**result, "rrf_score": 0.0, "bm25_score": 0.0, ...}
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        if chunk_id not in scores:
            scores[chunk_id] = {**result, "rrf_score": 0.0, "vector_score": 0.0, ...}
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)

    return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
```

**The formula:** `RRF(chunk) = Σ 1 / (k + rankᵢ + 1)` summed over every result list the chunk appears in (ranks are 0-indexed here, hence `+1`).

**Why k=60?** This constant was established empirically by Cormack, Clarke, and Buettcher (2009) in a SIGIR paper evaluating fusion methods across TREC tracks. k=60 dampens top-rank dominance — a chunk ranked #1 in one list scores `1/61 ≈ 0.0164`, while a chunk ranked #20 scores `1/81 ≈ 0.0123`. The difference is meaningful but not overwhelming, which prevents one strong signal from completely swamping the other. A chunk appearing in both lists scores roughly twice a chunk in only one.

**Score defaults for cross-list entries.** A chunk found by vector search but not BM25 gets `bm25_score: 0.0` (not `None`) and vice versa. This keeps all score fields as floats for safe downstream arithmetic in the UI layer.

---

## 8. Generation

### Model choice: claude-haiku-4-5-20251001

Haiku is the fastest and cheapest Claude model. For a RAG application the generation step is latency-sensitive — users see a spinner while waiting. Haiku's ~300ms median first-token latency is well-suited to interactive use. The model is easily swappable: change `MODEL` in `rag/generation.py`.

### Prompt design

```python
SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using ONLY the provided "
    "context chunks. If the answer is not in the context, say "
    "'I don't have enough context to answer that.' Be concise and technical."
)

def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{i+1}] (source: {filename}, chunk {chunk_index})\n{content}"
        for i, c in enumerate(chunks)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
```

**Numbered citations.** Each context block is prefixed `[1]`, `[2]`... with its source filename and chunk index. This gives Claude the scaffolding to say "according to [2]..." and makes it easy for the UI to map citations back to retrieved chunks.

**Strict grounding instruction.** The system prompt instructs Claude to answer *only* from context and to explicitly refuse when the answer isn't there. This is the standard technique for preventing hallucination in RAG systems — the model should not draw on parametric memory.

**The Anthropic client is a singleton.** `_get_client()` uses double-checked locking (same pattern as the DB pool) to construct one `anthropic.Anthropic` instance per process. The client maintains an HTTP connection pool internally; rebuilding it per-query would pay a TLS handshake on every request.

**Content block type guard.** The response is accessed via:

```python
text_blocks = [b for b in response.content if b.type == "text"]
if not text_blocks:
    raise ValueError(f"No text content in Claude response: {response.content}")
```

`b.type == "text"` uses the SDK's discriminated union (`TextBlock.type == "text"`) rather than `hasattr(b, "text")`. If the API returns a tool-use block or a thinking block (neither of which has a `text` field in the same sense), the guard catches it rather than raising an opaque `AttributeError`.

---

## 9. Thread Safety Patterns

Streamlit runs handler code in multiple threads simultaneously when users interact concurrently. Three module-level singletons need initialization guards:

| Singleton | Module | Pattern |
|---|---|---|
| `_pool` — psycopg2 connection pool | `db/connection.py` | double-checked lock |
| `_model` — SentenceTransformer | `rag/ingest.py` | double-checked lock |
| `_client` — Anthropic HTTP client | `rag/generation.py` | double-checked lock |

All three use the same pattern:

```python
_thing = None
_thing_lock = threading.Lock()

def _get_thing():
    global _thing
    if _thing is None:           # fast path: no lock on the common case
        with _thing_lock:
            if _thing is None:   # safety check: second thread may have beaten us
                _thing = ExpensiveThing()
    return _thing
```

The fast path (outer check) executes without acquiring a lock on every call after initialization — important because `embed_chunks` and `generate_answer` are called on every query. The inner check prevents double-initialization when two threads race through the outer check simultaneously before either acquires the lock.

**`_registered_conns` in `rag/retrieval.py`** uses `weakref.WeakSet` rather than a plain `set`. Storing connection objects directly (not their `id()` addresses) means the set entry is automatically removed when a connection is garbage-collected. A `set[int]` of addresses would accumulate stale entries indefinitely and could incorrectly skip `register_vector` if a new connection happened to reuse a closed connection's memory address.

---

## 10. Tradeoffs and Known Limitations

**Fixed-size chunking.** 512-character windows are simple and predictable but split sentences at arbitrary points. 50-character overlap mitigates this for most content. Semantic/recursive chunking (split on `\n\n` → `\n` → `. `) would improve retrieval quality on structured documents at the cost of variable chunk sizes and a heavier NLP dependency.

**ivfflat recall.** The ivfflat index with `lists=100, probes=1` provides ~95% recall on well-distributed data. On small tables (< 300 rows) the query planner ignores the index and does a sequential scan — this is correct behaviour, not a bug. For higher recall at scale, `probes` can be increased or the index type can be changed to `hnsw` (higher build cost, better recall at query time).

**No re-ranking.** The pipeline retrieves top-20 from each search method and fuses to top-5. A cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied to the RRF top-5 would improve precision further, at the cost of ~100ms additional latency per query.

**No chunking deduplication.** If the same document is uploaded twice, all chunks are inserted again. Production systems typically hash document content at ingest time and skip re-insertion on collision.

**Context window cap.** Five chunks at 512 characters each is ~2,500 characters of context — well within Haiku's 200K-token window. If chunk sizes are increased substantially, the prompt size should be monitored to stay within model limits.

**Single-language BM25.** `plainto_tsquery('english', ...)` uses the English text search configuration. Non-English documents will still be indexed and searchable but stop-word removal and stemming will be incorrect.

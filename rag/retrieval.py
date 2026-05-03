import numpy as np
from pgvector.psycopg2 import register_vector

_registered_conns: set[int] = set()

VECTOR_SEARCH_SQL = """
    WITH q AS (SELECT %s::vector AS vec)
    SELECT c.id, c.content, c.chunk_index, d.filename,
           1 - (c.embedding <=> q.vec) AS vector_score
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    CROSS JOIN q
    ORDER BY c.embedding <=> q.vec
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
    conn_id = id(conn)
    if conn_id not in _registered_conns:
        register_vector(conn)
        _registered_conns.add(conn_id)
    with conn.cursor() as cur:
        cur.execute(VECTOR_SEARCH_SQL, (query_embedding.tolist(), k))
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
            scores[chunk_id] = {**result, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None, "bm25_score": 0.0}
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[chunk_id]["vector_rank"] = rank + 1
        scores[chunk_id]["vector_score"] = result.get("vector_score", 0.0)

    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        if chunk_id not in scores:
            scores[chunk_id] = {**result, "rrf_score": 0.0, "vector_rank": None, "bm25_rank": None, "vector_score": 0.0}
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

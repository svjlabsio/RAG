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

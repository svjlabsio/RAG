import os

import pytest
from rag.ingest import parse_document, chunk_text, embed_chunks


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


def test_chunk_text_empty_string():
    assert chunk_text("") == []


def test_chunk_text_exact_boundary():
    # Text exactly equal to chunk_size still produces a second chunk because
    # the window advances by (chunk_size - overlap), so start=462 < 512.
    text = "a" * 512
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert chunks[0] == text          # first chunk is the full text
    assert chunks[-1] == text[462:]   # last chunk is the tail overlap


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_parse_document_unknown_extension_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(b"binary data", "file.xyz")


def test_embed_chunks_returns_correct_shape():
    embeddings = embed_chunks(["hello world", "foo bar baz"])
    assert embeddings.shape == (2, 384)


def test_embed_chunks_single():
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

            # On assertion failure, db_conn()'s rollback handles cleanup automatically.
            # This explicit DELETE runs on success to keep the test DB clean.
            cur.execute("DELETE FROM documents WHERE id = %s", (str(doc_id),))

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

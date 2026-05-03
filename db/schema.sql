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
  doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,
  embedding   vector(384),
  chunk_index INT,
  metadata    JSONB DEFAULT '{}',
  ts_content  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- ivfflat: approximate nearest-neighbor. Requires >= 300 rows (lists*3) before
-- the planner uses the index; sequential scan is used on smaller tables (correct behavior).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_ts_idx
  ON chunks USING GIN (ts_content);

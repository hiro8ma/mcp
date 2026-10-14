-- Run with psql -v ON_ERROR_STOP=1 -f this-file.sql on a lab database.
-- Requires vector extension. Each CONCURRENTLY command must be sent separately.
-- CREATE SCHEMA fails if the name already exists: existing data is retained.
CREATE SCHEMA partition_hnsw_demo;

CREATE TABLE partition_hnsw_demo.chunks (
    id bigint GENERATED ALWAYS AS IDENTITY,
    content text NOT NULL,
    embedding vector(3) NOT NULL,
    created_at timestamptz NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE partition_hnsw_demo.chunks_202603
PARTITION OF partition_hnsw_demo.chunks
FOR VALUES FROM ('2026-03-01 00:00:00+00') TO ('2026-04-01 00:00:00+00');
CREATE TABLE partition_hnsw_demo.chunks_202604
PARTITION OF partition_hnsw_demo.chunks
FOR VALUES FROM ('2026-04-01 00:00:00+00') TO ('2026-05-01 00:00:00+00');

INSERT INTO partition_hnsw_demo.chunks (content, embedding, created_at) VALUES
    ('March', '[1,0,0]', '2026-03-15 00:00:00+00'),
    ('April boundary', '[0.9,0.1,0]', '2026-04-01 00:00:00+00');

CREATE INDEX chunks_embedding_idx ON ONLY partition_hnsw_demo.chunks
USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);

DO $$ BEGIN
    IF (SELECT indisvalid FROM pg_index WHERE indexrelid='partition_hnsw_demo.chunks_embedding_idx'::regclass) THEN
        RAISE EXCEPTION 'Parent must be invalid before existing child indexes are attached';
    END IF;
END $$;

CREATE INDEX CONCURRENTLY chunks_202603_embedding_idx
ON partition_hnsw_demo.chunks_202603
USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
ALTER INDEX partition_hnsw_demo.chunks_embedding_idx
ATTACH PARTITION partition_hnsw_demo.chunks_202603_embedding_idx;

DO $$ BEGIN
    IF (SELECT indisvalid FROM pg_index WHERE indexrelid='partition_hnsw_demo.chunks_embedding_idx'::regclass) THEN
        RAISE EXCEPTION 'Parent must remain invalid while April is unattached';
    END IF;
END $$;

CREATE INDEX CONCURRENTLY chunks_202604_embedding_idx
ON partition_hnsw_demo.chunks_202604
USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
ALTER INDEX partition_hnsw_demo.chunks_embedding_idx
ATTACH PARTITION partition_hnsw_demo.chunks_202604_embedding_idx;

DO $$ BEGIN
    IF NOT (SELECT indisvalid FROM pg_index WHERE indexrelid='partition_hnsw_demo.chunks_embedding_idx'::regclass) THEN
        RAISE EXCEPTION 'Parent must be valid after all existing children are attached';
    END IF;
END $$;

-- A new empty partition inherits the parent's index, without a separate build.
CREATE TABLE partition_hnsw_demo.chunks_202605
PARTITION OF partition_hnsw_demo.chunks
FOR VALUES FROM ('2026-05-01 00:00:00+00') TO ('2026-06-01 00:00:00+00');

DO $$ BEGIN
    IF (SELECT count(*) FROM pg_inherits WHERE inhparent='partition_hnsw_demo.chunks_embedding_idx'::regclass) <> 3 THEN
        RAISE EXCEPTION 'The new May partition must have an attached child index';
    END IF;
    IF NOT (SELECT indisvalid FROM pg_index WHERE indexrelid='partition_hnsw_demo.chunks_embedding_idx'::regclass) THEN
        RAISE EXCEPTION 'Parent must remain valid after adding May';
    END IF;
    IF (SELECT tableoid FROM partition_hnsw_demo.chunks WHERE content='April boundary') <> 'partition_hnsw_demo.chunks_202604'::regclass THEN
        RAISE EXCEPTION 'RANGE upper bounds are exclusive';
    END IF;
END $$;

-- Analyze the parent explicitly; autovacuum does not analyze partitioned parents.
ANALYZE partition_hnsw_demo.chunks;
SELECT id, content FROM partition_hnsw_demo.chunks
WHERE created_at >= '2026-04-01 00:00:00+00'
  AND created_at < '2026-05-01 00:00:00+00'
ORDER BY embedding <=> '[1,0,0]'::vector LIMIT 10;

-- Only this example's successfully-created schema is removed.
DROP SCHEMA partition_hnsw_demo CASCADE;

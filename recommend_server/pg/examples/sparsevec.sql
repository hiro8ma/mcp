-- Sparse retrieval smoke example (pgvector >= 0.7).
-- Prerequisite: CREATE EXTENSION vector in the target database.
-- All demonstration data and indexes disappear at COMMIT.
BEGIN;

CREATE TEMP TABLE sparsevec_demo (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content text NOT NULL,
    embedding sparsevec(10000) NOT NULL
) ON COMMIT DROP;

-- Literal indexes are 1-based; 10000 is the full dimension, not nnz.
INSERT INTO sparsevec_demo (content, embedding) VALUES
    ('PostgreSQL extension', '{1:0.8,42:0.3,1337:0.1}/10000'),
    ('Vector distance functions', '{1:0.7,100:0.5,2048:0.2}/10000');

-- Use the metric defined by the model. SPLADE typically uses inner product.
CREATE INDEX sparsevec_demo_ip_idx ON sparsevec_demo
USING hnsw (embedding sparsevec_ip_ops)
WITH (m = 16, ef_construction = 64);

SELECT id, content,
       -(embedding <#> '{1:0.8,42:0.3}/10000'::sparsevec) AS inner_product
FROM sparsevec_demo
ORDER BY embedding <#> '{1:0.8,42:0.3}/10000'::sparsevec
LIMIT 10;

-- For cosine retrieval, both the operator and index opclass must match.
CREATE INDEX sparsevec_demo_cosine_idx ON sparsevec_demo
USING hnsw (embedding sparsevec_cosine_ops);

SELECT id, content,
       embedding <=> '{1:0.8,42:0.3}/10000'::sparsevec AS cosine_distance
FROM sparsevec_demo
ORDER BY embedding <=> '{1:0.8,42:0.3}/10000'::sparsevec
LIMIT 10;

DO $$
DECLARE
    best_id bigint;
    score double precision;
BEGIN
    SELECT id, -(embedding <#> '{1:0.8,42:0.3}/10000'::sparsevec)
    INTO best_id, score
    FROM sparsevec_demo
    ORDER BY embedding <#> '{1:0.8,42:0.3}/10000'::sparsevec
    LIMIT 1;
    IF best_id <> 1 OR abs(score - 0.73) > 0.00001 THEN
        RAISE EXCEPTION 'Unexpected sparse inner-product result: %, %', best_id, score;
    END IF;
END;
$$;

COMMIT;

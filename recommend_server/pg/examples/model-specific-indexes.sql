-- Independent mixed-dimension example; the current store uses vector(n).
-- Run with psql -v ON_ERROR_STOP=1 -f this-file.sql on a lab database.
CREATE SCHEMA model_evolution_demo;
CREATE TABLE model_evolution_demo.embeddings (
    chunk_id bigint NOT NULL,
    model_version text NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (chunk_id, model_version),
    CHECK ((model_version='model-a-v1' AND vector_dims(embedding)=3)
        OR (model_version='model-b-v1' AND vector_dims(embedding)=4))
);

INSERT INTO model_evolution_demo.embeddings VALUES
    (1, 'model-a-v1', '[1,0.1,0]'),
    (2, 'model-a-v1', '[0.1,1,0]'),
    (1, 'model-b-v1', '[0,1,0.1,0]'),
    (2, 'model-b-v1', '[1,0,0,0.1]');

CREATE INDEX model_a_hnsw ON model_evolution_demo.embeddings
USING hnsw ((embedding::vector(3)) vector_cosine_ops)
WHERE model_version='model-a-v1';
CREATE INDEX model_b_hnsw ON model_evolution_demo.embeddings
USING hnsw ((embedding::vector(4)) vector_cosine_ops)
WHERE model_version='model-b-v1';

DO $$ DECLARE best_id bigint; BEGIN
    SELECT chunk_id INTO best_id FROM model_evolution_demo.embeddings
    WHERE model_version='model-a-v1'
    ORDER BY embedding::vector(3) <=> '[1,0,0]'::vector(3) LIMIT 1;
    IF best_id <> 1 THEN RAISE EXCEPTION 'Unexpected model-a result'; END IF;
    SELECT chunk_id INTO best_id FROM model_evolution_demo.embeddings
    WHERE model_version='model-b-v1'
    ORDER BY embedding::vector(4) <=> '[0,1,0,0]'::vector(4) LIMIT 1;
    IF best_id <> 1 THEN RAISE EXCEPTION 'Unexpected model-b result'; END IF;
    BEGIN
        INSERT INTO model_evolution_demo.embeddings VALUES (3,'model-a-v1','[1,0,0,0]');
        RAISE EXCEPTION 'Wrong model dimension was accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
END $$;

ANALYZE model_evolution_demo.embeddings;
-- Match the model predicate and cast to the partial expression index.
SELECT chunk_id FROM model_evolution_demo.embeddings
WHERE model_version='model-a-v1'
ORDER BY embedding::vector(3) <=> '[1,0,0]'::vector(3) LIMIT 10;
DROP SCHEMA model_evolution_demo CASCADE;

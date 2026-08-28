-- 環境の動作確認。
--
-- 「拡張が入っている」ことと「使える状態になっている」ことは別になる。
-- 存在の確認で終えると、索引が使われないまま本番に出る。
\set ON_ERROR_STOP on
\timing off

\echo '=== 1. 拡張 ==='
SELECT extname, extversion FROM pg_extension
WHERE extname IN ('vector', 'pgroonga', 'pg_stat_statements') ORDER BY extname;

\echo ''
\echo '=== 2. 検証用データ ==='
-- 件数を少なくすると、索引があってもプランナが Seq Scan を選ぶ。
-- 索引の確認にならないため、選ばれる規模を用意する。
DROP TABLE IF EXISTS smoke_test;
CREATE TABLE smoke_test (
    id        int PRIMARY KEY,
    content   text NOT NULL,
    embedding vector(3) NOT NULL
);

INSERT INTO smoke_test (id, content, embedding)
SELECT i,
       'sample document ' || i,
       ARRAY[sin(i * 0.7), cos(i * 1.1), sin(i * 0.3)]::vector(3)
FROM generate_series(1, 5000) i;

CREATE INDEX smoke_test_embedding_hnsw
    ON smoke_test USING hnsw (embedding vector_l2_ops);
CREATE INDEX smoke_test_content_pgroonga
    ON smoke_test USING pgroonga (content);
ANALYZE smoke_test;

SELECT count(*) AS 行数 FROM smoke_test;

\echo ''
\echo '=== 3. 索引 ==='
SELECT indexname, am.amname AS 方式
FROM pg_indexes i
JOIN pg_class c ON c.relname = i.indexname
JOIN pg_am am ON am.oid = c.relam
WHERE i.tablename = 'smoke_test' ORDER BY indexname;

\echo ''
\echo '=== 4. ベクトル検索 ==='
SELECT id, content, embedding <-> '[1.0, 0.0, 0.0]' AS l2距離
FROM smoke_test ORDER BY embedding <-> '[1.0, 0.0, 0.0]' LIMIT 3;

\echo ''
\echo '=== 5. 索引が使える状態か ==='
-- 素の EXPLAIN で Seq Scan が出たとき、原因は 2 つに分かれる。
--
--   索引が使えない       opclass が対応しない、索引の作成に失敗している
--   索引を使う価値がない  行数が少ない、絞り込んだ結果が小さい
--
-- 後者は Seq Scan が正しい選択なので、直す対象ではない。
-- enable_seqscan = off で強制し、それでも Index Scan にならなければ前者になる。
-- 素の EXPLAIN だけでは切り分けられない。
SET enable_seqscan = off;
EXPLAIN (COSTS off)
SELECT id FROM smoke_test ORDER BY embedding <-> '[1.0, 0.0, 0.0]' LIMIT 1;
RESET enable_seqscan;

\echo ''
\echo '--- 通常のプランナの選択（参考） ---'
EXPLAIN (COSTS off)
SELECT id FROM smoke_test ORDER BY embedding <-> '[1.0, 0.0, 0.0]' LIMIT 1;

\echo ''
\echo '=== 6. opclass と演算子の対応 ==='
-- 索引はコサインではなく L2 で作ってある。コサイン距離 <=> を投げると
-- 強制しても索引が使えない。結果は正しいまま Seq Scan に落ちるため、
-- 遅くなったことに気づけない。この確認で事前に検出する。
SET enable_seqscan = off;
\echo '--- L2 <-> （索引と一致）---'
EXPLAIN (COSTS off)
SELECT id FROM smoke_test ORDER BY embedding <-> '[1.0, 0.0, 0.0]' LIMIT 1;
\echo '--- コサイン <=> （索引と不一致）---'
EXPLAIN (COSTS off)
SELECT id FROM smoke_test ORDER BY embedding <=> '[1.0, 0.0, 0.0]' LIMIT 1;
RESET enable_seqscan;

\echo ''
\echo '=== 7. PGroonga 全文検索 ==='
SELECT id, content FROM smoke_test WHERE content &@ 'sample' LIMIT 3;

\echo ''
\echo '--- 日本語の検索（PGroonga を入れる理由）---'
INSERT INTO smoke_test (id, content, embedding) VALUES
    (90001, '経費精算の申請締め日は毎月25日です', '[0,0,0]'),
    (90002, 'リモートワークは週3日まで可能です', '[0,0,0]');
-- 語の区切りが空白で示されない日本語では、単純な部分一致では取りこぼす。
SELECT id, content FROM smoke_test WHERE content &@ '経費';

\echo ''
\echo '=== 8. 設定 ==='
SELECT name, setting, unit FROM pg_settings
WHERE name IN ('shared_buffers', 'maintenance_work_mem', 'work_mem',
               'max_parallel_maintenance_workers', 'random_page_cost')
ORDER BY name;

DROP TABLE smoke_test;
\echo ''
\echo '=== 確認完了 ==='

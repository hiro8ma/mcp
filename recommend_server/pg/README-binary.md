# Binary candidate search experiment

This benchmark compares exact cosine search, vector HNSW, and binary HNSW
candidates reranked with the original vectors. It requires **pgvector >= 0.8**.

```bash
python -m pg.bench_binary --tenant demo --category shoes --top-k 10 --oversample 1 4 16 --repeats 5
```

Run from `recommend_server` with its environment. Omit `--category` for all
categories. Omitting `--tenant` explicitly selects the owner connection and a
cross-tenant benchmark. Specifying a tenant selects the application connection
and preserves RLS. Index construction and cleanup use the owner connection.
Only uniquely named indexes created by this invocation are removed; existing
indexes are retained. Run against a lab dataset without concurrent writes so
that ground truth remains comparable across measurements.

Exact queries enable sequential scans and disable index scans and bitmap scans
locally, including existing ANN indexes. All methods apply the same tenant/category conditions. The binary
method materializes candidates and reranks them in one SQL statement and snapshot.
Its output includes actual candidate count, candidate coverage of exact top-k,
final Recall@k, returned count, mean latency, and nearest-rank p95. Recall uses
the actual exact result count when fewer than k eligible rows exist.

Candidate coverage is an upper bound on reranking recall: reranking cannot
recover omitted candidates. Distance ties at the exact top-k boundary can affect
identity-based recall. Increasing `LIMIT` alone does not guarantee enough ANN
candidates. `ef_search` is capped at pgvector's upper bound of 1000 and
`iterative_scan=strict_order` allows further exploration. Scan and memory limits
can still cause underfilled results; inspect the measured count.

Latency excludes connection opening, model encoding, version checks, and settings,
but includes SQL execution, fetch, and candidate diagnostics. The repeated eight
example queries are a controlled experiment, not a production latency estimate.
Inspect execution plans to confirm the planner chooses the HNSW index; small or
highly selective datasets may use exact scans. Both HNSW variants use the same
iterative policy. Building an index does not force its use.

Binary payload uses roughly 1/32 of float32 vector payload. Original vectors are
retained for reranking, and index graph overhead also remains, so neither total
storage nor index size is promised to be 1/32. Reported sizes are individual index
sizes, not total database storage.

Tests use deterministic 3D vectors and connection-local temporary tables/indexes;
no model download or application-table migration is needed. Configure a lab DB
with pgvector installed, then run:

```bash
PG_BINARY_TEST_DSN='<lab DSN>' python -m pytest pg/test_bench_binary.py -q
```

Without that variable, database tests skip explicitly. Tests cover fresh-connection
extension settings, exact ground truth with an existing ANN index, filter isolation
including application-role RLS, a candidate omission that
reranking cannot repair, deletion, small/empty result sets, and candidate requests
beyond the `ef_search` cap.

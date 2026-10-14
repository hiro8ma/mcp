"""Measure binary HNSW candidates and exact cosine reranking (pgvector >= 0.8).

A binary vector uses roughly 1/32 of the float32 payload, but this experiment
retains the original vectors. Neither total storage nor HNSW index size is
therefore promised to shrink by 1/32. Reranking cannot recover omitted candidates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import math
from statistics import mean
import time
import uuid

from psycopg import sql
from psycopg.pq import TransactionStatus

TEXTS = [
    "黒いレザーのブーツを探している", "秋冬に着るウールのコート", "リネンの白いシャツ",
    "ネイビーのデニムパンツ", "レザーのトートバッグ", "オーバーサイズのニット",
    "カーキのナイロンブルゾン", "ベージュのローファー",
]


@dataclass(frozen=True)
class Filters:
    tenant: str | None = None
    category: str | None = None

    def clause(self):
        parts, params = [], []
        for column, value in (("tenant_id", self.tenant), ("category", self.category)):
            if value is not None:
                parts.append(sql.SQL("{} = %s").format(sql.Identifier(column)))
                params.append(value)
        return (sql.SQL(" AND ").join(parts) if parts else sql.SQL("TRUE")), params


@dataclass(frozen=True)
class SearchResult:
    ids: tuple[tuple[str, str], ...]
    elapsed_ms: float
    candidates: tuple[tuple[str, str], ...] | None = None


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def ef_for_candidates(candidate_limit: int) -> int:
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    return min(1000, max(40, candidate_limit * 2))


def require_pgvector(conn) -> None:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not row or tuple(int(p) for p in row[0].split(".")[:2]) < (0, 8):
        raise RuntimeError("binary benchmark requires pgvector >= 0.8 for iterative scans")
    # A fresh backend has not necessarily loaded the extension library yet.
    # Invoke its input function before reading extension-specific GUC defaults.
    # LOAD would require privileges that the application role should not have.
    conn.execute("SELECT vector_dims('[1]'::vector)")


@contextmanager
def local_settings(conn, settings):
    """Restore settings even when the caller already owns a transaction."""
    with conn.transaction():
        previous = {}
        for name, value in settings.items():
            previous[name] = conn.execute("SELECT current_setting(%s)", (name,)).fetchone()[0]
            conn.execute("SELECT set_config(%s, %s, true)", (name, str(value)))
        try:
            yield
        finally:
            # An aborted statement must roll back before more SQL can run.
            # The transaction/savepoint rollback restores local settings itself.
            if conn.info.transaction_status != TransactionStatus.INERROR:
                for name, value in previous.items():
                    conn.execute("SELECT set_config(%s, %s, true)", (name, value))


def eligible_count(conn, filters: Filters = Filters(), *, table: str = "items") -> int:
    where, params = filters.clause()
    return int(conn.execute(sql.SQL("SELECT count(*) FROM {} WHERE {}").format(
        sql.Identifier(table), where), params).fetchone()[0])


def cosine_query(filters: Filters = Filters(), *, table: str = "items", exact: bool = False):
    where, params = filters.clause()
    # Stable ties for exact ground truth; ANN retains a direct distance ORDER BY.
    order = sql.SQL(", tenant_id, item_id") if exact else sql.SQL("")
    query = sql.SQL(
        "SELECT tenant_id, item_id FROM {} WHERE {} "
        "ORDER BY embedding <=> %s::vector{} LIMIT %s"
    ).format(sql.Identifier(table), where, order)
    return query, params


def exact_search(conn, vector, top_k: int, filters: Filters = Filters(), *, table: str = "items"):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query, params = cosine_query(filters, table=table, exact=True)
    # Disable all paths to ANN indexes, including indexes left by other experiments.
    with local_settings(conn, {"enable_seqscan": "on", "enable_indexscan": "off",
                               "enable_bitmapscan": "off"}):
        started = time.perf_counter()
        rows = conn.execute(query, [*params, list(vector), top_k]).fetchall()
        elapsed = (time.perf_counter() - started) * 1000
    return SearchResult(tuple(rows), elapsed)


def hnsw_search(conn, vector, top_k: int, filters: Filters = Filters(), *, table: str = "items"):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    require_pgvector(conn)
    query, params = cosine_query(filters, table=table)
    with local_settings(conn, {"hnsw.ef_search": ef_for_candidates(top_k),
                               "hnsw.iterative_scan": "strict_order"}):
        started = time.perf_counter()
        rows = conn.execute(query, [*params, list(vector), top_k]).fetchall()
        elapsed = (time.perf_counter() - started) * 1000
    return SearchResult(tuple(rows), elapsed)


def two_stage_query(dim: int, filters: Filters = Filters(), *, table: str = "items"):
    if dim <= 0:
        raise ValueError("dim must be positive")
    where, params = filters.clause()
    query = sql.SQL("""
        WITH candidates AS MATERIALIZED (
            SELECT tenant_id, item_id, embedding FROM {table} WHERE {where}
            ORDER BY binary_quantize(embedding)::bit({dim})
                <~> binary_quantize(%s::vector)::bit({dim})
            LIMIT %s
        ), reranked AS MATERIALIZED (
            SELECT tenant_id, item_id FROM candidates
            ORDER BY embedding <=> %s::vector, tenant_id, item_id LIMIT %s
        )
        SELECT
            COALESCE((SELECT jsonb_agg(jsonb_build_array(tenant_id, item_id))
                      FROM candidates), '[]'::jsonb),
            COALESCE((SELECT jsonb_agg(jsonb_build_array(tenant_id, item_id))
                      FROM reranked), '[]'::jsonb)
    """).format(table=sql.Identifier(table), where=where, dim=sql.Literal(dim))
    return query, params


def two_stage_search(conn, vector, top_k: int, candidate_limit: int, dim: int,
                     filters: Filters = Filters(), *, table: str = "items"):
    if top_k <= 0 or candidate_limit < top_k:
        raise ValueError("top_k must be positive and candidate_limit must be >= top_k")
    require_pgvector(conn)
    query, params = two_stage_query(dim, filters, table=table)
    with local_settings(conn, {"hnsw.ef_search": ef_for_candidates(candidate_limit),
                               "hnsw.iterative_scan": "strict_order"}):
        started = time.perf_counter()
        candidates, ids = conn.execute(
            query, [*params, list(vector), candidate_limit, list(vector), top_k]
        ).fetchone()
        elapsed = (time.perf_counter() - started) * 1000
    return SearchResult(tuple(map(tuple, ids)), elapsed, tuple(map(tuple, candidates)))


def recall(truth, got) -> float:
    return len(set(truth) & set(got)) / len(truth) if truth else 1.0


def percentile95(values) -> float:
    """Nearest-rank p95, valid even for small samples."""
    if not values:
        raise ValueError("latency samples must not be empty")
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def summarize(truths, results):
    if not results or len(truths) != len(results):
        raise ValueError("truths and results must have equal nonzero lengths")
    return {
        "recall": mean(recall(t, r.ids) for t, r in zip(truths, results)),
        "candidate_coverage": (mean(recall(t, r.candidates) for t, r in zip(truths, results))
                               if all(r.candidates is not None for r in results) else None),
        "candidate_count": (mean(len(r.candidates) for r in results)
                            if all(r.candidates is not None for r in results) else None),
        "result_count": mean(len(r.ids) for r in results),
        "mean_ms": mean(r.elapsed_ms for r in results),
        "p95_ms": percentile95([r.elapsed_ms for r in results]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--oversample", type=positive_int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--tenant", help="Use app-role RLS; omission explicitly benchmarks all tenants as owner")
    parser.add_argument("--category")
    args = parser.parse_args()
    # Import the model only after validating input and eligible data.
    from . import store

    filters = Filters(args.tenant, args.category)
    def connection():
        return store.connect(args.tenant, owner=args.tenant is None)

    with connection() as conn:
        require_pgvector(conn)
        count = eligible_count(conn, filters)
    if not count:
        raise SystemExit("No eligible items for the supplied filters; nothing to benchmark")
    from . import embed
    dim = embed.dimension()
    vectors = embed.encode(TEXTS)
    truths = []
    for vector in vectors:
        with connection() as conn:
            result = exact_search(conn, vector, args.top_k, filters)
        truths.append(result.ids)

    # Unique names and cleanup only for indexes created by this invocation.
    suffix = uuid.uuid4().hex[:12]
    created = []
    try:
        print(f"{count} eligible rows / {dim} dimensions / top-{args.top_k}")
        print("Access: app-role RLS" if args.tenant is not None else "Access: owner, cross-tenant benchmark")
        for kind, expression in (
            ("vector", sql.SQL("embedding vector_cosine_ops")),
            ("binary", sql.SQL("(binary_quantize(embedding)::bit({})) bit_hamming_ops").format(sql.Literal(dim))),
        ):
            name = f"binary_bench_{kind}_{suffix}"
            with store.connect(owner=True) as conn:
                started = time.perf_counter()
                conn.execute(sql.SQL("CREATE INDEX {} ON items USING hnsw ({})").format(
                    sql.Identifier(name), expression))
                conn.commit()
                created.append(name)
                seconds = time.perf_counter() - started
                size = conn.execute("SELECT pg_relation_size(%s::regclass)", (name,)).fetchone()[0]
            print(f"{kind} index: {size} bytes / build {seconds:.2f}s")

        def report(label, results, repeated_truths):
            stats = summarize(repeated_truths, results)
            coverage = stats["candidate_coverage"]
            candidates = stats["candidate_count"]
            print(f"{label}: candidates={candidates if candidates is not None else '-'} "
                  f"coverage={coverage if coverage is not None else '-'} "
                  f"returned={stats['result_count']:.1f} Recall@k={stats['recall']:.3f} "
                  f"mean={stats['mean_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms")

        repeated_truths = truths * args.repeats
        repeated_vectors = list(vectors) * args.repeats
        exact_samples = []
        for vector in repeated_vectors:
            with connection() as conn:
                exact_samples.append(exact_search(conn, vector, args.top_k, filters))
        report("Exact", exact_samples, repeated_truths)
        baseline = []
        for vector in repeated_vectors:
            with connection() as conn:
                baseline.append(hnsw_search(conn, vector, args.top_k, filters))
        report("Vector HNSW (strict iterative)", baseline, repeated_truths)
        for oversample in args.oversample:
            limit = args.top_k * oversample
            results = []
            for vector in repeated_vectors:
                with connection() as conn:
                    results.append(two_stage_search(conn, vector, args.top_k, limit, dim, filters))
            report(f"Binary candidates={limit}, ef={ef_for_candidates(limit)}", results, repeated_truths)
        print("Candidate coverage is the reranking recall upper bound; absent candidates cannot be recovered.")
        print("Iterative scans remain bounded by pgvector scan/memory limits; inspect actual candidate counts.")
        print("Timings exclude connection/model/settings work and include candidate diagnostics; inspect plans to confirm HNSW use.")
    finally:
        for name in created:
            with store.connect(owner=True) as conn:
                conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(name)))
                conn.commit()


if __name__ == "__main__":
    main()

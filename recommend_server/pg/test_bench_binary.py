"""Deterministic binary-search integration tests; no embedding model required.

Set PG_BINARY_TEST_DSN to an expendable PostgreSQL database with pgvector >= 0.8.
All test data and indexes live in connection-local temporary tables.
"""

from __future__ import annotations

import argparse
import os
import uuid

import psycopg
from pgvector.psycopg import register_vector
import pytest

from . import bench_binary as bench

QUERY = [1.0, 0.1, 0.1]


@pytest.fixture
def db():
    dsn = os.getenv("PG_BINARY_TEST_DSN")
    if not dsn:
        pytest.skip("PG_BINARY_TEST_DSN is required for PostgreSQL integration tests")
    with psycopg.connect(dsn) as conn:
        bench.require_pgvector(conn)
        register_vector(conn)
        conn.execute("""
            CREATE TEMP TABLE binary_test_items (
                tenant_id text NOT NULL, item_id text NOT NULL, category text,
                embedding vector(3) NOT NULL
            )
        """)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO binary_test_items VALUES (%s, %s, %s, %s)",
                [
                    # The closest eligible vector has one different sign bit.
                    ("a", "nearest", "keep", [1.0, -0.01, 0.1]),
                    ("a", "hamming-first", "keep", [0.01, 1.0, 1.0]),
                    ("a", "other-category", "drop", QUERY),
                    ("b", "other-tenant", "keep", QUERY),
                ],
            )
        conn.execute("CREATE INDEX binary_test_vec_idx ON binary_test_items USING hnsw (embedding vector_cosine_ops)")
        conn.execute("CREATE INDEX binary_test_bit_idx ON binary_test_items USING hnsw ((binary_quantize(embedding)::bit(3)) bit_hamming_ops)")
        conn.commit()
        yield conn


def test_validation_and_summary():
    for value in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            bench.positive_int(value)
    assert bench.ef_for_candidates(800) == 1000
    truths = [(('a', 'one'), ('a', 'two'))]
    result = bench.SearchResult((('a', 'one'),), 4.0, (('a', 'one'),))
    stats = bench.summarize(truths, [result])
    assert stats["recall"] == stats["candidate_coverage"] == 0.5
    assert stats["candidate_count"] == stats["result_count"] == 1
    assert bench.percentile95(range(1, 21)) == 19
    assert bench.recall([], []) == 1.0


def test_fresh_connection_loads_hnsw_settings(db):
    with psycopg.connect(os.environ["PG_BINARY_TEST_DSN"]) as fresh:
        bench.require_pgvector(fresh)
        before = fresh.execute("SHOW hnsw.ef_search").fetchone()[0]
        with bench.local_settings(fresh, {"hnsw.ef_search": 100,
                                          "hnsw.iterative_scan": "strict_order"}):
            assert fresh.execute("SHOW hnsw.ef_search").fetchone()[0] == "100"
        assert fresh.execute("SHOW hnsw.ef_search").fetchone()[0] == before


def test_exact_ignores_existing_ann_and_restores_settings(db):
    filters = bench.Filters("a", "keep")
    db.execute("SET enable_seqscan = off")
    old = db.execute("SHOW enable_indexscan").fetchone()[0]
    with bench.local_settings(db, {"enable_seqscan": "on", "enable_indexscan": "off",
                                   "enable_bitmapscan": "off"}):
        query, params = bench.cosine_query(filters, table="binary_test_items", exact=True)
        from psycopg import sql
        plan = db.execute(sql.SQL("EXPLAIN (FORMAT JSON) ") + query,
                          [*params, QUERY, 1]).fetchone()[0][0]["Plan"]
        def nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from nodes(child)
        assert any(n["Node Type"] == "Seq Scan" for n in nodes(plan))
        assert not any("Index Name" in n for n in nodes(plan))
    truth = bench.exact_search(db, QUERY, 1, filters, table="binary_test_items")
    assert truth.ids == (("a", "nearest"),)
    assert db.execute("SHOW enable_indexscan").fetchone()[0] == old
    assert db.execute("SHOW enable_seqscan").fetchone()[0] == "off"


def test_candidates_bound_rerank_recall_and_preserve_filters(db):
    filters = bench.Filters("a", "keep")
    # Make the indexed path observable rather than accidentally testing a seq scan.
    db.execute("SET enable_seqscan = off")
    db.execute("SET enable_bitmapscan = off")
    truth = bench.exact_search(db, QUERY, 1, filters, table="binary_test_items")
    narrow = bench.two_stage_search(db, QUERY, 1, 1, 3, filters, table="binary_test_items")
    assert narrow.candidates == (("a", "hamming-first"),)
    assert narrow.ids == narrow.candidates
    assert bench.recall(truth.ids, narrow.ids) == 0.0
    wide = bench.two_stage_search(db, QUERY, 1, 2, 3, filters, table="binary_test_items")
    assert set(wide.candidates) == {("a", "nearest"), ("a", "hamming-first")}
    assert wide.ids == truth.ids
    assert bench.recall(truth.ids, wide.ids) <= bench.recall(truth.ids, wide.candidates)
    assert len(wide.ids) <= 1 and len(wide.candidates) <= 2


def test_deleted_rows_and_small_eligible_sets(db):
    filters = bench.Filters("a", "keep")
    db.execute("DELETE FROM binary_test_items WHERE item_id = 'nearest'")
    truth = bench.exact_search(db, QUERY, 10, filters, table="binary_test_items")
    result = bench.two_stage_search(db, QUERY, 10, 20, 3, filters, table="binary_test_items")
    assert truth.ids == result.ids == (("a", "hamming-first"),)
    assert result.candidates == truth.ids
    assert bench.recall(truth.ids, result.ids) == 1.0
    empty = bench.Filters("absent", "keep")
    assert bench.eligible_count(db, empty, table="binary_test_items") == 0
    assert bench.two_stage_search(db, QUERY, 10, 20, 3, empty,
                                  table="binary_test_items").candidates == ()


def test_large_candidate_request_respects_cap_and_reports_actual_count(db):
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO binary_test_items VALUES (%s, %s, %s, %s)",
            [("many", f"item-{i:04d}", "keep", [1.0, 0.1 + i / 2000, 0.2])
             for i in range(1100)],
        )
    db.execute("ANALYZE binary_test_items")
    db.execute("SET enable_seqscan = off")
    db.execute("SET enable_bitmapscan = off")
    result = bench.two_stage_search(db, QUERY, 10, 1100, 3, bench.Filters("many"),
                                    table="binary_test_items")
    # All rows share the same bit pattern; even iterative scans can underfill.
    # Requests above the GUC maximum must remain valid, without claiming recall.
    assert bench.ef_for_candidates(1100) == 1000
    assert 0 < len(result.candidates) <= bench.eligible_count(db, bench.Filters("many"), table="binary_test_items")
    assert len(result.ids) == 10
    assert all(tenant == "many" for tenant, _ in result.candidates)


def test_failed_statement_preserves_error_and_restores_local_settings(db):
    before = db.execute("SHOW enable_indexscan").fetchone()[0]
    with pytest.raises(psycopg.errors.DivisionByZero):
        with bench.local_settings(db, {"enable_indexscan": "off"}):
            db.execute("SELECT 1 / 0")
    assert db.execute("SHOW enable_indexscan").fetchone()[0] == before


def test_app_role_rls_applies_to_truth_and_binary_candidates(db):
    from psycopg import sql

    can_create_role = db.execute(
        "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname=current_user"
    ).fetchone()[0]
    if not can_create_role:
        pytest.skip("RLS role test requires CREATEROLE in the disposable test DB")
    role = "binary_test_" + uuid.uuid4().hex[:12]
    identifier = sql.Identifier(role)
    db.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(identifier))
    try:
        db.execute("ALTER TABLE binary_test_items ENABLE ROW LEVEL SECURITY")
        db.execute("ALTER TABLE binary_test_items FORCE ROW LEVEL SECURITY")
        db.execute("CREATE POLICY binary_test_tenant ON binary_test_items "
                   "USING (tenant_id = current_setting('app.tenant_id', true))")
        db.execute(sql.SQL("GRANT SELECT ON binary_test_items TO {}").format(identifier))
        with db.transaction():
            db.execute(sql.SQL("SET LOCAL ROLE {}").format(identifier))
            db.execute("SELECT set_config('app.tenant_id', 'a', true)")
            filters = bench.Filters(category="keep")  # RLS supplies tenant isolation.
            truth = bench.exact_search(db, QUERY, 10, filters, table="binary_test_items")
            result = bench.two_stage_search(db, QUERY, 10, 20, 3, filters,
                                            table="binary_test_items")
            assert set(truth.ids) == {("a", "nearest"), ("a", "hamming-first")}
            assert set(result.ids) == set(result.candidates) == set(truth.ids)
            db.execute("RESET ROLE")
    finally:
        db.execute(sql.SQL("REVOKE SELECT ON binary_test_items FROM {}").format(identifier))
        db.execute(sql.SQL("DROP ROLE {}").format(identifier))

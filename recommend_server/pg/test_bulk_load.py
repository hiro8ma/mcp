"""Binary COPY integration tests on a disposable pgvector database."""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
import pytest

from .bulk_load import load_initial_items
from .store import Item, _to_tuple


@pytest.fixture
def db():
    dsn = os.getenv("PG_BINARY_TEST_DSN")
    if not dsn:
        pytest.skip("PG_BINARY_TEST_DSN must identify a disposable pgvector database")
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("""
            CREATE TEMP TABLE bulk_test_items (
                tenant_id text NOT NULL, item_id text NOT NULL,
                title text NOT NULL, description text NOT NULL DEFAULT '',
                category text, tags text[] NOT NULL DEFAULT '{}',
                embedding vector(3) NOT NULL,
                PRIMARY KEY (tenant_id,item_id)
            )
        """)
        yield conn


def test_binary_copy_round_trip_and_post_load_index(db):
    vectors = [np.array([0.3, -0.2, 0.8], dtype=np.float32),
               np.array([-0.1, 0.4, 0.9], dtype=np.float32)]
    rows = ((Item(str(i), f"商品 {i}", category=None if i == 0 else "manual",
                  tags=("日本語", "example")), v) for i, v in enumerate(vectors))
    original_memory = db.execute("SHOW maintenance_work_mem").fetchone()[0]
    result = load_initial_items(db, "tenant-a", rows, table="bulk_test_items", build_index=True)
    assert result.rows == 2 and result.index_name
    data = db.execute("SELECT tenant_id,item_id,category,tags,embedding FROM bulk_test_items ORDER BY item_id").fetchall()
    assert len(data) == 2 and all(row[0] == "tenant-a" for row in data)
    assert data[0][2] is None and data[0][3] == ["日本語", "example"]
    for row, vector in zip(data, vectors):
        np.testing.assert_array_equal(np.asarray(_to_tuple(row[4]), dtype=np.float32), vector)
    index = db.execute("SELECT i.indisvalid, am.amname FROM pg_index i "
                       "JOIN pg_class c ON c.oid=i.indexrelid JOIN pg_am am ON am.oid=c.relam "
                       "WHERE i.indexrelid=%s::regclass", (result.index_name,)).fetchone()
    assert index == (True, "hnsw")
    assert db.execute("SELECT reltuples FROM pg_class WHERE oid='bulk_test_items'::regclass").fetchone()[0] == 2
    assert db.execute("SHOW maintenance_work_mem").fetchone()[0] == original_memory


@pytest.mark.parametrize("bad", [[1.0, 2.0], [float('nan'), 1.0, 0.0], [0.0, 0.0, 0.0]])
def test_invalid_later_row_rolls_back_entire_copy(db, bad):
    rows = [(Item("ok", "valid"), [1.0, 0.0, 0.0]), (Item("bad", "invalid"), bad)]
    with pytest.raises(ValueError):
        load_initial_items(db, "tenant-a", iter(rows), table="bulk_test_items", build_index=True)
    assert db.execute("SELECT count(*) FROM bulk_test_items").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM pg_indexes WHERE tablename='bulk_test_items' AND indexdef LIKE '%hnsw%'").fetchone()[0] == 0


def test_duplicate_and_empty_inputs_are_atomic(db):
    item = Item("duplicate", "item")
    with pytest.raises(psycopg.errors.UniqueViolation):
        load_initial_items(db, "tenant-a", [(item, [1, 0, 0]), (item, [0, 1, 0])], table="bulk_test_items")
    assert db.execute("SELECT count(*) FROM bulk_test_items").fetchone()[0] == 0
    with pytest.raises(ValueError, match="at least one"):
        load_initial_items(db, "tenant-a", [], table="bulk_test_items")


def test_existing_rows_and_ann_indexes_are_preserved(db):
    row = [(Item("first", "item"), [1, 0, 0])]
    load_initial_items(db, "tenant-a", row, table="bulk_test_items")
    with pytest.raises(ValueError, match="empty table"):
        load_initial_items(db, "tenant-b", row, table="bulk_test_items")
    assert db.execute("SELECT tenant_id FROM bulk_test_items").fetchall() == [("tenant-a",)]
    db.execute("DELETE FROM bulk_test_items")
    db.execute("CREATE INDEX bulk_existing_hnsw ON bulk_test_items USING hnsw (embedding vector_cosine_ops)")
    with pytest.raises(ValueError, match="existing ANN index"):
        load_initial_items(db, "tenant-a", row, table="bulk_test_items")
    assert db.execute("SELECT to_regclass('bulk_existing_hnsw') IS NOT NULL").fetchone()[0]


def test_post_copy_build_failure_rolls_back_loaded_data(db):
    with pytest.raises(psycopg.Error, match="maintenance_work_mem"):
        load_initial_items(db, "tenant-a", [(Item("item", "item"), [1, 0, 0])],
                           table="bulk_test_items", build_index=True,
                           maintenance_work_mem="not-a-memory-size")
    assert db.execute("SELECT count(*) FROM bulk_test_items").fetchone()[0] == 0


def test_caller_transaction_retains_commit_control(db):
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction():
            result = load_initial_items(db, "tenant-a", [(Item("item", "item"), [1, 0, 0])],
                                        table="bulk_test_items", build_index=True)
            assert db.execute("SELECT count(*) FROM bulk_test_items").fetchone()[0] == 1
            raise RuntimeError("caller rollback")
    assert db.execute("SELECT count(*) FROM bulk_test_items").fetchone()[0] == 0
    assert db.execute("SELECT to_regclass(%s)", (result.index_name,)).fetchone()[0] is None


def test_forced_rls_rejects_owner_without_bypass(db):
    if not db.execute("SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()[0]:
        pytest.skip("CREATEROLE required for the RLS regression test")
    role = "bulk_test_" + uuid.uuid4().hex[:12]
    identifier = sql.Identifier(role)
    db.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(identifier))
    try:
        db.execute("ALTER TABLE bulk_test_items ENABLE ROW LEVEL SECURITY")
        db.execute("ALTER TABLE bulk_test_items FORCE ROW LEVEL SECURITY")
        db.execute(sql.SQL("ALTER TABLE bulk_test_items OWNER TO {}").format(identifier))
        with db.transaction():
            db.execute(sql.SQL("SET LOCAL ROLE {}").format(identifier))
            with pytest.raises(PermissionError, match="RLS"):
                load_initial_items(db, "tenant-a", [(Item("no", "item"), [1, 0, 0])], table="bulk_test_items")
            db.execute("RESET ROLE")
        assert db.execute("SELECT relrowsecurity AND relforcerowsecurity FROM pg_class "
                          "WHERE oid='bulk_test_items'::regclass").fetchone()[0]
    finally:
        db.execute("DROP TABLE bulk_test_items")
        db.execute(sql.SQL("DROP ROLE {}").format(identifier))

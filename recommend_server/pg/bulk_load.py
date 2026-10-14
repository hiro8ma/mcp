"""Administrative initial loading with binary COPY; never an application UPSERT.

Use an empty, unpartitioned items-shaped table without ANN indexes. Existing
application data, policies and indexes are retained. COPY requires a role to
which row-level security does not apply. Timings include input iteration and
validation; embedding inference should be measured separately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import time
from typing import Iterable, Sequence, TYPE_CHECKING
import uuid

import numpy as np
import psycopg
from psycopg import sql
from pgvector import Vector
from pgvector.psycopg import register_vector

if TYPE_CHECKING:
    from .store import Item


@dataclass(frozen=True)
class LoadResult:
    rows: int
    copy_seconds: float
    index_seconds: float
    analyze_seconds: float
    index_name: str | None


def load_initial_items(
    conn: psycopg.Connection,
    tenant_id: str,
    rows: Iterable[tuple[Item, Sequence[float]]],
    *,
    table: str = "items",
    build_index: bool = False,
    maintenance_work_mem: str = "64MB",
    parallel_workers: int = 0,
) -> LoadResult:
    """Load one tenant into an empty items-shaped table, atomically.

    Own a transaction (or a savepoint if the caller owns one). Hold a write lock
    through COPY/index/ANALYZE to make the empty-table check race-free. This is
    an offline initial-load operation, not online maintenance. A caller-owned
    outer transaction still needs to commit after this function returns.
    """
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be nonempty")
    if parallel_workers < 0:
        raise ValueError("parallel_workers must be nonnegative")
    identifier = sql.Identifier(table)
    with conn.transaction():
        register_vector(conn)
        schema = conn.execute(
            "SELECT c.relkind, a.atttypid = 'vector'::regtype, a.atttypmod, "
            "row_security_active(c.oid) "
            "FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid "
            "WHERE c.oid = %s::regclass AND a.attname = 'embedding' "
            "AND NOT a.attisdropped", (table,),
        ).fetchone()
        if not schema or schema[0] != "r" or not schema[1] or schema[2] <= 0:
            raise ValueError("initial load requires an unpartitioned table with embedding vector(n)")
        if schema[3]:
            raise PermissionError("COPY FROM is unsupported when RLS applies; use an exempt administrative role or staging plus INSERT")
        dimension = schema[2]
        if build_index and dimension > 2000:
            raise ValueError("vector HNSW supports at most 2000 dimensions; select an indexing representation first")
        # Preserve policies and every existing index. Serialize initial writers.
        conn.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(identifier))
        if conn.execute(sql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(identifier)).fetchone()[0]:
            raise ValueError("initial load requires an empty table; use UPSERT or staging for existing data")
        if conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
            "JOIN pg_am am ON am.oid=c.relam WHERE i.indrelid=%s::regclass "
            "AND am.amname IN ('hnsw','ivfflat'))", (table,),
        ).fetchone()[0]:
            raise ValueError("initial load requires no existing ANN index; prepare a new table instead of dropping live indexes")

        count = 0
        started = time.perf_counter()
        statement = sql.SQL(
            "COPY {} (tenant_id,item_id,title,description,category,tags,embedding) "
            "FROM STDIN WITH (FORMAT BINARY)"
        ).format(identifier)
        with conn.cursor() as cur:
            with cur.copy(statement) as copy:
                copy.set_types(["text", "text", "text", "text", "text", "text[]", "vector"])
                for item, embedding in rows:
                    vector = np.asarray(embedding, dtype=np.float32)
                    if vector.ndim != 1 or len(vector) != dimension:
                        raise ValueError(f"row {count}: expected {dimension} dimensions")
                    if not np.isfinite(vector).all() or not np.any(vector != 0):
                        raise ValueError(f"row {count}: embedding must be finite and nonzero for cosine retrieval")
                    copy.write_row((tenant_id, item.item_id, item.title, item.description,
                                    item.category, list(item.tags), Vector(vector)))
                    count += 1
        copy_seconds = time.perf_counter() - started
        if not count:
            raise ValueError("initial load requires at least one row")

        index_name, index_seconds = None, 0.0
        if build_index:
            conn.execute("SELECT set_config('maintenance_work_mem', %s, true)", (maintenance_work_mem,))
            conn.execute("SELECT set_config('max_parallel_maintenance_workers', %s, true)",
                         (str(parallel_workers),))
            index_name = "bulk_hnsw_" + uuid.uuid4().hex[:12]
            started = time.perf_counter()
            conn.execute(sql.SQL(
                "CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops) "
                "WITH (m=16, ef_construction=64)"
            ).format(sql.Identifier(index_name), identifier))
            index_seconds = time.perf_counter() - started
        started = time.perf_counter()
        conn.execute(sql.SQL("ANALYZE {}").format(identifier))
        analyze_seconds = time.perf_counter() - started
    return LoadResult(count, copy_seconds, index_seconds, analyze_seconds, index_name)


def synthetic_rows(n: int, dimension: int):
    """Deterministic lab data; these vectors do not encode the item text."""
    from .store import Item

    rng = np.random.default_rng(42)
    for index in range(n):
        vector = rng.standard_normal(dimension).astype(np.float32)
        vector /= np.linalg.norm(vector)
        yield Item(item_id=f"bulk-{index}", title=f"Synthetic item {index}"), vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True,
                        help="Acknowledge that generated vectors are lab data, not model embeddings")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--table", default="items")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--maintenance-work-mem", default="64MB")
    parser.add_argument("--parallel-workers", type=int, default=0)
    args = parser.parse_args()
    if args.rows <= 0 or args.dimension <= 0:
        parser.error("rows and dimension must be positive")
    dsn = os.environ.get("PG_OWNER_DSN")
    if not dsn:
        parser.error("PG_OWNER_DSN must identify the administrative lab database")
    with psycopg.connect(dsn, autocommit=True) as conn:
        result = load_initial_items(
            conn, args.tenant, synthetic_rows(args.rows, args.dimension),
            table=args.table, build_index=args.build_index,
            maintenance_work_mem=args.maintenance_work_mem,
            parallel_workers=args.parallel_workers,
        )
    print(f"loaded={result.rows} COPY={result.copy_seconds:.3f}s "
          f"index={result.index_seconds:.3f}s ANALYZE={result.analyze_seconds:.3f}s "
          f"index_name={result.index_name or '-'}")


if __name__ == "__main__":
    main()

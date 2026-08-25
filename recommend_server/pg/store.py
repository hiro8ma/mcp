"""pgvector を使ったアイテムストア。

ベクトル検索の索引は HNSW と IVFFlat を切り替えられる。
どちらも近似最近傍なので、再現率と速度のトレードオフを測るために両方を残している。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector

DSN = os.getenv("PG_DSN", "postgresql://postgres:lab@localhost:5433/ixlab")
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str
    description: str = ""
    category: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    item_id: str
    title: str
    category: str | None
    similarity: float


@contextmanager
def connect(tenant_id: str | None = None) -> Iterator[psycopg.Connection]:
    """接続を開き、テナントを指定した場合は RLS 用のセッション変数を立てる。"""
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        if tenant_id is not None:
            # SET LOCAL ではなく SET を使う。コネクション単位で有効にしたいため。
            conn.execute("SELECT set_config('app.tenant_id', %s, false)", (tenant_id,))
        yield conn


def init_schema() -> None:
    schema = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema, encoding="utf-8") as f:
        sql = f.read()
    with connect() as conn:
        conn.execute(sql)
        conn.commit()


def upsert(tenant_id: str, items: Sequence[Item], embeddings: Sequence[Sequence[float]]) -> int:
    """アイテムを一括で登録または更新する。"""
    if len(items) != len(embeddings):
        raise ValueError("items と embeddings の長さが一致しません")

    rows = [
        (tenant_id, it.item_id, it.title, it.description, it.category, list(it.tags), list(vec))
        for it, vec in zip(items, embeddings)
    ]
    with connect(tenant_id) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO items (tenant_id, item_id, title, description, category, tags, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, item_id) DO UPDATE SET
                    title       = EXCLUDED.title,
                    description = EXCLUDED.description,
                    category    = EXCLUDED.category,
                    tags        = EXCLUDED.tags,
                    embedding   = EXCLUDED.embedding,
                    updated_at  = now()
                """,
                rows,
            )
        conn.commit()
    return len(rows)


def search(
    tenant_id: str,
    embedding: Sequence[float],
    top_k: int = 100,
    category: str | None = None,
    exclude_item_id: str | None = None,
) -> list[Hit]:
    """コサイン距離で近傍を取る。

    `<=>` はコサイン距離なので、類似度は 1 から引いて求める。
    """
    sql = [
        "SELECT item_id, title, category, 1 - (embedding <=> %s::vector) AS similarity",
        "FROM items",
        "WHERE tenant_id = %s",
    ]
    params: list[object] = [list(embedding), tenant_id]

    if category is not None:
        sql.append("AND category = %s")
        params.append(category)
    if exclude_item_id is not None:
        sql.append("AND item_id <> %s")
        params.append(exclude_item_id)

    sql.append("ORDER BY embedding <=> %s::vector")
    params.append(list(embedding))
    sql.append("LIMIT %s")
    params.append(top_k)

    with connect(tenant_id) as conn:
        rows = conn.execute("\n".join(sql), params).fetchall()

    return [Hit(item_id=r[0], title=r[1], category=r[2], similarity=round(float(r[3]), 4)) for r in rows]


def get_embedding(tenant_id: str, item_id: str) -> list[float] | None:
    with connect(tenant_id) as conn:
        row = conn.execute(
            "SELECT embedding FROM items WHERE tenant_id = %s AND item_id = %s",
            (tenant_id, item_id),
        ).fetchone()
    return list(row[0]) if row else None


def create_index(kind: str, *, lists: int = 100, m: int = 16, ef_construction: int = 64) -> None:
    """索引を張り直す。kind は "hnsw" か "ivfflat" か "none"。"""
    with connect() as conn:
        conn.execute("DROP INDEX IF EXISTS items_embedding_idx")
        if kind == "hnsw":
            conn.execute(
                f"CREATE INDEX items_embedding_idx ON items "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {m}, ef_construction = {ef_construction})"
            )
        elif kind == "ivfflat":
            conn.execute(
                f"CREATE INDEX items_embedding_idx ON items "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            )
        elif kind != "none":
            raise ValueError(f"未知の索引種別: {kind}")
        conn.commit()


def count(tenant_id: str | None = None) -> int:
    with connect(tenant_id) as conn:
        if tenant_id is None:
            row = conn.execute("SELECT count(*) FROM items").fetchone()
        else:
            row = conn.execute("SELECT count(*) FROM items WHERE tenant_id = %s", (tenant_id,)).fetchone()
    return int(row[0])


def tenants() -> list[str]:
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT tenant_id FROM items ORDER BY tenant_id").fetchall()
    return [r[0] for r in rows]

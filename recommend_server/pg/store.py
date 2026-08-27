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

# 接続は 2 系統に分ける。
#
# OWNER_DSN はスキーマ作成と索引の張り替え用。テーブル所有者の権限が要る。
# APP_DSN は検索と投入用で、非特権ロールを使う。
#
# 分けるのは Row Level Security の都合による。RLS は superuser とテーブル所有者を
# バイパスするため、所有者で接続したままだとテナント分離が働かない。
# 所有者の接続を管理操作だけに閉じ込めることで、通常の読み書きが必ず RLS を通る。
OWNER_DSN = os.getenv("PG_OWNER_DSN", "postgresql://postgres:lab@localhost:15433/ixlab")
APP_DSN = os.getenv("PG_APP_DSN", "postgresql://app_user:lab@localhost:15433/ixlab")


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
    # クラスタリングのように埋め込みを必要とする処理向け。
    # 既定では載せない。1 件あたり数百次元あり、返す件数が多いと無駄が大きい。
    embedding: tuple[float, ...] | None = None


def _to_tuple(v) -> tuple[float, ...]:
    for attr in ("to_list", "tolist"):
        fn = getattr(v, attr, None)
        if callable(fn):
            return tuple(fn())
    return tuple(v)


@contextmanager
def connect(tenant_id: str | None = None, *, owner: bool = False) -> Iterator[psycopg.Connection]:
    """接続を開き、テナントを指定した場合は RLS 用のセッション変数を立てる。

    owner=True はスキーマ作成と索引の張り替え専用。所有者権限が要る操作に限る。
    通常の読み書きは非特権ロールで接続し、RLS を通す。
    """
    with psycopg.connect(OWNER_DSN if owner else APP_DSN) as conn:
        register_vector(conn)
        if tenant_id is not None:
            # 第 3 引数 false は「トランザクションではなくセッションに設定する」意味。
            # ただしこの接続は with を抜けた時点で閉じるため、実質は 1 呼び出しの寿命しかない。
            # 接続をプールして使い回す形にするなら、ここの選択が意味を持つ。
            conn.execute("SELECT set_config('app.tenant_id', %s, false)", (tenant_id,))
        yield conn


def init_schema() -> None:
    """スキーマとアプリ用ロールを作る。所有者権限で実行する。

    ベクトルの次元は埋め込みモデルから引く。DDL に数字を直書きすると、
    モデルを差し替えたときに投入時まで不整合に気づけない。
    """
    from . import embed

    here = os.path.dirname(__file__)
    with connect(owner=True) as conn:
        with open(os.path.join(here, "schema.sql"), encoding="utf-8") as f:
            conn.execute(f.read().replace("{{DIM}}", str(embed.dimension())))
        with open(os.path.join(here, "setup_role.sql"), encoding="utf-8") as f:
            conn.execute(f.read())
        # 行動履歴は setup_role のあとに作る。GRANT が app_user の存在を前提にする。
        with open(os.path.join(here, "schema_interactions.sql"), encoding="utf-8") as f:
            conn.execute(f.read())
        conn.commit()
    assert_dimension_matches()


def assert_dimension_matches() -> None:
    """既存テーブルの次元と、いま使うモデルの次元が一致することを確かめる。

    次元が同じ別モデルに差し替えた場合はここでは検出できない。
    そちらは items.embedding_model に記録して照合する。
    """
    from . import embed

    want = embed.dimension()
    with connect(owner=True) as conn:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'items'::regclass AND attname = 'embedding'"
        ).fetchone()
    if row and row[0] not in (-1, want):
        raise RuntimeError(
            f"既存テーブルの次元 {row[0]} と、モデル {embed.MODEL_NAME} の次元 {want} が食い違う。"
            f"テーブルを作り直すか、モデルを戻す必要がある。"
        )


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
    with_embeddings: bool = False,
) -> list[Hit]:
    """コサイン距離で近傍を取る。

    `<=>` はコサイン距離なので、類似度は 1 から引いて求める。

    with_embeddings=True にすると各行の埋め込みも返す。
    このクエリが読んだ行にすでに載っているものなので、
    呼び出し元が 1 件ずつ取り直すより 1 クエリで済む。
    """
    cols = "item_id, title, category, 1 - (embedding <=> %s::vector) AS similarity"
    if with_embeddings:
        cols += ", embedding"

    sql = [
        f"SELECT {cols}",
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

    return [
        Hit(
            item_id=r[0],
            title=r[1],
            category=r[2],
            similarity=round(float(r[3]), 4),
            # pgvector が返す Vector は直接反復できないため list に落とす。
            # numpy 配列で返る場合もあるので tolist / to_list の両方を見る。
            embedding=_to_tuple(r[4]) if with_embeddings else None,
        )
        for r in rows
    ]


def get_embedding(tenant_id: str, item_id: str) -> list[float] | None:
    with connect(tenant_id) as conn:
        row = conn.execute(
            "SELECT embedding FROM items WHERE tenant_id = %s AND item_id = %s",
            (tenant_id, item_id),
        ).fetchone()
    # pgvector が返す Vector は list() で直接展開できない。
    # search() と同じ _to_tuple を通す。
    return list(_to_tuple(row[0])) if row else None


def get_embeddings(tenant_id: str, item_ids: Sequence[str]) -> dict[str, list[float]]:
    """複数アイテムの埋め込みをまとめて引く。

    履歴からユーザープロファイルを作る用途では 1 ユーザーあたり数十件を読む。
    1 件ずつ問い合わせると接続の開閉がその回数だけ発生する。
    """
    if not item_ids:
        return {}

    with connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, embedding FROM items WHERE tenant_id = %s AND item_id = ANY(%s)",
            (tenant_id, list(item_ids)),
        ).fetchall()
    return {r[0]: list(_to_tuple(r[1])) for r in rows}


def count(tenant_id: str | None = None) -> int:
    if tenant_id is None:
        # 全テナント横断の件数は RLS を通せないため所有者権限で読む。
        with connect(owner=True) as conn:
            return int(conn.execute("SELECT count(*) FROM items").fetchone()[0])

    with connect(tenant_id) as conn:
        row = conn.execute("SELECT count(*) FROM items WHERE tenant_id = %s", (tenant_id,)).fetchone()
    return int(row[0])


def tenants() -> list[str]:
    """全テナントの一覧。RLS を通さないため所有者権限で読む。"""
    with connect(owner=True) as conn:
        rows = conn.execute("SELECT DISTINCT tenant_id FROM items ORDER BY tenant_id").fetchall()
    return [r[0] for r in rows]

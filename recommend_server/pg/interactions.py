"""行動履歴の読み書き。協調フィルタリングの入力になる。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import store


@dataclass(frozen=True)
class Event:
    user_id: str
    item_id: str
    rating: float = 1.0


def record(tenant_id: str, events: Sequence[Event]) -> int:
    """行動を一括で記録する。同じ組み合わせは上書きする。"""
    if not events:
        return 0

    rows = [(tenant_id, e.user_id, e.item_id, e.rating) for e in events]
    with store.connect(tenant_id) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO interactions (tenant_id, user_id, item_id, rating) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, user_id, item_id) DO UPDATE SET rating = EXCLUDED.rating",
                rows,
            )
        conn.commit()
    return len(rows)


def history(tenant_id: str, user_id: str) -> dict[str, float]:
    """1 ユーザーの履歴を {item_id: rating} で返す。"""
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, rating FROM interactions WHERE tenant_id = %s AND user_id = %s",
            (tenant_id, user_id),
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def neighbors_raw(tenant_id: str, item_ids: Iterable[str], exclude_user: str) -> dict[str, dict[str, float]]:
    """指定アイテム群に触れた他ユーザーの履歴をまとめて引く。

    ユーザー間型は「対象ユーザーと 1 つでもアイテムを共有する相手」しか
    類似度が 0 より大きくならない。全ユーザーを舐めるのではなく、
    共有アイテムから逆引きして候補を絞る。
    """
    ids = list(item_ids)
    if not ids:
        return {}

    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            """
            SELECT i.user_id, i.item_id, i.rating
            FROM interactions i
            WHERE i.tenant_id = %s
              AND i.user_id <> %s
              AND i.user_id IN (
                  SELECT DISTINCT user_id FROM interactions
                  WHERE tenant_id = %s AND item_id = ANY(%s) AND user_id <> %s
              )
            """,
            (tenant_id, exclude_user, tenant_id, ids, exclude_user),
        ).fetchall()

    out: dict[str, dict[str, float]] = {}
    for user_id, item_id, rating in rows:
        out.setdefault(user_id, {})[item_id] = float(rating)
    return out


def item_co_users(tenant_id: str, item_ids: Sequence[str]) -> dict[str, set[str]]:
    """アイテムごとに、触れたユーザーの集合を返す。アイテム間型で使う。"""
    if not item_ids:
        return {}

    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, user_id FROM interactions "
            "WHERE tenant_id = %s AND item_id = ANY(%s)",
            (tenant_id, list(item_ids)),
        ).fetchall()

    out: dict[str, set[str]] = {}
    for item_id, user_id in rows:
        out.setdefault(item_id, set()).add(user_id)
    return out


def all_item_users(tenant_id: str) -> dict[str, set[str]]:
    """全アイテムについて、触れたユーザーの集合を返す。"""
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, user_id FROM interactions WHERE tenant_id = %s",
            (tenant_id,),
        ).fetchall()

    out: dict[str, set[str]] = {}
    for item_id, user_id in rows:
        out.setdefault(item_id, set()).add(user_id)
    return out


def popularity(tenant_id: str) -> dict[str, int]:
    """アイテムごとの利用ユーザー数。人気バイアスの計測に使う。"""
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, count(DISTINCT user_id) FROM interactions "
            "WHERE tenant_id = %s GROUP BY item_id",
            (tenant_id,),
        ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def users(tenant_id: str) -> list[str]:
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM interactions WHERE tenant_id = %s ORDER BY user_id",
            (tenant_id,),
        ).fetchall()
    return [r[0] for r in rows]

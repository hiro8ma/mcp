"""比較の基準線。

推薦方式の数値は、それ単体では良し悪しを判断できない。
「人気バイアス 0.76 倍」が低いのか普通なのかは、何もしない推薦が
いくつになるかを知って初めて言える。

ランダムが中立（1.0 付近）、人気順が上限を示す。
実装した方式がこの 2 本の間のどこに位置するかで性質が読める。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import interactions, store


@dataclass(frozen=True)
class Rec:
    item_id: str
    score: float
    why: str


def _catalog(tenant_id: str) -> list[str]:
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id FROM items WHERE tenant_id = %s ORDER BY item_id", (tenant_id,)
        ).fetchall()
    return [r[0] for r in rows]


def random_items(tenant_id: str, user_id: str, top_k: int = 10, seed: int = 0) -> list[Rec]:
    """履歴を除いてランダムに選ぶ。

    人気バイアスは 1.0 前後になるはず。カタログから一様に選べば、
    推薦されたアイテムの平均人気度はカタログ平均に一致する。
    ここが 1.0 から大きく外れるなら計測側を疑う。
    """
    seen = set(interactions.history(tenant_id, user_id))
    pool = [i for i in _catalog(tenant_id) if i not in seen]
    if not pool:
        return []

    # ユーザーごとに違う結果を出しつつ、実行ごとには再現するよう種を混ぜる。
    rng = random.Random(f"{seed}:{user_id}")
    picked = rng.sample(pool, k=min(top_k, len(pool)))
    return [Rec(item_id=i, score=0.0, why="ランダム") for i in picked]


def most_popular(tenant_id: str, user_id: str, top_k: int = 10) -> list[Rec]:
    """利用者数の多い順に選ぶ。個人化を一切しない。

    的中率で測るとこの方式が高得点を出しやすい。実際よく当たるため。
    しかし全ユーザーに同じものを出しているので、推薦としての価値はない。
    的中率だけを見ていると、この差が指標に現れない。
    """
    seen = set(interactions.history(tenant_id, user_id))
    pop = interactions.popularity(tenant_id)
    ranked = sorted(pop.items(), key=lambda x: (-x[1], x[0]))
    out = [
        Rec(item_id=i, score=float(c), why=f"利用者 {c} 人")
        for i, c in ranked
        if i not in seen
    ]
    return out[:top_k]

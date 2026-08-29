"""履歴の量に応じて方式を切り替えるハイブリッド推薦。

教材の選択指針を実装に落とす。

  サービス開始直後やデータ量が少ない段階では内容ベースを使うことが多い
  既存ユーザーには協調、新規ユーザーや新規アイテムには内容ベース

協調フィルタリングは他ユーザーとの共通アイテムが無いと類似度が 0 になり、
候補が 1 件も出ない。内容ベースはアイテム特徴があれば履歴 1 件でも推薦できる。
この差が出る境界を実測で決める。

境界を定数で決め打ちしないのは、適切な値がカタログの規模と
1 ユーザーあたりの履歴の量で変わるため。measure_threshold で実測する。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import collaborative, content_based, interactions


@dataclass(frozen=True)
class Rec:
    item_id: str
    score: float
    why: str
    source: str  # どちらの方式が出したか。運用で切り分けるために残す


def recommend(
    tenant_id: str,
    user_id: str,
    top_k: int = 10,
    min_history: int = 3,
) -> list[Rec]:
    """履歴が min_history 件未満なら内容ベース、以上なら協調を使う。

    協調で候補が出なかった場合も内容ベースに落とす。
    件数の条件を満たしていても、他ユーザーと共通アイテムが無ければ
    協調は何も返さない。件数だけで判断すると空の結果を返すことになる。
    """
    hist = interactions.history(tenant_id, user_id)

    if len(hist) >= min_history:
        recs = collaborative.item_based(tenant_id, user_id, top_k=top_k)
        if recs:
            return [Rec(r.item_id, r.score, r.why, "協調") for r in recs]

    recs = content_based.recommend(tenant_id, user_id, top_k=top_k)
    return [Rec(r.item_id, r.score, r.why, "内容ベース") for r in recs]


def blend(
    tenant_id: str,
    user_id: str,
    top_k: int = 10,
    new_item_slots: int = 2,
) -> list[Rec]:
    """協調の結果に、行動ログの無いアイテムを内容ベースから混ぜる。

    ユーザー単位の切り替え（recommend）では新規アイテムの問題が解けない。
    新規アイテムを届ける相手は既存ユーザーであり、そのユーザーは
    履歴が足りているため常に協調へ流れる。実測でハイブリッドの
    新規アイテム登場数が 0 件になった。

    枠を確保して混ぜる。new_item_slots 件を行動ログの無いアイテムに割り当て、
    残りを協調で埋める。探索の枠を明示的に取る形になる。
    """
    touched = set(interactions.popularity(tenant_id))

    fresh: list[Rec] = []
    for r in content_based.recommend(tenant_id, user_id, top_k=top_k * 5):
        if r.item_id not in touched:
            fresh.append(Rec(r.item_id, r.score, r.why, "内容ベース(新規)"))
        if len(fresh) >= new_item_slots:
            break

    rest = top_k - len(fresh)
    main = [Rec(r.item_id, r.score, r.why, "協調")
            for r in collaborative.item_based(tenant_id, user_id, top_k=rest)]
    if not main:
        main = [Rec(r.item_id, r.score, r.why, "内容ベース")
                for r in content_based.recommend(tenant_id, user_id, top_k=rest)]

    return (main + fresh)[:top_k]


def measure_threshold(tenant_id: str, users: list[str], top_k: int = 10) -> dict:
    """履歴の件数ごとに、各方式が推薦を出せる割合を測る。

    切り替えの境界は、協調が出せるようになる件数で決まる。
    定数で置くと、カタログの規模が変わったときに合わなくなる。
    """
    buckets: dict[int, dict[str, int]] = {}

    for u in users:
        n = len(interactions.history(tenant_id, u))
        b = buckets.setdefault(n, {"users": 0, "協調": 0, "内容ベース": 0, "ハイブリッド": 0})
        b["users"] += 1
        if collaborative.item_based(tenant_id, u, top_k=top_k):
            b["協調"] += 1
        if content_based.recommend(tenant_id, u, top_k=top_k):
            b["内容ベース"] += 1
        if recommend(tenant_id, u, top_k=top_k):
            b["ハイブリッド"] += 1

    return buckets

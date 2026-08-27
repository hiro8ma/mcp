"""内容ベースフィルタリング。

アイテムの中身（embedding）とユーザープロファイルを照合する。
store.search() はすでにアイテム側を担っているため、ここで補うのは
ユーザープロファイルの獲得にあたる。

教材が「購入履歴で最も多く出現する特徴」を採る形で説明している部分は、
embedding では「好んだアイテムのベクトルの平均」になる。

属性の一致数で測る方式との違いは、意味の近さを連続値で扱える点。
一致数だと「ミステリー」と「サスペンス」は別物として 0 点になり、
SF と同じ扱いになる。embedding なら距離が近く出る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import interactions, store


@dataclass(frozen=True)
class Rec:
    item_id: str
    score: float
    why: str


def profile_from_history(tenant_id: str, user_id: str) -> list[float] | None:
    """履歴からユーザープロファイルを作る。

    好んだアイテムの埋め込みを rating で重み付けして平均する。
    平均ベクトルは「そのユーザーの好みの重心」にあたる。

    重心 1 点にまとめる方式には既知の弱点がある。
    好みが複数方向に分かれているユーザー（仕事用と休日用で全く違う等）では、
    重心がどちらでもない中間を指す。実運用では履歴をクラスタに分けて
    複数のプロファイルを持たせることが多い。ここでは単一重心のみ実装する。
    """
    hist = interactions.history(tenant_id, user_id)
    if not hist:
        return None

    embs = store.get_embeddings(tenant_id, list(hist))

    acc: list[float] | None = None
    total = 0.0
    for item_id, rating in hist.items():
        emb = embs.get(item_id)
        if emb is None:
            continue
        if acc is None:
            acc = [0.0] * len(emb)
        for i, v in enumerate(emb):
            acc[i] += v * rating
        total += rating

    if acc is None or total == 0:
        return None

    mean = [v / total for v in acc]
    # コサイン距離で引くので長さは効かないが、正規化しておくと
    # 他のベクトルと足し合わせるときに扱いやすい。
    norm = math.sqrt(sum(v * v for v in mean))
    return [v / norm for v in mean] if norm else mean


def recommend(tenant_id: str, user_id: str, top_k: int = 10) -> list[Rec]:
    """履歴から作ったプロファイルに近いアイテムを推薦する。

    履歴にあるアイテムは除く。除かないと、自分が既に持っているものが
    最も似ているアイテムとして上位を占める。
    """
    profile = profile_from_history(tenant_id, user_id)
    if profile is None:
        return []

    seen = set(interactions.history(tenant_id, user_id))
    hits = store.search(tenant_id, profile, top_k=top_k + len(seen))

    out = [
        Rec(item_id=h.item_id, score=h.similarity, why=f"好みの重心との類似度 {h.similarity}")
        for h in hits
        if h.item_id not in seen
    ]
    return out[:top_k]

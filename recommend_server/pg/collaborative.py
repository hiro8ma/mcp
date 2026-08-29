"""協調フィルタリング（メモリベース法）。

内容ベースフィルタリングと違い、アイテムの中身を一切見ない。
誰が何に触れたかの共起だけで推薦する。

メモリベース法と呼ぶのは、推薦のたびに蓄積データをその場で読んで計算するため。
事前にモデルを作らないので、新しい行動が即座に反映される。
代償として推薦 1 回あたりの計算量がデータ量に比例する。
（事前に規則性を学習しておくのがモデルベース法で、こちらは未実装）
"""

from __future__ import annotations

import math
import typing
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from . import interactions


@dataclass(frozen=True)
class Rec:
    item_id: str
    score: float
    # why は推薦理由。協調フィルタリングは中身を見ないため、
    # 「なぜこれが出たか」を人が読める形にしないと運用で説明できない。
    why: str


# Similarity は 2 人の履歴から類似度を返す。
Similarity = typing.Callable[[dict[str, float], dict[str, float]], float]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """2 つの履歴ベクトルのコサイン類似度。

    共通アイテムだけで内積を取り、ノルムは各自の全履歴で割る。
    共通が多くても、片方が大量に何でも触れているなら類似度は下がる。

    非購入は見ない。値域は 0 以上になるため、
    「好みが正反対」と「共通点が無い」を区別できない。
    区別したい場合は _pearson を使う。ただし下の注意を読むこと。
    """
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0

    dot = sum(a[i] * b[i] for i in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    """共通アイテム数を和集合で割る。

    コサインと違い評価値の大小を見ない。購入したかどうかだけの
    暗黙のフィードバックでは、評価値が全て 1.0 なので差が出にくい。
    """
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _pearson_over(catalog: Sequence[str]) -> Similarity:
    """非購入を 0 とみなし、カタログ全体で相関を取る類似度を返す。

    教材が「購入しなかったアイテムは好んでいない」と置く形にあたる。
    値域が -1 から 1 になるため、好みが正反対のユーザーを負の値で表せる。

    ただしこの前提が成り立つのは、ユーザーがカタログをおおむね見ている場合に限る。
    暗黙のフィードバック（購買・閲覧）では、非購入の大半は「嫌い」ではなく
    「見ていない」を意味する。実測した構成ではカタログ 1176 件に対して
    1 人あたり平均 9.5 件で、99.2% が非購入だった。
    この状態で相関を取ると、値のほとんどが「両者とも触れていない」ことに
    由来する。共起ではなく共通の欠落を測ることになる。

    カタログが小さく、ユーザーが全件を評価している（明示的な評価がある）
    場合には適切な選択になる。
    """

    def sim(a: dict[str, float], b: dict[str, float]) -> float:
        va = [a.get(i, 0.0) for i in catalog]
        vb = [b.get(i, 0.0) for i in catalog]
        n = len(catalog)
        if n == 0:
            return 0.0

        ma, mb = sum(va) / n, sum(vb) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(va, vb))
        da = math.sqrt(sum((x - ma) ** 2 for x in va))
        db = math.sqrt(sum((y - mb) ** 2 for y in vb))
        return num / (da * db) if da and db else 0.0

    return sim


def user_based(
    tenant_id: str,
    user_id: str,
    top_k: int = 10,
    neighbors: int = 20,
    min_similarity: float = 0.0,
    similarity: Similarity | None = None,
) -> list[Rec]:
    """ユーザー間型メモリベース法。

    嗜好が似ているユーザーを探し、その人たちが好んでいて対象ユーザーが
    まだ触れていないアイテムを推薦する。

    候補アイテムの得点は「類似ユーザーの類似度 × 評価」の合計。
    類似度で重みを付けるのは、より近い人の評価を強く効かせるため。

    合計を取るので、多くの類似ユーザーが触れているアイテムほど高得点になる。
    これが人気バイアスの発生源になる。人気アイテムは誰の履歴にも入っているため、
    誰と似ていても上位に来る。evaluate.py で実際に測れる。
    """
    mine = interactions.history(tenant_id, user_id)
    if not mine:
        return []

    others = interactions.neighbors_raw(tenant_id, mine.keys(), exclude_user=user_id)

    sim = similarity or _cosine
    sims = [(uid, sim(mine, hist)) for uid, hist in others.items()]
    sims = [(uid, s) for uid, s in sims if s > min_similarity]
    sims.sort(key=lambda x: x[1], reverse=True)
    sims = sims[:neighbors]

    scores: dict[str, float] = defaultdict(float)
    supporters: dict[str, int] = defaultdict(int)
    for uid, sim in sims:
        for item_id, rating in others[uid].items():
            if item_id in mine:
                continue
            scores[item_id] += sim * rating
            supporters[item_id] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        Rec(item_id=i, score=round(s, 4), why=f"嗜好の近いユーザー {supporters[i]} 人が利用")
        for i, s in ranked
    ]


def item_based(
    tenant_id: str,
    user_id: str,
    top_k: int = 10,
) -> list[Rec]:
    """アイテム間型メモリベース法。

    ユーザー同士ではなくアイテム同士の類似度を使う。
    2 つのアイテムの類似度は「両方に触れたユーザー数」から測る。

    ユーザー間型より実運用で好まれることが多い。ユーザーの嗜好は日々変わるが、
    アイテム同士の関係は動きにくいため、計算結果を使い回しやすい。
    ユーザー数がアイテム数より桁違いに多いサービスでは計算量でも有利になる。
    """
    mine = interactions.history(tenant_id, user_id)
    if not mine:
        return []

    item_users = interactions.all_item_users(tenant_id)

    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, set[str]] = defaultdict(set)
    for seed_id, rating in mine.items():
        seed_users = item_users.get(seed_id, set())
        if not seed_users:
            continue

        for cand_id, cand_users in item_users.items():
            if cand_id in mine:
                continue
            shared = len(seed_users & cand_users)
            if shared == 0:
                continue
            # コサイン類似度。共起数を両アイテムの利用者数で正規化する。
            # 正規化しないと、単に利用者が多いアイテムが常に似ていることになる。
            sim = shared / math.sqrt(len(seed_users) * len(cand_users))
            scores[cand_id] += sim * rating
            reasons[cand_id].add(seed_id)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        Rec(
            item_id=i,
            score=round(s, 4),
            why=f"履歴の {len(reasons[i])} 件と併用が多い",
        )
        for i, s in ranked
    ]

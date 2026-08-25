"""検索結果を棚(シェルフ)の形に組み替える。

近傍を大量に取ってから並べ替えるのが要点になる。
上位 10 件をそのまま出すと同系統の商品ばかりが並び、面としての情報量が落ちる。
100 件から 300 件を取ってグループに割り、グループごとに代表を出すと幅が出る。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

from .store import Hit

GroupBy = Literal["category", "cluster"]


@dataclass(frozen=True)
class Shelf:
    title: str
    items: list[Hit]


def by_category(hits: Sequence[Hit], per_shelf: int, max_shelves: int) -> list[Shelf]:
    buckets: dict[str, list[Hit]] = defaultdict(list)
    for h in hits:
        buckets[h.category or "その他"].append(h)

    # 候補が多いグループほど関連が強いとみなして先に並べる。
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [Shelf(title=name, items=group[:per_shelf]) for name, group in ordered[:max_shelves]]


def by_cluster(
    hits: Sequence[Hit],
    per_shelf: int,
    max_shelves: int,
) -> list[Shelf]:
    """埋め込みを k-means で割ってグループにする。

    カテゴリ分けと違い、カタログ側の分類に縛られない切り口が出る。
    そのかわりグループ名を機械的に決められないため、代表アイテムの題名を借りる。

    候補が max_shelves より少ない場合は、カテゴリ分けに切り替えるのではなく
    クラスタ数のほうを候補数まで下げる。k-means の制約は n_samples >= n_clusters
    というだけで、要求された「埋め込みの近さで割る」という軸は保てる。
    軸ごと変えてしまうと、呼び出し元は cluster で組んだつもりでカテゴリ棚を受け取る。
    """
    import numpy as np
    from sklearn.cluster import KMeans

    usable = [h for h in hits if h.embedding is not None]
    if not usable:
        raise ValueError(
            "埋め込みが載っていない。store.search(..., with_embeddings=True) で取得する必要がある"
        )

    k = min(max_shelves, len(usable))
    matrix = np.asarray([h.embedding for h in usable], dtype="float32")
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(matrix)

    buckets: dict[int, list[Hit]] = defaultdict(list)
    for label, hit in zip(km.labels_, usable):
        buckets[int(label)].append(hit)

    shelves: list[Shelf] = []
    for label, group in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True):
        group.sort(key=lambda h: h.similarity, reverse=True)
        shelves.append(Shelf(title=f"{group[0].title} に近い系統", items=group[:per_shelf]))
    return shelves

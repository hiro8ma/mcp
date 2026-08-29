"""3 つの推薦方式を、精度ではなくカバレッジと人気バイアスで比べる。

推薦の良し悪しを的中率だけで測ると、人気アイテムを出し続ける実装が最も高得点になる。
実際よく当たるためで、指標としては正しい。しかしそれを推薦と呼ぶ意味はない。

ここで測るのは次の 3 つ。

  カタログカバレッジ  全ユーザーへの推薦に何種類のアイテムが登場したか
  人気バイアス        推薦されたアイテムの人気度が、カタログ平均の何倍か
  ジニ係数            推薦回数の偏り。0 が完全に均等、1 が 1 品目に集中

人気バイアスの中立点は 1.00 ではない。どの方式も自分の履歴を推薦から除くが、
履歴には人気アイテムが集中している（実測でカタログ平均の 30.7 倍）ため、
除いた残りは平均人気度が下がる。実測ではランダム推薦が 0.78 倍になった。
除外しない条件では 0.96 倍で、指標そのものは正しい。

そのため中立点はランダム基準の実測値で決める。表には対ランダム比も出す。

的中率と違い、いずれも正解ラベルが要らない。運用に載せた推薦を
そのまま監視できるため、日々のダッシュボードに向く。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from . import baselines, collaborative, content_based, interactions, store


@dataclass
class Result:
    name: str
    coverage: float
    popularity_lift: float
    gini: float
    users_served: int
    empty: int


def gini(counts: list[int]) -> float:
    """推薦回数の偏りを測る。

    0 に近いほど多様なアイテムが出ており、1 に近いほど一部に集中している。
    カバレッジは「登場したか」の 0/1 しか見ないため、1 品目が 1000 回出て
    他が 1 回ずつでも高く出る。偏りの度合いはジニ係数で補う。
    """
    if not counts:
        return 0.0
    xs = sorted(counts)
    n = len(xs)
    total = sum(xs)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n


def evaluate(tenant_id: str, name: str, recommend, users: list[str], catalog_size: int,
             pop: dict[str, int]) -> Result:
    """1 方式を評価する。

    pop はアイテムごとの利用ユーザー数。利用実績のないアイテムは含まれないため、
    人気度を引くときは pop.get(i, 0) で 0 に落とす。
    """
    shown = Counter()
    empty = 0
    served = 0

    # 分母はカタログ全体で平均する。pop に載っているのは利用実績のある
    # アイテムだけなので、pop.values() の平均を使うと分母が過大になる。
    # 分子は未利用アイテムを 0 として数えており、母集団が食い違う。
    #
    # 実際にこれで測ったとき、ランダム推薦の人気バイアスが 0.56 倍になった。
    # カタログから一様に選べば定義上 1.0 付近になるはずで、
    # ずれの比 846/1176 = 0.72 がちょうど母集団の食い違いに一致した。
    mean_pop = sum(pop.values()) / catalog_size if catalog_size else 0.0

    for u in users:
        recs = recommend(tenant_id, u)
        if not recs:
            empty += 1
            continue
        served += 1
        for r in recs:
            shown[r.item_id] += 1

    if not shown:
        return Result(name, 0.0, 0.0, 0.0, served, empty)

    # 推薦された回数で重み付けした平均人気度。
    # 種類ではなく回数で見るのは、同じ人気アイテムが何度も出る影響を捉えるため。
    weighted = sum(pop.get(i, 0) * c for i, c in shown.items()) / sum(shown.values())

    return Result(
        name=name,
        coverage=len(shown) / catalog_size,
        popularity_lift=weighted / mean_pop if mean_pop else 0.0,
        gini=gini(list(shown.values())),
        users_served=served,
        empty=empty,
    )


# pearson はカタログ全体を要するため、テナントごとに 1 度だけ作る。
_pearson_cache: dict[str, object] = {}


def _pearson(tenant_id: str):
    if tenant_id not in _pearson_cache:
        with store.connect(owner=True) as conn:
            catalog = [r[0] for r in conn.execute(
                "SELECT item_id FROM items WHERE tenant_id = %s ORDER BY item_id",
                (tenant_id,)).fetchall()]
        _pearson_cache[tenant_id] = collaborative._pearson_over(catalog)
    return _pearson_cache[tenant_id]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sample-users", type=int, default=100)
    args = p.parse_args()

    catalog_size = store.count(args.tenant)
    all_users = interactions.users(args.tenant)
    users = all_users[: args.sample_users]
    pop = interactions.popularity(args.tenant)

    if not users:
        raise SystemExit("行動ログがない。先に seed_interactions.py を実行する")

    strategies = [
        # 基準線を 2 本置く。これが無いと「人気バイアス 0.76x」が低いのか
        # 普通なのか判断できない。ランダムが中立、人気順が上限にあたる。
        ("[基準] ランダム",
         lambda t, u: baselines.random_items(t, u, top_k=args.top_k)),
        ("[基準] 人気順",
         lambda t, u: baselines.most_popular(t, u, top_k=args.top_k)),
        ("内容ベース（履歴の重心）",
         lambda t, u: content_based.recommend(t, u, top_k=args.top_k)),
        ("協調 ユーザー間型 cos",
         lambda t, u: collaborative.user_based(t, u, top_k=args.top_k)),
        # 類似度の定義で結果が変わるかを見る。
        # 教材は非購入を「嫌い」と置く形（pearson 相当）を前提にしている。
        ("協調 ユーザー間型 jaccard",
         lambda t, u: collaborative.user_based(
             t, u, top_k=args.top_k, similarity=collaborative._jaccard)),
        ("協調 ユーザー間型 pearson",
         lambda t, u: collaborative.user_based(
             t, u, top_k=args.top_k, similarity=_pearson(t))),
        ("協調 アイテム間型",
         lambda t, u: collaborative.item_based(t, u, top_k=args.top_k)),
    ]

    print(f"テナント {args.tenant} / カタログ {catalog_size} 件 / "
          f"ユーザー {len(all_users)} 人中 {len(users)} 人で評価 / top-{args.top_k}\n")

    results = [
        evaluate(args.tenant, name, fn, users, catalog_size, pop)
        for name, fn in strategies
    ]

    # 中立点はランダム基準の実測値。1.00 ではない（履歴の除外で下がる）。
    neutral = next((r.popularity_lift for r in results if r.name.startswith("[基準] ランダム")), 0.0)

    print(f"{'方式':<24}{'カバレッジ':>12}{'人気バイアス':>13}{'対ランダム':>12}"
          f"{'ジニ係数':>11}{'推薦不可':>10}")
    print("-" * 84)
    for r in results:
        rel = f"{r.popularity_lift / neutral:>10.1f}x" if neutral else f"{'-':>11}"
        print(f"  {r.name:<22}{r.coverage:>11.1%}{r.popularity_lift:>12.2f}x{rel}"
              f"{r.gini:>11.3f}{r.empty:>10}")
    print("-" * 84)
    print(f"  人気バイアス  推薦アイテムの平均人気度がカタログ平均の何倍か")
    print(f"  対ランダム    ランダム基準（実測 {neutral:.2f}x）を 1.0 としたときの倍率。"
          f"これが中立点になる")
    print("  ジニ係数      推薦回数の偏り。0 が均等、1 が一部に集中")
    print("  推薦不可      履歴が足りず 1 件も出せなかったユーザー数")


if __name__ == "__main__":
    main()

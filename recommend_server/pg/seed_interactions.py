"""合成の行動履歴を投入する。

人気バイアスは元データが偏っていないと再現しない。均等に散らしたログで測ると
「バイアスは出ませんでした」という結論になるが、それは実データを模していないだけ。

実サービスの利用分布はロングテールになる。少数のアイテムに利用が集中し、
大多数はほとんど触れられない。ここでは 2 つの機構でそれを作る。

  1. 嗜好  ユーザーごとに好むカテゴリと属性（色・素材）があり、それに合うものを選ぶ
  2. 追従  すでに人気のあるアイテムほど選ばれやすい（優先的選択）

嗜好をカテゴリだけにすると、カテゴリ内では一様になる。約 200 件から
次の 1 件を当てる問題になり、どの方式も偶然の水準（5% 前後）に潰れる。
内容ベースが学ぶべき構造が無いため、予測精度の比較にならない。
実測で follow_ratio=0.2 のとき全方式が Recall 0.03 前後になった。

色や素材まで好みを持たせると、埋め込みが捉えられる構造が入る。
実サービスでも「黒い服が好き」「リネンを好む」という一貫性はある。

2 が無いと、カテゴリ内では均等になりロングテールにならない。

追従の強さは alpha で調整する。重みを count^alpha にするため、
alpha=1 の素朴な優先的選択では偏りの成長が遅く、イベント数が少ないうちは
ほぼ均等なままになる。実測では alpha=1・follow_ratio=0.5 で
上位 1% の占有率が 3.4% にしかならず、実サービスとかけ離れていた。

投入後に必ず分布を確認する。偏っていないデータで人気バイアスを測ると
「バイアスは出ませんでした」という結論が出るが、それはデータが実態を
模していないだけで、アルゴリズムの性質を測ったことにはならない。
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from . import interactions, store
from .seed import COLORS, MATERIALS


def build_events(
    item_ids_by_category: dict[str, list[str]],
    n_users: int,
    events_per_user: tuple[int, int],
    follow_ratio: float,
    attrs_by_item: dict[str, tuple[str, ...]] | None = None,
    colors: list[str] | None = None,
    materials: list[str] | None = None,
    alpha: float = 1.0,
    seed: int = 42,
) -> list[interactions.Event]:
    """嗜好と追従を混ぜて行動ログを作る。

    follow_ratio は「人気に引かれて選ぶ割合」。0 なら嗜好だけで選ぶ。
    """
    rng = random.Random(seed)
    attrs_by_item = attrs_by_item or {}
    colors = colors or []
    materials = materials or []
    categories = list(item_ids_by_category)
    all_items = [i for ids in item_ids_by_category.values() for i in ids]

    # 優先的選択のための累積カウント。1 から始めて未利用アイテムも選ばれ得るようにする。
    picked = Counter({i: 1 for i in all_items})

    events: list[interactions.Event] = []
    for u in range(n_users):
        user_id = f"user-{u:04d}"
        # 各ユーザーは 1〜2 個のカテゴリと、色・素材の好みを持つ。
        taste = rng.sample(categories, k=rng.choice([1, 1, 2]))
        pool = [i for c in taste for i in item_ids_by_category[c]]

        # 属性の好みに合うものを重く扱う。合わないものも選ばれ得るが確率は低い。
        # 完全に絞ると嗜好の外に一切出なくなり、実サービスと離れる。
        if attrs_by_item:
            fav = {
                "color": rng.choice(colors) if colors else None,
                "material": rng.choice(materials) if materials else None,
            }
            weighted = []
            for i in pool:
                a = attrs_by_item.get(i, ())
                score = 1
                if fav["color"] and fav["color"] in a:
                    score += 6
                if fav["material"] and fav["material"] in a:
                    score += 6
                weighted.extend([i] * score)
            pool = weighted or pool

        n = rng.randint(*events_per_user)
        chosen: set[str] = set()
        for _ in range(n):
            if rng.random() < follow_ratio:
                # 人気に追従する。重み付き抽選なので、すでに多く選ばれたものほど出やすい。
                # alpha を上げると差が指数的に開き、現実に近いロングテールになる。
                weights = [picked[i] ** alpha for i in all_items]
                item = rng.choices(all_items, weights=weights, k=1)[0]
            else:
                item = rng.choice(pool)

            if item in chosen:
                continue
            chosen.add(item)
            picked[item] += 1
            events.append(interactions.Event(user_id=user_id, item_id=item, rating=1.0))

    return events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--max-events", type=int, default=25)
    p.add_argument("--follow-ratio", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=2.5,
                   help="人気への追従の強さ。大きいほど偏る")
    p.add_argument("--reset", action="store_true", help="既存の行動ログを消してから投入")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with store.connect(args.tenant) as conn:
        rows = conn.execute(
            "SELECT category, item_id, tags FROM items WHERE tenant_id = %s", (args.tenant,)
        ).fetchall()

    if not rows:
        raise SystemExit(f"テナント {args.tenant} にアイテムがない。先に seed.py を実行する")

    by_cat: dict[str, list[str]] = {}
    attrs: dict[str, tuple[str, ...]] = {}
    for cat, item_id, tags in rows:
        by_cat.setdefault(cat or "未分類", []).append(item_id)
        attrs[item_id] = tuple(tags or ())

    if args.reset:
        with store.connect(args.tenant) as conn:
            conn.execute("DELETE FROM interactions WHERE tenant_id = %s", (args.tenant,))
            conn.commit()

    events = build_events(
        by_cat,
        n_users=args.users,
        attrs_by_item=attrs,
        colors=COLORS,
        materials=MATERIALS,
        events_per_user=(args.min_events, args.max_events),
        follow_ratio=args.follow_ratio,
        alpha=args.alpha,
        seed=args.seed,
    )
    n = interactions.record(args.tenant, events)

    pop = interactions.popularity(args.tenant)
    counts = sorted(pop.values(), reverse=True)
    total_items = len(rows)
    touched = len(pop)
    total_events = sum(counts)

    print(f"行動ログ {n} 件を投入（ユーザー {args.users} 人）")
    print(f"  カタログ {total_items} 件のうち利用されたのは {touched} 件 "
          f"({touched / total_items:.1%})")
    for pct in (1, 5, 10, 20):
        k = max(1, total_items * pct // 100)
        share = sum(counts[:k]) / total_events if total_events else 0
        print(f"  上位 {pct:>2}% のアイテムが全利用の {share:>5.1%} を占める")

    # 実サービスの利用分布では上位 1% が 2 割前後を占めることが多い。
    # そこに届いていないデータで人気バイアスを測っても意味がない。
    top1 = sum(counts[: max(1, total_items // 100)]) / total_events if total_events else 0
    if top1 < 0.10:
        print(f"\n  警告: 上位 1% の占有率が {top1:.1%} しかない。分布が均等に近い。")
        print("  この状態で人気バイアスを測っても、アルゴリズムの性質は現れない。")
        print("  --alpha か --follow-ratio を上げて偏りを強める。")


if __name__ == "__main__":
    main()

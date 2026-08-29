"""予測精度と被覆率を測る。

教材の比較表は協調フィルタリングを予測精度 ○、被覆率 × とし、
内容ベースを予測精度 △、被覆率 ○ としている。
どちらも自分のデータで測っていなかったため確かめる。

被覆率は 2 通りに分けて測る。教材の根拠が
「サービス内に存在するアイテムを推薦候補にできる」であるのに対し、
実測していたのは「実際に推薦に登場した割合」だった。
別のものを比べていた可能性がある。

  候補被覆率  原理上その方式で推薦され得るアイテムの割合
  実現被覆率  実際に推薦に登場したアイテムの割合

内容ベースは全アイテムが候補になり得るが、重心の近傍しか出ない。
協調は行動ログの無いアイテムが候補にすらならない。
候補で見るか実現で見るかで順序が入れ替わる。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from . import baselines, collaborative, content_based, interactions, store


@dataclass
class Accuracy:
    name: str
    recall: float      # 隠した 1 件を上位 K に含められた割合
    mrr: float         # 含められた場合の順位の逆数の平均
    evaluated: int     # 評価できたユーザー数
    skipped: int       # 履歴が足りず評価できなかった数


def leave_one_out(tenant_id: str, users: list[str], recommend, top_k: int,
                  seed: int = 0) -> Accuracy:
    """各ユーザーの履歴から 1 件隠し、それを当てられるかを測る。

    隠した 1 件だけが正解になる。ユーザーが実際に触れていない
    アイテムを推薦しても不正解として数えるため、真の精度より低く出る。
    方式間の比較には使えるが、絶対値を精度として読んではいけない。
    """
    rng = random.Random(seed)
    hits, rr, n, skipped = 0, 0.0, 0, 0

    for u in users:
        hist = interactions.history(tenant_id, u)
        if len(hist) < 2:
            skipped += 1
            continue

        held = rng.choice(sorted(hist))
        with store.connect(tenant_id) as conn:
            conn.execute(
                "DELETE FROM interactions WHERE tenant_id=%s AND user_id=%s AND item_id=%s",
                (tenant_id, u, held))
            conn.commit()
        try:
            recs = recommend(tenant_id, u)
            ids = [r.item_id for r in recs][:top_k]
            n += 1
            if held in ids:
                hits += 1
                rr += 1.0 / (ids.index(held) + 1)
        finally:
            # 隠した 1 件を必ず戻す。戻し損ねると後続の測定が別のデータになる。
            interactions.record(tenant_id, [interactions.Event(u, held, hist[held])])

    if n == 0:
        return Accuracy(name="", recall=0, mrr=0, evaluated=0, skipped=skipped)
    return Accuracy(name="", recall=hits / n, mrr=rr / n, evaluated=n, skipped=skipped)


def candidate_coverage(tenant_id: str) -> dict[str, float]:
    """原理上推薦され得るアイテムの割合。

    内容ベースは全アイテムが埋め込みを持つため候補になり得る。
    協調は行動ログのあるアイテムしか候補にならない。
    """
    total = store.count(tenant_id)
    touched = len(interactions.popularity(tenant_id))
    return {"内容ベース": 1.0, "協調": touched / total if total else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sample-users", type=int, default=60)
    args = p.parse_args()

    users = interactions.users(args.tenant)[: args.sample_users]
    if not users:
        raise SystemExit("行動ログがない")

    strategies = [
        ("[基準] ランダム", lambda t, u: baselines.random_items(t, u, top_k=args.top_k)),
        ("[基準] 人気順", lambda t, u: baselines.most_popular(t, u, top_k=args.top_k)),
        ("内容ベース", lambda t, u: content_based.recommend(t, u, top_k=args.top_k)),
        ("協調 ユーザー間型", lambda t, u: collaborative.user_based(t, u, top_k=args.top_k)),
        ("協調 アイテム間型", lambda t, u: collaborative.item_based(t, u, top_k=args.top_k)),
    ]

    print(f"テナント {args.tenant} / ユーザー {len(users)} 人 / top-{args.top_k}")
    print("各ユーザーの履歴から 1 件を隠し、上位 K に戻せるかで測る\n")
    print(f"{'方式':<24}{'Recall@K':>10}{'MRR':>9}{'評価数':>8}")
    print("-" * 52)
    for name, fn in strategies:
        a = leave_one_out(args.tenant, users, fn, args.top_k)
        print(f"  {name:<22}{a.recall:>10.3f}{a.mrr:>9.3f}{a.evaluated:>8}")
    print("-" * 52)
    print("  隠した 1 件だけを正解とするため、絶対値は真の精度より低く出る")
    print("  方式間の比較にのみ使う")

    print()
    cc = candidate_coverage(args.tenant)
    total = store.count(args.tenant)
    print(f"{'方式':<24}{'候補被覆率':>12}")
    print("-" * 40)
    for k, v in cc.items():
        print(f"  {k:<22}{v:>11.1%}")
    print("-" * 40)
    print(f"  カタログ {total} 件のうち、原理上推薦され得る割合")
    print("  内容ベースは全件が埋め込みを持つため候補になり得る")
    print("  協調は行動ログのあるアイテムしか候補にならない")


if __name__ == "__main__":
    main()

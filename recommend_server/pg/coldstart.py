"""コールドスタートを測る。

教材はアルゴリズム選択の中心にコールドスタートを置く。
新規ユーザーと新規アイテムの 2 方向がある。

  新規ユーザー  履歴が無い、または少ない。協調は類似ユーザーを見つけられない
  新規アイテム  行動ログが無い。協調は候補にすら入れられない

どちらも合成データで作れる。履歴の件数を変えたユーザーを足し、
行動ログの無いアイテムが推薦に登場するかを見る。
"""

from __future__ import annotations

import argparse

from . import content_based, collaborative, hybrid, interactions, store


def add_synthetic_users(tenant_id: str, sizes: list[int], seed: int = 99) -> list[str]:
    """指定した件数の履歴を持つユーザーを作る。

    既存ユーザーが触れているアイテムから選ぶ。まったく誰も触れていない
    アイテムだけを与えると、協調が出せないのが履歴の少なさによるのか
    アイテムの新しさによるのか分からなくなる。
    """
    import random

    rng = random.Random(seed)
    pop = sorted(interactions.popularity(tenant_id))
    if not pop:
        raise SystemExit("行動ログがない")

    created = []
    for n in sizes:
        uid = f"cold-{n:02d}"
        with store.connect(tenant_id) as conn:
            conn.execute("DELETE FROM interactions WHERE tenant_id=%s AND user_id=%s",
                         (tenant_id, uid))
            conn.commit()
        if n > 0:
            interactions.record(tenant_id, [
                interactions.Event(uid, i, 1.0) for i in rng.sample(pop, k=min(n, len(pop)))])
        created.append(uid)
    return created


def new_item_reachability(tenant_id: str, users: list[str], top_k: int = 10) -> dict:
    """行動ログの無いアイテムが推薦に登場するかを測る。

    協調は原理上出せない。内容ベースは埋め込みがあれば出せる。
    「出せるはず」ではなく実際に登場するかを見る。
    """
    touched = set(interactions.popularity(tenant_id))
    with store.connect(tenant_id) as conn:
        catalog = {r[0] for r in conn.execute(
            "SELECT item_id FROM items WHERE tenant_id=%s", (tenant_id,)).fetchall()}
    untouched = catalog - touched

    out = {"新規アイテム数": len(untouched), "カタログ": len(catalog)}
    for name, fn in (("内容ベース", content_based.recommend),
                     ("協調 アイテム間型", collaborative.item_based),
                     ("ハイブリッド(切替)", hybrid.recommend),
                     ("ハイブリッド(枠確保)", hybrid.blend)):
        shown = set()
        for u in users:
            shown.update(r.item_id for r in fn(tenant_id, u, top_k=top_k))
        out[name] = len(shown & untouched)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sample-users", type=int, default=40)
    args = p.parse_args()

    print("=== 新規ユーザー 履歴の件数ごとに推薦を出せるか ===\n")
    sizes = [0, 1, 2, 3, 5, 10]
    cold = add_synthetic_users(args.tenant, sizes)

    print(f"{'履歴':>5}{'内容ベース':>12}{'協調':>10}{'ハイブリッド':>14}")
    print("-" * 44)
    for n, uid in zip(sizes, cold):
        cb = len(content_based.recommend(args.tenant, uid, top_k=args.top_k))
        cf = len(collaborative.item_based(args.tenant, uid, top_k=args.top_k))
        hy = hybrid.recommend(args.tenant, uid, top_k=args.top_k)
        src = hy[0].source if hy else "-"
        print(f"{n:>5}{cb:>10} 件{cf:>8} 件{len(hy):>8} 件 ({src})")
    print("-" * 44)
    print("  履歴 0 件ではどの方式も推薦を出せない。プロファイルが無いため")

    print()
    print("=== 新規アイテム 行動ログが無いアイテムが推薦に登場するか ===\n")
    users = interactions.users(args.tenant)[: args.sample_users]
    users = [u for u in users if not u.startswith("cold-")]
    r = new_item_reachability(args.tenant, users, args.top_k)

    print(f"  カタログ {r['カタログ']} 件のうち行動ログが無いのは {r['新規アイテム数']} 件")
    print()
    print(f"{'方式':<22}{'新規アイテムの登場数':>22}")
    print("-" * 46)
    for k in ("内容ベース", "協調 アイテム間型", "ハイブリッド(切替)", "ハイブリッド(枠確保)"):
        print(f"  {k:<20}{r[k]:>18} 件")
    print("-" * 46)
    print("  協調は行動ログの無いアイテムを候補にできないため 0 件になるはず")

    print()
    print("=== 切り替えの境界 ===\n")
    b = hybrid.measure_threshold(args.tenant, users, args.top_k)
    print(f"{'履歴':>5}{'人数':>6}{'協調が出せた':>14}{'内容ベースが出せた':>20}")
    print("-" * 50)
    for n in sorted(b):
        d = b[n]
        print(f"{n:>5}{d['users']:>6}{d['協調'] / d['users']:>13.0%}"
              f"{d['内容ベース'] / d['users']:>19.0%}")
    print("-" * 50)


if __name__ == "__main__":
    main()

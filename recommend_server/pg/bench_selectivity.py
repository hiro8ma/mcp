"""フィルタの選択率と再現率の関係を測る。

「pgvector 1 台で捌ける規模」を決める変数は行数だけではない。
社内 RAG で先に効くのは、部署別・権限別のフィルタで
1 グループが全体の何 % を占めるかという選択率のほうになる。

HNSW は ef_search 件の候補を集めてから WHERE を適用する。
選択率 s なら、候補のうち残るのは期待値で ef_search × s 件。
top_k 件を返すには ef_search ≳ top_k / s が要る計算になる。

この関係を実測で確かめる。成り立つなら、選択率から必要な ef_search を見積もれる。
"""

from __future__ import annotations

import argparse

from . import bench_filtered, embed, store

TEXTS = [
    "黒いレザーのブーツを探している",
    "秋冬に着るウールのコート",
    "リネンの白いシャツ",
    "ネイビーのデニムパンツ",
    "レザーのトートバッグ",
]


def set_tenant_count(n: int) -> None:
    """bench-* テナントを n 件だけ残す。demo と t-demo は触らない。"""
    with store.connect(owner=True) as conn:
        # LIKE のパターンも引数で渡す。SQL に直書きすると psycopg が
        # %' をプレースホルダとして解釈して落ちる。
        conn.execute(
            "DELETE FROM items WHERE tenant_id LIKE %s AND tenant_id > %s",
            ("bench-%", f"bench-{n - 1:02d}"),
        )
        conn.commit()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--counts", type=int, nargs="+", default=[15, 11, 7, 3, 1])
    args = p.parse_args()

    vecs = embed.encode(TEXTS)

    print(f"{'テナント数':>10}{'全行数':>9}{'選択率':>9}"
          f"{'ef=40':>9}{'ef=100':>9}{'ef=400':>9}{'理論値':>9}")
    print("-" * 66)

    for n in sorted(args.counts, reverse=True):
        set_tenant_count(n)
        bench_filtered.rebuild_hnsw()

        with store.connect(owner=True) as conn:
            total = conn.execute("SELECT count(*) FROM items").fetchone()[0]
            target = conn.execute(
                "SELECT count(*) FROM items WHERE tenant_id = %s", (args.tenant,)
            ).fetchone()[0]
        sel = target / total

        # 索引なしの厳密検索を正解にする。
        bench_filtered.drop_index()
        truths = [bench_filtered.query(args.tenant, v, args.top_k)[0] for v in vecs]
        bench_filtered.rebuild_hnsw()

        cells = []
        for ef in (40, 100, 400):
            rs = []
            for v, truth in zip(vecs, truths):
                got, _ = bench_filtered.query(
                    args.tenant, v, args.top_k, iterative="off", ef_search=ef
                )
                rs.append(bench_filtered.recall(truth, got))
            cells.append(sum(rs) / len(rs))

        # top_k 件返すのに必要な ef_search の見積もり。
        need = args.top_k / sel
        print(f"{n:>10}{total:>9}{sel:>8.1%}"
              f"{cells[0]:>9.2f}{cells[1]:>9.2f}{cells[2]:>9.2f}{need:>9.0f}")

    print("-" * 66)
    print("  理論値  top_k 件を返すのに必要な ef_search の下限（top_k / 選択率）")
    print("  この値を下回る ef_search では、返却件数が top_k に届かない")


if __name__ == "__main__":
    main()

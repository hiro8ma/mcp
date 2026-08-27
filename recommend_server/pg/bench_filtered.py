"""フィルタ付きベクトル検索の再現率を測る。

「SQL なのでベクトル検索に WHERE や JOIN をそのまま組み合わせられる」は
pgvector の利点として語られる。書けるのは事実だが、正しい答えが返るかは別問題になる。

WHERE tenant_id = $1 ORDER BY embedding <=> $2 LIMIT 10 を投げると、
プランナは 2 つの経路から選ぶ。

  embedding の索引     グラフを全テナント横断で辿り、あとからテナントで絞る
                       ef_search 件しか候補を持たないため、絞ると足りなくなる
  tenant_id の索引     先に絞ってから距離計算。厳密だが件数に比例して遅くなる

前者を選ぶと取りこぼしが出る。しかもエラーにならず、件数の少ない結果が返るだけなので
アプリからは気づけない。テナント数が増えるほど（1 テナントの占める割合が下がるほど）
劣化する。pgvector 0.8 の iterative_scan は候補が足りなければ探索を続ける緩和策だが、
既定は off になっている。

厳密検索（索引なし）を正解として、索引ありの結果と突き合わせる。
"""

from __future__ import annotations

import argparse
import time

from . import embed, store


def rebuild_hnsw(m: int = 16, ef_construction: int = 64) -> None:
    with store.connect(owner=True) as conn:
        conn.execute("DROP INDEX IF EXISTS items_embedding_hnsw")
        conn.execute(
            f"CREATE INDEX items_embedding_hnsw ON items "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction})"
        )
        conn.commit()


def drop_index() -> None:
    with store.connect(owner=True) as conn:
        conn.execute("DROP INDEX IF EXISTS items_embedding_hnsw")
        conn.commit()


def query(tenant: str, vec: list[float], top_k: int, *,
          iterative: str | None = None, ef_search: int | None = None) -> tuple[list[str], float]:
    """1 クエリ投げて item_id の並びと所要時間を返す。"""
    with store.connect(tenant) as conn:
        if ef_search is not None:
            conn.execute(f"SET hnsw.ef_search = {ef_search}")
        if iterative is not None:
            conn.execute(f"SET hnsw.iterative_scan = {iterative}")

        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT item_id FROM items WHERE tenant_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (tenant, vec, top_k),
        ).fetchall()
        ms = (time.perf_counter() - t0) * 1000
    return [r[0] for r in rows], ms


def explain_plan(tenant: str, vec: list[float], top_k: int) -> None:
    """実行計画から索引の使用状況とフィルタの除外件数を出す。

    Rows Removed by Filter が小さいなら、対象テナントの行が
    たまたま近傍に固まっている。データの作り方を疑う。
    """
    with store.connect(tenant) as conn:
        rows = conn.execute(
            "EXPLAIN (ANALYZE) SELECT item_id FROM items WHERE tenant_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (tenant, vec, top_k),
        ).fetchall()

    print("実行計画（要点）")
    for (line,) in rows:
        t = line.strip()
        if t.startswith("Order By:"):
            continue  # ベクトルの全要素が出て読めなくなる
        if any(k in t for k in ("Index Scan", "Seq Scan", "Rows Removed", "Filter:", "Execution Time")):
            print("  ", t[:110])


def recall(truth: list[str], got: list[str]) -> float:
    if not truth:
        return 1.0
    return len(set(truth) & set(got)) / len(truth)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--queries", type=int, default=20)
    args = p.parse_args()

    with store.connect(owner=True) as conn:
        dist = conn.execute(
            "SELECT tenant_id, count(*) FROM items GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    total = sum(c for _, c in dist)
    target = dict(dist).get(args.tenant, 0)
    if target == 0:
        raise SystemExit(f"テナント {args.tenant} にデータがない")

    print(f"全体 {total} 件 / テナント {args.tenant} は {target} 件 "
          f"({target / total:.1%})")
    print(f"  テナント構成: {dict(dist)}\n")

    texts = [f"{c}を探している" for c in
             ("黒いレザーのブーツ", "秋冬のウールコート", "リネンのシャツ",
              "ネイビーのデニム", "レザーのトートバッグ")]
    vecs = (embed.encode(texts) * (args.queries // len(texts) + 1))[: args.queries]

    # 索引なしの厳密検索を正解にする。
    drop_index()
    truths = []
    exact_ms = 0.0
    for v in vecs:
        ids, ms = query(args.tenant, v, args.top_k)
        truths.append(ids)
        exact_ms += ms
    exact_ms /= len(vecs)

    rebuild_hnsw()

    configs = [
        ("HNSW 既定 (iterative off)", {"iterative": "off", "ef_search": 40}),
        ("HNSW ef_search=100", {"iterative": "off", "ef_search": 100}),
        ("HNSW ef_search=400", {"iterative": "off", "ef_search": 400}),
        ("HNSW iterative=relaxed_order", {"iterative": "relaxed_order", "ef_search": 40}),
        ("HNSW iterative=strict_order", {"iterative": "strict_order", "ef_search": 40}),
    ]

    print(f"{'設定':<32}{'再現率':>9}{'返却件数':>10}{'ms':>9}{'対厳密':>9}")
    print("-" * 70)
    print(f"  {'厳密検索（索引なし）':<30}{1.0:>8.3f}{args.top_k:>10}{exact_ms:>9.1f}{1.0:>8.2f}x")

    for name, kw in configs:
        rs, ms_sum, counts = [], 0.0, []
        for v, truth in zip(vecs, truths):
            ids, ms = query(args.tenant, v, args.top_k, **kw)
            rs.append(recall(truth, ids))
            counts.append(len(ids))
            ms_sum += ms
        avg_ms = ms_sum / len(vecs)
        print(f"  {name:<30}{sum(rs)/len(rs):>8.3f}"
              f"{sum(counts)/len(counts):>10.1f}{avg_ms:>9.1f}"
              f"{exact_ms/avg_ms:>8.2f}x")

    print("-" * 70)
    print("  返却件数が top-k を下回るのが取りこぼし。エラーにならないため気づけない")
    print("  再現率は厳密検索の結果を正解としたときの一致率")

    # 索引が本当に使われたかを実行計画で確かめる。
    # 索引を張っただけでプランナが選ぶとは限らず、選ばなければ
    # 「再現率 1.000」は索引の性能ではなく厳密検索の結果になる。
    print()
    explain_plan(args.tenant, vecs[0], args.top_k)


if __name__ == "__main__":
    main()

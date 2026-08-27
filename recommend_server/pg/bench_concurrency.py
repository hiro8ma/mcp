"""同時実行数を上げたときの遅延を測る。

「規模そのものより、同時実行クエリ数や更新の偏りで詰まるケースの方が実務では多い」
という指摘を確かめる。件数を増やすベンチはよく見るが、
同時実行を上げるベンチは少ない。実際に詰まるのは後者という主張になる。

ベクトル検索は 1 クエリあたりの CPU 消費が大きい。距離計算が次元数に比例し、
HNSW ならグラフを辿る分も乗る。通常の btree 検索と違って I/O 待ちではなく
CPU で詰まるため、コア数を超えた同時実行で急激に劣化する。

平均ではなく p95 を見る。平均は詰まりを隠す。
"""

from __future__ import annotations

import argparse
import os
import statistics
import threading
import time

from . import embed, store

TEXTS = [
    "黒いレザーのブーツを探している",
    "秋冬に着るウールのコート",
    "リネンの白いシャツ",
    "ネイビーのデニムパンツ",
    "レザーのトートバッグ",
]


def worker(tenant: str, vecs: list[list[float]], top_k: int, deadline: float,
           out: list[float], barrier: threading.Barrier) -> None:
    """接続を 1 本開いて、締め切りまでクエリを投げ続ける。

    接続の確立をループ内に含めない。測りたいのは検索の遅延であって
    接続コストではない。
    """
    with store.connect(tenant) as conn:
        barrier.wait()  # 全スレッドが接続を終えてから一斉に開始する
        i = 0
        while time.perf_counter() < deadline:
            v = vecs[i % len(vecs)]
            i += 1
            t0 = time.perf_counter()
            conn.execute(
                "SELECT item_id FROM items WHERE tenant_id = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (tenant, v, top_k),
            ).fetchall()
            out.append((time.perf_counter() - t0) * 1000)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(len(s) * p))
    return s[k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    vecs = embed.encode(TEXTS)
    cores = os.cpu_count() or 1

    with store.connect(owner=True) as conn:
        n = int(conn.execute("SELECT count(*) FROM items").fetchone()[0])
        has_idx = conn.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename='items' AND indexdef LIKE '%hnsw%'"
        ).fetchone()[0]

    print(f"{n} 行 / 索引 {'あり' if has_idx else 'なし（厳密検索）'} / "
          f"クライアント側の論理コア {cores}\n")
    print(f"{'同時実行':>8}{'スループット':>13}{'平均 ms':>10}{'p50':>9}{'p95':>9}{'p99':>9}")
    print("-" * 60)

    base_tps = None
    for c in args.levels:
        results: list[list[float]] = [[] for _ in range(c)]
        barrier = threading.Barrier(c)
        deadline = time.perf_counter() + args.seconds + 1.0

        threads = [
            threading.Thread(target=worker,
                             args=(args.tenant, vecs, args.top_k, deadline, results[i], barrier))
            for i in range(c)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0

        lat = [x for r in results for x in r]
        tps = len(lat) / elapsed
        if base_tps is None:
            base_tps = tps

        print(f"{c:>8}{tps:>12.0f}/s{statistics.fmean(lat):>10.2f}"
              f"{percentile(lat, 0.50):>9.2f}{percentile(lat, 0.95):>9.2f}"
              f"{percentile(lat, 0.99):>9.2f}")

    print("-" * 60)
    print("  p95 を見る。平均は詰まりを隠す")
    print("  スループットが同時実行数に比例しなくなった点が飽和の境界")


if __name__ == "__main__":
    main()

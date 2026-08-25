"""索引ごとの再現率と速度を測る。

索引なしの結果を正解とみなし、HNSW と IVFFlat がそれを何割再現できるかを見る。
近似最近傍なので、速くなるかわりに取りこぼしが出る。その量を数字にするのが目的。
"""

from __future__ import annotations

import argparse
import statistics
import time

from . import embed, seed, store

QUERIES = [
    "秋冬に着られる暖かいアウター",
    "オフィスに履いていける革靴",
    "休日に使える大きめのバッグ",
    "重ね着しやすい薄手のニット",
    "きれいめに見えるパンツ",
    "普段使いのスニーカー",
    "差し色になる小物",
    "オーバーサイズのシャツ",
]


def _run(tenant: str, vectors: list[list[float]], top_k: int) -> tuple[list[list[str]], float]:
    latencies: list[float] = []
    results: list[list[str]] = []
    for vec in vectors:
        t0 = time.perf_counter()
        hits = store.search(tenant, vec, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append([h.item_id for h in hits])
    return results, statistics.median(latencies)


def recall(truth: list[list[str]], got: list[list[str]]) -> float:
    scores = []
    for t, g in zip(truth, got):
        if not t:
            continue
        scores.append(len(set(t) & set(g)) / len(t))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="tenant-a")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--lists", type=int, default=100)
    args = p.parse_args()

    total = store.count(args.tenant)
    if total == 0:
        raise SystemExit(f"tenant={args.tenant} にデータがありません。先に seed を実行してください")

    print(f"対象 {total} 件 / top_k={args.top_k} / クエリ {len(QUERIES)} 本\n")
    vectors = embed.encode(QUERIES)

    store.create_index("none")
    truth, base_ms = _run(args.tenant, vectors, args.top_k)
    print(f"{'索引':<10}{'再現率':>10}{'中央値 ms':>12}{'対 exact':>12}")
    print(f"{'なし':<10}{1.0:>10.3f}{base_ms:>12.1f}{'1.00x':>12}")

    for kind, kwargs in (("ivfflat", {"lists": args.lists}), ("hnsw", {})):
        store.create_index(kind, **kwargs)
        got, ms = _run(args.tenant, vectors, args.top_k)
        print(f"{kind:<10}{recall(truth, got):>10.3f}{ms:>12.1f}{base_ms / ms:>11.2f}x")


if __name__ == "__main__":
    main()

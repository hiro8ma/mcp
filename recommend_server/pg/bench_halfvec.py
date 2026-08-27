"""halfvec の索引サイズ・再現率・速度を vector と比べる。

halfvec は 16 bit 浮動小数のベクトル型。1 要素 4 バイトの vector に対して 2 バイトで済む。
索引サイズが半分になるなら、メモリに載る件数が倍になり、実効的な規模上限が変わる。

精度の損失は、埋め込みの各要素が -1 〜 1 程度の範囲に収まることを前提にすると小さい。
ただし「小さい」が「無視できる」かは、実際に再現率を測らないと言えない。

索引だけ halfvec にして本体は vector のまま残す構成も取れる。

    CREATE INDEX ... ON items USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops)

この形なら格納精度は保ったまま索引だけ縮む。ただしクエリ側も同じキャストで
書かないと索引が使われない。書き忘れても正しい結果は返るため、
遅くなっていることに気づきにくい。
"""

from __future__ import annotations

import argparse
import time

from . import embed, store

TEXTS = [
    "黒いレザーのブーツを探している",
    "秋冬に着るウールのコート",
    "リネンの白いシャツ",
    "ネイビーのデニムパンツ",
    "レザーのトートバッグ",
    "オーバーサイズのニット",
    "カーキのナイロンブルゾン",
    "ベージュのローファー",
]

INDEXES = {
    "vector (hnsw)": (
        "CREATE INDEX bench_idx ON items USING hnsw (embedding vector_cosine_ops)",
        "SELECT tenant_id, item_id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
    ),
    "halfvec (hnsw)": (
        "CREATE INDEX bench_idx ON items USING hnsw "
        "((embedding::halfvec(DIM)) halfvec_cosine_ops)",
        "SELECT tenant_id, item_id FROM items "
        "ORDER BY embedding::halfvec(DIM) <=> %s::halfvec(DIM) LIMIT %s",
    ),
}


def drop() -> None:
    with store.connect(owner=True) as conn:
        conn.execute("DROP INDEX IF EXISTS bench_idx")
        conn.commit()


def build(ddl: str) -> float:
    drop()
    with store.connect(owner=True) as conn:
        t0 = time.perf_counter()
        conn.execute(ddl)
        conn.commit()
    return time.perf_counter() - t0


def index_size() -> str:
    with store.connect(owner=True) as conn:
        return conn.execute("SELECT pg_size_pretty(pg_relation_size('bench_idx'))").fetchone()[0]


def index_bytes() -> int:
    with store.connect(owner=True) as conn:
        return int(conn.execute("SELECT pg_relation_size('bench_idx')").fetchone()[0])


def run(sql: str, vecs: list[list[float]], top_k: int) -> tuple[list[set], float]:
    out, total = [], 0.0
    with store.connect(owner=True) as conn:
        for v in vecs:
            t0 = time.perf_counter()
            rows = conn.execute(sql, (v, top_k)).fetchall()
            total += time.perf_counter() - t0
            out.append({(r[0], r[1]) for r in rows})
    return out, total / len(vecs) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    dim = embed.dimension()
    vecs = embed.encode(TEXTS)

    with store.connect(owner=True) as conn:
        n = int(conn.execute("SELECT count(*) FROM items").fetchone()[0])
        table = conn.execute("SELECT pg_size_pretty(pg_relation_size('items'))").fetchone()[0]
    print(f"{n} 行 / {dim} 次元 / テーブル本体 {table}\n")

    # 索引なしの厳密検索を正解にする。
    drop()
    truth, exact_ms = run(
        "SELECT tenant_id, item_id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
        vecs, args.top_k)
    print(f"{'索引':<18}{'サイズ':>10}{'構築 s':>9}{'再現率':>9}{'検索 ms':>10}{'対厳密':>9}")
    print("-" * 66)
    print(f"  {'なし（厳密）':<16}{'-':>10}{'-':>9}{1.0:>9.3f}{exact_ms:>10.2f}{1.0:>8.2f}x")

    sizes = {}
    for name, (ddl, sql) in INDEXES.items():
        ddl = ddl.replace("DIM", str(dim))
        sql = sql.replace("DIM", str(dim))
        secs = build(ddl)
        sizes[name] = index_bytes()
        got, ms = run(sql, vecs, args.top_k)
        rec = sum(len(t & g) / len(t) for t, g in zip(truth, got)) / len(truth)
        print(f"  {name:<16}{index_size():>10}{secs:>9.2f}{rec:>9.3f}{ms:>10.2f}"
              f"{exact_ms / ms:>8.2f}x")

    drop()
    print("-" * 66)
    if len(sizes) == 2:
        a, b = sizes["vector (hnsw)"], sizes["halfvec (hnsw)"]
        print(f"  索引サイズ比 halfvec / vector = {b / a:.2f}"
              f"（{a:,} → {b:,} バイト）")
    print("  再現率は索引なしの厳密検索を正解としたときの一致率")


if __name__ == "__main__":
    main()

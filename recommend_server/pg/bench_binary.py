"""binary_quantize による二段構えの検索を測る。

bit 型は 1 要素 1 ビットで、vector の 1/32 に縮む。
ただしビット化で情報が落ちるため、bit だけで最終結果を決めると精度が足りない。

そこで二段に分ける。

  1 段目  bit 索引（ハミング距離）で候補を多めに集める
  2 段目  集めた候補だけを vector の厳密な距離で並べ直す

索引は小さいまま、最終順位は元の精度で決まる。
候補数（oversample）が効き幅を決める。少なすぎると 1 段目で正解を取りこぼし、
2 段目で拾い直せない。多すぎると 2 段目の距離計算が増えて速度の利点が消える。

ここで ef_search を候補数に合わせて上げる必要がある。
LIMIT を大きく書いても、HNSW は ef_search 件しか候補を持たないため、
それを超える件数は返らない。実測では ef_search=40 のとき LIMIT 320 に対して
43 件しか返らなかった。警告もエラーも出ない。

ef_search は精度のつまみであると同時に、返却件数の上限でもある。
リランキングは候補を多めに取ることが前提の手法なので、
これを知らないと手法そのものが働かない。

本体は vector のまま持ち、索引だけ式インデックスで bit にする。

    CREATE INDEX ... USING hnsw ((binary_quantize(embedding)::bit(384)) bit_hamming_ops)
"""

from __future__ import annotations

import argparse
import time

from . import embed, store

TEXTS = [
    "黒いレザーのブーツを探している", "秋冬に着るウールのコート", "リネンの白いシャツ",
    "ネイビーのデニムパンツ", "レザーのトートバッグ", "オーバーサイズのニット",
    "カーキのナイロンブルゾン", "ベージュのローファー",
]


def drop_all() -> None:
    with store.connect(owner=True) as conn:
        for n in ("bq_idx", "vec_idx"):
            conn.execute(f"DROP INDEX IF EXISTS {n}")
        conn.commit()


def size_of(idx: str) -> tuple[int, str]:
    with store.connect(owner=True) as conn:
        b = int(conn.execute(f"SELECT pg_relation_size('{idx}')").fetchone()[0])
        p = conn.execute(f"SELECT pg_size_pretty(pg_relation_size('{idx}'))").fetchone()[0]
    return b, p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--oversample", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = p.parse_args()

    dim = embed.dimension()
    vecs = embed.encode(TEXTS)

    with store.connect(owner=True) as conn:
        n = int(conn.execute("SELECT count(*) FROM items").fetchone()[0])

    drop_all()

    # 索引なしの厳密検索を正解にする。
    def exact(v):
        with store.connect(owner=True) as conn:
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT tenant_id, item_id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
                (v, args.top_k)).fetchall()
            return {(a, b) for a, b in rows}, (time.perf_counter() - t0) * 1000

    truths, exact_ms = [], 0.0
    for v in vecs:
        t, ms = exact(v)
        truths.append(t); exact_ms += ms
    exact_ms /= len(vecs)

    with store.connect(owner=True) as conn:
        t0 = time.perf_counter()
        conn.execute(f"CREATE INDEX vec_idx ON items USING hnsw (embedding vector_cosine_ops)")
        conn.commit()
        vec_build = time.perf_counter() - t0
        vec_bytes, vec_pretty = size_of("vec_idx")

        t0 = time.perf_counter()
        conn.execute(
            f"CREATE INDEX bq_idx ON items USING hnsw "
            f"((binary_quantize(embedding)::bit({dim})) bit_hamming_ops)")
        conn.commit()
        bq_build = time.perf_counter() - t0
        bq_bytes, bq_pretty = size_of("bq_idx")

    print(f"{n} 行 / {dim} 次元 / top-{args.top_k}\n")
    print(f"{'索引':<26}{'サイズ':>10}{'構築 s':>9}")
    print("-" * 47)
    print(f"  {'vector (hnsw cosine)':<24}{vec_pretty:>10}{vec_build:>9.2f}")
    print(f"  {'bit (hnsw hamming)':<24}{bq_pretty:>10}{bq_build:>9.2f}")
    print(f"  {'比':<24}{bq_bytes / vec_bytes:>9.2f}x")

    # vector の HNSW だけで引いた場合。
    def hnsw_only(v):
        with store.connect(owner=True) as conn:
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT tenant_id, item_id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
                (v, args.top_k)).fetchall()
            return {(a, b) for a, b in rows}, (time.perf_counter() - t0) * 1000

    # 二段構え。bit で候補を集め、vector で並べ直す。
    def two_stage(v, cand):
        with store.connect(owner=True) as conn:
            # 候補数を満たすには ef_search を候補数以上にする。
            # 余裕を持たせないとグラフの探索が候補数ちょうどに届かない。
            conn.execute(f"SET hnsw.ef_search = {max(40, cand * 2)}")
            t0 = time.perf_counter()
            rows = conn.execute(
                f"""
                SELECT tenant_id, item_id FROM (
                    SELECT tenant_id, item_id, embedding
                    FROM items
                    ORDER BY binary_quantize(embedding)::bit({dim})
                             <~> binary_quantize(%s::vector)::bit({dim})
                    LIMIT %s
                ) c
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (v, cand, v, args.top_k)).fetchall()
            return {(a, b) for a, b in rows}, (time.perf_counter() - t0) * 1000

    def report(label, fn):
        rs, ms = [], 0.0
        for v, truth in zip(vecs, truths):
            got, t = fn(v)
            rs.append(len(truth & got) / len(truth)); ms += t
        ms /= len(vecs)
        print(f"  {label:<28}{sum(rs)/len(rs):>9.3f}{ms:>10.2f}{exact_ms/ms:>9.2f}x")

    print()
    print(f"{'方式':<30}{'再現率':>9}{'ms':>10}{'対厳密':>10}")
    print("-" * 60)
    print(f"  {'厳密検索（索引なし相当）':<28}{1.0:>9.3f}{exact_ms:>10.2f}{1.0:>9.2f}x")
    report("vector HNSW のみ", hnsw_only)
    for o in args.oversample:
        c = args.top_k * o
        report(f"二段構え 候補 {c} 件 (ef={max(40, c * 2)})", lambda v, c=c: two_stage(v, c))

    drop_all()
    print("-" * 60)
    print("  候補数が少ないと 1 段目で落ちた正解を 2 段目で拾えない")
    print("  ef_search を候補数に合わせて上げないと、LIMIT を書いても候補が集まらない")


if __name__ == "__main__":
    main()

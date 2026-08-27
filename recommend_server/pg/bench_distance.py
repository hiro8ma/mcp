"""距離演算子ごとの速度と、subvector による次元削減の再現率を測る。

前半 正規化済みベクトルでは内積が最速か
    コサイン距離は計算の中で正規化を含む。すでに長さ 1 なら冗長な計算になるため、
    内積のほうが速いとされる。ただし索引を対応する opclass で作り直さないと
    Seq Scan に落ちて逆に遅くなる。索引ごと作り分けて比べる。

後半 subvector で次元を削っても順位が保たれるか
    subvector(v, 1, n) は先頭 n 次元を切り出す。索引上限を超えるモデルへの対応や、
    小次元で粗く絞ってから全次元で並べ直す階層検索の土台になる。

    ただしこれが成り立つのは Matryoshka Embeddings のように
    「先頭の次元に重要な情報が集まるよう学習されたモデル」に限る。
    通常のモデルでは全次元に情報が分散しているため、先頭を切り出す根拠がない。
    測定に使う all-MiniLM-L6-v2 は Matryoshka 学習ではないので、
    どの程度落ちるかを確かめる。
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

OPS = [
    ("<=>", "vector_cosine_ops", "コサイン"),
    ("<#>", "vector_ip_ops", "内積"),
    ("<->", "vector_l2_ops", "L2"),
]


def drop(name: str = "d_idx") -> None:
    with store.connect(owner=True) as conn:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
        conn.commit()


def measure(op: str, opclass: str | None, vecs, top_k: int, repeat: int) -> tuple[float, bool]:
    """索引を作って計測する。opclass=None なら索引なし。

    索引が実際に使われたかを EXPLAIN で確かめる。使われないまま速度だけ比べると、
    Seq Scan 同士を比べていることになる。
    """
    drop()
    if opclass:
        with store.connect(owner=True) as conn:
            conn.execute(f"CREATE INDEX d_idx ON items USING hnsw (embedding {opclass})")
            conn.commit()

    used = True
    with store.connect(owner=True) as conn:
        if opclass:
            plan = conn.execute(
                f"EXPLAIN SELECT item_id FROM items ORDER BY embedding {op} %s::vector LIMIT %s",
                (vecs[0], top_k)).fetchall()
            used = any("Index Scan" in r[0] for r in plan)

        # 1 度空回ししてキャッシュを温める。初回の物理読み取りを混ぜない。
        for v in vecs:
            conn.execute(
                f"SELECT item_id FROM items ORDER BY embedding {op} %s::vector LIMIT %s",
                (v, top_k)).fetchall()

        t0 = time.perf_counter()
        for _ in range(repeat):
            for v in vecs:
                conn.execute(
                    f"SELECT item_id FROM items ORDER BY embedding {op} %s::vector LIMIT %s",
                    (v, top_k)).fetchall()
        ms = (time.perf_counter() - t0) / (repeat * len(vecs)) * 1000
    return ms, used


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--dims", type=int, nargs="+", default=[384, 256, 192, 128, 64, 32])
    args = p.parse_args()

    dim = embed.dimension()
    vecs = embed.encode(TEXTS)

    with store.connect(owner=True) as conn:
        n = int(conn.execute("SELECT count(*) FROM items").fetchone()[0])
    print(f"{n} 行 / {dim} 次元 / top-{args.top_k} / {args.repeat} 回平均\n")

    print("=== 距離演算子ごとの速度（索引は opclass を合わせて作り直す）===")
    print(f"{'演算子':<20}{'索引':>10}{'ms':>10}{'コサイン比':>12}")
    print("-" * 54)
    base = None
    for op, opclass, label in OPS:
        ms, used = measure(op, opclass, vecs, args.top_k, args.repeat)
        if base is None:
            base = ms
        print(f"  {label + ' ' + op:<18}{'使用' if used else 'Seq Scan':>10}"
              f"{ms:>10.3f}{base / ms:>11.2f}x")

    # opclass を合わせない場合。
    drop()
    with store.connect(owner=True) as conn:
        conn.execute("CREATE INDEX d_idx ON items USING hnsw (embedding vector_cosine_ops)")
        conn.commit()
    ms, used = measure("<#>", None, vecs, args.top_k, args.repeat) if False else (None, None)
    with store.connect(owner=True) as conn:
        plan = conn.execute(
            "EXPLAIN SELECT item_id FROM items ORDER BY embedding <#> %s::vector LIMIT %s",
            (vecs[0], args.top_k)).fetchall()
        used = any("Index Scan" in r[0] for r in plan)
        t0 = time.perf_counter()
        for _ in range(args.repeat):
            for v in vecs:
                conn.execute(
                    "SELECT item_id FROM items ORDER BY embedding <#> %s::vector LIMIT %s",
                    (v, args.top_k)).fetchall()
        ms = (time.perf_counter() - t0) / (args.repeat * len(vecs)) * 1000
    print(f"  {'内積 <#> (コサイン索引)':<18}{'使用' if used else 'Seq Scan':>10}"
          f"{ms:>10.3f}{base / ms:>11.2f}x")

    drop()
    print("-" * 54)
    print("  opclass を合わせないと索引が使われず、演算子を変えた意味が消える")

    print("\n=== subvector で次元を削ったときの再現率 ===")
    print(f"  モデル {embed.MODEL_NAME}（Matryoshka 学習ではない）\n")

    with store.connect(owner=True) as conn:
        truth = []
        for v in vecs:
            rows = conn.execute(
                "SELECT tenant_id, item_id FROM items ORDER BY embedding <=> %s::vector LIMIT %s",
                (v, args.top_k)).fetchall()
            truth.append({(a, b) for a, b in rows})

        print(f"{'次元':>8}{'再現率':>10}{'索引サイズ':>12}")
        print("-" * 34)
        for d in args.dims:
            conn.execute("DROP INDEX IF EXISTS sv_idx")
            conn.execute(
                f"CREATE INDEX sv_idx ON items USING hnsw "
                f"((subvector(embedding, 1, {d})::vector({d})) vector_cosine_ops)")
            conn.commit()
            size = conn.execute("SELECT pg_size_pretty(pg_relation_size('sv_idx'))").fetchone()[0]

            rs = []
            for v, t in zip(vecs, truth):
                rows = conn.execute(
                    f"SELECT tenant_id, item_id FROM items "
                    f"ORDER BY subvector(embedding, 1, {d})::vector({d}) "
                    f"<=> subvector(%s::vector, 1, {d})::vector({d}) LIMIT %s",
                    (v, args.top_k)).fetchall()
                got = {(a, b) for a, b in rows}
                rs.append(len(t & got) / len(t))
            print(f"{d:>8}{sum(rs) / len(rs):>10.3f}{size:>12}")

        conn.execute("DROP INDEX IF EXISTS sv_idx")
        conn.commit()

    print("-" * 34)
    print("  再現率は全 384 次元の厳密検索を正解としたときの一致率")
    print("  Matryoshka 学習でないモデルでは先頭次元に情報が集まっていない")


if __name__ == "__main__":
    main()

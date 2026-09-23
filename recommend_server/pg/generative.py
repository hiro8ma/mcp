"""生成型推薦の最小の形。次のアイテムを Semantic ID（SID）として生成する。

TIGER（NeurIPS 2023）の構成を、学習の軽い形に置き換えて試す。

  1. アイテムの埋め込みを残差 k-means で L 段に量子化し、コードの組（SID）にする
  2. 同じ SID を持つアイテムは、末尾に重複を区別する番号を足して一意にする
  3. 履歴のアイテムの SID を文脈に、次のアイテムの SID をトークンごとに予測する
  4. 実在する SID の接頭辞の木（trie）だけをたどるビームサーチで上位 K 件を出す

TIGER は系列を Transformer で学習する。ここでは数え上げの n-gram で代える。
行動ログに順序が無く（seed_interactions は一括投入で created_at がほぼ同じ）、
系列モデルが学べる「直前のアイテム」がそもそも無いため。
文脈は履歴の全アイテムとし、各アイテムを文脈にした確率を平均する。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from . import interactions, store

SID = tuple[int, ...]


@dataclass(frozen=True)
class Rec:
    item_id: str
    score: float
    why: str


def residual_kmeans(emb: np.ndarray, levels: int, k: int,
                    seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """段ごとに k-means をかけ、残差を次の段に渡す。"""
    residual = np.asarray(emb, dtype=np.float64).copy()
    codes = np.zeros((len(residual), levels), dtype=int)
    books: list[np.ndarray] = []
    for level in range(levels):
        km = KMeans(n_clusters=min(k, len(residual)), n_init=4, random_state=seed).fit(residual)
        codes[:, level] = km.labels_
        books.append(km.cluster_centers_)
        residual -= km.cluster_centers_[km.labels_]
    return codes, books


def dedupe(codes: Iterable[Sequence[int]]) -> list[SID]:
    """TIGER と同じく、全アイテムの末尾に同じコード内での通し番号を足す。"""
    seen: Counter[SID] = Counter()
    out: list[SID] = []
    for row in codes:
        key = tuple(int(c) for c in row)
        out.append(key + (seen[key],))
        seen[key] += 1
    return out


@dataclass(frozen=True)
class Quality:
    levels: int
    k: int
    usage: tuple[float, ...]
    perplexity: tuple[float, ...]
    collision: float
    max_bucket: int
    recon: float
    variance: float


def quality(emb: np.ndarray, levels: int, k: int, seed: int = 0) -> Quality:
    """段ごとの使用率、衝突率、再構成誤差を測る。

    衝突率は、L 段のコードが他のアイテムと重なったアイテムの割合。
    k-means は全コードを必ず使うため、使用率だけでは偏りが見えない。
    コード分布の perplexity（実効的に使われているコード数）を K で割った値も出す。
    再構成誤差は、各段の中心の和で埋め込みを戻したときの二乗誤差の平均。
    比べる基準として、平均ベクトル 1 本で戻したときの誤差（分散）も返す。
    """
    codes, books = residual_kmeans(emb, levels, k, seed)
    approx = sum(books[level][codes[:, level]] for level in range(levels))
    groups = Counter(map(tuple, codes.tolist()))
    emb = np.asarray(emb)

    def perplexity(col: np.ndarray) -> float:
        p = np.bincount(col) / len(col)
        p = p[p > 0]
        return float(np.exp(-np.sum(p * np.log(p))))

    return Quality(
        levels=levels,
        k=k,
        usage=tuple(len(np.unique(codes[:, level])) / k for level in range(levels)),
        perplexity=tuple(perplexity(codes[:, level]) / k for level in range(levels)),
        collision=sum(c for c in groups.values() if c > 1) / len(codes),
        max_bucket=max(groups.values()),
        recon=float(np.mean(np.sum((emb - approx) ** 2, axis=1))),
        variance=float(np.mean(np.sum((emb - emb.mean(axis=0)) ** 2, axis=1))),
    )


class Trie:
    def __init__(self, sids: Iterable[SID]):
        self._next: dict[SID, set[int]] = defaultdict(set)
        for sid in sids:
            for i, tok in enumerate(sid):
                self._next[sid[:i]].add(tok)

    def next_tokens(self, prefix: SID) -> list[int]:
        return sorted(self._next.get(prefix, ()))


class SidNgram:
    """P(次の SID のトークン | 文脈アイテムの SID の接頭辞, 生成済みの接頭辞)。

    文脈は粗い順（接頭辞なし）から細かい順（アイテムそのもの）へ重ね、
    細かい文脈の数が少なければ粗い文脈の確率に寄せる（Dirichlet 平滑化）。
    細かい文脈は併用の記憶、粗い文脈は内容の近さによる一般化にあたる。
    """

    def __init__(self, sids: dict[str, SID], histories: dict[str, set[str]],
                 beta: float = 1.0):
        self.sids = sids
        self.items = {sid: item_id for item_id, sid in sids.items()}
        self.trie = Trie(sids.values())
        self.depth = len(next(iter(sids.values())))
        self.beta = beta
        self.histories = histories
        self.counts: Counter[tuple] = Counter()
        for hist in histories.values():
            self.counts.update(self._pairs(hist))

    def _pairs(self, hist: Iterable[str]) -> Iterable[tuple]:
        known = [self.sids[i] for i in hist if i in self.sids]
        for a in known:
            for b in known:
                if a == b:
                    continue
                for m in range(self.depth + 1):
                    ctx = a[:m]
                    for level in range(self.depth):
                        yield (ctx, b[:level], b[level])
                        yield (ctx, b[:level], None)

    def own_counts(self, user_id: str) -> Counter[tuple]:
        """学習に入った本人の分。推薦時に引いて、本人の正解を覚えた状態を避ける。"""
        return Counter(self._pairs(self.histories.get(user_id, ())))

    def prob(self, ctx: SID, prefix: SID, token: int, minus: Counter[tuple]) -> float:
        p = 1.0 / len(self.trie.next_tokens(prefix))
        for m in range(self.depth + 1):
            c = ctx[:m]
            n = self.counts[(c, prefix, None)] - minus[(c, prefix, None)]
            if n <= 0:
                break
            x = self.counts[(c, prefix, token)] - minus[(c, prefix, token)]
            p = (x + self.beta * p) / (n + self.beta)
        return p

    def generate(self, contexts: Sequence[SID], exclude: set[SID], top_k: int,
                 beam: int = 50, minus: Counter[tuple] | None = None) -> list[tuple[SID, float]]:
        """trie に載った子だけを展開するビームサーチ。スコアは文脈ごとの確率の平均。"""
        if not contexts:
            return []
        minus = minus or Counter()
        width = max(beam, top_k + len(exclude))
        beams: list[tuple[SID, list[float]]] = [((), [1.0] * len(contexts))]
        for _ in range(self.depth):
            cand = []
            for prefix, probs in beams:
                for tok in self.trie.next_tokens(prefix):
                    cand.append((prefix + (tok,), [
                        p * self.prob(c, prefix, tok, minus) for c, p in zip(contexts, probs)]))
            cand.sort(key=lambda x: (-sum(x[1]), x[0]))
            beams = cand[:width]
        out = [(sid, sum(ps) / len(ps)) for sid, ps in beams if sid not in exclude]
        return out[:top_k]


def _catalog_embeddings(tenant_id: str) -> tuple[list[str], np.ndarray]:
    with store.connect(tenant_id) as conn:
        rows = conn.execute(
            "SELECT item_id, embedding FROM items WHERE tenant_id = %s ORDER BY item_id",
            (tenant_id,)).fetchall()
    return [r[0] for r in rows], np.array([store._to_tuple(r[1]) for r in rows])


_models: dict[tuple[str, int, int], SidNgram] = {}


def model(tenant_id: str, levels: int = 3, k: int = 32) -> SidNgram:
    key = (tenant_id, levels, k)
    if key not in _models:
        ids, emb = _catalog_embeddings(tenant_id)
        codes, _ = residual_kmeans(emb, levels, k)
        histories: dict[str, set[str]] = defaultdict(set)
        for item_id, users in interactions.all_item_users(tenant_id).items():
            for u in users:
                histories[u].add(item_id)
        _models[key] = SidNgram(dict(zip(ids, dedupe(codes))), dict(histories))
    return _models[key]


def recommend(tenant_id: str, user_id: str, top_k: int = 10, levels: int = 3,
              k: int = 32, beam: int = 50) -> list[Rec]:
    m = model(tenant_id, levels, k)
    mine = interactions.history(tenant_id, user_id)
    contexts = [m.sids[i] for i in mine if i in m.sids]
    exclude = set(contexts)
    return [
        Rec(item_id=m.items[sid], score=round(p, 6), why=f"SID {sid} を生成")
        for sid, p in m.generate(contexts, exclude, top_k, beam, m.own_counts(user_id))
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", default="demo")
    p.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--k", type=int, nargs="+", default=[16, 32, 64])
    args = p.parse_args()

    ids, emb = _catalog_embeddings(args.tenant)
    print(f"テナント {args.tenant} / カタログ {len(ids)} 件 / 埋め込み {emb.shape[1]} 次元\n")
    print(f"{'L':>3}{'K':>5}  {'使用率':<8}{'実効利用率（段ごと）':<24}"
          f"{'衝突率':>8}{'最大重複':>8}{'再構成誤差':>11}{'対分散':>8}")
    print("-" * 84)
    for levels in args.levels:
        for k in args.k:
            q = quality(emb, levels, k)
            used = f"{min(q.usage):.0%}"
            ppl = " / ".join(f"{u:.0%}" for u in q.perplexity)
            print(f"{levels:>3}{k:>5}  {used:<8}{ppl:<26}{q.collision:>8.1%}{q.max_bucket:>8}"
                  f"{q.recon:>11.4f}{q.recon / q.variance:>8.1%}")
    print("-" * 84)
    print("  使用率      全段のうち最も低い段の、使われたコードの割合")
    print("  実効利用率  コード分布の perplexity を K で割った値。偏るほど下がる")
    print("  衝突率      L 段のコードが他のアイテムと重なったアイテムの割合")
    print("  対分散      再構成誤差を、平均ベクトル 1 本で戻したときの誤差で割った値")

if __name__ == "__main__":
    main()

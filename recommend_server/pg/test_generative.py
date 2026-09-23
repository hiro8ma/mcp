"""生成型推薦（SID）の検証。DB を使わない部分だけを対象にする。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import generative


class TestDedupe:
    def test_同じコードのアイテムにも一意のSIDが付く(self):
        sids = generative.dedupe([[1, 2], [1, 2], [3, 4], [1, 2]])
        assert sids == [(1, 2, 0), (1, 2, 1), (3, 4, 0), (1, 2, 2)]

    def test_同じ埋め込みを量子化しても一意になる(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(20, 8))
        emb[1] = emb[0]
        emb[2] = emb[0]
        codes, _ = generative.residual_kmeans(emb, levels=2, k=4)

        assert len({tuple(c) for c in codes.tolist()}) < len(emb)
        assert len(set(generative.dedupe(codes))) == len(emb)


SIDS = {"a": (0, 0, 0), "b": (0, 1, 0), "c": (1, 0, 0)}


class TestTrieConstraint:
    def model(self):
        return generative.SidNgram(SIDS, {"u1": {"a", "b"}, "u2": {"b", "c"}})

    def test_カタログにないSIDを生成しない(self):
        """各段の値は (0,1) と (0,1) が出るが、(1,1,0) のような組は実在しない。

        上位 K をカタログより多く求めても、実在する SID だけを返す。
        """
        m = self.model()
        out = m.generate([SIDS["a"]], exclude=set(), top_k=10)

        assert {sid for sid, _ in out} == set(SIDS.values())

    def test_履歴のアイテムを除く(self):
        m = self.model()
        out = m.generate([SIDS["a"]], exclude={SIDS["a"]}, top_k=10)

        assert {sid for sid, _ in out} == {SIDS["b"], SIDS["c"]}

    def test_併用の多いアイテムが先に出る(self):
        m = self.model()
        out = m.generate([SIDS["a"]], exclude={SIDS["a"]}, top_k=1)

        assert out[0][0] == SIDS["b"]


class TestOwnCounts:
    def test_本人の分を引くと本人の併用を覚えていない(self):
        """leave-one-out で本人の正解を学習に含めない仕組みを確かめる。

        a と b の併用は u1 だけにある。u1 の分を引けば、a を文脈にしても
        b と c を区別する材料が残らず、同じ確率になる。
        """
        sids = {"a": (0, 0), "b": (1, 0), "c": (2, 0)}
        m = generative.SidNgram(sids, {"u1": {"a", "b"}})

        with_own = m.prob(sids["a"], (), 1, Counter())
        without = m.prob(sids["a"], (), 1, m.own_counts("u1"))

        assert with_own > without
        assert without == m.prob(sids["a"], (), 2, m.own_counts("u1"))

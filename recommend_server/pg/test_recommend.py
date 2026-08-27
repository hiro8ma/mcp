"""推薦アルゴリズムの検証。DB を使わない部分だけを対象にする。

類似度計算と偏りの指標は純粋な計算なので、DB 無しで確かめられる。
推薦が「それらしい結果」を返すことと、計算が正しいことは別なので、
値が決まっている入力で照合する。
"""

from __future__ import annotations

import math

import pytest

from . import collaborative, evaluate


class TestCosine:
    def test_共通アイテムが無ければ0(self):
        assert collaborative._cosine({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_完全に一致すれば1(self):
        h = {"a": 1.0, "b": 1.0}
        assert collaborative._cosine(h, dict(h)) == pytest.approx(1.0)

    def test_片方が広く浅いと類似度が下がる(self):
        """共通アイテム数が同じでも、片方が何にでも触れているなら類似度は下がる。

        ノルムを各自の全履歴で取るため。この正規化が無いと、
        大量に行動しているユーザーが誰とでも似ていることになる。
        """
        me = {"a": 1.0, "b": 1.0}
        narrow = {"a": 1.0, "b": 1.0, "c": 1.0}
        wide = {"a": 1.0, "b": 1.0, **{f"x{i}": 1.0 for i in range(50)}}

        assert collaborative._cosine(me, narrow) > collaborative._cosine(me, wide)

    def test_手計算と一致する(self):
        # 共通は a のみ。dot=1。|me|=sqrt(2), |other|=sqrt(2)
        me = {"a": 1.0, "b": 1.0}
        other = {"a": 1.0, "c": 1.0}
        assert collaborative._cosine(me, other) == pytest.approx(1 / 2)

    def test_評価値の重みが効く(self):
        me = {"a": 5.0, "b": 1.0}
        same = {"a": 5.0, "b": 1.0}
        flipped = {"a": 1.0, "b": 5.0}
        assert collaborative._cosine(me, same) > collaborative._cosine(me, flipped)


class TestGini:
    def test_完全に均等なら0(self):
        assert evaluate.gini([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_空なら0(self):
        assert evaluate.gini([]) == 0.0

    def test_1品目に集中するほど1に近づく(self):
        even = evaluate.gini([10] * 10)
        skewed = evaluate.gini([100] + [1] * 9)
        assert skewed > even
        assert skewed < 1.0

    def test_偏りが強いほど値が大きい(self):
        assert evaluate.gini([10, 10, 10, 10]) < evaluate.gini([1, 2, 3, 34])

    def test_カバレッジでは見えない偏りを捉える(self):
        """カバレッジは「登場したか」の 0/1 しか見ない。

        10 品目すべてが登場していればカバレッジは同じ 100% になるが、
        1 品目が 1000 回で他が 1 回ずつなら実質 1 品目しか出していない。
        ジニ係数はこの差を数値にする。
        """
        balanced = [100] * 10
        concentrated = [1000] + [1] * 9
        assert len(balanced) == len(concentrated)  # カバレッジは同じ
        assert evaluate.gini(concentrated) > evaluate.gini(balanced) * 3


class TestItemSimilarityNormalization:
    def test_共起数だけでは人気アイテムが常に似てしまう(self):
        """アイテム間型のコサイン正規化が何を防ぐかを確かめる。

        人気アイテム（利用者 1000 人）とニッチなアイテム（利用者 10 人）が
        10 人分共起しているとする。共起数は同じ 10 でも、
        ニッチ同士のほうが関係は強い。正規化しないとこれが区別できない。
        """
        shared = 10
        # 人気アイテムとの共起
        popular = shared / math.sqrt(10 * 1000)
        # ニッチ同士の共起
        niche = shared / math.sqrt(10 * 10)

        assert niche > popular
        assert niche == pytest.approx(1.0)

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


class TestSimilarityDefinitions:
    """教材の図 4-13 の例を数値で固定する。

      ユーザー1  本○ Tシャツ○ 眼鏡×   （推薦を受け取る）
      ユーザー2  本× Tシャツ× 眼鏡○   正反対
      ユーザー3  本○ Tシャツ○ 眼鏡×   全て一致
      ユーザー4  本○ Tシャツ× 眼鏡○   1 つ一致
      ユーザー5  本○ Tシャツ○ 眼鏡○   2 つ一致

    教材は「購入しなかったアイテムは好んでいない」と置く。
    この前提を採るかどうかで、正反対のユーザーの扱いが変わる。
    """

    ITEMS = ["本", "Tシャツ", "眼鏡"]
    BOUGHT = {
        1: {"本", "Tシャツ"},
        2: {"眼鏡"},
        3: {"本", "Tシャツ"},
        4: {"本", "眼鏡"},
        5: {"本", "Tシャツ", "眼鏡"},
    }

    def hist(self, uid):
        return {i: 1.0 for i in self.BOUGHT[uid]}

    def test_教材の順位はどの定義でも一致する(self):
        """似ている順は 3 > 5 > 4 > 2。定義を変えても順位は変わらない。"""
        pearson = collaborative._pearson_over(self.ITEMS)
        for name, sim in (("cosine", collaborative._cosine),
                          ("jaccard", collaborative._jaccard),
                          ("pearson", pearson)):
            scores = {u: sim(self.hist(1), self.hist(u)) for u in (2, 3, 4, 5)}
            order = sorted(scores, key=lambda u: scores[u], reverse=True)
            assert order == [3, 5, 4, 2], f"{name} の順位が {order}"

    def test_購入のみを見る定義は正反対を表現できない(self):
        """値域が 0 以上なので、正反対と無関係が同じ値になる。

        ユーザー2 は教材が「正反対」と呼ぶ相手だが、
        共通アイテムが無いだけの相手と区別がつかない。
        """
        assert collaborative._cosine(self.hist(1), self.hist(2)) == 0.0
        assert collaborative._jaccard(self.hist(1), self.hist(2)) == 0.0

    def test_非購入を見る定義は正反対を負で表す(self):
        pearson = collaborative._pearson_over(self.ITEMS)
        assert pearson(self.hist(1), self.hist(2)) == pytest.approx(-1.0)

    def test_定義によって符号が逆になる相手がいる(self):
        """ユーザー4 は cosine で +0.5、pearson で -0.5 になる。

        min_similarity > 0 で足切りすると、cosine だけが採用する。
        順位が同じでも、閾値の扱いで結果が変わる。
        """
        pearson = collaborative._pearson_over(self.ITEMS)
        c = collaborative._cosine(self.hist(1), self.hist(4))
        p = pearson(self.hist(1), self.hist(4))

        assert c > 0 and p < 0, f"cosine {c} / pearson {p}"

    def test_疎なデータでは正反対を表現できなくなる(self):
        """非購入を「嫌い」と読めるのは、ほぼ全件を見ている場合に限る。

        カタログが大きく 1 人が数件しか触れないと、
        平均が 0 に近づき中心化がほとんど効かない。
        共通点の無い相手の相関が 0 に潰れ、
        正反対を表す負の値が出せなくなる。

        実測では 1176 件のカタログで 1 人平均 8.1 件（非購入 99.3%）、
        共通アイテムが無い相手の pearson が -0.008 だった。
        """
        catalog = [f"item-{i:04d}" for i in range(1000)]
        pearson = collaborative._pearson_over(catalog)

        # 3 アイテムなら正反対は -1.0 になる。
        dense = collaborative._pearson_over(self.ITEMS)
        assert dense(self.hist(1), self.hist(2)) == pytest.approx(-1.0)

        # 同じ「重なりが無い」関係でも、疎なカタログでは 0 近くになる。
        a = {c: 1.0 for c in catalog[:8]}
        b = {c: 1.0 for c in catalog[8:16]}
        sparse = pearson(a, b)

        assert -0.02 < sparse < 0, f"疎なデータでの pearson = {sparse}"
        assert abs(sparse) < 0.05, (
            f"疎なデータで正反対が {sparse} と表現できてしまっている。"
            "この前提が崩れると教材の議論が成り立つ")

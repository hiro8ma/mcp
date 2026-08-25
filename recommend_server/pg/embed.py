"""埋め込みの生成。

ローカルで完結するモデルを使う。API キーもネットワークも要らない。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def dimension() -> int:
    """モデルが出すベクトルの次元。

    スキーマの vector(N) はこの値から組み立てる。
    定数を別に置くと、モデルを変えたときに投入時まで不整合に気づけない。
    しかも失敗するのは行ごとの次元不一致エラーで、原因を辿りにくい。
    """
    return int(_model().get_sentence_embedding_dimension())


def encode(texts: Sequence[str]) -> list[list[float]]:
    vectors = _model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def encode_one(text: str) -> list[float]:
    return encode([text])[0]

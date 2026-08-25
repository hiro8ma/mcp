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


def encode(texts: Sequence[str]) -> list[list[float]]:
    vectors = _model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def encode_one(text: str) -> list[float]:
    return encode([text])[0]

"""
完全一致キャッシュ — AIエンジニアリング Ch10 Step 4 の実装。

同じクエリに対するモデル呼び出しを回避し、レイテンシーとコストを削減。
LRU（Least Recently Used）削除ポリシーを使用。
"""

import hashlib
import json
import time
from collections import OrderedDict


class ResponseCache:
    """LRUベースの完全一致キャッシュ。"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Args:
            max_size: キャッシュの最大エントリ数
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, system_prompt: str, user_prompt: str) -> str:
        """システムプロンプト+ユーザープロンプトからキャッシュキーを生成。"""
        content = f"{system_prompt}|||{user_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, system_prompt: str, user_prompt: str) -> str | None:
        """キャッシュから応答を取得。ヒットしなければNone。"""
        key = self._make_key(system_prompt, user_prompt)

        if key in self._cache:
            entry = self._cache[key]
            # TTLチェック
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                # LRU: アクセスされたので末尾に移動
                self._cache.move_to_end(key)
                self.hits += 1
                return entry["response"]
            else:
                # 期限切れ
                del self._cache[key]

        self.misses += 1
        return None

    def put(self, system_prompt: str, user_prompt: str, response: str):
        """応答をキャッシュに保存。"""
        key = self._make_key(system_prompt, user_prompt)

        # 既存エントリがあれば更新
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = {"response": response, "timestamp": time.time()}
            return

        # 最大サイズ超過時はLRUで削除
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = {"response": response, "timestamp": time.time()}

    def stats(self) -> dict:
        """キャッシュの統計情報。"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


# グローバルキャッシュインスタンス
_cache = ResponseCache(max_size=200, ttl_seconds=3600)


def get_cached_response(system_prompt: str, user_prompt: str) -> str | None:
    """キャッシュから応答を取得。"""
    return _cache.get(system_prompt, user_prompt)


def cache_response(system_prompt: str, user_prompt: str, response: str):
    """応答をキャッシュに保存。"""
    _cache.put(system_prompt, user_prompt, response)


def get_cache_stats() -> dict:
    """キャッシュ統計を取得。"""
    return _cache.stats()

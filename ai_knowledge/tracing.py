"""
トレーシング基盤 — AIエンジニアリング Ch10 モニタリングの実装。

各リクエストの実行パス全体を記録:
- 入力ガードレール
- キャッシュチェック
- モデル推論
- 出力ガードレール

Langfuse統合の準備も含む。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    """トレース内の1ステップ。"""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    """1リクエストのトレース全体。"""

    trace_id: str
    user_prompt: str
    system_prompt: str
    spans: list[Span] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    response: str = ""
    cache_hit: bool = False
    guardrail_triggered: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_prompt": self.user_prompt[:100],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "cache_hit": self.cache_hit,
            "guardrail_triggered": self.guardrail_triggered,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
            "warnings": self.warnings,
        }


class Tracer:
    """リクエストトレーサー。"""

    def __init__(self, log_dir: str = "traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._trace_count = 0
        self._total_duration = 0.0
        self._cache_hits = 0

    def new_trace(self, user_prompt: str, system_prompt: str) -> Trace:
        """新しいトレースを開始。"""
        self._trace_count += 1
        trace = Trace(
            trace_id=f"trace-{self._trace_count:06d}",
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            start_time=time.time(),
        )
        return trace

    def start_span(self, trace: Trace, name: str, **metadata) -> Span:
        """スパンを開始。"""
        span = Span(name=name, start_time=time.time(), metadata=metadata)
        trace.spans.append(span)
        return span

    def end_span(self, span: Span, **metadata):
        """スパンを終了。"""
        span.end_time = time.time()
        span.metadata.update(metadata)

    def end_trace(self, trace: Trace, response: str = ""):
        """トレースを終了してログに記録。"""
        trace.end_time = time.time()
        trace.response = response
        self._total_duration += trace.total_duration_ms
        if trace.cache_hit:
            self._cache_hits += 1

        # ログファイルに追記
        log_file = self.log_dir / "traces.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        """トレーサーの統計。"""
        avg_duration = (
            self._total_duration / self._trace_count
            if self._trace_count > 0
            else 0
        )
        return {
            "total_traces": self._trace_count,
            "avg_duration_ms": round(avg_duration, 1),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                f"{self._cache_hits / self._trace_count:.1%}"
                if self._trace_count > 0
                else "0%"
            ),
        }


# グローバルトレーサーインスタンス
tracer = Tracer()

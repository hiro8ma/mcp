#!/usr/bin/env python3
"""
Utility functions for MCP Agent
共通ユーティリティ関数
"""

import sys
import os
import io
from typing import Any


def safe_str(obj: Any, use_repr: bool = False) -> str:
    """
    オブジェクトをサロゲート文字を除去して文字列化
    """
    text = repr(obj) if use_repr else str(obj)
    if not isinstance(text, str):
        return text

    if sys.platform == "win32":
        try:
            return text.encode('cp932', errors='replace').decode('cp932')
        except Exception:
            pass

    return ''.join(
        char if not (0xD800 <= ord(char) <= 0xDFFF) else '?'
        for char in text
    )


def setup_windows_encoding():
    """Windows環境でのUnicode対応設定"""
    if sys.platform != "win32":
        return

    os.environ["PYTHONIOENCODING"] = "utf-8"

    for stream_name in ['stdout', 'stderr']:
        stream = getattr(sys, stream_name)
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            wrapper = io.TextIOWrapper(
                stream.buffer,
                encoding='utf-8',
                errors='replace'
            )
            setattr(sys, stream_name, wrapper)


class Logger:
    """統一されたログ出力クラス"""

    LEVEL_PRIORITY = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40
    }

    def __init__(self, verbose: bool = True, log_level: str = "INFO"):
        self.verbose = verbose
        self.log_level = log_level.upper()
        self.min_priority = self.LEVEL_PRIORITY.get(self.log_level, 20)

    def should_log(self, level: str) -> bool:
        level = level.upper()
        priority = self.LEVEL_PRIORITY.get(level, 20)
        return priority >= self.min_priority

    def ulog(self, message: str, level: str = "info", always_print: bool = False, show_level: bool = False) -> None:
        parts = level.split(':', 1)
        log_level = parts[0]
        prefix_key = parts[1] if len(parts) > 1 else None

        if not self.should_log(log_level) and not always_print:
            return

        if self.verbose or always_print:
            if prefix_key:
                prefixes = {
                    "session": "[セッション]",
                    "request": "[リクエスト]",
                    "interrupt": "[中断]",
                    "warning": "[警告]",
                    "error": "[エラー]",
                    "info": "[情報]",
                    "retry": "[リトライ]",
                    "connection": "[接続管理]",
                    "init": "[初期化]",
                    "collection": "[収集]",
                    "correction": "[修正]",
                    "success": "[成功]",
                    "classification": "[分類]",
                    "completed": "[完了]",
                    "failed": "[失敗]",
                    "param": "[パラメータ]",
                    "analysis": "[分析]",
                    "llm_judgment": "[LLM判断]",
                    "startup": "[起動]",
                    "command": "[コマンド]",
                }
                prefix = prefixes.get(prefix_key, f"[{prefix_key.upper()}]")
                if show_level:
                    level_prefix = f"[{log_level.upper()}]"
                    print(f"{level_prefix} {prefix} {message}")
                else:
                    print(f"{prefix} {message}")
            else:
                if show_level:
                    level_prefix = f"[{log_level.upper()}]"
                    print(f"{level_prefix} {message}")
                else:
                    print(message)

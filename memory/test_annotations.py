"""ツールの注釈が MCP のツール一覧に載ることを確かめる。

注釈の無いツールは、MCP の仕様の既定で読み取り専用ではなく破壊的とみなされる。
クライアントが注釈で絞り込むと、読み取りだけのツールまで外れる。
"""

import asyncio

from fastmcp import Client

from memory_server import mcp


def list_annotations() -> dict[str, dict]:
    async def run() -> dict[str, dict]:
        async with Client(mcp) as client:
            tools = await client.list_tools()
            return {t.name: (t.annotations.model_dump(exclude_none=True) if t.annotations else {}) for t in tools}

    return asyncio.run(run())


def test_every_tool_declares_annotations() -> None:
    got = list_annotations()
    assert set(got) == {"remember", "recall", "forget", "memory_stats"}
    assert all(got.values()), got


def test_only_stats_is_read_only_and_only_forget_is_destructive() -> None:
    got = list_annotations()
    assert [n for n, a in got.items() if a.get("readOnlyHint")] == ["memory_stats"]
    assert [n for n, a in got.items() if a.get("destructiveHint")] == ["forget"]
    # recall は検索のたびに参照の時刻を書き換えるので、読み取り専用にしない。
    assert got["recall"]["readOnlyHint"] is False

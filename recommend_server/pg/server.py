#!/usr/bin/env python3
"""pgvector を裏に置いたレコメンド MCP サーバー。

同じディレクトリの Chroma 版(recommend_server.py)と機能は揃えつつ、
テナント分離と棚組みを足してある。
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from fastmcp import FastMCP

from . import embed, shelf, store

mcp = FastMCP("Recommend Server (pgvector)")


@mcp.tool()
def add_item(
    tenant_id: str,
    item_id: str,
    title: str,
    description: str = "",
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """アイテムを 1 件登録または更新する。"""
    item = store.Item(
        item_id=item_id,
        title=title,
        description=description,
        category=category,
        tags=tuple(tags or ()),
    )
    vector = embed.encode_one(f"{title}. {description}")
    store.upsert(tenant_id, [item], [vector])
    return {"status": "ok", "tenant_id": tenant_id, "item_id": item_id, "title": title}


@mcp.tool()
def recommend(tenant_id: str, item_id: str, top_k: int = 10) -> dict[str, Any]:
    """指定アイテムに近いものを返す。"""
    vector = store.get_embedding(tenant_id, item_id)
    if vector is None:
        return {"error": f"アイテム '{item_id}' が見つかりません(tenant={tenant_id})"}

    hits = store.search(tenant_id, vector, top_k=top_k, exclude_item_id=item_id)
    return {
        "base_item_id": item_id,
        "recommendations": [h.__dict__ for h in hits],
    }


@mcp.tool()
def search(
    tenant_id: str,
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
) -> dict[str, Any]:
    """自然文でアイテムを検索する。"""
    vector = embed.encode_one(query)
    hits = store.search(tenant_id, vector, top_k=top_k, category=category)
    return {"query": query, "count": len(hits), "results": [h.__dict__ for h in hits]}


@mcp.tool()
def build_shelves(
    tenant_id: str,
    query: str,
    candidates: int = 200,
    per_shelf: int = 8,
    max_shelves: int = 6,
    group_by: str = "category",
) -> dict[str, Any]:
    """候補を大量に取ってから棚に組み直す。

    group_by が "category" ならカタログの分類、"cluster" なら埋め込みの近さで割る。
    """
    vector = embed.encode_one(query)
    hits = store.search(tenant_id, vector, top_k=candidates)
    if not hits:
        return {"query": query, "shelves": [], "candidates": 0}

    if group_by == "cluster":
        vectors = [store.get_embedding(tenant_id, h.item_id) or [] for h in hits]
        shelves = shelf.by_cluster(hits, vectors, per_shelf, max_shelves)
    else:
        shelves = shelf.by_category(hits, per_shelf, max_shelves)

    return {
        "query": query,
        "candidates": len(hits),
        "group_by": group_by,
        "shelves": [{"title": s.title, "items": [h.__dict__ for h in s.items]} for s in shelves],
    }


@mcp.tool()
def get_stats() -> dict[str, Any]:
    """テナントごとの登録件数を返す。"""
    names = store.tenants()
    return {
        "tenants": {name: store.count(name) for name in names},
        "total": store.count(),
    }


def main() -> None:
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)
    else:
        mcp.run()


if __name__ == "__main__":
    main()

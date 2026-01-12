#!/usr/bin/env python3
"""
Recommendation MCP Server
コサイン類似度を使ったレコメンデーションサーバー
"""

import os
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
import chromadb
from chromadb.config import Settings

mcp = FastMCP("Recommend Server")

# データ保存先
DATA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")

# ChromaDB クライアント（永続化）
client = chromadb.PersistentClient(
    path=DATA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# コレクション取得（SentenceTransformers使用）
collection = client.get_or_create_collection(
    name="items",
    metadata={"hnsw:space": "cosine"}
)


@mcp.tool()
def add_item(
    item_id: str,
    title: str,
    description: str,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    アイテムを追加してベクトル化。

    Args:
        item_id: 一意のアイテムID
        title: アイテムのタイトル
        description: アイテムの説明文
        category: カテゴリ（オプション）
        tags: タグリスト（オプション）

    Returns:
        追加結果
    """
    # 埋め込み用テキスト
    text = f"{title}. {description}"

    # メタデータ
    metadata = {
        "title": title,
        "description": description,
    }
    if category:
        metadata["category"] = category
    if tags:
        metadata["tags"] = ",".join(tags)

    # 既存チェック
    existing = collection.get(ids=[item_id])
    if existing["ids"]:
        # 更新
        collection.update(
            ids=[item_id],
            documents=[text],
            metadatas=[metadata]
        )
        return {"status": "updated", "item_id": item_id, "title": title}

    # 新規追加
    collection.add(
        ids=[item_id],
        documents=[text],
        metadatas=[metadata]
    )

    return {"status": "added", "item_id": item_id, "title": title}


@mcp.tool()
def recommend(item_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    指定アイテムに類似したアイテムを取得。

    Args:
        item_id: 基準となるアイテムのID
        top_k: 取得する類似アイテム数（デフォルト: 5）

    Returns:
        類似アイテムのリスト
    """
    # 指定アイテムを取得
    item = collection.get(ids=[item_id], include=["documents", "metadatas"])

    if not item["ids"]:
        return {"error": f"アイテム '{item_id}' が見つかりません"}

    document = item["documents"][0]

    # 類似検索（自身を除くため+1）
    results = collection.query(
        query_texts=[document],
        n_results=top_k + 1,
        include=["metadatas", "distances"]
    )

    # 自身を除外して整形
    recommendations = []
    for i, rid in enumerate(results["ids"][0]):
        if rid == item_id:
            continue
        if len(recommendations) >= top_k:
            break

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance  # コサイン距離→類似度

        recommendations.append({
            "item_id": rid,
            "title": metadata.get("title", ""),
            "category": metadata.get("category", ""),
            "similarity": round(similarity, 4)
        })

    base_item = item["metadatas"][0]
    return {
        "base_item": {
            "item_id": item_id,
            "title": base_item.get("title", "")
        },
        "recommendations": recommendations
    }


@mcp.tool()
def search(query: str, top_k: int = 5, category: Optional[str] = None) -> Dict[str, Any]:
    """
    テキストで類似アイテムを検索。

    Args:
        query: 検索クエリ（自然文OK）
        top_k: 取得件数（デフォルト: 5）
        category: カテゴリでフィルタ（オプション）

    Returns:
        検索結果
    """
    # フィルタ条件
    where = None
    if category:
        where = {"category": category}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances"]
    )

    items = []
    for i, item_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance

        items.append({
            "item_id": item_id,
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "category": metadata.get("category", ""),
            "similarity": round(similarity, 4)
        })

    return {
        "query": query,
        "results": items,
        "count": len(items)
    }


@mcp.tool()
def list_items(limit: int = 20, category: Optional[str] = None) -> Dict[str, Any]:
    """
    登録済みアイテムを一覧表示。

    Args:
        limit: 取得件数（デフォルト: 20）
        category: カテゴリでフィルタ（オプション）

    Returns:
        アイテム一覧
    """
    where = None
    if category:
        where = {"category": category}

    results = collection.get(
        where=where,
        limit=limit,
        include=["metadatas"]
    )

    items = []
    for i, item_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i]
        items.append({
            "item_id": item_id,
            "title": metadata.get("title", ""),
            "category": metadata.get("category", ""),
            "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else []
        })

    return {
        "items": items,
        "count": len(items),
        "total": collection.count()
    }


@mcp.tool()
def delete_item(item_id: str) -> Dict[str, Any]:
    """
    アイテムを削除。

    Args:
        item_id: 削除するアイテムのID

    Returns:
        削除結果
    """
    existing = collection.get(ids=[item_id])
    if not existing["ids"]:
        return {"error": f"アイテム '{item_id}' が見つかりません"}

    collection.delete(ids=[item_id])

    return {"status": "deleted", "item_id": item_id}


@mcp.tool()
def get_stats() -> Dict[str, Any]:
    """
    サーバーの統計情報を取得。

    Returns:
        統計情報
    """
    total = collection.count()

    # カテゴリ別集計
    all_items = collection.get(include=["metadatas"])
    categories = {}
    for metadata in all_items["metadatas"]:
        cat = metadata.get("category", "未分類")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_items": total,
        "categories": categories,
        "data_dir": DATA_DIR
    }


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
    else:
        mcp.run()

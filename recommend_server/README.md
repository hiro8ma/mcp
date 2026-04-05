# Recommend Server

コサイン類似度を使ったレコメンデーションMCPサーバー

## Setup

```bash
uv sync
```

## Usage

```bash
make run      # stdio mode
make inspect  # MCP Inspector
make http     # HTTP mode (port 8001)
```

## Tools

| ツール | 説明 |
|--------|------|
| `add_item` | アイテム追加（自動ベクトル化） |
| `recommend` | 類似アイテム取得 |
| `search` | テキストで類似検索 |
| `list_items` | アイテム一覧 |
| `delete_item` | アイテム削除 |
| `get_stats` | 統計情報 |

## Example

```python
# アイテム追加
add_item("iphone-15", "iPhone 15 Pro", "A17チップ搭載スマートフォン", "スマートフォン")

# 類似アイテム取得
recommend("iphone-15", top_k=3)
# → ["MacBook Pro", "iPad Pro", ...]

# テキスト検索
search("Apple製品", top_k=5)
```

## Tech Stack

- **ChromaDB**: ベクトルデータベース（永続化対応）
- **all-MiniLM-L6-v2**: Embeddingモデル（ローカル実行）

## Search Characteristics

This server implements **vector search** using ChromaDB.

| Aspect | Detail |
|---|---|
| **Engine** | ChromaDB with HNSW index (cosine distance) |
| **Embedding** | sentence-transformers all-MiniLM-L6-v2 |
| **Matching** | Approximate Nearest Neighbor (HNSW) + cosine similarity |
| **Strengths** | Persistent storage, fast ANN search, metadata filtering, semantic matching |
| **Weaknesses** | Requires ChromaDB process, higher memory for HNSW index |
| **Query style** | Natural language descriptions |

### Comparison with memory/ server

| Feature | recommend_server | memory/ |
|---|---|---|
| Storage | ChromaDB (persistent) | SQLite (persistent) |
| Full-text search | No | Yes (FTS5 trigram) |
| Vector search | Yes (HNSW) | Yes (ONNX brute-force) |
| Temporal decay | No | Yes (14-day half-life) |
| Use case | Item recommendations | Long-term memory with recency |

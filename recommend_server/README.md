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

## 推薦アルゴリズムの比較（pg/）

`pg/` には内容ベースフィルタリングと協調フィルタリングの両方が入っている。
同じデータで比べられるようにしてあり、精度ではなくカバレッジと人気バイアスで測る。

| モジュール | 方式 |
|---|---|
| `content_based.py` | 内容ベース。履歴の埋め込みを平均してユーザープロファイルを作る |
| `collaborative.py` | 協調（メモリベース法）。ユーザー間型とアイテム間型 |
| `baselines.py` | 基準線。ランダムと人気順 |
| `evaluate.py` | カバレッジ・人気バイアス・ジニ係数の計測 |

```bash
uv run python -m pg.seed --tenant demo --limit 2000
uv run python -m pg.seed_interactions --tenant demo --users 300 --reset
uv run python -m pg.evaluate --tenant demo --top-k 10
```

実測結果（カタログ 1176 件・ユーザー 300 人・上位 1% が全利用の 35.4%）

```
方式                     カバレッジ   人気バイアス   対ランダム   ジニ係数
[基準] ランダム             56.6%      0.78x       1.0x     0.231
[基準] 人気順                1.5%     14.11x      18.1x     0.359
内容ベース（履歴の重心）        19.2%      1.06x       1.4x     0.599
協調 ユーザー間型            22.5%     10.60x      13.6x     0.617
協調 アイテム間型            38.4%      9.43x      12.1x     0.449
```

協調フィルタリングは人気バイアスが中立の 10 倍を超える。候補の得点を
「類似ユーザーの類似度 × 評価」の合計で出すため、多くの履歴に入っている
人気アイテムが誰に対しても上位に来る。構造上そうなるので、補正しない限り消えない。

内容ベースはカバレッジが最も低いが人気バイアスは無い。似たものばかり出して
カタログの狭い一角に集中するためで、原因が協調と逆になる。

### 人気バイアスの中立点は 1.0 ではない

どの方式も自分の履歴を推薦から除く。履歴には人気アイテムが集中しており、
実測では履歴 1 件あたりの人気度がカタログ平均の 30.7 倍だった。
除いた残りは平均人気度が下がるため、ランダム推薦でも 0.78 倍になる。
（除外しない条件では 0.96 倍で、指標そのものは正しい）

そのため中立点はランダム基準の実測値で決める。`evaluate.py` の対ランダム列がそれ。

### 測る前に分布を確かめる

人気バイアスは元データが偏っていないと再現しない。素朴な優先的選択では
偏りの成長が遅く、実測では上位 1% の占有率が 3.4% にしかならなかった。
`seed_interactions.py` は重みを `利用回数^alpha` にして偏りを強め、
投入時に分布を表示する。基準を満たさなければ警告を出す。

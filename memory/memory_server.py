"""
長期記憶MCPサーバー — セッション横断で学び・判断・経験を蓄積し検索する。

設計思想:
- 外部依存ゼロ: SQLite 1ファイルで完結
- LLMスキップ: 保存にLLM呼び出しを使わない（ONNX Embeddingのみ）
- 複合検索: ベクトル類似度 + テキスト検索(FTS5) + 時間減衰
"""

import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("memory-server")

DB_PATH = Path(__file__).parent / "memory.db"
MODEL_DIR = Path(__file__).parent / "model"
HALF_LIFE_DAYS = 14  # 時間減衰の半減期

# Embedding
_session = None

# Reranker
_reranker_model = None
_reranker_tokenizer = None


def _get_embedding_session():
    """ONNX Runtimeセッションを遅延初期化。"""
    global _session
    if _session is not None:
        return _session

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = MODEL_DIR / "model.onnx"
        tokenizer_path = MODEL_DIR / "tokenizer.json"

        if not model_path.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            # all-MiniLM-L6-v2 — 軽量で高速、384次元
            hf_hub_download(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                filename="onnx/model.onnx",
                local_dir=MODEL_DIR,
            )
            hf_hub_download(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                filename="tokenizer.json",
                local_dir=MODEL_DIR,
            )

        _session = ort.InferenceSession(str(model_path))
        return _session
    except Exception:
        return None


def _get_reranker():
    """Qwen3-Reranker-0.6B を lazy-load。"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch  # noqa: F811

        model_name = "Qwen/Qwen3-Reranker-0.6B"
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        _reranker_model.eval()
    return _reranker_model, _reranker_tokenizer


def _rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """検索結果を Qwen3-Reranker-0.6B でリランクする。

    リランカーが利用できない場合はそのまま返す（graceful degradation）。
    """
    if not results:
        return results

    try:
        model, tokenizer = _get_reranker()
    except Exception:
        return results

    import torch

    pairs = [[query, r["content"]] for r in results]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    for result, score in zip(results, scores):
        result["rerank_score"] = score

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def _embed(text: str) -> list[float] | None:
    """テキストをベクトル化。モデルがなければNoneを返す。"""
    session = _get_embedding_session()
    if session is None:
        return None

    try:
        from tokenizers import Tokenizer

        tokenizer_path = MODEL_DIR / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encoded = tokenizer.encode(text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling
        embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (embeddings * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1)
        pooled = summed / counts

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / norm

        return normalized[0].tolist()
    except Exception:
        return None


def _init_db():
    """データベースとテーブルを初期化。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # メインテーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]',
            embedding BLOB,
            created_at REAL NOT NULL,
            accessed_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0
        )
    """)

    # FTS5テキスト検索（trigram tokenizer で日本語対応）
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, category, tags, content=memories, content_rowid=id, tokenize='trigram')
    """)

    # FTS同期トリガー
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, tags)
            VALUES (new.id, new.content, new.category, new.tags);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, tags)
            VALUES ('delete', old.id, old.content, old.category, old.tags);
        END
    """)

    conn.commit()
    return conn


def _temporal_decay(created_at: float, accessed_at: float) -> float:
    """時間減衰スコアを計算。半減期ベースの指数減衰 + アクセスリセット。"""
    now = time.time()
    # 最終アクセスからの経過日数を使う（アクセスされるほど生き残る）
    days_since = (now - accessed_at) / 86400
    decay = math.pow(0.5, days_since / HALF_LIFE_DAYS)
    return decay


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """コサイン類似度。"""
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@mcp.tool()
def remember(content: str, category: str = "general", tags: list[str] | None = None) -> str:
    """
    記憶を保存します。会話・学び・判断・経験などを蓄積してください。

    Args:
        content: 記憶する内容
        category: カテゴリ（learning, decision, experience, insight等）
        tags: タグのリスト
    """
    conn = _init_db()
    now = time.time()
    tags_json = json.dumps(tags or [], ensure_ascii=False)

    embedding = _embed(content)
    embedding_blob = np.array(embedding).tobytes() if embedding else None

    conn.execute(
        """INSERT INTO memories (content, category, tags, embedding, created_at, accessed_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (content, category, tags_json, embedding_blob, now, now),
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()

    return f"記憶を保存しました（全{count}件）"


@mcp.tool()
def recall(query: str, limit: int = 5, category: str | None = None, rerank: bool = True) -> str:
    """
    記憶を検索します。関連する過去の学び・判断・経験を取得します。

    Args:
        query: 検索クエリ
        limit: 取得件数（デフォルト5）
        category: カテゴリで絞り込み（省略時は全カテゴリ）
        rerank: Qwen3-Reranker によるリランクを適用するか（デフォルトTrue）
    """
    conn = _init_db()
    results = {}

    # 1. FTS5テキスト検索（trigram tokenizer用）
    try:
        # trigramではクエリ全体を引用符で囲んで部分文字列検索
        # さらに個別キーワードでもOR検索（日本語はスペース分割できないのでそのまま）
        fts_query = f'"{query}"'
        rows = conn.execute(
            """SELECT rowid, content, category, tags, rank
               FROM memories_fts
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit * 2),
        ).fetchall()

        # ヒットしなければ、短いキーワード（3文字以上）で再検索
        if not rows and len(query) > 3:
            # 3文字ずつのサブストリングで検索
            sub_queries = []
            for i in range(0, len(query) - 2, 3):
                sub = query[i:i+3]
                sub_queries.append(f'"{sub}"')
            if sub_queries:
                fts_query = " OR ".join(sub_queries)
                rows = conn.execute(
                    """SELECT rowid, content, category, tags, rank
                       FROM memories_fts
                       WHERE memories_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, limit * 2),
                ).fetchall()

        for rank_idx, row in enumerate(rows):
            rid = row[0]
            if rid not in results:
                results[rid] = {"content": row[1], "category": row[2], "tags": row[3], "score": 0.0}
            # FTSスコア（順位ベース、正規化）
            results[rid]["score"] += 1.0 / (rank_idx + 1)
    except Exception:
        pass

    # 2. ベクトル検索
    query_embedding = _embed(query)
    if query_embedding:
        rows = conn.execute(
            "SELECT id, content, category, tags, embedding, accessed_at FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in rows:
            rid, content, cat, tags, emb_blob, accessed_at = row
            if emb_blob:
                emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
                sim = _cosine_similarity(query_embedding, emb)
                scored.append((rid, content, cat, tags, sim, accessed_at))

        scored.sort(key=lambda x: x[4], reverse=True)
        for rank_idx, item in enumerate(scored[: limit * 2]):
            rid, content, cat, tags, sim, accessed_at = item
            if rid not in results:
                results[rid] = {"content": content, "category": cat, "tags": tags, "score": 0.0}
            # ベクトルスコア（順位ベース、正規化）
            results[rid]["score"] += 1.0 / (rank_idx + 1)

    # 3. 時間減衰を適用
    for rid in results:
        row = conn.execute(
            "SELECT created_at, accessed_at FROM memories WHERE id = ?", (rid,)
        ).fetchone()
        if row:
            decay = _temporal_decay(row[0], row[1])
            results[rid]["score"] *= decay

            # アクセスカウント更新
            conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (time.time(), rid),
            )

    conn.commit()

    # カテゴリフィルタ
    if category:
        results = {k: v for k, v in results.items() if v["category"] == category}

    # スコア順でソート（リランク前の候補取得）
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]

    # 4. Qwen3-Reranker によるリランク
    if rerank and sorted_results:
        candidates = [{"rid": rid, **data} for rid, data in sorted_results]
        reranked = _rerank(query, candidates, top_k=limit)
        sorted_results = [(c["rid"], c) for c in reranked]

    if not sorted_results:
        conn.close()
        return "関連する記憶が見つかりませんでした"

    output = []
    for rid, data in sorted_results:
        tags = json.loads(data["tags"]) if isinstance(data["tags"], str) else data["tags"]
        tags_str = ", ".join(tags) if tags else ""
        score_parts = [f"score: {data['score']:.3f}"]
        if "rerank_score" in data:
            score_parts.append(f"rerank: {data['rerank_score']:.3f}")
        output.append(
            f"[{data['category']}] {data['content']}"
            + (f" (tags: {tags_str})" if tags_str else "")
            + f" ({', '.join(score_parts)})"
        )

    conn.close()
    return "\n---\n".join(output)


@mcp.tool()
def forget(memory_id: int | None = None, older_than_days: int | None = None) -> str:
    """
    記憶を削除します。IDで個別削除、または経過日数で一括削除できます。

    Args:
        memory_id: 削除する記憶のID（省略時はolder_than_daysで一括削除）
        older_than_days: 指定日数以上アクセスされていない記憶を一括削除
    """
    conn = _init_db()

    if memory_id is not None:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        conn.close()
        return f"記憶 ID:{memory_id} を削除しました"

    if older_than_days is not None:
        cutoff = time.time() - (older_than_days * 86400)
        cursor = conn.execute(
            "DELETE FROM memories WHERE accessed_at < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return f"{older_than_days}日以上アクセスされていない記憶を{deleted}件削除しました"

    return "memory_id または older_than_days を指定してください"


@mcp.tool()
def memory_stats() -> str:
    """記憶の統計情報を返します。"""
    conn = _init_db()

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    categories = conn.execute(
        "SELECT category, COUNT(*) FROM memories GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    with_embedding = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    cat_str = "\n".join(f"  {cat}: {count}件" for cat, count in categories)
    return f"""記憶の統計:
- 総数: {total}件
- ベクトル化済み: {with_embedding}件
- カテゴリ別:
{cat_str}"""


if __name__ == "__main__":
    mcp.run()

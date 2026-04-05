# Memory MCP Server

Long-term memory server with hybrid search (full-text + vector similarity) and temporal decay.

## Features

- **SQLite + FTS5** — Full-text search with trigram tokenizer (Japanese support)
- **ONNX Embeddings** — all-MiniLM-L6-v2 (384-dim vectors, lazy-loaded)
- **Temporal Decay** — Exponential decay with 14-day half-life. Accessed memories extend their lifetime
- **Hybrid Search** — Combines FTS5 text matching + vector cosine similarity + recency scoring

## Tools

| Tool | Description |
|---|---|
| `remember` | Store a memory with optional category and tags |
| `recall` | Search memories by query. Hybrid scoring: text relevance + vector similarity + temporal decay |
| `forget` | Delete by ID or bulk delete older than N days |
| `memory_stats` | Statistics per category (count, storage) |

## How Search Works

```
Query → FTS5 trigram search (Japanese-aware)
     → ONNX embedding → cosine similarity with stored vectors
     → Temporal decay: score × 0.5^(days_since_access / 14)
     → Rank by combined score
     → Top-K results returned
```

## Search Characteristics

This server implements **hybrid search**, combining full-text search and vector search to leverage the strengths of both approaches.

### Full-Text Search (FTS5 + Trigram)

| Aspect | Detail |
|---|---|
| **Engine** | SQLite FTS5 with trigram tokenizer |
| **Matching** | Token-level exact matching |
| **Strengths** | Exact keyword match, specific terms, proper nouns, Japanese text support |
| **Weaknesses** | Vocabulary mismatch ("car" won't match "vehicle"), no semantic understanding |
| **Query style** | Short keywords (2-4 terms) |

### Vector Search (ONNX Embeddings)

| Aspect | Detail |
|---|---|
| **Engine** | ONNX Runtime + all-MiniLM-L6-v2 (384-dim) |
| **Matching** | Cosine similarity between embedding vectors |
| **Strengths** | Semantic similarity, handles paraphrasing, cross-lingual matching |
| **Weaknesses** | Higher memory usage, may miss exact terms in favor of related concepts |
| **Query style** | Natural language sentences |

### Why Hybrid?

Neither approach alone is sufficient:
- Full-text search misses semantically similar content with different wording
- Vector search may rank loosely related content above exact matches

By combining both scores with temporal decay weighting, the hybrid approach provides more robust recall across different query patterns.

## Quick Start

```bash
make setup    # Install dependencies + download ONNX model
make run      # Run server (stdio)
make inspect  # MCP Inspector
make test     # Run smoke test (remember → recall → stats)
```

## Architecture

```
┌──────────────┐     ┌──────────────┐
│ MCP Client   │────▶│ FastMCP      │
│ (Claude etc) │◀────│ memory_server│
└──────────────┘     └──────┬───────┘
                            │
                   ┌────────┴────────┐
                   │   SQLite DB     │
                   │ ┌─────────────┐ │
                   │ │ memories    │ │  (content, category, tags, embedding, timestamps)
                   │ ├─────────────┤ │
                   │ │ memories_fts│ │  (FTS5 virtual table, trigram tokenizer)
                   │ └─────────────┘ │
                   └─────────────────┘
                            │
                   ┌────────┴────────┐
                   │  ONNX Runtime   │
                   │  MiniLM-L6-v2   │  (384-dim embedding, lazy-loaded)
                   └─────────────────┘
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `MEMORY_DB_PATH` | `memory.db` | SQLite database path |
| `ONNX_MODEL_DIR` | `onnx_model/` | ONNX model directory |

## Tech Stack

- FastMCP
- SQLite + FTS5 (trigram tokenizer)
- ONNX Runtime + all-MiniLM-L6-v2
- Python 3.10+

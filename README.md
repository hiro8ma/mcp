# MCP Servers — Model Context Protocol Research

A collection of MCP (Model Context Protocol) server implementations built with [FastMCP](https://github.com/jlowin/fastmcp). 14 servers covering AI/ML, external APIs, development tools, and educational notebooks.

## Role in the Ecosystem

This repo is the **Tool provider layer**. MCP servers here are consumed by AI agents implemented in [`../agent/`](../agent/) (Go / Python / TypeScript across multiple frameworks). See `../agent/` for the AI agent side (LLM invocation, Tool Dispatch loop, conversation history, prompt management).

## Servers

### AI / ML

| Server | Description | Tools |
|---|---|---|
| `ai_knowledge/` | AI engineering Q&A with fine-tuned Gemma 3 4B. 3-layer guardrails (injection detection, PII masking, output quality), LRU cache, Langfuse tracing | `ask_ai_engineering`, `quiz_ai_engineering`, `explain_concept` |
| `memory/` | Long-term memory with SQLite + FTS5 (Japanese trigram) + ONNX embeddings. Temporal decay (14-day half-life) | `remember`, `recall`, `forget`, `memory_stats` |
| `image_classifier/` | CNN-based MNIST digit classifier (PyTorch) | `classify_digit`, `get_model_info` |
| `recommend_server/` | Cosine similarity recommendations with ChromaDB | `add_item`, `recommend`, `search`, `list_items`, `delete_item`, `get_stats` |

### External Integration

| Server | Description | Tools |
|---|---|---|
| `external_api/` | Weather (OpenWeatherMap), News (NewsAPI), IP geolocation | `get_weather`, `get_weather_forecast`, `get_latest_news`, `search_news`, `get_ip_info` |
| `openapi/` | OpenAPI spec parser + dynamic API execution | `list_endpoints`, `get_endpoint_detail`, `call_api` |
| `gateway/` | LiteLLM proxy for unified LLM API access (Gemini / Claude / GPT fallback). Pinned to v1.82.6 for supply chain security | — |

### Development Tools

| Server | Description | Tools |
|---|---|---|
| `universal_tools/` | Web search (Tavily), webpage content extraction, Python sandbox (AST-inspected, resource-limited) | `web_search`, `get_webpage_content`, `execute_python`, `execute_python_basic` |
| `calc/` | Basic arithmetic operations | `add`, `subtract`, `multiply`, `divide`, `power`, `square_root`, `circle_area` |
| `design_system/` | Design tokens, components, and icons reference | `get_components`, `get_style_types`, `get_design_tokens`, `get_icon_list`, `get_icon_detail` |

### Orchestration

| Server | Description |
|---|---|
| `agent/` | Multi-MCP orchestration with LLM task decomposition |
| `client/` | LLM-integrated MCP client with Gradio UI |
| `ai_platform/` | FastAPI integration platform (experimental) |

### Educational

| Directory | Description |
|---|---|
| `transformer/notebooks/` | 9 Jupyter notebooks: Seq2Seq → Transformer architecture (Embedding, Multi-Head Attention, Positional Encoding, Feed Forward, Decoder) |
| `image_classifier/notebooks/` | 10 Jupyter notebooks: CNN fundamentals (Convolution, Backpropagation, Chain Rule) |

## Quick Start

```bash
cd <server_dir>
make setup    # Install dependencies
make run      # Run server (stdio)
make inspect  # MCP Inspector (browser)
make http     # HTTP mode (if supported)
```

## Claude Desktop Integration

Copy `claude_desktop_config.json` to `~/Library/Application Support/Claude/`:

```bash
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

## Architecture Highlights

- **33 tools** across 10+ servers
- **Hybrid search** in memory server (FTS5 + ONNX vector similarity)
- **3-layer guardrails** in ai_knowledge (injection detection / PII masking / output quality)
- **Temporal decay** in memory (14-day half-life, access extends lifetime)
- **Supply chain security** — LiteLLM pinned to safe version (1.82.6)
- **Observable** — Langfuse tracing integration

## Tech Stack

- **Framework:** FastMCP, MCP SDK
- **Language:** Python 3.10+
- **ML:** PyTorch, ONNX Runtime, MLX, sentence-transformers
- **Vector DB:** ChromaDB
- **Search:** SQLite FTS5 (trigram tokenizer)
- **Observability:** Langfuse
- **Package Manager:** uv

# MCP Research Repository

This repository contains MCP (Model Context Protocol) server implementations.

## Structure

- `calc/` - Calculator MCP server (FastMCP)
- `recommend_server/` - Recommendation server with cosine similarity
- `external_api/` - Weather, News, IP info APIs
- `universal_tools/` - Web search, Python sandbox execution

## Development

Each server directory has its own `uv` environment.

```bash
cd <server_dir>
make run      # Run server (stdio)
make inspect  # MCP Inspector
make http     # HTTP mode (if supported)
```

## MCP Configuration

- `claude_desktop_config.json` - Claude Desktop config (symlinked)
- `.claude/settings.json` - Claude Code config

## Notebooks

`image_classifier/notebooks/` と `transformer/notebooks/` で AI の数理を学習する Jupyter Notebook シリーズを管理している。

### 前提

- ユーザーから画像（書籍のページ等）やテキストで学習材料が渡される
- 基礎学力や理解力に自信がない人でも理解できるレベルを目指す
- 「読んで終わり」ではなく、コードを動かして体感できるアウトプット型

### 構成ルール

1. **目次を冒頭に置く** — 全体像を最初に把握できるようにする
2. **概念 → 図解 → コード → まとめ** の順で進める
3. **1ノートブック = 1テーマ** — 詰め込みすぎない
4. **「次のステップ」で次回への橋渡しを書く**
5. ファイル名は `01_topic_name.ipynb` の連番形式

### 説明スタイル

- 専門用語は初出時に必ず日本語で噛み砕いて説明する
- 数式は「何を意味しているか」を日本語で先に説明してからコードで示す
- 表やテーブルを積極的に使って比較・整理する
- 「なぜそうなるのか？」を省略しない
- 具体的な数値例を必ず入れる（抽象的な説明だけにしない）

### コードスタイル

- numpy / matplotlib を基本とする（必要に応じて PyTorch）
- コードにはコメントを多めに書く
- 可視化は日本語ラベル・タイトルを使う
- print 文で計算の途中経過を表示し、何が起きているか追えるようにする

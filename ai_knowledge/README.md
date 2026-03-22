# AI Knowledge Assistant MCP Server

FT済みローカルLLM（Gemma 3 4B + LoRA）を使った、AIエンジニアリング知識Q&AのMCPサーバー。

## Tools

| Tool | 説明 |
|---|---|
| `ask_ai_engineering` | AIエンジニアリングに関する質問に回答 |
| `quiz_ai_engineering` | トピックについてクイズ形式で出題 |
| `explain_concept` | 概念を指定された深さで説明 |

## セットアップ

```bash
# 依存関係インストール
uv sync

# ft/ リポジトリでFT済みアダプターを生成しておく
cd ../../ft && make prepare && make train
```

## 使い方

```bash
# MCPサーバー起動
make run

# MCP Inspectorで確認
make inspect
```

## 必要なもの

- Apple Silicon Mac（MLX使用）
- FT済みアダプター（`../../ft/adapters/`）

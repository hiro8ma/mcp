# LLM Gateway

LiteLLMベースのLLMゲートウェイ。複数のLLMプロバイダー（Gemini, Claude, GPT, ローカルモデル）を
OpenAI互換の統一APIで利用可能にする。

## 機能

- **統一API**: OpenAI互換インターフェースで全プロバイダーにアクセス
- **フェイルオーバー**: プロバイダー障害時に自動的に代替プロバイダーに切り替え
- **コスト最適化ルーティング**: コストベースのルーティング戦略
- **コスト追跡**: API呼び出しのコストを自動記録

## セットアップ

```bash
# 依存関係インストール
make setup

# 環境変数設定
cp .env.example .env
# .env を編集して API キーを設定
```

## 使い方

```bash
# ゲートウェイ起動（http://localhost:4000）
make run

# テストリクエスト送信
make test
```

## モデル一覧

| モデル名 | プロバイダー | 用途 |
|---|---|---|
| `gemini` | Google Gemini | メイン（GCP軸） |
| `claude` | Anthropic | 高品質タスク |
| `gpt` | OpenAI | 汎用 |
| `local` | Ollama | ローカル推論（コスト$0） |

## 他コンポーネントとの連携

```python
# ai_knowledge_server.py からゲートウェイ経由でモデルを呼ぶ場合
import openai
client = openai.OpenAI(base_url="http://localhost:4000", api_key="your-master-key")
response = client.chat.completions.create(model="gemini", messages=[...])
```

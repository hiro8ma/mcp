.PHONY: help setup run test

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール
	uv sync

run: ## LiteLLMゲートウェイを起動（http://localhost:4000）
	uv run litellm --config litellm_config.yaml --port 4000

test: ## ゲートウェイにテストリクエストを送信
	curl -s http://localhost:4000/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $${LITELLM_MASTER_KEY}" \
		-d '{"model": "gemini", "messages": [{"role": "user", "content": "Hello"}]}' | python3 -m json.tool

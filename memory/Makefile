.PHONY: help setup run test

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール
	uv sync

run: ## MCPサーバーを起動
	uv run python memory_server.py

test: ## 動作テスト（remember → recall → stats）
	@echo "=== remember ==="
	uv run python -c "from memory_server import remember; print(remember('Self-Attentionは全トークン間の関係を並列計算する', 'learning', ['transformer', 'attention']))"
	@echo "=== recall ==="
	uv run python -c "from memory_server import recall; print(recall('Attentionの仕組み'))"
	@echo "=== stats ==="
	uv run python -c "from memory_server import memory_stats; print(memory_stats())"

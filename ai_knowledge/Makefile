.PHONY: run inspect http

run: ## MCPサーバーを起動（stdio）
	uv run python ai_knowledge_server.py

inspect: ## MCP Inspectorで起動
	npx @modelcontextprotocol/inspector uv run python ai_knowledge_server.py

http: ## HTTPモードで起動
	uv run python ai_knowledge_server.py --http

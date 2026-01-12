.PHONY: run inspect http

run:
	uv run python recommend_server.py

inspect:
	npx @modelcontextprotocol/inspector uv run python recommend_server.py

http:
	uv run python recommend_server.py --http

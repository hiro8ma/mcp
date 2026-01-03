.PHONY: run inspect http init

run:
	uv run python database_server.py

inspect:
	npx @modelcontextprotocol/inspector uv run python database_server.py

http:
	uv run python database_server.py --http

init:
	uv run python database_server.py --init

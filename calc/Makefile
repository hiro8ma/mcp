.PHONY: run inspect http

run:
	uv run python calculator_server.py

inspect:
	npx @modelcontextprotocol/inspector uv run python calculator_server.py

http:
	uv run python calculator_server.py --http

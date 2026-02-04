.PHONY: run inspect http install

run:
	uv run python image_classifier_server.py

inspect:
	npx @modelcontextprotocol/inspector uv run python image_classifier_server.py

http:
	uv run python image_classifier_server.py --http

install:
	uv sync

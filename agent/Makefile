.PHONY: run inspect help

# Run the MCP Agent REPL
run:
	uv run mcp_agent.py

# Run with MCP Inspector (for debugging individual tools)
inspect:
	npx @anthropic/mcp-inspector

help:
	@echo "Available commands:"
	@echo "  make run      - Start the MCP Agent REPL"
	@echo "  make inspect  - Run MCP Inspector for debugging"

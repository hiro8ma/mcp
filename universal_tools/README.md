# Universal Tools MCP Server

MCP server for web search and Python code execution.

## Tools

- `web_search` - Web search via Tavily API
- `get_webpage_content` - Fetch webpage content
- `execute_python` - Execute Python in sandbox (secure)
- `execute_python_basic` - Execute Python (basic)

## Setup

```bash
cp .env.example .env
# Set TAVILY_API_KEY
```

## Usage

```bash
make run      # Run server (stdio)
make inspect  # MCP Inspector
```

## Security

Python execution uses AST inspection and resource limits (CPU 2s, memory 256MB).

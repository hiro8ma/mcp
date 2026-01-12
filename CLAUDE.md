# MCP Research Repository

This repository contains MCP (Model Context Protocol) server implementations.

## Structure

- `calc/` - Calculator MCP server (FastMCP)
- `recommend_server/` - Recommendation server with cosine similarity
- `external_api/` - Weather, News, IP info APIs
- `universal_tools/` - Web search, Python sandbox execution

## Development

Each server directory has its own `uv` environment.

```bash
cd <server_dir>
make run      # Run server (stdio)
make inspect  # MCP Inspector
make http     # HTTP mode (if supported)
```

## MCP Configuration

- `claude_desktop_config.json` - Claude Desktop config (symlinked)
- `.claude/settings.json` - Claude Code config

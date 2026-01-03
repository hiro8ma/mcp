# Database MCP Server

SQLite database MCP server with safety checks.

## Tools

- `query` - Execute SELECT queries (read-only)
- `list_tables` - List all tables

## Usage

```bash
make init     # Create sample database
make run      # Run server (stdio)
make inspect  # MCP Inspector
```

## Safety

Only SELECT queries are allowed. INSERT, UPDATE, DELETE, DROP are blocked.

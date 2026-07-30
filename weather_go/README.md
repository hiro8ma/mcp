# weather_go — Open-Meteo MCP Server (Go)

Weather MCP server built with the official Go SDK ([modelcontextprotocol/go-sdk](https://github.com/modelcontextprotocol/go-sdk) v1.7.0). No API key required (Open-Meteo). Built following the `mcp-builder-go` skill guide.

## Tools

| Tool | Description | Annotations |
|---|---|---|
| `get_current_weather` | Current weather for a city (temperature, feels-like, humidity, wind, precipitation) | readOnly, openWorld |
| `get_weekly_forecast` | 7-day forecast (max/min temperature, precipitation probability, wind) | readOnly, openWorld |

City names work in Japanese and English (geocoded via Open-Meteo geocoding API). WMO weather codes are converted to Japanese labels.

## Run

```bash
# stdio (for Claude Code / Claude Desktop)
go run .

# Streamable HTTP (for remote MCP clients, e.g. genkit's MCP plugin)
go run . -http :19920
```

`.mcp.json` example:

```json
{
  "mcpServers": {
    "weather": {
      "command": "go",
      "args": ["run", "."],
      "cwd": "/path/to/mcp/weather_go"
    }
  }
}
```

## Test

```bash
go test ./...   # handlers tested against httptest fakes (no network)
```

Verified end-to-end (2026-07-30): genkit agent → Streamable HTTP → this server → Open-Meteo live API.

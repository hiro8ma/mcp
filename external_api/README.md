# External API MCP Server

MCP server for external APIs (Weather, News, IP info).

## Tools

- `get_weather` - Get weather by city (OpenWeatherMap)
- `get_forecast` - Get 5-day forecast
- `get_news` - Get top headlines (NewsAPI)
- `search_news` - Search news articles
- `get_ip_info` - Get IP geolocation info

## Setup

```bash
cp .env.example .env
# Set OPENWEATHERMAP_API_KEY and NEWSAPI_KEY
```

## Usage

```bash
make run      # Run server (stdio)
make inspect  # MCP Inspector
```

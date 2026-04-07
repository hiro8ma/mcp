#!/usr/bin/env python3
"""
External API MCP Server
"""

import os
import requests
from datetime import datetime
from typing import Optional
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("External API Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def make_api_request(url: str, params: dict = None, headers: dict = None) -> dict:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("APIリクエストがタイムアウトしました")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"APIリクエストエラー: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"ネットワークエラー: {e}")


@mcp.tool()
def get_weather(city: str, country_code: str = "JP") -> dict:
    """Purpose: Fetch current weather conditions for a given city via OpenWeatherMap.
    Use when: The user asks about current weather, temperature, humidity, or wind for a specific city.
    Do not use when: The user asks about future weather (use get_weather_forecast instead).
    Notes: Returns temperature in Celsius. Defaults to Japan (JP) country code. Requires OPENWEATHER_API_KEY."""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap APIキーが設定されていません")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country_code}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ja"
    }

    data = make_api_request(url, params)

    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "weather_main": data["weather"][0]["main"],
        "weather_description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "visibility": data.get("visibility", 0) / 1000,
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
def get_weather_forecast(city: str, days: int = 5, country_code: str = "JP") -> dict:
    """Purpose: Fetch a multi-day weather forecast (up to 5 days, 3-hour intervals) for a given city.
    Use when: The user asks about upcoming weather, whether to bring an umbrella tomorrow, or multi-day planning.
    Do not use when: The user only needs current conditions (use get_weather instead).
    Notes: Days parameter must be 1-5. Returns 3-hour interval forecasts grouped by day. Requires OPENWEATHER_API_KEY."""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeatherMap APIキーが設定されていません")

    if days < 1 or days > 5:
        raise ValueError("予報日数は1-5日の範囲で指定してください")

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": f"{city},{country_code}",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ja"
    }

    data = make_api_request(url, params)

    daily_forecasts = []
    current_date = None
    daily_data = []

    for item in data["list"][:days * 8]:
        forecast_date = datetime.fromtimestamp(item["dt"]).date()

        if current_date != forecast_date:
            if daily_data:
                daily_forecasts.append({
                    "date": current_date.isoformat(),
                    "forecasts": daily_data
                })
            current_date = forecast_date
            daily_data = []

        daily_data.append({
            "time": datetime.fromtimestamp(item["dt"]).strftime("%H:%M"),
            "temperature": item["main"]["temp"],
            "weather": item["weather"][0]["description"],
            "rain_probability": item.get("pop", 0) * 100
        })

    if daily_data:
        daily_forecasts.append({
            "date": current_date.isoformat(),
            "forecasts": daily_data
        })

    return {
        "city": data["city"]["name"],
        "country": data["city"]["country"],
        "daily_forecasts": daily_forecasts[:days]
    }


@mcp.tool()
def get_latest_news(category: str = "general", country: str = "us", limit: int = 5) -> dict:
    """Purpose: Fetch top headlines by category and country via NewsAPI.
    Use when: The user wants to browse trending or top news in a specific category (general, business, technology, etc.).
    Do not use when: The user wants to search for news about a specific topic or keyword (use search_news instead).
    Notes: Categories: general, business, entertainment, health, science, sports, technology. Max 20 articles. Requires NEWS_API_KEY."""
    if not NEWS_API_KEY:
        raise ValueError("NewsAPI APIキーが設定されていません")

    if limit > 20:
        limit = 20

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "category": category,
        "country": country,
        "pageSize": limit
    }

    data = make_api_request(url, params)

    articles = []
    for article in data["articles"]:
        articles.append({
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"],
            "author": article.get("author", "不明")
        })

    return {
        "category": category,
        "country": country,
        "total_results": data["totalResults"],
        "articles": articles,
        "fetched_at": datetime.now().isoformat()
    }


@mcp.tool()
def search_news(query: str, language: str = "en", limit: int = 5) -> dict:
    """Purpose: Search news articles by keyword across all sources via NewsAPI.
    Use when: The user wants news about a specific topic, event, company, or keyword.
    Do not use when: The user wants general top headlines without a specific query (use get_latest_news instead).
    Notes: Sorted by publication date (newest first). Max 20 articles. Supports language filtering. Requires NEWS_API_KEY."""
    if not NEWS_API_KEY:
        raise ValueError("NewsAPI APIキーが設定されていません")

    if limit > 20:
        limit = 20

    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": limit
    }

    data = make_api_request(url, params)

    articles = []
    for article in data["articles"]:
        articles.append({
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"]
        })

    return {
        "query": query,
        "language": language,
        "total_results": data["totalResults"],
        "articles": articles,
        "fetched_at": datetime.now().isoformat()
    }


@mcp.tool()
def get_ip_info(ip_address: Optional[str] = None) -> dict:
    """Purpose: Look up geolocation and ISP information for an IP address.
    Use when: The user wants to know the geographic location, ISP, or organization behind an IP address.
    Do not use when: The user needs weather or news information (use the appropriate weather/news tools).
    Notes: If no IP is provided, returns info for the server's own public IP. Uses the free ip-api.com service (no API key required)."""
    if ip_address:
        url = f"http://ip-api.com/json/{ip_address}"
    else:
        url = "http://ip-api.com/json/"

    data = make_api_request(url)

    if data["status"] == "fail":
        raise Exception(f"IP情報取得エラー: {data.get('message', 'Unknown error')}")

    return {
        "ip": data["query"],
        "country": data["country"],
        "country_code": data["countryCode"],
        "region": data["regionName"],
        "city": data["city"],
        "zip": data["zip"],
        "latitude": data["lat"],
        "longitude": data["lon"],
        "timezone": data["timezone"],
        "isp": data["isp"],
        "organization": data["org"]
    }


if __name__ == "__main__":
    import sys
    print("[*] External API Server")
    print(f"  OpenWeather: {'OK' if OPENWEATHER_API_KEY else 'NG'}")
    print(f"  NewsAPI: {'OK' if NEWS_API_KEY else 'NG'}")
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)
    else:
        mcp.run()

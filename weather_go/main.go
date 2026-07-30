// weather_go は Open-Meteo API をラップする MCP サーバー（公式 Go SDK 使用）。
// デフォルトは stdio、-http でリモート接続用の Streamable HTTP になる。
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type cityInput struct {
	City string `json:"city" jsonschema:"都市名。日本語・英語どちらでも指定できる（例: 東京、Osaka）"`
}

type currentOutput struct {
	City          string  `json:"city" jsonschema:"解決された都市名"`
	Country       string  `json:"country" jsonschema:"国名"`
	Weather       string  `json:"weather" jsonschema:"天気の日本語表記"`
	Temperature   float64 `json:"temperatureC" jsonschema:"気温（℃）"`
	FeelsLike     float64 `json:"feelsLikeC" jsonschema:"体感温度（℃）"`
	Humidity      float64 `json:"humidityPercent" jsonschema:"湿度（%）"`
	Precipitation float64 `json:"precipitationMm" jsonschema:"降水量（mm）"`
	WindSpeed     float64 `json:"windSpeedKmh" jsonschema:"風速（km/h）"`
}

type dailyOutput struct {
	Date                     string  `json:"date" jsonschema:"日付（YYYY-MM-DD）"`
	Weather                  string  `json:"weather" jsonschema:"天気の日本語表記"`
	TemperatureMax           float64 `json:"temperatureMaxC" jsonschema:"最高気温（℃）"`
	TemperatureMin           float64 `json:"temperatureMinC" jsonschema:"最低気温（℃）"`
	PrecipitationProbability int     `json:"precipitationProbabilityPercent" jsonschema:"降水確率（%）"`
	WindSpeedMax             float64 `json:"windSpeedMaxKmh" jsonschema:"最大風速（km/h）"`
}

type weeklyOutput struct {
	City    string        `json:"city" jsonschema:"解決された都市名"`
	Country string        `json:"country" jsonschema:"国名"`
	Days    []dailyOutput `json:"days" jsonschema:"7 日間の予報"`
}

func newServer(svc *weatherService) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "weather", Version: "v0.1.0"}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "get_current_weather",
		Description: "指定都市の現在の天気（気温・体感温度・湿度・風速・降水量）を取得する",
		Annotations: &mcp.ToolAnnotations{
			Title:         "現在の天気",
			ReadOnlyHint:  true,
			OpenWorldHint: ptr(true),
		},
	}, svc.getCurrentWeather)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "get_weekly_forecast",
		Description: "指定都市の 7 日間天気予報（最高/最低気温・降水確率・風速）を取得する",
		Annotations: &mcp.ToolAnnotations{
			Title:         "週間予報",
			ReadOnlyHint:  true,
			OpenWorldHint: ptr(true),
		},
	}, svc.getWeeklyForecast)

	return server
}

func (s *weatherService) getCurrentWeather(ctx context.Context, _ *mcp.CallToolRequest, in cityInput) (*mcp.CallToolResult, currentOutput, error) {
	loc, err := s.geocode(ctx, in.City)
	if err != nil {
		return nil, currentOutput{}, err
	}
	cur, err := s.current(ctx, loc)
	if err != nil {
		return nil, currentOutput{}, err
	}
	return nil, currentOutput{
		City:          loc.Name,
		Country:       loc.Country,
		Weather:       wmoText(cur.WeatherCode),
		Temperature:   cur.Temperature,
		FeelsLike:     cur.FeelsLike,
		Humidity:      cur.Humidity,
		Precipitation: cur.Precipitation,
		WindSpeed:     cur.WindSpeed,
	}, nil
}

func (s *weatherService) getWeeklyForecast(ctx context.Context, _ *mcp.CallToolRequest, in cityInput) (*mcp.CallToolResult, weeklyOutput, error) {
	loc, err := s.geocode(ctx, in.City)
	if err != nil {
		return nil, weeklyOutput{}, err
	}
	daily, err := s.weekly(ctx, loc)
	if err != nil {
		return nil, weeklyOutput{}, err
	}
	out := weeklyOutput{City: loc.Name, Country: loc.Country}
	for i := range daily.Time {
		day := dailyOutput{Date: daily.Time[i]}
		if i < len(daily.WeatherCode) {
			day.Weather = wmoText(daily.WeatherCode[i])
		}
		if i < len(daily.TemperatureMax) {
			day.TemperatureMax = daily.TemperatureMax[i]
		}
		if i < len(daily.TemperatureMin) {
			day.TemperatureMin = daily.TemperatureMin[i]
		}
		if i < len(daily.PrecipitationProbability) {
			day.PrecipitationProbability = daily.PrecipitationProbability[i]
		}
		if i < len(daily.WindSpeedMax) {
			day.WindSpeedMax = daily.WindSpeedMax[i]
		}
		out.Days = append(out.Days, day)
	}
	return nil, out, nil
}

func ptr[T any](v T) *T { return &v }

func main() {
	httpAddr := flag.String("http", "", "Streamable HTTP で待ち受けるアドレス（例 :19920）。未指定なら stdio")
	flag.Parse()

	server := newServer(newWeatherService())

	if *httpAddr != "" {
		handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
		log.Printf("weather MCP server (streamable http) listening on %s", *httpAddr)
		if err := http.ListenAndServe(*httpAddr, handler); err != nil {
			log.Fatal(err)
		}
		return
	}
	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(fmt.Errorf("run: %w", err))
	}
}

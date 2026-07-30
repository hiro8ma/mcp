package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

const (
	defaultGeocodingBaseURL = "https://geocoding-api.open-meteo.com"
	defaultForecastBaseURL  = "https://api.open-meteo.com"
)

// weatherService は Open-Meteo API のクライアント。ベース URL はテストで差し替える。
type weatherService struct {
	httpClient       *http.Client
	geocodingBaseURL string
	forecastBaseURL  string
}

func newWeatherService() *weatherService {
	return &weatherService{
		httpClient:       &http.Client{Timeout: 10 * time.Second},
		geocodingBaseURL: defaultGeocodingBaseURL,
		forecastBaseURL:  defaultForecastBaseURL,
	}
}

type location struct {
	Name      string  `json:"name"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Country   string  `json:"country"`
}

func (s *weatherService) geocode(ctx context.Context, city string) (*location, error) {
	q := url.Values{"name": {city}, "count": {"1"}, "language": {"ja"}}
	var resp struct {
		Results []location `json:"results"`
	}
	if err := s.getJSON(ctx, s.geocodingBaseURL+"/v1/search?"+q.Encode(), &resp); err != nil {
		return nil, err
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("city not found: %s。都市名を確認してください（日本語・英語どちらでも指定できます）", city)
	}
	return &resp.Results[0], nil
}

type currentWeather struct {
	Temperature   float64 `json:"temperature_2m"`
	FeelsLike     float64 `json:"apparent_temperature"`
	Humidity      float64 `json:"relative_humidity_2m"`
	Precipitation float64 `json:"precipitation"`
	WindSpeed     float64 `json:"wind_speed_10m"`
	WeatherCode   int     `json:"weather_code"`
}

func (s *weatherService) current(ctx context.Context, loc *location) (*currentWeather, error) {
	q := url.Values{
		"latitude":  {fmt.Sprintf("%f", loc.Latitude)},
		"longitude": {fmt.Sprintf("%f", loc.Longitude)},
		"current":   {"temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"},
		"timezone":  {"auto"},
	}
	var resp struct {
		Current currentWeather `json:"current"`
	}
	if err := s.getJSON(ctx, s.forecastBaseURL+"/v1/forecast?"+q.Encode(), &resp); err != nil {
		return nil, err
	}
	return &resp.Current, nil
}

type dailyForecast struct {
	Time                     []string  `json:"time"`
	WeatherCode              []int     `json:"weather_code"`
	TemperatureMax           []float64 `json:"temperature_2m_max"`
	TemperatureMin           []float64 `json:"temperature_2m_min"`
	PrecipitationProbability []int     `json:"precipitation_probability_max"`
	WindSpeedMax             []float64 `json:"wind_speed_10m_max"`
}

func (s *weatherService) weekly(ctx context.Context, loc *location) (*dailyForecast, error) {
	q := url.Values{
		"latitude":  {fmt.Sprintf("%f", loc.Latitude)},
		"longitude": {fmt.Sprintf("%f", loc.Longitude)},
		"daily":     {"weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max"},
		"timezone":  {"auto"},
	}
	var resp struct {
		Daily dailyForecast `json:"daily"`
	}
	if err := s.getJSON(ctx, s.forecastBaseURL+"/v1/forecast?"+q.Encode(), &resp); err != nil {
		return nil, err
	}
	return &resp.Daily, nil
}

func (s *weatherService) getJSON(ctx context.Context, rawURL string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Open-Meteo API に接続できません: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Open-Meteo API がエラーを返しました（HTTP %d）。時間をおいて再試行してください", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// wmoText は WMO 天気コードを日本語表記へ変換する。
func wmoText(code int) string {
	switch {
	case code == 0:
		return "快晴"
	case code <= 2:
		return "晴れ"
	case code == 3:
		return "くもり"
	case code <= 48:
		return "霧"
	case code <= 57:
		return "霧雨"
	case code <= 67:
		return "雨"
	case code <= 77:
		return "雪"
	case code <= 82:
		return "にわか雨"
	case code <= 86:
		return "にわか雪"
	case code <= 99:
		return "雷雨"
	default:
		return fmt.Sprintf("不明（コード %d）", code)
	}
}

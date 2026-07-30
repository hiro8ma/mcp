package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newFakeService(t *testing.T) *weatherService {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/search", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("name") == "存在しない街" {
			_, _ = w.Write([]byte(`{"results":[]}`))
			return
		}
		_, _ = w.Write([]byte(`{"results":[{"name":"東京","latitude":35.68,"longitude":139.76,"country":"日本"}]}`))
	})
	mux.HandleFunc("/v1/forecast", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("current") != "" {
			_, _ = w.Write([]byte(`{"current":{"temperature_2m":31.5,"apparent_temperature":35.0,"relative_humidity_2m":68,"precipitation":0,"wind_speed_10m":9.7,"weather_code":1}}`))
			return
		}
		_, _ = w.Write([]byte(`{"daily":{"time":["2026-08-01","2026-08-02"],"weather_code":[3,61],"temperature_2m_max":[33.1,29.4],"temperature_2m_min":[26.0,25.2],"precipitation_probability_max":[10,80],"wind_speed_10m_max":[12.3,18.9]}}`))
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return &weatherService{
		httpClient:       ts.Client(),
		geocodingBaseURL: ts.URL,
		forecastBaseURL:  ts.URL,
	}
}

func TestGetCurrentWeather(t *testing.T) {
	svc := newFakeService(t)
	_, out, err := svc.getCurrentWeather(context.Background(), nil, cityInput{City: "東京"})
	if err != nil {
		t.Fatal(err)
	}
	if out.City != "東京" || out.Weather != "晴れ" || out.Temperature != 31.5 {
		t.Fatalf("unexpected output: %+v", out)
	}
}

func TestGetWeeklyForecast(t *testing.T) {
	svc := newFakeService(t)
	_, out, err := svc.getWeeklyForecast(context.Background(), nil, cityInput{City: "東京"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Days) != 2 {
		t.Fatalf("expected 2 days, got %d", len(out.Days))
	}
	if out.Days[1].Weather != "雨" || out.Days[1].PrecipitationProbability != 80 {
		t.Fatalf("unexpected day[1]: %+v", out.Days[1])
	}
}

func TestGeocodeNotFound(t *testing.T) {
	svc := newFakeService(t)
	_, _, err := svc.getCurrentWeather(context.Background(), nil, cityInput{City: "存在しない街"})
	if err == nil || !strings.Contains(err.Error(), "city not found") {
		t.Fatalf("expected city not found error, got %v", err)
	}
}

func TestWmoText(t *testing.T) {
	cases := map[int]string{0: "快晴", 2: "晴れ", 3: "くもり", 61: "雨", 95: "雷雨"}
	for code, want := range cases {
		if got := wmoText(code); got != want {
			t.Fatalf("wmoText(%d) = %s, want %s", code, got, want)
		}
	}
}

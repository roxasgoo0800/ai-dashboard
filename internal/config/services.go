package config

import (
	"net/url"
	"os"
	"strings"
)

func serviceURL(key, fallback string) string {
	value := strings.TrimRight(strings.TrimSpace(os.Getenv(key)), "/")
	if value == "" {
		return fallback
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return fallback
	}
	return value
}

func OllamaURL() string { return serviceURL("OLLAMA_URL", "http://127.0.0.1:11434") }

func ComfyUIURL() string { return serviceURL("COMFYUI_URL", "http://127.0.0.1:8188") }

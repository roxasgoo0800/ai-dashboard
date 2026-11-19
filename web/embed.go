package web

import (
	_ "embed"
	"net/http"
)

//go:embed index.html
var IndexHTML string

//go:embed login.html
var LoginHTML string

//go:embed assets/fonts/PlusJakartaSans-latin.woff2
var JakartaSansWOFF2 []byte

func HandleFontAsset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "font/woff2")
	w.Header().Set("Cache-Control", "public, max-age=86400")
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(JakartaSansWOFF2)
}

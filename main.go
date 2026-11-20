package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-dashboard/internal/auth"
	"ai-dashboard/internal/chat"
	"ai-dashboard/internal/telemetry"
	"ai-dashboard/web"
)

func loadProjectEnv() error {
	wd, err := os.Getwd()
	if err != nil {
		return err
	}
	return loadEnvFile(filepath.Join(wd, ".env"))
}

func loadEnvFile(path string) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for lineNo := 1; scanner.Scan(); lineNo++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		key = strings.TrimSpace(key)
		if !ok || key == "" {
			return fmt.Errorf(".env line %d: expected KEY=VALUE", lineNo)
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return fmt.Errorf(".env line %d: %w", lineNo, err)
		}
	}
	return scanner.Err()
}

func main() {
	if err := loadProjectEnv(); err != nil {
		log.Fatalf("Failed to load .env: %v", err)
	}

	addr := flag.String("addr", envOrDefault("APP_ADDR", "0.0.0.0:8095"), "HTTP listen address")
	mode := flag.String("mode", "hybrid", "Run mode: hybrid | backend | frontend")
	authMode := flag.String("auth-mode", os.Getenv("AUTH_MODE"), "Auth mode: development (username/password) | google")
	backendURL := flag.String("backend-url", os.Getenv("BACKEND_URL"), "Backend API URL when -mode=frontend")
	corsOrigins := flag.String("cors-origins", "", "Comma-separated allowed CORS origins for -mode=backend")
	setUser := flag.String("set-user", "", "Optional: create/update username at startup")
	setPass := flag.String("set-pass", "", "Optional: password for -set-user")
	delUser := flag.String("del-user", "", "Optional: delete username at startup")
	setOnly := flag.Bool("set-only", false, "Exit immediately after -set-user/-set-pass/-del-user")
	googleClientID := flag.String("google-client-id", os.Getenv("GOOGLE_CLIENT_ID"), "Google OAuth Client ID")
	allowedEmails := flag.String("allowed-emails", os.Getenv("ALLOWED_EMAILS"), "Comma-separated Google email whitelist")
	googleOnly := flag.Bool("google-only", false, "Disable username/password login (Google Auth only)")
	flag.Parse()

	runMode := strings.ToLower(strings.TrimSpace(*mode))
	if runMode != "hybrid" && runMode != "backend" && runMode != "frontend" {
		log.Fatalf("Invalid -mode %q (allowed: hybrid, backend, frontend)", *mode)
	}
	selectedAuthMode := strings.ToLower(strings.TrimSpace(*authMode))
	if selectedAuthMode == "" {
		selectedAuthMode = "development"
	}
	if selectedAuthMode != "development" && selectedAuthMode != "google" {
		log.Fatalf("Invalid auth mode %q (allowed: development, google)", selectedAuthMode)
	}

	if err := auth.InitAuthManager(auth.AuthOptions{
		Mode:           runMode,
		AuthMode:       selectedAuthMode,
		BackendURL:     *backendURL,
		CorsCSV:        *corsOrigins,
		SetUser:        *setUser,
		SetPass:        *setPass,
		DelUser:        *delUser,
		GoogleClientID: *googleClientID,
		AllowedEmails:  *allowedEmails,
		GoogleOnly:     *googleOnly || selectedAuthMode == "google",
	}); err != nil {
		log.Fatalf("Auth init failed: %v", err)
	}
	if *setOnly {
		return
	}

	http.HandleFunc("/api/health", auth.WithSecurity(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	http.HandleFunc("/api/assets/fonts/plus-jakarta-sans.woff2", auth.WithSecurity(web.HandleFontAsset))

	if runMode == "frontend" {
		startFrontendProxy(*addr, *backendURL)
		return
	}

	// Warm up CPU delta and start 1s background sampler (hybrid & backend modes)
	telemetry.ReadCPUMetrics()
	time.Sleep(150 * time.Millisecond)
	telemetry.StartSampler()

	// Auth routes
	http.HandleFunc("/api/auth/providers", auth.WithSecurity(auth.HandleAuthProviders))
	http.HandleFunc("/api/auth/google", auth.WithSecurity(auth.HandleAuthGoogle))
	http.HandleFunc("/api/auth/login", auth.WithSecurity(auth.HandleAuthLogin))
	http.HandleFunc("/api/auth/logout", auth.WithSecurity(auth.HandleAuthLogout))
	http.HandleFunc("/api/auth/me", auth.WithSecurity(auth.HandleAuthMe))
	http.HandleFunc("/api/auth/password", auth.WithAuth(auth.HandleAuthChangePassword))

	// Telemetry & system actions
	http.HandleFunc("/api/metrics", auth.WithAuth(telemetry.HandleMetrics))
	http.HandleFunc("/api/history", auth.WithAuth(telemetry.HandleHistory))
	http.HandleFunc("/api/stream", auth.WithAuth(telemetry.HandleStream))
	http.HandleFunc("/api/energy", auth.WithAuth(telemetry.HandleEnergyAPI))
	http.HandleFunc("/api/energy/settings", auth.WithAuth(telemetry.HandleEnergySettings))
	http.HandleFunc("/api/action/unload-ollama", auth.WithAuth(telemetry.HandleUnloadOllama))
	http.HandleFunc("/api/action/free-comfy", auth.WithAuth(telemetry.HandleFreeComfy))
	http.HandleFunc("/api/action/restart-ollama", auth.WithAuth(telemetry.HandleRestartOllama))
	http.HandleFunc("/api/action/restart-mcp", auth.WithAuth(telemetry.HandleRestartMCP))
	http.HandleFunc("/api/action/shutdown-pc", auth.WithAuth(telemetry.HandleShutdownPC))
	http.HandleFunc("/api/action/reboot-pc", auth.WithAuth(telemetry.HandleRebootPC))
	http.HandleFunc("/api/action/stop-comfy", auth.WithAuth(telemetry.HandleStopComfy))
	http.HandleFunc("/api/action/gpu-power", auth.WithAuth(telemetry.HandleGPUPower))

	// AI Chat Studio, /goal, Usage Calendar & Memory/Logs
	http.HandleFunc("/api/chat", auth.WithAuth(chat.HandleChat))
	http.HandleFunc("/api/chat-media", auth.WithAuth(chat.HandleChatMedia))
	http.HandleFunc("/api/chat-download", auth.WithAuth(chat.HandleChatDownload))
	http.HandleFunc("/api/chats", auth.WithAuth(chat.HandleChatsCollection))
	http.HandleFunc("/api/chats/", auth.WithAuth(chat.HandleChatSession))
	http.HandleFunc("/api/mcp/status", auth.WithAuth(chat.HandleMCPStatus))
	http.HandleFunc("/api/usage", auth.WithAuth(chat.HandleUsageAPI))
	http.HandleFunc("/api/memory", auth.WithAuth(chat.HandleMemoryStatus))
	http.HandleFunc("/api/memory/flush", auth.WithAuth(chat.HandleMemoryFlush))
	http.HandleFunc("/api/memory/logs", auth.WithAuth(chat.HandleMemoryLogs))

	// UI routes
	http.HandleFunc("/api/ui", auth.WithAuth(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = io.WriteString(w, web.IndexHTML)
	}))
	http.HandleFunc("/", auth.WithSecurity(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		if runMode == "backend" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"status":"online"}`)
			return
		}
		tok := auth.ExtractTokenFromRequest(r)
		if _, err := auth.VerifyJWT(tok); err == nil {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = io.WriteString(w, web.IndexHTML)
			return
		}
		if tok != "" {
			auth.ClearAuthCookies(w, r)
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, web.LoginHTML)
	}))

	fmt.Printf("========================================================================\n")
	fmt.Printf("  LOCAL AI CONTROL PLANE (Mode: %s | JWT + CSRF Enabled)\n", strings.ToUpper(runMode))
	fmt.Printf("  Listen : %s\n", *addr)
	fmt.Printf("  Local  : http://127.0.0.1:8095\n")
	fmt.Printf("  LAN    : http://192.168.1.7:8095\n")
	fmt.Printf("========================================================================\n")
	log.Fatal(http.ListenAndServe(*addr, nil))
}

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func startFrontendProxy(addr, backendRaw string) {
	target, err := url.Parse(strings.TrimRight(backendRaw, "/"))
	if err != nil || target.Scheme == "" || target.Host == "" {
		log.Fatalf("Invalid -backend-url %q", backendRaw)
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1
	origDirector := proxy.Director
	proxy.Director = func(req *http.Request) { origDirector(req) }
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, perr error) {
		log.Printf("[PROXY] backend error on %s %s: %v", r.Method, r.URL.Path, perr)
		auth.ApplySecurityHeaders(w, r)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, `{"error":"Backend service temporarily unavailable"}`)
	}

	uiClient := &http.Client{Timeout: 6 * time.Second}
	http.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	})
	http.HandleFunc("/", auth.WithSecurity(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		if c, err := r.Cookie(auth.AuthCookieName); err == nil && c.Value != "" {
			req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target.String()+"/api/ui", nil)
			if err == nil {
				req.Host = r.Host
				for _, hdr := range []string{"Cookie", "X-Forwarded-Proto", "Cf-Visitor"} {
					if v := r.Header.Get(hdr); v != "" {
						req.Header.Set(hdr, v)
					}
				}
				if resp, err := uiClient.Do(req); err == nil {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						w.Header().Set("Content-Type", "text/html; charset=utf-8")
						_, _ = io.Copy(w, resp.Body)
						return
					}
					if resp.StatusCode == http.StatusUnauthorized {
						auth.ClearAuthCookies(w, r)
					}
				}
			}
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, web.LoginHTML)
	}))

	fmt.Printf("========================================================================\n")
	fmt.Printf("  AI CONTROL PLANE (FRONTEND PROXY MODE)\n")
	fmt.Printf("  Listen : %s\n", addr)
	fmt.Printf("========================================================================\n")
	log.Fatal(http.ListenAndServe(addr, nil))
}

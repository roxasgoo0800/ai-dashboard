package auth

import (
	"crypto"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	AuthCookieName        = "ai_jwt"
	csrfCookieName        = "ai_csrf"
	pbkdf2Iterations      = 100000
	jwtTTL                = 12 * time.Hour
	maxLoginFails         = 5
	loginWindow           = 5 * time.Minute
	defaultGoogleClientID = "846480250188-knhuu7hqk16pibbcq1pvhlk6t99kisk3.apps.googleusercontent.com"
	googleJWKSURL         = "https://www.googleapis.com/oauth2/v3/certs"
)

var defaultAllowedEmails = []string{
	"roxas0088@gmail.com",
	"roxasgoo@gmail.com",
	"rizalfahmi0890@gmail.com",
	"roxas0800@gmail.com",
}

type UserRecord struct {
	Username     string `json:"username"`
	PasswordHash string `json:"password_hash"` // pbkdf2_sha256$iter$saltHex$hashHex
	Role         string `json:"role"`
	UpdatedAt    int64  `json:"updated_at"`
}

type AuthConfig struct {
	JWTSecret          string                `json:"jwt_secret"`
	Users              map[string]UserRecord `json:"users"`
	GoogleClientID     string                `json:"google_client_id,omitempty"`
	AllowedEmails      []string              `json:"allowed_emails,omitempty"`
	AllowPasswordLogin *bool                 `json:"allow_password_login,omitempty"`
}

type AuthOptions struct {
	Mode           string
	BackendURL     string
	CorsCSV        string
	SetUser        string
	SetPass        string
	DelUser        string
	GoogleClientID string
	AllowedEmails  string
	GoogleOnly     bool
}

type JWTClaims struct {
	Sub      string `json:"sub"`
	Role     string `json:"role"`
	AuthType string `json:"auth_type,omitempty"`
	JTI      string `json:"jti"`
	IAT      int64  `json:"iat"`
	EXP      int64  `json:"exp"`
}

type failBucket struct {
	Count       int
	WindowStart time.Time
	LockedUntil time.Time
}

type AuthManager struct {
	mu             sync.RWMutex
	cfgPath        string
	credPath       string
	cfg            AuthConfig
	revokedJTIs    map[string]int64
	failMu         sync.Mutex
	failMap        map[string]failBucket
	corsOrigins    map[string]bool
	runMode        string
	backendURL     string
	googleClientID string
	allowedEmails  map[string]bool
	allowPassword  bool
	googleCertsURL string
	jwksMu         sync.RWMutex
	jwksKeys       map[string]*rsa.PublicKey
	jwksExpiry     time.Time
	jwksLastFetch  time.Time
	httpClient     *http.Client
}

var authMgr *AuthManager

func appBaseDir() string {
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if !strings.HasPrefix(dir, "/tmp/") {
			return dir
		}
	}
	return "/home/rizal/Documents/ollama/ai-dashboard"
}

func randomHex(nBytes int) string {
	b := make([]byte, nBytes)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

func pbkdf2SHA256(password, salt []byte, iter, keyLen int) []byte {
	prf := func(key, data []byte) []byte {
		m := hmac.New(sha256.New, key)
		m.Write(data)
		return m.Sum(nil)
	}
	numBlocks := (keyLen + 31) / 32
	var dk []byte
	for block := 1; block <= numBlocks; block++ {
		var uInput []byte
		uInput = append(uInput, salt...)
		var idxBuf [4]byte
		binary.BigEndian.PutUint32(idxBuf[:], uint32(block))
		uInput = append(uInput, idxBuf[:]...)
		u := prf(password, uInput)
		t := make([]byte, len(u))
		copy(t, u)
		for i := 2; i <= iter; i++ {
			u = prf(password, u)
			for j := range t {
				t[j] ^= u[j]
			}
		}
		dk = append(dk, t...)
	}
	return dk[:keyLen]
}

func hashPassword(password string) string {
	salt := make([]byte, 16)
	_, _ = rand.Read(salt)
	dk := pbkdf2SHA256([]byte(password), salt, pbkdf2Iterations, 32)
	return fmt.Sprintf("pbkdf2_sha256$%d$%s$%s", pbkdf2Iterations, hex.EncodeToString(salt), hex.EncodeToString(dk))
}

func verifyPassword(password, stored string) bool {
	parts := strings.Split(stored, "$")
	if len(parts) != 4 || parts[0] != "pbkdf2_sha256" {
		return false
	}
	var iter int
	if _, err := fmt.Sscanf(parts[1], "%d", &iter); err != nil || iter < 10000 {
		return false
	}
	salt, err1 := hex.DecodeString(parts[2])
	expected, err2 := hex.DecodeString(parts[3])
	if err1 != nil || err2 != nil || len(expected) == 0 {
		return false
	}
	got := pbkdf2SHA256([]byte(password), salt, iter, len(expected))
	return subtle.ConstantTimeCompare(got, expected) == 1
}

func normalizeEmail(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func InitAuthManager(opts AuthOptions) error {
	base := appBaseDir()
	_ = os.MkdirAll(base, 0o755)
	m := &AuthManager{
		cfgPath:        filepath.Join(base, "auth_config.json"),
		credPath:       filepath.Join(base, "initial_credentials.txt"),
		revokedJTIs:    map[string]int64{},
		failMap:        map[string]failBucket{},
		corsOrigins:    map[string]bool{},
		allowedEmails:  map[string]bool{},
		allowPassword:  !opts.GoogleOnly,
		googleCertsURL: googleJWKSURL,
		jwksKeys:       map[string]*rsa.PublicKey{},
		httpClient:     &http.Client{Timeout: 8 * time.Second},
		runMode:        opts.Mode,
		backendURL:     opts.BackendURL,
	}
	for _, o := range strings.Split(opts.CorsCSV, ",") {
		o = strings.TrimRight(strings.TrimSpace(o), "/")
		if o != "" {
			m.corsOrigins[o] = true
		}
	}

	if opts.Mode == "frontend" {
		// Frontend proxy mode forwards all /api/* and auth checks to the backend PC.
		authMgr = m
		return nil
	}

	if raw, err := os.ReadFile(m.cfgPath); err == nil {
		_ = json.Unmarshal(raw, &m.cfg)
	}
	changed := false
	if len(m.cfg.JWTSecret) < 32 {
		m.cfg.JWTSecret = randomHex(32)
		changed = true
	}
	if m.cfg.Users == nil {
		m.cfg.Users = map[string]UserRecord{}
	}

	// Google OAuth Client ID setup (flag -> existing config -> default)
	if strings.TrimSpace(opts.GoogleClientID) != "" {
		m.cfg.GoogleClientID = strings.TrimSpace(opts.GoogleClientID)
		changed = true
	} else if strings.TrimSpace(m.cfg.GoogleClientID) == "" {
		m.cfg.GoogleClientID = defaultGoogleClientID
		changed = true
	}
	m.googleClientID = strings.TrimSpace(m.cfg.GoogleClientID)

	// Google Email Whitelist setup (never allows wildcard "*")
	if strings.TrimSpace(opts.AllowedEmails) != "" {
		var clean []string
		for _, e := range strings.Split(opts.AllowedEmails, ",") {
			ne := normalizeEmail(e)
			if ne != "" && ne != "*" && strings.Contains(ne, "@") {
				clean = append(clean, ne)
			}
		}
		if len(clean) > 0 {
			m.cfg.AllowedEmails = clean
			changed = true
		}
	} else if len(m.cfg.AllowedEmails) == 0 {
		m.cfg.AllowedEmails = append([]string(nil), defaultAllowedEmails...)
		changed = true
	}
	for _, e := range m.cfg.AllowedEmails {
		ne := normalizeEmail(e)
		if ne != "" && ne != "*" && strings.Contains(ne, "@") {
			m.allowedEmails[ne] = true
		}
	}

	if opts.GoogleOnly {
		f := false
		if m.cfg.AllowPasswordLogin == nil || *m.cfg.AllowPasswordLogin {
			m.cfg.AllowPasswordLogin = &f
			changed = true
		}
		m.allowPassword = false
	} else if m.cfg.AllowPasswordLogin != nil {
		m.allowPassword = *m.cfg.AllowPasswordLogin
	}

	if len(m.cfg.Users) == 0 && opts.SetUser == "" && m.allowPassword {
		initPass := randomHex(8)
		uname := "admin"
		m.cfg.Users[uname] = UserRecord{
			Username:     uname,
			PasswordHash: hashPassword(initPass),
			Role:         "admin",
			UpdatedAt:    time.Now().UnixMilli(),
		}
		changed = true
		note := fmt.Sprintf("Username: %s\nPassword: %s\n(Segera ganti password di dalam dashboard setelah login)\n", uname, initPass)
		_ = os.WriteFile(m.credPath, []byte(note), 0o600)
		fmt.Printf("[AUTH] Akun awal dibuat -> Username: %s | Password tersimpan di: %s\n", uname, m.credPath)
	}
	if opts.SetUser != "" && opts.SetPass != "" {
		m.cfg.Users[opts.SetUser] = UserRecord{
			Username:     opts.SetUser,
			PasswordHash: hashPassword(opts.SetPass),
			Role:         "admin",
			UpdatedAt:    time.Now().UnixMilli(),
		}
		changed = true
		fmt.Printf("[AUTH] Password untuk user '%s' berhasil diperbarui.\n", opts.SetUser)
	}
	if opts.DelUser != "" {
		if _, exists := m.cfg.Users[opts.DelUser]; exists && (len(m.cfg.Users) > 1 || len(m.allowedEmails) > 0) {
			delete(m.cfg.Users, opts.DelUser)
			changed = true
			if opts.DelUser == "admin" {
				_ = os.Remove(m.credPath)
			}
			fmt.Printf("[AUTH] User '%s' berhasil dihapus.\n", opts.DelUser)
		} else if len(m.cfg.Users) <= 1 {
			fmt.Printf("[AUTH] Peringatan: user '%s' tidak dihapus karena tidak ada metode login lain aktif.\n", opts.DelUser)
		}
	}
	if changed {
		if err := m.saveConfigLocked(); err != nil {
			return err
		}
	}
	authMgr = m
	return nil
}

func (m *AuthManager) saveConfigLocked() error {
	b, err := json.MarshalIndent(m.cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(m.cfgPath, b, 0o600)
}

func b64urlEncode(b []byte) string {
	return base64.RawURLEncoding.EncodeToString(b)
}

func b64urlDecode(s string) ([]byte, error) {
	return base64.RawURLEncoding.DecodeString(s)
}

func (m *AuthManager) signJWT(claims JWTClaims) (string, error) {
	hdrJSON := `{"alg":"HS256","typ":"JWT"}`
	payloadBytes, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	signingInput := b64urlEncode([]byte(hdrJSON)) + "." + b64urlEncode(payloadBytes)
	m.mu.RLock()
	secret := m.cfg.JWTSecret
	m.mu.RUnlock()
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signingInput))
	sig := b64urlEncode(mac.Sum(nil))
	return signingInput + "." + sig, nil
}

func (m *AuthManager) VerifyJWT(token string) (*JWTClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errors.New("invalid jwt format")
	}
	hdrBytes, err := b64urlDecode(parts[0])
	if err != nil {
		return nil, err
	}
	var hdr struct {
		Alg string `json:"alg"`
		Typ string `json:"typ"`
	}
	if json.Unmarshal(hdrBytes, &hdr) != nil || hdr.Alg != "HS256" {
		return nil, errors.New("unsupported jwt alg")
	}
	signingInput := parts[0] + "." + parts[1]
	sigBytes, err := b64urlDecode(parts[2])
	if err != nil {
		return nil, err
	}

	m.mu.RLock()
	secret := m.cfg.JWTSecret
	m.mu.RUnlock()

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signingInput))
	expectedSig := mac.Sum(nil)
	if !hmac.Equal(sigBytes, expectedSig) {
		return nil, errors.New("invalid signature")
	}

	payloadBytes, err := b64urlDecode(parts[1])
	if err != nil {
		return nil, err
	}
	var claims JWTClaims
	if err := json.Unmarshal(payloadBytes, &claims); err != nil {
		return nil, err
	}
	now := time.Now().Unix()
	if claims.EXP <= now || claims.Sub == "" || claims.JTI == "" {
		return nil, errors.New("token expired")
	}

	m.mu.RLock()
	_, revoked := m.revokedJTIs[claims.JTI]
	m.mu.RUnlock()
	if revoked {
		return nil, errors.New("token revoked")
	}
	return &claims, nil
}

func (m *AuthManager) deriveCSRF(jti string) string {
	m.mu.RLock()
	secret := m.cfg.JWTSecret
	m.mu.RUnlock()
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte("csrf:" + jti))
	return hex.EncodeToString(mac.Sum(nil))
}

func (m *AuthManager) verifyCSRF(claims *JWTClaims, headerToken string) bool {
	if claims == nil || headerToken == "" {
		return false
	}
	expected := m.deriveCSRF(claims.JTI)
	return subtle.ConstantTimeCompare([]byte(headerToken), []byte(expected)) == 1
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(strings.TrimSpace(r.RemoteAddr))
	if err == nil && host != "" {
		return host
	}
	return r.RemoteAddr
}

func (m *AuthManager) checkRateLimit(key string) (bool, int) {
	m.failMu.Lock()
	defer m.failMu.Unlock()
	now := time.Now()
	b, ok := m.failMap[key]
	if !ok {
		return true, 0
	}
	if !b.LockedUntil.IsZero() && now.Before(b.LockedUntil) {
		secs := int(b.LockedUntil.Sub(now).Seconds()) + 1
		return false, secs
	}
	if now.Sub(b.WindowStart) > loginWindow {
		delete(m.failMap, key)
	}
	return true, 0
}

func (m *AuthManager) recordLoginFailure(key string, limit int) {
	m.failMu.Lock()
	defer m.failMu.Unlock()
	now := time.Now()
	b, ok := m.failMap[key]
	if !ok || now.Sub(b.WindowStart) > loginWindow {
		b = failBucket{Count: 0, WindowStart: now}
	}
	b.Count++
	if b.Count >= limit {
		b.LockedUntil = now.Add(loginWindow)
	}
	m.failMap[key] = b
}

func (m *AuthManager) clearLoginFailure(key string) {
	m.failMu.Lock()
	delete(m.failMap, key)
	m.failMu.Unlock()
}

func ExtractTokenFromRequest(r *http.Request) string {
	authz := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(authz), "bearer ") {
		return strings.TrimSpace(authz[7:])
	}
	if c, err := r.Cookie(AuthCookieName); err == nil && c.Value != "" {
		return c.Value
	}
	return ""
}

func ApplySecurityHeaders(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Referrer-Policy", "same-origin")
	w.Header().Set("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
	w.Header().Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
	w.Header().Set("Content-Security-Policy", "default-src 'self' data: blob:; style-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/style; script-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/client; img-src 'self' data: blob: https://*.googleusercontent.com; media-src 'self' data: blob:; connect-src 'self' https://accounts.google.com/gsi/; frame-src 'self' data: blob: https://accounts.google.com/gsi/; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none';")

	origin := strings.TrimRight(r.Header.Get("Origin"), "/")
	if origin != "" && authMgr != nil && authMgr.isAllowedOrigin(origin, r) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-CSRF-Token")
	}
}

func (m *AuthManager) isAllowedOrigin(origin string, r *http.Request) bool {
	if origin == "" {
		return true
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if strings.EqualFold(u.Host, r.Host) {
		return true
	}
	if m.corsOrigins[origin] || m.corsOrigins["*"] {
		return true
	}
	return false
}

func isMutatingMethod(method string) bool {
	return method == http.MethodPost || method == http.MethodPut || method == http.MethodPatch || method == http.MethodDelete
}

func checkSameOriginOrCORS(r *http.Request) bool {
	origin := strings.TrimRight(r.Header.Get("Origin"), "/")
	if origin != "" {
		return authMgr.isAllowedOrigin(origin, r)
	}
	ref := r.Header.Get("Referer")
	if ref != "" {
		if u, err := url.Parse(ref); err == nil {
			refOrigin := u.Scheme + "://" + u.Host
			return authMgr.isAllowedOrigin(refOrigin, r)
		}
		return false
	}
	return true
}

func WithSecurity(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ApplySecurityHeaders(w, r)
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next(w, r)
	}
}

func WithAuth(next http.HandlerFunc) http.HandlerFunc {
	return WithSecurity(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") && !strings.HasSuffix(r.URL.Path, "/stream") {
			w.Header().Set("Cache-Control", "no-store")
		}
		tok := ExtractTokenFromRequest(r)
		claims, err := authMgr.VerifyJWT(tok)
		if err != nil {
			http.Error(w, `{"error":"Unauthorized"}`, http.StatusUnauthorized)
			return
		}
		if isMutatingMethod(r.Method) {
			if !checkSameOriginOrCORS(r) {
				http.Error(w, `{"error":"Forbidden origin"}`, http.StatusForbidden)
				return
			}
			// Require CSRF token whenever authenticated via cookie (or always for browser calls)
			usedBearer := strings.HasPrefix(strings.ToLower(strings.TrimSpace(r.Header.Get("Authorization"))), "bearer ")
			csrfHdr := r.Header.Get("X-CSRF-Token")
			if !usedBearer || csrfHdr != "" {
				if !authMgr.verifyCSRF(claims, csrfHdr) {
					http.Error(w, `{"error":"Invalid or missing CSRF token"}`, http.StatusForbidden)
					return
				}
			}
		}
		next(w, r)
	})
}

func isHTTPS(r *http.Request) bool {
	if r == nil {
		return false
	}
	if r.TLS != nil {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")), "https") {
		return true
	}
	if strings.Contains(strings.ToLower(r.Header.Get("Cf-Visitor")), `"https"`) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(r.Header.Get("Origin"))), "https://")
}

func setAuthCookies(w http.ResponseWriter, r *http.Request, jwtStr, csrfStr string) {
	maxAge := int(jwtTTL.Seconds())
	sec := isHTTPS(r)
	http.SetCookie(w, &http.Cookie{
		Name:     AuthCookieName,
		Value:    jwtStr,
		Path:     "/",
		MaxAge:   maxAge,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   sec,
	})
	http.SetCookie(w, &http.Cookie{
		Name:     csrfCookieName,
		Value:    csrfStr,
		Path:     "/",
		MaxAge:   maxAge,
		HttpOnly: false,
		SameSite: http.SameSiteLaxMode,
		Secure:   sec,
	})
}

func ClearAuthCookies(w http.ResponseWriter, r *http.Request) {
	sec := isHTTPS(r)
	http.SetCookie(w, &http.Cookie{
		Name:     AuthCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   sec,
	})
	http.SetCookie(w, &http.Cookie{
		Name:     csrfCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: false,
		SameSite: http.SameSiteLaxMode,
		Secure:   sec,
	})
}

func issueSession(w http.ResponseWriter, r *http.Request, sub, role, authType string) {
	now := time.Now()
	jti := randomHex(16)
	claims := JWTClaims{
		Sub:      sub,
		Role:     role,
		AuthType: authType,
		JTI:      jti,
		IAT:      now.Unix(),
		EXP:      now.Add(jwtTTL).Unix(),
	}
	jwtStr, err := authMgr.signJWT(claims)
	if err != nil {
		http.Error(w, `{"error":"Token signing error"}`, http.StatusInternalServerError)
		return
	}
	csrfStr := authMgr.deriveCSRF(jti)
	setAuthCookies(w, r, jwtStr, csrfStr)

	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok":         true,
		"username":   sub,
		"role":       role,
		"auth_type":  authType,
		"token":      jwtStr,
		"csrf_token": csrfStr,
		"expires_at": claims.EXP,
		"mode":       authMgr.runMode,
	})
}

func HandleAuthProviders(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"google_enabled":   authMgr != nil && authMgr.googleClientID != "",
		"google_client_id": authMgr.googleClientID,
		"password_enabled": authMgr != nil && authMgr.allowPassword,
	})
}

var errGoogleEmailDenied = errors.New("email not whitelisted")

type googleJWK struct {
	Kty string `json:"kty"`
	Alg string `json:"alg"`
	Use string `json:"use"`
	Kid string `json:"kid"`
	N   string `json:"n"`
	E   string `json:"e"`
}

func parseMaxAge(hdr string) time.Duration {
	for _, part := range strings.Split(hdr, ",") {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(strings.ToLower(part), "max-age=") {
			val := strings.TrimSpace(part[len("max-age="):])
			if sec, err := strconv.Atoi(val); err == nil && sec > 0 {
				d := time.Duration(sec) * time.Second
				if d < 5*time.Minute {
					return 5 * time.Minute
				}
				if d > 24*time.Hour {
					return 24 * time.Hour
				}
				return d
			}
		}
	}
	return time.Hour
}

func parseBoolish(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s == "true" || s == `"true"`
}

func (m *AuthManager) refreshGoogleJWKS(force bool) error {
	m.jwksMu.Lock()
	defer m.jwksMu.Unlock()

	now := time.Now()
	if !force && len(m.jwksKeys) > 0 && now.Before(m.jwksExpiry) {
		return nil
	}
	if force && now.Sub(m.jwksLastFetch) < 15*time.Second && len(m.jwksKeys) > 0 {
		return nil
	}

	resp, err := m.httpClient.Get(m.googleCertsURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("jwks HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 256*1024))
	if err != nil {
		return err
	}
	var payload struct {
		Keys []googleJWK `json:"keys"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return err
	}
	parsed := make(map[string]*rsa.PublicKey, len(payload.Keys))
	for _, k := range payload.Keys {
		if k.Kty != "RSA" || k.Kid == "" || (k.Alg != "" && k.Alg != "RS256") || (k.Use != "" && k.Use != "sig") {
			continue
		}
		nb, err1 := b64urlDecode(k.N)
		eb, err2 := b64urlDecode(k.E)
		if err1 != nil || err2 != nil || len(nb) == 0 || len(eb) == 0 {
			continue
		}
		nInt := new(big.Int).SetBytes(nb)
		eInt := int(new(big.Int).SetBytes(eb).Int64())
		if nInt.BitLen() < 1024 || eInt <= 1 {
			continue
		}
		parsed[k.Kid] = &rsa.PublicKey{N: nInt, E: eInt}
	}
	if len(parsed) == 0 {
		return errors.New("no valid RSA keys in Google JWKS")
	}
	m.jwksKeys = parsed
	m.jwksExpiry = now.Add(parseMaxAge(resp.Header.Get("Cache-Control")))
	m.jwksLastFetch = now
	return nil
}

func (m *AuthManager) getGooglePublicKey(kid string) (*rsa.PublicKey, error) {
	if kid == "" {
		return nil, errors.New("missing kid")
	}
	m.jwksMu.RLock()
	pub := m.jwksKeys[kid]
	fresh := pub != nil && time.Now().Before(m.jwksExpiry)
	m.jwksMu.RUnlock()
	if fresh {
		return pub, nil
	}

	if err := m.refreshGoogleJWKS(pub == nil); err != nil {
		return nil, err
	}
	m.jwksMu.RLock()
	pub = m.jwksKeys[kid]
	m.jwksMu.RUnlock()
	if pub == nil {
		return nil, fmt.Errorf("unknown Google JWK kid %q", kid)
	}
	return pub, nil
}

func (m *AuthManager) verifyGoogleIDToken(idToken string) (string, error) {
	if m.googleClientID == "" {
		return "", errors.New("google auth not configured")
	}
	parts := strings.Split(strings.TrimSpace(idToken), ".")
	if len(parts) != 3 {
		return "", errors.New("invalid google jwt format")
	}
	hdrBytes, err := b64urlDecode(parts[0])
	if err != nil {
		return "", err
	}
	var hdr struct {
		Alg string `json:"alg"`
		Kid string `json:"kid"`
	}
	if err := json.Unmarshal(hdrBytes, &hdr); err != nil || hdr.Alg != "RS256" || hdr.Kid == "" {
		return "", errors.New("unsupported google token header")
	}
	sigBytes, err := b64urlDecode(parts[2])
	if err != nil {
		return "", err
	}
	pubKey, err := m.getGooglePublicKey(hdr.Kid)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	if err := rsa.VerifyPKCS1v15(pubKey, crypto.SHA256, digest[:], sigBytes); err != nil {
		return "", errors.New("invalid google token signature")
	}

	payloadBytes, err := b64urlDecode(parts[1])
	if err != nil {
		return "", err
	}
	var claims struct {
		Iss           string          `json:"iss"`
		Aud           string          `json:"aud"`
		Email         string          `json:"email"`
		EmailVerified json.RawMessage `json:"email_verified"`
		Iat           int64           `json:"iat"`
		Exp           int64           `json:"exp"`
	}
	if err := json.Unmarshal(payloadBytes, &claims); err != nil {
		return "", err
	}
	if claims.Iss != "https://accounts.google.com" && claims.Iss != "accounts.google.com" {
		return "", errors.New("invalid issuer")
	}
	if claims.Aud != m.googleClientID {
		return "", errors.New("invalid audience")
	}
	now := time.Now().Unix()
	if claims.Exp <= now {
		return "", errors.New("google token expired")
	}
	if claims.Iat > now+300 {
		return "", errors.New("google token issued in the future")
	}
	if !parseBoolish(claims.EmailVerified) {
		return "", errors.New("google email not verified")
	}

	email := normalizeEmail(claims.Email)
	if email == "" || !m.allowedEmails[email] {
		return email, errGoogleEmailDenied
	}
	return email, nil
}

func HandleAuthGoogle(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	if !checkSameOriginOrCORS(r) {
		http.Error(w, `{"error":"Forbidden origin"}`, http.StatusForbidden)
		return
	}

	srcIP := clientIP(r)
	ipKey := srcIP + "|google"
	ipSprayKey := srcIP + "|*"
	for _, k := range []string{ipKey, ipSprayKey} {
		if ok, retrySec := authMgr.checkRateLimit(k); !ok {
			w.Header().Set("Retry-After", fmt.Sprintf("%d", retrySec))
			w.WriteHeader(http.StatusTooManyRequests)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": fmt.Sprintf("Terlalu banyak percobaan login. Coba lagi dalam %d detik.", retrySec),
			})
			return
		}
	}

	var body struct {
		IDToken string `json:"id_token"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 16384)).Decode(&body); err != nil || strings.TrimSpace(body.IDToken) == "" {
		http.Error(w, `{"error":"Google id_token wajib dikirim"}`, http.StatusBadRequest)
		return
	}

	email, err := authMgr.verifyGoogleIDToken(body.IDToken)
	if err != nil {
		authMgr.recordLoginFailure(ipKey, maxLoginFails)
		authMgr.recordLoginFailure(ipSprayKey, 20)
		w.WriteHeader(http.StatusUnauthorized)
		if errors.Is(err, errGoogleEmailDenied) {
			fmt.Printf("[AUTH] Google login ditolak (bukan whitelist): %q ip=%s\n", email, srcIP)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "Email Google tidak terdaftar atau tidak diizinkan mengakses dashboard ini."})
			return
		}
		fmt.Printf("[AUTH] Google login gagal (%v) ip=%s\n", err, srcIP)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": "Token Google tidak valid atau kedaluwarsa."})
		return
	}

	authMgr.clearLoginFailure(ipKey)
	authMgr.clearLoginFailure(ipSprayKey)
	fmt.Printf("[AUTH] Google login sukses: %s (ip=%s)\n", email, srcIP)
	issueSession(w, r, email, "admin", "google")
}

func HandleAuthLogin(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	if !authMgr.allowPassword {
		http.Error(w, `{"error":"Login password dinonaktifkan; silakan login menggunakan Google Sign-In"}`, http.StatusForbidden)
		return
	}
	if !checkSameOriginOrCORS(r) {
		http.Error(w, `{"error":"Forbidden origin"}`, http.StatusForbidden)
		return
	}

	var body struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&body); err != nil {
		http.Error(w, `{"error":"Invalid JSON"}`, http.StatusBadRequest)
		return
	}
	uname := strings.TrimSpace(body.Username)
	srcIP := clientIP(r)
	ipKey := srcIP + "|" + strings.ToLower(uname)
	ipSprayKey := srcIP + "|*"
	if ok, retrySec := authMgr.checkRateLimit(ipKey); !ok {
		w.Header().Set("Retry-After", fmt.Sprintf("%d", retrySec))
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Terlalu banyak percobaan gagal. Coba lagi dalam %d detik.", retrySec),
		})
		return
	}
	if ok, retrySec := authMgr.checkRateLimit(ipSprayKey); !ok {
		w.Header().Set("Retry-After", fmt.Sprintf("%d", retrySec))
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Terlalu banyak percobaan gagal. Coba lagi dalam %d detik.", retrySec),
		})
		return
	}

	authMgr.mu.RLock()
	rec, exists := authMgr.cfg.Users[uname]
	authMgr.mu.RUnlock()

	if !exists || !verifyPassword(body.Password, rec.PasswordHash) {
		authMgr.recordLoginFailure(ipKey, maxLoginFails)
		authMgr.recordLoginFailure(ipSprayKey, 20)
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": "Username atau password salah"})
		return
	}
	authMgr.clearLoginFailure(ipKey)
	authMgr.clearLoginFailure(ipSprayKey)

	issueSession(w, r, rec.Username, rec.Role, "password")
}

func HandleAuthMe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	tok := ExtractTokenFromRequest(r)
	claims, err := authMgr.VerifyJWT(tok)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"authenticated": false,
		})
		return
	}
	authType := claims.AuthType
	if authType == "" {
		authType = "password"
	}
	csrfStr := authMgr.deriveCSRF(claims.JTI)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"authenticated": true,
		"username":      claims.Sub,
		"role":          claims.Role,
		"auth_type":     authType,
		"csrf_token":    csrfStr,
		"expires_at":    claims.EXP,
		"mode":          authMgr.runMode,
	})
}

func HandleAuthLogout(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	tok := ExtractTokenFromRequest(r)
	if claims, err := authMgr.VerifyJWT(tok); err == nil {
		if !authMgr.verifyCSRF(claims, r.Header.Get("X-CSRF-Token")) {
			http.Error(w, `{"error":"Invalid CSRF token"}`, http.StatusForbidden)
			return
		}
		authMgr.mu.Lock()
		authMgr.revokedJTIs[claims.JTI] = claims.EXP
		now := time.Now().Unix()
		for k, exp := range authMgr.revokedJTIs {
			if exp < now {
				delete(authMgr.revokedJTIs, k)
			}
		}
		authMgr.mu.Unlock()
	}
	ClearAuthCookies(w, r)
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func HandleAuthChangePassword(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	tok := ExtractTokenFromRequest(r)
	claims, err := authMgr.VerifyJWT(tok)
	if err != nil {
		http.Error(w, `{"error":"Unauthorized"}`, http.StatusUnauthorized)
		return
	}
	if claims.AuthType == "google" {
		http.Error(w, `{"error":"Akun login Google tidak menggunakan password lokal"}`, http.StatusForbidden)
		return
	}
	if !authMgr.verifyCSRF(claims, r.Header.Get("X-CSRF-Token")) {
		http.Error(w, `{"error":"Invalid CSRF token"}`, http.StatusForbidden)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 16384)
	var body struct {
		CurrentPassword string `json:"current_password"`
		NewPassword     string `json:"new_password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, `{"error":"Invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if len(body.NewPassword) < 8 {
		http.Error(w, `{"error":"Password baru minimal 8 karakter"}`, http.StatusBadRequest)
		return
	}

	authMgr.mu.Lock()
	defer authMgr.mu.Unlock()
	rec, ok := authMgr.cfg.Users[claims.Sub]
	if !ok || !verifyPassword(body.CurrentPassword, rec.PasswordHash) {
		http.Error(w, `{"error":"Password lama salah"}`, http.StatusUnauthorized)
		return
	}
	rec.PasswordHash = hashPassword(body.NewPassword)
	rec.UpdatedAt = time.Now().UnixMilli()
	authMgr.cfg.Users[claims.Sub] = rec
	if err := authMgr.saveConfigLocked(); err != nil {
		http.Error(w, `{"error":"Gagal menyimpan konfigurasi"}`, http.StatusInternalServerError)
		return
	}
	_ = os.Remove(authMgr.credPath)
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func VerifyJWT(tok string) (*JWTClaims, error) {
	return authMgr.VerifyJWT(tok)
}

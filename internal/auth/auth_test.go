package auth

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func signTestGoogleJWT(t *testing.T, priv *rsa.PrivateKey, alg, kid string, claims map[string]any) string {
	t.Helper()
	hdrBytes, _ := json.Marshal(map[string]string{"alg": alg, "typ": "JWT", "kid": kid})
	payBytes, _ := json.Marshal(claims)
	signingInput := b64urlEncode(hdrBytes) + "." + b64urlEncode(payBytes)
	sum := sha256.Sum256([]byte(signingInput))
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, sum[:])
	if err != nil {
		t.Fatalf("sign failed: %v", err)
	}
	return signingInput + "." + b64urlEncode(sig)
}

func newTestAuthManager(t *testing.T, jwksURL string, allowPwd bool) *AuthManager {
	t.Helper()
	dir := t.TempDir()
	m := &AuthManager{
		cfgPath:        filepath.Join(dir, "auth_config.json"),
		credPath:       filepath.Join(dir, "initial_credentials.txt"),
		revokedJTIs:    map[string]int64{},
		failMap:        map[string]failBucket{},
		corsOrigins:    map[string]bool{},
		allowedEmails:  map[string]bool{},
		allowPassword:  allowPwd,
		googleClientID: defaultGoogleClientID,
		googleCertsURL: jwksURL,
		jwksKeys:       map[string]*rsa.PublicKey{},
		httpClient:     &http.Client{Timeout: 3 * time.Second},
		runMode:        "hybrid",
	}
	m.cfg.JWTSecret = randomHex(32)
	for _, e := range defaultAllowedEmails {
		m.allowedEmails[e] = true
	}
	authMgr = m
	return m
}

func TestGoogleAuthFlowAndWhitelist(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	jwksSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=3600")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"keys": []map[string]string{
				{
					"kty": "RSA",
					"alg": "RS256",
					"use": "sig",
					"kid": "test-kid-1",
					"n":   b64urlEncode(priv.PublicKey.N.Bytes()),
					"e":   b64urlEncode(big.NewInt(int64(priv.PublicKey.E)).Bytes()),
				},
			},
		})
	}))
	defer jwksSrv.Close()

	origMgr := authMgr
	defer func() { authMgr = origMgr }()
	m := newTestAuthManager(t, jwksSrv.URL, false)

	now := time.Now().Unix()

	// 1. All 4 whitelisted emails (plus mixed case) must succeed
	allowedInputs := []string{
		"roxas0088@gmail.com",
		"roxasgoo@gmail.com",
		"RizalFahmi0890@gmail.com",
		"roxas0800@gmail.com",
	}
	for _, em := range allowedInputs {
		tok := signTestGoogleJWT(t, priv, "RS256", "test-kid-1", map[string]any{
			"iss":            "https://accounts.google.com",
			"aud":            defaultGoogleClientID,
			"email":          em,
			"email_verified": true,
			"iat":            now,
			"exp":            now + 600,
		})
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/auth/google", bytes.NewBufferString(`{"id_token":"`+tok+`"}`))
		req.RemoteAddr = "127.0.0.1:12345"
		WithSecurity(HandleAuthGoogle)(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("expected 200 for allowed email %s, got %d (%s)", em, rr.Code, rr.Body.String())
		}
		var res map[string]any
		_ = json.Unmarshal(rr.Body.Bytes(), &res)
		if res["auth_type"] != "google" || res["username"] != strings.ToLower(em) {
			t.Fatalf("unexpected payload for %s: %+v", em, res)
		}

		// Verify cookies set and /api/auth/me works
		cookies := rr.Result().Cookies()
		var jwtCookie, csrfCookie *http.Cookie
		for _, c := range cookies {
			if c.Name == AuthCookieName {
				jwtCookie = c
			} else if c.Name == csrfCookieName {
				csrfCookie = c
			}
		}
		if jwtCookie == nil || !jwtCookie.HttpOnly || csrfCookie == nil {
			t.Fatalf("missing expected cookies for %s", em)
		}

		// Verify Google session cannot use local password change
		pwReq := httptest.NewRequest(http.MethodPost, "/api/auth/password", bytes.NewBufferString(`{"current_password":"a","new_password":"longpassword"}`))
		pwReq.AddCookie(jwtCookie)
		pwReq.Header.Set("X-CSRF-Token", csrfCookie.Value)
		pwRec := httptest.NewRecorder()
		WithAuth(HandleAuthChangePassword)(pwRec, pwReq)
		if pwRec.Code != http.StatusForbidden {
			t.Fatalf("expected 403 on password change for google session, got %d", pwRec.Code)
		}

		// Verify logout revokes token
		loReq := httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil)
		loReq.AddCookie(jwtCookie)
		loReq.Header.Set("X-CSRF-Token", csrfCookie.Value)
		loRec := httptest.NewRecorder()
		WithSecurity(HandleAuthLogout)(loRec, loReq)
		if loRec.Code != http.StatusOK {
			t.Fatalf("expected 200 logout, got %d", loRec.Code)
		}
		meReq := httptest.NewRequest(http.MethodGet, "/api/auth/me", nil)
		meReq.AddCookie(jwtCookie)
		meRec := httptest.NewRecorder()
		HandleAuthMe(meRec, meReq)
		if meRec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 after logout revocation, got %d", meRec.Code)
		}
	}

	// 2. Non-whitelisted email, unverified email, wrong aud, wrong iss, expired, tampered, wrong kid -> 401
	badCases := []struct {
		name   string
		alg    string
		kid    string
		claims map[string]any
		tamper bool
	}{
		{"non_whitelisted", "RS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": defaultGoogleClientID, "email": "attacker@gmail.com", "email_verified": true, "iat": now, "exp": now + 300}, false},
		{"unverified_email", "RS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": defaultGoogleClientID, "email": "roxas0800@gmail.com", "email_verified": false, "iat": now, "exp": now + 300}, false},
		{"wrong_aud", "RS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": "wrong-client.apps.googleusercontent.com", "email": "roxas0800@gmail.com", "email_verified": true, "iat": now, "exp": now + 300}, false},
		{"wrong_iss", "RS256", "test-kid-1", map[string]any{"iss": "https://evil.example", "aud": defaultGoogleClientID, "email": "roxas0800@gmail.com", "email_verified": true, "iat": now, "exp": now + 300}, false},
		{"expired", "RS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": defaultGoogleClientID, "email": "roxas0800@gmail.com", "email_verified": true, "iat": now - 900, "exp": now - 10}, false},
		{"tampered_sig", "RS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": defaultGoogleClientID, "email": "roxas0800@gmail.com", "email_verified": true, "iat": now, "exp": now + 300}, true},
		{"alg_downgrade", "HS256", "test-kid-1", map[string]any{"iss": "https://accounts.google.com", "aud": defaultGoogleClientID, "email": "roxas0800@gmail.com", "email_verified": true, "iat": now, "exp": now + 300}, false},
	}

	for i, bc := range badCases {
		tok := signTestGoogleJWT(t, priv, bc.alg, bc.kid, bc.claims)
		if bc.tamper {
			tok = tok[:len(tok)-4] + "AAAA"
		}
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/auth/google", bytes.NewBufferString(`{"id_token":"`+tok+`"}`))
		req.RemoteAddr = "10.0.0." + string(rune('1'+i)) + ":1234"
		WithSecurity(HandleAuthGoogle)(rr, req)
		if rr.Code != http.StatusUnauthorized {
			t.Fatalf("%s: expected 401, got %d (%s)", bc.name, rr.Code, rr.Body.String())
		}
	}

	// 3. Providers endpoint must never leak whitelist emails
	provRec := httptest.NewRecorder()
	WithSecurity(HandleAuthProviders)(provRec, httptest.NewRequest(http.MethodGet, "/api/auth/providers", nil))
	if provRec.Code != http.StatusOK || strings.Contains(provRec.Body.String(), "roxas") {
		t.Fatalf("providers leaked or failed: %d %s", provRec.Code, provRec.Body.String())
	}

	// 4. Password login disabled when allowPassword=false
	m.allowPassword = false
	pwdRec := httptest.NewRecorder()
	WithSecurity(HandleAuthLogin)(pwdRec, httptest.NewRequest(http.MethodPost, "/api/auth/login", bytes.NewBufferString(`{"username":"rizal","password":"x"}`)))
	if pwdRec.Code != http.StatusForbidden {
		t.Fatalf("expected 403 when password login disabled, got %d", pwdRec.Code)
	}
}

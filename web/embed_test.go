package web

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPlusJakartaSansFontAndRPGThemeAssets(t *testing.T) {
	if len(JakartaSansWOFF2) < 10000 {
		t.Fatalf("expected embedded PlusJakartaSans-latin.woff2 >= 10KB, got %d bytes", len(JakartaSansWOFF2))
	}

	req := httptest.NewRequest(http.MethodGet, "/api/assets/fonts/plus-jakarta-sans.woff2", nil)
	rr := httptest.NewRecorder()
	HandleFontAsset(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from HandleFontAsset, got %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "font/woff2" {
		t.Fatalf("expected Content-Type font/woff2, got %q", ct)
	}
	if rr.Body.Len() != len(JakartaSansWOFF2) {
		t.Fatalf("expected body len %d, got %d", len(JakartaSansWOFF2), rr.Body.Len())
	}

	badReq := httptest.NewRequest(http.MethodPost, "/api/assets/fonts/plus-jakarta-sans.woff2", nil)
	badRR := httptest.NewRecorder()
	HandleFontAsset(badRR, badReq)
	if badRR.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST font asset, got %d", badRR.Code)
	}

	for name, htmlDoc := range map[string]string{"IndexHTML": IndexHTML, "LoginHTML": LoginHTML} {
		if !strings.Contains(htmlDoc, "font-family: 'Plus Jakarta Sans'") || !strings.Contains(htmlDoc, "/api/assets/fonts/plus-jakarta-sans.woff2") {
			t.Fatalf("%s missing self-hosted Plus Jakarta Sans @font-face", name)
		}
		if !strings.Contains(htmlDoc, `shape-rendering="crispEdges"`) {
			t.Fatalf("%s missing 32-bit pixel-art SVG sprite assets", name)
		}
	}
	if !strings.Contains(IndexHTML, `--sans: 'Plus Jakarta Sans'`) || !strings.Contains(IndexHTML, `:root[data-theme="light"]`) {
		t.Fatalf("IndexHTML missing Plus Jakarta Sans --sans token or light GBA parchment theme")
	}
	if !strings.Contains(IndexHTML, "pixelMageAvatarSvg") || !strings.Contains(IndexHTML, "rpgSpriteBob") {
		t.Fatalf("IndexHTML missing RPG pixel mage avatar sprite helper/animation")
	}
	if strings.Contains(IndexHTML, `<svg class="ico`) || strings.Contains(LoginHTML, `<svg class="ico`) {
		t.Fatalf("found mixed flat vector <svg class=\"ico\"> icon; all UI icons must be cohesive 32-bit pixel art")
	}
	for _, rpgSym := range []string{
		"rpg-event-panel",
		"rpg-error-card",
		"pixelWorkstationDioramaSvg",
		"rpgEventIconSvg",
		"parseRpgGoalEventLine",
		"gpuCoreFrame",
		"workstation-scene-frame",
		"view-usage",
		"view-memory",
		"rpg-ambient-bg",
		"rail-wol-card",
		"rpg-media-frame",
		"usageCalendarGrid",
		"triggerMemoryFlush",
	} {
		if !strings.Contains(IndexHTML, rpgSym) {
			t.Fatalf("IndexHTML missing expected 32-bit RPG art direction component %q", rpgSym)
		}
	}
	for _, tok := range []string{"#e6d3bd", "#1b3b6f", "#7a1c2e", "#c24914", "#1a1310"} {
		if !strings.Contains(IndexHTML, tok) {
			t.Fatalf("IndexHTML missing expected theme color token %s", tok)
		}
	}
	for _, uiTok := range []string{"toggleSlopBtn", "toggleBrainstormBtn", "Agent Anti-Slop (/ai-slop)", "Agent Brainstorm (/brainstorm)"} {
		if !strings.Contains(IndexHTML, uiTok) {
			t.Fatalf("IndexHTML missing agent UI element %q", uiTok)
		}
	}
}

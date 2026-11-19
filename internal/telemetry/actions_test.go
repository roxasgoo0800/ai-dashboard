package telemetry

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSystemPowerActionValidation(t *testing.T) {
	origCheck := systemPowerCheck
	origExec := systemPowerExec
	defer func() {
		systemPowerCheck = origCheck
		systemPowerExec = origExec
	}()
	systemPowerCheck = func() error { return nil }
	systemPowerExec = func(subcmd string) error { return nil }

	rrGet := httptest.NewRecorder()
	HandleSystemPowerAction(rrGet, httptest.NewRequest(http.MethodGet, "/api/action/shutdown-pc", nil), "poweroff", "bye")
	if rrGet.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET, got %d", rrGet.Code)
	}

	rrBad := httptest.NewRecorder()
	HandleSystemPowerAction(rrBad, httptest.NewRequest(http.MethodPost, "/api/action/shutdown-pc", bytes.NewBufferString(`{"confirm":false}`)), "poweroff", "bye")
	if rrBad.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 when confirm=false, got %d", rrBad.Code)
	}

	rrOk := httptest.NewRecorder()
	HandleSystemPowerAction(rrOk, httptest.NewRequest(http.MethodPost, "/api/action/shutdown-pc", bytes.NewBufferString(`{"confirm":true}`)), "poweroff", "PC shutting down")
	if rrOk.Code != http.StatusOK {
		t.Fatalf("expected 200 when confirm=true, got %d (%s)", rrOk.Code, rrOk.Body.String())
	}
}

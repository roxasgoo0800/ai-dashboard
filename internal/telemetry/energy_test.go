package telemetry

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestEnergyKWhAndIDRCostCalculation(t *testing.T) {
	energyMu.Lock()
	prevOverride := energyPathOverride
	prevLoaded := energyLoaded
	prevStore := energyStore
	energyPathOverride = filepath.Join(t.TempDir(), "energy_log.json")
	energyLoaded = true
	today := time.Now().In(time.Local).Format("2006-01-02")
	month := time.Now().In(time.Local).Format("2006-01")
	energyStore = energyStoreFile{
		Settings: EnergySettings{
			TariffIDRPerKWh: 1500.0,
			SystemBaseWatts: 55.0,
			VoltageV:        220.0,
		},
		Daily: map[string]DailyEnergyRecord{
			today: {
				Date:          today,
				GPUWh:         600.0,
				TotalWh:       1000.0, // 1.0 kWh => Rp 1.500
				PeakWatts:     210.5,
				PeakGPUWatts:  165.0,
				ActiveSeconds: 3600,
				Samples:       3600,
			},
		},
	}
	energyMu.Unlock()
	defer func() {
		energyMu.Lock()
		energyPathOverride = prevOverride
		energyLoaded = prevLoaded
		energyStore = prevStore
		energyMu.Unlock()
	}()

	req := httptest.NewRequest(http.MethodGet, "/api/energy?month="+month, nil)
	rr := httptest.NewRecorder()
	HandleEnergyAPI(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from HandleEnergyAPI, got %d", rr.Code)
	}
	var res EnergySummaryResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
		t.Fatalf("decode EnergySummaryResponse: %v", err)
	}
	if res.TodayKWh != 1.0 || res.TodayCostIDR != 1500 {
		t.Fatalf("expected 1.0 kWh and Rp 1500, got kwh=%v idr=%v", res.TodayKWh, res.TodayCostIDR)
	}
	if len(res.Daily) != 1 || res.Daily[0].CostIDR != 1500 {
		t.Fatalf("unexpected daily energy entries: %+v", res.Daily)
	}

	// Update settings via POST /api/energy/settings
	sReq := httptest.NewRequest(http.MethodPost, "/api/energy/settings", strings.NewReader(`{"tariff_idr_per_kwh":1699.53,"system_base_watts":60,"voltage_v":220}`))
	sRR := httptest.NewRecorder()
	HandleEnergySettings(sRR, sReq)
	if sRR.Code != http.StatusOK {
		t.Fatalf("expected 200 from HandleEnergySettings, got %d: %s", sRR.Code, sRR.Body.String())
	}
}

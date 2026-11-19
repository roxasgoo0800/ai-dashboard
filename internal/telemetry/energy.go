package telemetry

import (
	"encoding/json"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	defaultTariffIDRPerKWh = 1444.70 // PLN R-1/TR 1.300VA - 2.200VA
	defaultSystemBaseWatts = 65.0    // CPU + Motherboard + RAM + NVMe + Fans + PSU overhead baseline
	defaultVoltageV        = 220.0   // Standard Indonesia PLN AC Voltage
)

type EnergySettings struct {
	TariffIDRPerKWh float64 `json:"tariff_idr_per_kwh"`
	SystemBaseWatts float64 `json:"system_base_watts"`
	VoltageV        float64 `json:"voltage_v"`
}

type DailyEnergyRecord struct {
	Date          string  `json:"date"`
	GPUWh         float64 `json:"gpu_wh"`
	TotalWh       float64 `json:"total_wh"`
	PeakWatts     float64 `json:"peak_watts"`
	PeakGPUWatts  float64 `json:"peak_gpu_watts"`
	ActiveSeconds int64   `json:"active_seconds"`
	Samples       int64   `json:"samples"`
}

type energyStoreFile struct {
	Settings EnergySettings               `json:"settings"`
	Daily    map[string]DailyEnergyRecord `json:"daily"`
}

type DailyEnergyView struct {
	Date         string  `json:"date"`
	GPUKWh       float64 `json:"gpu_kwh"`
	TotalKWh     float64 `json:"total_kwh"`
	CostIDR      float64 `json:"cost_idr"`
	PeakWatts    float64 `json:"peak_watts"`
	PeakGPUWatts float64 `json:"peak_gpu_watts"`
	AvgWatts     float64 `json:"avg_watts"`
	ActiveHours  float64 `json:"active_hours"`
}

type EnergySummaryResponse struct {
	Month                 string            `json:"month"`
	Today                 string            `json:"today"`
	Settings              EnergySettings    `json:"settings"`
	CurrentGPUWatts       float64           `json:"current_gpu_watts"`
	CurrentSystemWatts    float64           `json:"current_system_watts"`
	CurrentTotalWatts     float64           `json:"current_total_watts"`
	CurrentAmps           float64           `json:"current_amps"`
	TodayKWh              float64           `json:"today_kwh"`
	TodayCostIDR          float64           `json:"today_cost_idr"`
	MonthKWh              float64           `json:"month_kwh"`
	MonthCostIDR          float64           `json:"month_cost_idr"`
	TotalKWh              float64           `json:"total_kwh"`
	TotalCostIDR          float64           `json:"total_cost_idr"`
	ProjectedMonthKWh     float64           `json:"projected_month_kwh"`
	ProjectedMonthCostIDR float64           `json:"projected_month_cost_idr"`
	Daily                 []DailyEnergyView `json:"daily"`
}

var (
	energyMu          sync.Mutex
	energyLoaded      bool
	energyStore       energyStoreFile
	energyLastSave    time.Time
	energyLastSample  time.Time
	energyPathOverride string
)

func energyLogPath() string {
	if energyPathOverride != "" {
		return energyPathOverride
	}
	dir := "/home/rizal/Documents/ollama/ai-dashboard/chats"
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), "chats")
		if os.MkdirAll(cand, 0o755) == nil {
			return filepath.Join(cand, "energy_log.json")
		}
	}
	_ = os.MkdirAll(dir, 0o755)
	return filepath.Join(dir, "energy_log.json")
}

func normalizeEnergySettings(s EnergySettings) EnergySettings {
	if s.TariffIDRPerKWh <= 0 {
		s.TariffIDRPerKWh = defaultTariffIDRPerKWh
	}
	if s.SystemBaseWatts < 0 {
		s.SystemBaseWatts = defaultSystemBaseWatts
	}
	if s.SystemBaseWatts == 0 {
		s.SystemBaseWatts = defaultSystemBaseWatts
	}
	if s.VoltageV <= 0 {
		s.VoltageV = defaultVoltageV
	}
	return s
}

func loadEnergyStoreLocked() {
	if energyLoaded {
		return
	}
	energyStore = energyStoreFile{
		Settings: normalizeEnergySettings(EnergySettings{}),
		Daily:    map[string]DailyEnergyRecord{},
	}
	if b, err := os.ReadFile(energyLogPath()); err == nil {
		var loaded energyStoreFile
		if json.Unmarshal(b, &loaded) == nil {
			energyStore.Settings = normalizeEnergySettings(loaded.Settings)
			if loaded.Daily != nil {
				energyStore.Daily = loaded.Daily
			}
		}
	}
	energyLoaded = true
}

func saveEnergyStoreLocked() {
	b, err := json.MarshalIndent(energyStore, "", "  ")
	if err != nil {
		return
	}
	p := energyLogPath()
	_ = os.MkdirAll(filepath.Dir(p), 0o755)
	tmp := p + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, p)
	}
	energyLastSave = time.Now()
}

func computeSystemWatts(baseWatts, cpuPct float64) float64 {
	if cpuPct < 0 {
		cpuPct = 0
	}
	if cpuPct > 100 {
		cpuPct = 100
	}
	// Base motherboard/RAM/fans + dynamic CPU load (up to +45W at 100% CPU)
	return baseWatts + (cpuPct/100.0)*45.0
}

func recordEnergySample(snap DashboardSnapshot) {
	now := time.Now().In(time.Local)
	dateStr := now.Format("2006-01-02")

	energyMu.Lock()
	defer energyMu.Unlock()
	loadEnergyStoreLocked()

	dtSec := 1.0
	if !energyLastSample.IsZero() {
		d := now.Sub(energyLastSample).Seconds()
		if d > 0.1 && d <= 10.0 {
			dtSec = d
		}
	}
	energyLastSample = now

	gpuW := math.Max(0, snap.GPU.PowerDrawW)
	sysW := computeSystemWatts(energyStore.Settings.SystemBaseWatts, snap.CPU.UsagePct)
	totalW := gpuW + sysW

	rec := energyStore.Daily[dateStr]
	rec.Date = dateStr
	rec.GPUWh += (gpuW * dtSec) / 3600.0
	rec.TotalWh += (totalW * dtSec) / 3600.0
	if totalW > rec.PeakWatts {
		rec.PeakWatts = math.Round(totalW*10) / 10
	}
	if gpuW > rec.PeakGPUWatts {
		rec.PeakGPUWatts = math.Round(gpuW*10) / 10
	}
	rec.ActiveSeconds += int64(math.Round(dtSec))
	rec.Samples++
	energyStore.Daily[dateStr] = rec

	if energyLastSave.IsZero() || now.Sub(energyLastSave) >= 20*time.Second {
		saveEnergyStoreLocked()
	}
}

func BuildEnergySummary(monthStr string) EnergySummaryResponse {
	now := time.Now().In(time.Local)
	todayStr := now.Format("2006-01-02")
	monthStr = strings.TrimSpace(monthStr)
	if len(monthStr) != 7 || monthStr[4] != '-' {
		monthStr = now.Format("2006-01")
	}
	tMonth, err := time.Parse("2006-01", monthStr)
	if err != nil {
		monthStr = now.Format("2006-01")
		tMonth, _ = time.Parse("2006-01", monthStr)
	}
	daysInMonth := time.Date(tMonth.Year(), tMonth.Month()+1, 0, 0, 0, 0, 0, time.Local).Day()

	snap := GetLatestSnapshot()

	energyMu.Lock()
	loadEnergyStoreLocked()
	settings := energyStore.Settings
	dailyCopy := make([]DailyEnergyRecord, 0, len(energyStore.Daily))
	for _, rec := range energyStore.Daily {
		dailyCopy = append(dailyCopy, rec)
	}
	energyMu.Unlock()

	gpuW := math.Round(math.Max(0, snap.GPU.PowerDrawW)*10) / 10
	sysW := math.Round(computeSystemWatts(settings.SystemBaseWatts, snap.CPU.UsagePct)*10) / 10
	totalW := math.Round((gpuW+sysW)*10) / 10
	amps := 0.0
	if settings.VoltageV > 0 {
		amps = math.Round((totalW/settings.VoltageV)*100) / 100
	}

	resp := EnergySummaryResponse{
		Month:              monthStr,
		Today:              todayStr,
		Settings:           settings,
		CurrentGPUWatts:    gpuW,
		CurrentSystemWatts: sysW,
		CurrentTotalWatts:  totalW,
		CurrentAmps:        amps,
		Daily:              []DailyEnergyView{},
	}

	var monthActiveDays int
	round4 := func(v float64) float64 { return math.Round(v*10000) / 10000 }
	round0 := func(v float64) float64 { return math.Round(v) }

	for _, rec := range dailyCopy {
		totKWh := rec.TotalWh / 1000.0
		gpuKWh := rec.GPUWh / 1000.0
		cost := totKWh * settings.TariffIDRPerKWh

		resp.TotalKWh += totKWh
		resp.TotalCostIDR += cost
		if rec.Date == todayStr {
			resp.TodayKWh = round4(totKWh)
			resp.TodayCostIDR = round0(cost)
		}
		if strings.HasPrefix(rec.Date, monthStr+"-") {
			resp.MonthKWh += totKWh
			resp.MonthCostIDR += cost
			if rec.ActiveSeconds > 0 {
				monthActiveDays++
			}
			hrs := float64(rec.ActiveSeconds) / 3600.0
			avgW := 0.0
			if hrs > 0 {
				avgW = math.Round((rec.TotalWh/hrs)*10) / 10
			}
			resp.Daily = append(resp.Daily, DailyEnergyView{
				Date:         rec.Date,
				GPUKWh:       round4(gpuKWh),
				TotalKWh:     round4(totKWh),
				CostIDR:      round0(cost),
				PeakWatts:    rec.PeakWatts,
				PeakGPUWatts: rec.PeakGPUWatts,
				AvgWatts:     avgW,
				ActiveHours:  math.Round(hrs*100) / 100,
			})
		}
	}

	sort.Slice(resp.Daily, func(i, j int) bool { return resp.Daily[i].Date < resp.Daily[j].Date })
	resp.TotalKWh = round4(resp.TotalKWh)
	resp.TotalCostIDR = round0(resp.TotalCostIDR)
	resp.MonthKWh = round4(resp.MonthKWh)
	resp.MonthCostIDR = round0(resp.MonthCostIDR)

	if monthActiveDays > 0 {
		avgDailyKWh := resp.MonthKWh / float64(monthActiveDays)
		resp.ProjectedMonthKWh = round4(avgDailyKWh * float64(daysInMonth))
		resp.ProjectedMonthCostIDR = round0(resp.ProjectedMonthKWh * settings.TariffIDRPerKWh)
	}
	return resp
}

func HandleEnergyAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	month := r.URL.Query().Get("month")
	res := BuildEnergySummary(month)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(res)
}

func HandleEnergySettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	var req EnergySettings
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if req.TariffIDRPerKWh < 100 || req.TariffIDRPerKWh > 20000 {
		http.Error(w, "Tarif PLN tidak valid (100 - 20000 Rp/kWh)", http.StatusBadRequest)
		return
	}
	if req.SystemBaseWatts < 10 || req.SystemBaseWatts > 1000 {
		http.Error(w, "Base system watts tidak valid (10 - 1000 W)", http.StatusBadRequest)
		return
	}
	if req.VoltageV <= 0 {
		req.VoltageV = defaultVoltageV
	}
	if req.VoltageV < 100 || req.VoltageV > 260 {
		http.Error(w, "Tegangan tidak valid (100 - 260 V)", http.StatusBadRequest)
		return
	}

	energyMu.Lock()
	loadEnergyStoreLocked()
	energyStore.Settings = req
	saveEnergyStoreLocked()
	energyMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "settings": req})
}

package chat

import (
	"ai-dashboard/internal/config"

	"ai-dashboard/internal/telemetry"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"
)

// UsageEvent tracks a single completed LLM generation turn/cycle.
type UsageEvent struct {
	Key         string  `json:"key"`
	TimestampMS int64   `json:"timestamp_ms"`
	Date        string  `json:"date"` // YYYY-MM-DD (local time)
	Model       string  `json:"model"`
	ChatID      string  `json:"chat_id,omitempty"`
	Tokens      int     `json:"tokens"`
	TokPerSec   float64 `json:"tok_per_sec,omitempty"`
	ElapsedSec  float64 `json:"elapsed_sec,omitempty"`
	GoalCycle   int     `json:"goal_cycle,omitempty"`
	Agent       string  `json:"agent,omitempty"`
}

type DailyUsageItem struct {
	Date     string         `json:"date"`
	Tokens   int            `json:"tokens"`
	Requests int            `json:"requests"`
	AvgTPS   float64        `json:"avg_tps"`
	Models   map[string]int `json:"models"`
}

type ModelUsageItem struct {
	Model        string  `json:"model"`
	Tokens       int     `json:"tokens"`
	Requests     int     `json:"requests"`
	AvgTPS       float64 `json:"avg_tps"`
	SharePct     float64 `json:"share_pct"`
	LastUsedDate string  `json:"last_used_date"`
}

type UsagePeakDay struct {
	Date   string `json:"date"`
	Tokens int    `json:"tokens"`
}

type UsageRecommendation struct {
	Kind   string `json:"kind"` // tip, cost, perf, focus
	Title  string `json:"title"`
	Detail string `json:"detail"`
}

type UsageSummaryResponse struct {
	Month              string                `json:"month"` // YYYY-MM
	Today              string                `json:"today"` // YYYY-MM-DD
	TotalTokens        int                   `json:"total_tokens"`
	MonthTokens        int                   `json:"month_tokens"`
	TodayTokens        int                   `json:"today_tokens"`
	TotalRequests      int                   `json:"total_requests"`
	MonthRequests      int                   `json:"month_requests"`
	AvgTokPerSec       float64               `json:"avg_tok_per_sec"`
	ActiveDays         int                   `json:"active_days"`
	StreakDays         int                   `json:"streak_days"`
	GoalRuns           int                   `json:"goal_runs"`
	PeakDay            UsagePeakDay          `json:"peak_day"`
	EstCloudSavingsUSD float64               `json:"est_cloud_savings_usd"`
	Daily              []DailyUsageItem      `json:"daily"`
	ByModel            []ModelUsageItem      `json:"by_model"`
	MonthByModel       []ModelUsageItem      `json:"month_by_model"`
	Recommendations    []UsageRecommendation `json:"recommendations"`
}

var (
	usageMu          sync.Mutex
	usageEvents      map[string]UsageEvent
	usageLoaded      bool
	usageDirOverride string
)

func usageBaseDir() string {
	if usageDirOverride != "" {
		return usageDirOverride
	}
	return getChatsDir()
}

func usageLogPath() string {
	return filepath.Join(usageBaseDir(), "usage_log.json")
}

func loadUsageEventsLocked() {
	if usageLoaded {
		return
	}
	usageEvents = map[string]UsageEvent{}
	if b, err := os.ReadFile(usageLogPath()); err == nil {
		var list []UsageEvent
		if json.Unmarshal(b, &list) == nil {
			for _, ev := range list {
				if ev.Key != "" && ev.Tokens > 0 {
					usageEvents[ev.Key] = ev
				}
			}
		}
	}
	usageLoaded = true
}

func saveUsageEventsLocked() {
	list := make([]UsageEvent, 0, len(usageEvents))
	for _, ev := range usageEvents {
		list = append(list, ev)
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].TimestampMS != list[j].TimestampMS {
			return list[i].TimestampMS < list[j].TimestampMS
		}
		return list[i].Key < list[j].Key
	})
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return
	}
	tmp := usageLogPath() + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, usageLogPath())
	}
}

func recordUsageEvent(ev UsageEvent) {
	if ev.Tokens <= 0 {
		return
	}
	if ev.TimestampMS <= 0 {
		ev.TimestampMS = time.Now().UnixMilli()
	}
	if ev.Date == "" {
		ev.Date = time.UnixMilli(ev.TimestampMS).In(time.Local).Format("2006-01-02")
	}
	if strings.TrimSpace(ev.Model) == "" {
		ev.Model = "unknown"
	}
	if ev.Key == "" {
		ev.Key = fmt.Sprintf("ev_%d_%s", ev.TimestampMS, ev.Model)
	}
	usageMu.Lock()
	defer usageMu.Unlock()
	loadUsageEventsLocked()
	prev, exists := usageEvents[ev.Key]
	if exists && prev.Tokens == ev.Tokens && prev.Model == ev.Model && prev.Date == ev.Date {
		return
	}
	usageEvents[ev.Key] = ev
	saveUsageEventsLocked()
}

// syncUsageFromChatsLocked scans saved chat sessions and merges any historical
// assistant messages that have recorded token counts so existing sessions show
// up immediately in the calendar and per-model breakdown.
func syncUsageFromChatsLocked() {
	loadUsageEventsLocked()
	dir := usageBaseDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	changed := false
	for _, ent := range entries {
		if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") || ent.Name() == "usage_log.json" {
			continue
		}
		p := filepath.Join(dir, ent.Name())
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var sess ChatSession
		if json.Unmarshal(b, &sess) != nil {
			continue
		}
		if sess.ID == "" {
			sess.ID = strings.TrimSuffix(ent.Name(), ".json")
		}
		baseTS := sess.UpdatedAt
		if baseTS <= 0 {
			baseTS = sess.CreatedAt
		}
		if baseTS <= 0 {
			if info, err := ent.Info(); err == nil {
				baseTS = info.ModTime().UnixMilli()
			} else {
				baseTS = time.Now().UnixMilli()
			}
		}
		model := strings.TrimSpace(sess.Model)
		if model == "" {
			model = "local-model"
		}
		for idx, m := range sess.Messages {
			if m.Role != "assistant" || m.Tokens <= 0 {
				continue
			}
			key := fmt.Sprintf("chat:%s:%d", sess.ID, idx)
			ts := m.CreatedAt
			if ts <= 0 {
				ts = baseTS
			}
			ev := UsageEvent{
				Key:         key,
				TimestampMS: ts,
				Date:        time.UnixMilli(ts).In(time.Local).Format("2006-01-02"),
				Model:       model,
				ChatID:      sess.ID,
				Tokens:      m.Tokens,
				TokPerSec:   m.TokPerSec,
				ElapsedSec:  m.ElapsedSec,
				GoalCycle:   m.GoalCycle,
				Agent:       sess.Agent,
			}
			prev, ok := usageEvents[key]
			if !ok || prev.Tokens != ev.Tokens || prev.Model != ev.Model {
				usageEvents[key] = ev
				changed = true
			}
		}
	}
	if changed {
		saveUsageEventsLocked()
	}
}

func buildUsageSummary(monthStr string) UsageSummaryResponse {
	now := time.Now().In(time.Local)
	todayStr := now.Format("2006-01-02")
	monthStr = strings.TrimSpace(monthStr)
	if len(monthStr) != 7 || monthStr[4] != '-' {
		monthStr = now.Format("2006-01")
	}
	if _, err := time.Parse("2006-01", monthStr); err != nil {
		monthStr = now.Format("2006-01")
	}

	usageMu.Lock()
	syncUsageFromChatsLocked()
	all := make([]UsageEvent, 0, len(usageEvents))
	for _, ev := range usageEvents {
		all = append(all, ev)
	}
	usageMu.Unlock()

	type modelAgg struct {
		tokens   int
		requests int
		tpsSum   float64
		tpsCount int
		lastDate string
	}
	type dayAgg struct {
		tokens   int
		requests int
		tpsSum   float64
		tpsCount int
		models   map[string]int
	}

	allModels := map[string]*modelAgg{}
	monthModels := map[string]*modelAgg{}
	monthDays := map[string]*dayAgg{}
	allActiveDates := map[string]bool{}

	resp := UsageSummaryResponse{
		Month: monthStr,
		Today: todayStr,
	}

	var totalTPSSum float64
	var totalTPSCount int

	for _, ev := range all {
		if ev.Tokens <= 0 {
			continue
		}
		resp.TotalTokens += ev.Tokens
		resp.TotalRequests++
		if ev.GoalCycle > 0 {
			resp.GoalRuns++
		}
		if ev.TokPerSec > 0 {
			totalTPSSum += ev.TokPerSec
			totalTPSCount++
		}
		if ev.Date != "" {
			allActiveDates[ev.Date] = true
		}
		if ev.Date == todayStr {
			resp.TodayTokens += ev.Tokens
		}

		// All-time model aggregate
		ma := allModels[ev.Model]
		if ma == nil {
			ma = &modelAgg{}
			allModels[ev.Model] = ma
		}
		ma.tokens += ev.Tokens
		ma.requests++
		if ev.TokPerSec > 0 {
			ma.tpsSum += ev.TokPerSec
			ma.tpsCount++
		}
		if ev.Date > ma.lastDate {
			ma.lastDate = ev.Date
		}

		// Selected month aggregate
		if strings.HasPrefix(ev.Date, monthStr+"-") {
			resp.MonthTokens += ev.Tokens
			resp.MonthRequests++

			da := monthDays[ev.Date]
			if da == nil {
				da = &dayAgg{models: map[string]int{}}
				monthDays[ev.Date] = da
			}
			da.tokens += ev.Tokens
			da.requests++
			if ev.TokPerSec > 0 {
				da.tpsSum += ev.TokPerSec
				da.tpsCount++
			}
			da.models[ev.Model] += ev.Tokens

			mm := monthModels[ev.Model]
			if mm == nil {
				mm = &modelAgg{}
				monthModels[ev.Model] = mm
			}
			mm.tokens += ev.Tokens
			mm.requests++
			if ev.TokPerSec > 0 {
				mm.tpsSum += ev.TokPerSec
				mm.tpsCount++
			}
			if ev.Date > mm.lastDate {
				mm.lastDate = ev.Date
			}
		}
	}

	if totalTPSCount > 0 {
		resp.AvgTokPerSec = math.Round((totalTPSSum/float64(totalTPSCount))*10) / 10
	}
	resp.ActiveDays = len(allActiveDates)

	// Calculate active day streak ending today (or yesterday if nothing yet today)
	streak := 0
	cursor := now
	if !allActiveDates[todayStr] {
		cursor = cursor.AddDate(0, 0, -1)
	}
	for {
		ds := cursor.Format("2006-01-02")
		if !allActiveDates[ds] {
			break
		}
		streak++
		cursor = cursor.AddDate(0, 0, -1)
		if streak > 3650 {
			break
		}
	}
	resp.StreakDays = streak

	// Estimated cloud API savings (~$0.008 per 1K blended output/context tokens)
	resp.EstCloudSavingsUSD = math.Round((float64(resp.TotalTokens)/1000.0)*0.008*100) / 100

	// Build sorted daily list for the selected month + peak day
	dailyList := make([]DailyUsageItem, 0, len(monthDays))
	for dt, da := range monthDays {
		avgTPS := 0.0
		if da.tpsCount > 0 {
			avgTPS = math.Round((da.tpsSum/float64(da.tpsCount))*10) / 10
		}
		dailyList = append(dailyList, DailyUsageItem{
			Date:     dt,
			Tokens:   da.tokens,
			Requests: da.requests,
			AvgTPS:   avgTPS,
			Models:   da.models,
		})
		if da.tokens > resp.PeakDay.Tokens {
			resp.PeakDay = UsagePeakDay{Date: dt, Tokens: da.tokens}
		}
	}
	sort.Slice(dailyList, func(i, j int) bool { return dailyList[i].Date < dailyList[j].Date })
	resp.Daily = dailyList

	buildModelList := func(src map[string]*modelAgg, denom int) []ModelUsageItem {
		out := make([]ModelUsageItem, 0, len(src))
		for name, a := range src {
			avgTPS := 0.0
			if a.tpsCount > 0 {
				avgTPS = math.Round((a.tpsSum/float64(a.tpsCount))*10) / 10
			}
			share := 0.0
			if denom > 0 {
				share = math.Round((float64(a.tokens)/float64(denom))*1000) / 10
			}
			out = append(out, ModelUsageItem{
				Model:        name,
				Tokens:       a.tokens,
				Requests:     a.requests,
				AvgTPS:       avgTPS,
				SharePct:     share,
				LastUsedDate: a.lastDate,
			})
		}
		sort.Slice(out, func(i, j int) bool {
			if out[i].Tokens != out[j].Tokens {
				return out[i].Tokens > out[j].Tokens
			}
			return out[i].Model < out[j].Model
		})
		return out
	}

	resp.ByModel = buildModelList(allModels, resp.TotalTokens)
	resp.MonthByModel = buildModelList(monthModels, resp.MonthTokens)

	// Smart actionable usage recommendations
	recs := []UsageRecommendation{}
	if resp.TotalTokens == 0 {
		recs = append(recs, UsageRecommendation{
			Kind:   "tip",
			Title:  "Mulai Catat Aktivitas Token Harian",
			Detail: "Setiap respons dari AI Chat Studio dan /goal Quest akan otomatis dicatat ke kalender harian dan breakdown per-model di halaman ini.",
		})
	} else {
		if len(resp.ByModel) > 0 {
			top := resp.ByModel[0]
			if top.SharePct >= 75 && len(resp.ByModel) == 1 {
				recs = append(recs, UsageRecommendation{
					Kind:   "focus",
					Title:  fmt.Sprintf("Routing Model Pintar (%s = %.0f%%)", top.Model, top.SharePct),
					Detail: "Anda memakai 1 model untuk semua tugas. Pertimbangkan model ringan (7B-8B) untuk Q&A cepat/brainstorm dan simpan model besar khusus /goal coding agar VRAM lebih hemat & respons lebih cepat.",
				})
			} else {
				recs = append(recs, UsageRecommendation{
					Kind:   "focus",
					Title:  fmt.Sprintf("Model Utama: %s (%.1f%% • %.1f tok/s)", top.Model, top.SharePct, top.AvgTPS),
					Detail: fmt.Sprintf("Total %d request dengan %d token. Pantau kecepatan tok/s di tabel model untuk mendeteksi spill-over VRAM ke RAM.", top.Requests, top.Tokens),
				})
			}
		}
		if resp.EstCloudSavingsUSD >= 0.01 {
			recs = append(recs, UsageRecommendation{
				Kind:   "cost",
				Title:  fmt.Sprintf("Estimasi Hemat Biaya Cloud: ~$%.2f", resp.EstCloudSavingsUSD),
				Detail: fmt.Sprintf("Menjalankan %d token secara lokal di GPU RTX 3060 menjaga privasi codebase sekaligus menghemat biaya API cloud berbayar.", resp.TotalTokens),
			})
		}
		if resp.GoalRuns > 0 {
			recs = append(recs, UsageRecommendation{
				Kind:   "perf",
				Title:  fmt.Sprintf("Budget Konteks /goal (%d cycle tercatat)", resp.GoalRuns),
				Detail: "Gunakan mode /goal dengan deskripsi target terukur dan flush VRAM melalui menu Memory & Flush setelah sesi panjang agar KV-Cache tetap segar.",
			})
		}
		if resp.AvgTokPerSec > 0 && resp.AvgTokPerSec < 22 {
			recs = append(recs, UsageRecommendation{
				Kind:   "tip",
				Title:  fmt.Sprintf("Throughput Rata-Rata %.1f tok/s", resp.AvgTokPerSec),
				Detail: "Kecepatan di bawah 22 tok/s biasanya menandakan sebagian layer jatuh ke RAM. Coba flush ComfyUI/VRAM di menu Memory & Flush atau turunkan context size.",
			})
		} else if resp.PeakDay.Tokens > 0 {
			recs = append(recs, UsageRecommendation{
				Kind:   "tip",
				Title:  fmt.Sprintf("Hari Tersibuk Bulan Ini: %s (%d tok)", resp.PeakDay.Date, resp.PeakDay.Tokens),
				Detail: "Klik tanggal mana pun pada kalender di atas untuk melihat rincian token per-model dan jumlah request pada hari tersebut.",
			})
		}
	}
	resp.Recommendations = recs
	return resp
}

func HandleUsageAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	month := r.URL.Query().Get("month")
	summary := buildUsageSummary(month)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(summary)
}

// --- Memory Inspector & Multi-Tier Flush System ---

type GoRuntimeMemInfo struct {
	AllocMB     float64 `json:"alloc_mb"`
	HeapInuseMB float64 `json:"heap_inuse_mb"`
	SysMB       float64 `json:"sys_mb"`
	NumGC       uint32  `json:"num_gc"`
	Goroutines  int     `json:"goroutines"`
}

type ChatStoreMemInfo struct {
	SessionCount int     `json:"session_count"`
	DiskMB       float64 `json:"disk_mb"`
	ActiveRuns   int     `json:"active_runs"`
}

type MemoryFlushLogEntry struct {
	Timestamp    string  `json:"timestamp"`
	Target       string  `json:"target"`
	OK           bool    `json:"ok"`
	FreedVRAMMiB float64 `json:"freed_vram_mib"`
	FreedRAMMiB  float64 `json:"freed_ram_mib"`
	FreedHeapMB  float64 `json:"freed_heap_mb"`
	Message      string  `json:"message"`
}

type MemoryStatusResponse struct {
	RAM          telemetry.RAMMetrics        `json:"ram"`
	GPU          telemetry.GPUMetrics        `json:"gpu"`
	OllamaModels []telemetry.OllamaModelInfo `json:"ollama_models"`
	Comfy        telemetry.ComfyMetrics      `json:"comfy"`
	Runtime      GoRuntimeMemInfo            `json:"runtime"`
	ChatStore    ChatStoreMemInfo            `json:"chat_store"`
	TopProcs     []telemetry.ProcessMemInfo  `json:"top_procs"`
	RecentFlush  []MemoryFlushLogEntry       `json:"recent_flush"`
}

var (
	flushLogMu    sync.Mutex
	recentFlushes []MemoryFlushLogEntry
)

func recordFlushLog(entry MemoryFlushLogEntry) {
	flushLogMu.Lock()
	defer flushLogMu.Unlock()
	recentFlushes = append([]MemoryFlushLogEntry{entry}, recentFlushes...)
	if len(recentFlushes) > 10 {
		recentFlushes = recentFlushes[:10]
	}
}

func readGoRuntimeMem() GoRuntimeMemInfo {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	toMB := func(b uint64) float64 {
		return math.Round((float64(b)/(1024*1024))*10) / 10
	}
	return GoRuntimeMemInfo{
		AllocMB:     toMB(ms.Alloc),
		HeapInuseMB: toMB(ms.HeapInuse),
		SysMB:       toMB(ms.Sys),
		NumGC:       ms.NumGC,
		Goroutines:  runtime.NumGoroutine(),
	}
}

func readChatStoreMem() ChatStoreMemInfo {
	dir := getChatsDir()
	entries, err := os.ReadDir(dir)
	count := 0
	var totalBytes int64
	if err == nil {
		for _, ent := range entries {
			if !ent.IsDir() && strings.HasSuffix(ent.Name(), ".json") && ent.Name() != "usage_log.json" {
				count++
				if info, err := ent.Info(); err == nil {
					totalBytes += info.Size()
				}
			}
		}
	}
	activeRunsMu.RLock()
	activeCount := 0
	for _, run := range activeRuns {
		run.mu.RLock()
		if run.Running {
			activeCount++
		}
		run.mu.RUnlock()
	}
	activeRunsMu.RUnlock()
	return ChatStoreMemInfo{
		SessionCount: count,
		DiskMB:       math.Round((float64(totalBytes)/(1024*1024))*100) / 100,
		ActiveRuns:   activeCount,
	}
}

func collectMemoryStatus() MemoryStatusResponse {
	snap := telemetry.GetLatestSnapshot()
	ram := telemetry.ReadRAMMetrics()
	gpu := telemetry.ReadGPUMetrics()
	if !gpu.Online && snap.GPU.Online {
		gpu = snap.GPU
	}

	flushLogMu.Lock()
	flushes := make([]MemoryFlushLogEntry, len(recentFlushes))
	copy(flushes, recentFlushes)
	flushLogMu.Unlock()

	models := snap.Ollama.Models
	if models == nil {
		models = []telemetry.OllamaModelInfo{}
	}
	procs := snap.TopProcs
	if procs == nil {
		procs = []telemetry.ProcessMemInfo{}
	}

	return MemoryStatusResponse{
		RAM:          ram,
		GPU:          gpu,
		OllamaModels: models,
		Comfy:        snap.Comfy,
		Runtime:      readGoRuntimeMem(),
		ChatStore:    readChatStoreMem(),
		TopProcs:     procs,
		RecentFlush:  flushes,
	}
}

func HandleMemoryStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	st := collectMemoryStatus()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(st)
}

func unloadOllamaModels(targetModel string) (int, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	if strings.TrimSpace(targetModel) != "" {
		payload, _ := json.Marshal(map[string]any{"model": targetModel, "keep_alive": 0})
		resp, err := client.Post(config.OllamaURL()+"/api/generate", "application/json", bytes.NewReader(payload))
		if err != nil {
			return 0, err
		}
		resp.Body.Close()
		return 1, nil
	}
	resp, err := client.Get(config.OllamaURL() + "/api/ps")
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	var psData struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &psData); err != nil {
		return 0, err
	}
	unloaded := 0
	for _, m := range psData.Models {
		payload, _ := json.Marshal(map[string]any{"model": m.Name, "keep_alive": 0})
		if r2, err := client.Post(config.OllamaURL()+"/api/generate", "application/json", bytes.NewReader(payload)); err == nil {
			r2.Body.Close()
			unloaded++
		}
	}
	return unloaded, nil
}

func flushComfyMemory() bool {
	client := &http.Client{Timeout: 4 * time.Second}
	payload := []byte(`{"unload_models":true,"free_memory":true}`)
	resp, err := client.Post(config.ComfyUIURL()+"/free", "application/json", bytes.NewReader(payload))
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func HandleMemoryFlush(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Target string `json:"target"`
		Model  string `json:"model"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&req)
	target := strings.TrimSpace(strings.ToLower(req.Target))
	if target == "" {
		target = "all_safe"
	}

	beforeRAM := telemetry.ReadRAMMetrics()
	beforeGPU := telemetry.ReadGPUMetrics()
	beforeHeap := readGoRuntimeMem()

	ok := true
	var notes []string

	switch target {
	case "vram":
		n, err := unloadOllamaModels(req.Model)
		if err != nil {
			ok = false
			notes = append(notes, "Gagal menghubungi Ollama: "+err.Error())
		} else if req.Model != "" {
			notes = append(notes, fmt.Sprintf("Model %s dilepas dari GPU VRAM.", req.Model))
		} else if n > 0 {
			notes = append(notes, fmt.Sprintf("%d model Ollama dilepas dari GPU VRAM.", n))
		} else {
			notes = append(notes, "Tidak ada model Ollama yang sedang aktif di VRAM.")
		}

	case "comfy":
		if flushComfyMemory() {
			notes = append(notes, "Cache & model ComfyUI berhasil di-flush dari VRAM/RAM.")
		} else {
			notes = append(notes, "ComfyUI sedang tidak aktif (standby) — tidak ada VRAM ComfyUI untuk dilepas.")
		}

	case "go_gc":
		runtime.GC()
		debug.FreeOSMemory()
		notes = append(notes, "Go Runtime Garbage Collection & FreeOSMemory dijalankan.")

	case "page_cache":
		_ = exec.Command("sync").Run()
		runtime.GC()
		debug.FreeOSMemory()
		out, err := exec.Command("sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches").CombinedOutput()
		if err == nil {
			notes = append(notes, "Linux Page Cache, Dentries & Inodes berhasil dibersihkan (drop_caches=3).")
		} else {
			ok = false
			msg := strings.TrimSpace(string(out))
			if msg == "" {
				msg = err.Error()
			}
			notes = append(notes, "sync + Go GC dijalankan, namun drop_caches butuh izin sudoers NOPASSWD (/proc/sys/vm/drop_caches): "+msg)
		}

	case "all_safe":
		if n, err := unloadOllamaModels(""); err == nil {
			if n > 0 {
				notes = append(notes, fmt.Sprintf("Ollama: %d model dilepas", n))
			} else {
				notes = append(notes, "Ollama: VRAM sudah bersih")
			}
		} else {
			notes = append(notes, "Ollama tidak terjangkau")
		}
		if flushComfyMemory() {
			notes = append(notes, "ComfyUI: memory flushed")
		}
		runtime.GC()
		debug.FreeOSMemory()
		notes = append(notes, "Go Heap: GC + FreeOSMemory selesai")

	default:
		http.Error(w, "Unsupported flush target (use: vram, comfy, go_gc, page_cache, all_safe)", http.StatusBadRequest)
		return
	}

	time.Sleep(320 * time.Millisecond)

	afterRAM := telemetry.ReadRAMMetrics()
	afterGPU := telemetry.ReadGPUMetrics()
	afterHeap := readGoRuntimeMem()

	freedVRAM := math.Round(math.Max(0, beforeGPU.VRAMUsedMiB-afterGPU.VRAMUsedMiB)*10) / 10
	freedRAM := math.Round(math.Max(0, beforeRAM.UsedMiB-afterRAM.UsedMiB)*10) / 10
	freedHeap := math.Round(math.Max(0, beforeHeap.AllocMB-afterHeap.AllocMB)*10) / 10

	entry := MemoryFlushLogEntry{
		Timestamp:    time.Now().In(time.Local).Format("15:04:05"),
		Target:       target,
		OK:           ok,
		FreedVRAMMiB: freedVRAM,
		FreedRAMMiB:  freedRAM,
		FreedHeapMB:  freedHeap,
		Message:      strings.Join(notes, " • "),
	}
	recordFlushLog(entry)

	status := collectMemoryStatus()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok":             ok,
		"target":         target,
		"message":        entry.Message,
		"freed_vram_mib": freedVRAM,
		"freed_ram_mib":  freedRAM,
		"freed_heap_mb":  freedHeap,
		"status":         status,
	})
}

// --- Live Ollama, ComfyUI & Per-Model Logs + Automated Error Mitigation ---

type RuntimeLogLine struct {
	Source    string `json:"source"` // ollama, comfyui, model, system
	Model     string `json:"model,omitempty"`
	Level     string `json:"level"` // error, warn, info
	Timestamp string `json:"timestamp,omitempty"`
	Message   string `json:"message"`
}

type MitigationAlert struct {
	Severity    string `json:"severity"` // error, warn, ok
	Source      string `json:"source"`
	Title       string `json:"title"`
	Evidence    string `json:"evidence,omitempty"`
	Mitigation  string `json:"mitigation"`
	ActionType  string `json:"action_type,omitempty"` // flush_vram, flush_comfy, flush_all, restart_ollama, restart_mcp
	ActionLabel string `json:"action_label,omitempty"`
}

type ModelHealthRow struct {
	Name         string  `json:"name"`
	Engine       string  `json:"engine"`
	SizeGB       float64 `json:"size_gb"`
	LoadedVRAMGB float64 `json:"loaded_vram_gb"`
	GPUPct       int     `json:"gpu_pct"`
	Status       string  `json:"status"`
	ErrorCount   int     `json:"error_count"`
	LastEvent    string  `json:"last_event,omitempty"`
}

type MemoryLogsResponse struct {
	GeneratedAt string            `json:"generated_at"`
	ErrorCount  int               `json:"error_count"`
	WarnCount   int               `json:"warn_count"`
	Lines       []RuntimeLogLine  `json:"lines"`
	Mitigations []MitigationAlert `json:"mitigations"`
	ModelHealth []ModelHealthRow  `json:"model_health"`
}

var (
	reLogTimeISO     = regexp.MustCompile(`(\d{4}[/-]\d{2}[/-]\d{2}[T ]\d{2}:\d{2}:\d{2})`)
	reLogTimeShort   = regexp.MustCompile(`\b(\d{2}:\d{2}:\d{2})\b`)
	rePartialOffload = regexp.MustCompile(`(?i)offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU`)
	reANSIStrip      = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
)

func classifyLogLevel(line string) string {
	low := strings.ToLower(line)
	if strings.Contains(low, "level=error") ||
		strings.Contains(low, "traceback") ||
		strings.Contains(low, "out of memory") ||
		strings.Contains(low, "cuda error") ||
		strings.Contains(low, "fatal") ||
		strings.Contains(low, "panic:") ||
		strings.Contains(low, "connection refused") ||
		strings.Contains(low, " 500 ") ||
		strings.Contains(low, "error:") ||
		strings.Contains(low, "failed to ") {
		return "error"
	}
	if strings.Contains(low, "level=warn") ||
		strings.Contains(low, "warning") ||
		strings.Contains(low, "timeout") ||
		strings.Contains(low, "context canceled") ||
		strings.Contains(low, "truncating") ||
		strings.Contains(low, "loop guard") ||
		strings.Contains(low, "retry") ||
		strings.Contains(low, " 499 ") ||
		strings.Contains(low, " 400 ") {
		return "warn"
	}
	return "info"
}

func extractLogTimestamp(line string) string {
	if m := reLogTimeISO.FindStringSubmatch(line); len(m) == 2 {
		t := m[1]
		if len(t) >= 19 {
			return t[11:19]
		}
		return t
	}
	if m := reLogTimeShort.FindStringSubmatch(line); len(m) == 2 {
		return m[1]
	}
	return ""
}

func detectModelInLogLine(line string, knownModels []string) string {
	low := strings.ToLower(line)
	for _, m := range knownModels {
		if m != "" && strings.Contains(low, strings.ToLower(m)) {
			return m
		}
	}
	if strings.Contains(low, "safetensors") || strings.Contains(low, "comfyui") || strings.Contains(low, "ksampler") {
		return "ComfyUI"
	}
	return ""
}

func readTailLines(path string, maxBytes int64, maxLines int) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil
	}
	size := info.Size()
	start := int64(0)
	if size > maxBytes {
		start = size - maxBytes
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil
	}
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil
	}
	parts := strings.Split(string(raw), "\n")
	if start > 0 && len(parts) > 1 {
		parts = parts[1:] // drop partial first line
	}
	out := make([]string, 0, maxLines)
	for _, p := range parts {
		clean := strings.TrimSpace(reANSIStrip.ReplaceAllString(p, ""))
		if clean == "" {
			continue
		}
		out = append(out, clean)
	}
	if len(out) > maxLines {
		out = out[len(out)-maxLines:]
	}
	return out
}

func analyzeLogLinesForMitigations(lines []RuntimeLogLine, ram telemetry.RAMMetrics, gpu telemetry.GPUMetrics) []MitigationAlert {
	alerts := []MitigationAlert{}
	seenKey := map[string]bool{}
	addOnce := func(key string, a MitigationAlert) {
		if seenKey[key] {
			return
		}
		seenKey[key] = true
		alerts = append(alerts, a)
	}

	for i := len(lines) - 1; i >= 0; i-- {
		ln := lines[i]
		low := strings.ToLower(ln.Message)
		snippet := ln.Message
		if len(snippet) > 150 {
			snippet = snippet[:150] + "..."
		}

		if strings.Contains(low, "out of memory") || strings.Contains(low, "unable to allocate cuda") || strings.Contains(low, "cuda error") || strings.Contains(low, "oom") {
			actType := "flush_vram"
			actLbl := "Flush Ollama VRAM"
			if ln.Source == "comfyui" {
				actType = "flush_comfy"
				actLbl = "Free ComfyUI Cache"
			}
			addOnce("oom_"+ln.Source, MitigationAlert{
				Severity:    "error",
				Source:      ln.Source,
				Title:       "GPU Out of Memory (VRAM Penuh)",
				Evidence:    snippet,
				Mitigation:  "VRAM RTX 3060 (12 GB) kehabisan ruang saat memuat model atau render. Lepas model yang tidak dipakai atau bersihkan cache ComfyUI sebelum menjalankan ulang.",
				ActionType:  actType,
				ActionLabel: actLbl,
			})
		}

		if m := rePartialOffload.FindStringSubmatch(ln.Message); len(m) == 3 {
			var onGPU, total int
			_, _ = fmt.Sscanf(m[1], "%d", &onGPU)
			_, _ = fmt.Sscanf(m[2], "%d", &total)
			if total > 0 && onGPU < total {
				addOnce("partial_offload", MitigationAlert{
					Severity:    "warn",
					Source:      "ollama",
					Title:       fmt.Sprintf("Partial GPU Offload (%d/%d Layers di GPU)", onGPU, total),
					Evidence:    snippet,
					Mitigation:  "Sebagian layer model jatuh ke System RAM sehingga kecepatan tok/s turun. Flush ComfyUI/VRAM lain atau gunakan model quantisasi lebih kecil agar 100% muat di RTX 3060.",
					ActionType:  "flush_all",
					ActionLabel: "Flush All Safe",
				})
			}
		}

		if strings.Contains(low, "connection refused") || strings.Contains(low, "dial tcp 127.0.0.1:11434") || strings.Contains(low, "eof") && ln.Source == "model" {
			addOnce("conn_refused", MitigationAlert{
				Severity:    "error",
				Source:      ln.Source,
				Title:       "Koneksi ke Daemon Ollama / MCP Terputus",
				Evidence:    snippet,
				Mitigation:  "Container Ollama atau layanan MCP tidak merespons. Lakukan restart daemon untuk memulihkan endpoint lokal.",
				ActionType:  "restart_ollama",
				ActionLabel: "Restart Ollama",
			})
		}

		if strings.Contains(low, "model requires more system memory") || strings.Contains(low, "not enough memory") {
			addOnce("sys_ram_low", MitigationAlert{
				Severity:    "error",
				Source:      ln.Source,
				Title:       "System RAM Tidak Cukup untuk Memuat Model",
				Evidence:    snippet,
				Mitigation:  "Total RAM bebas lebih kecil dari kebutuhan model. Jalankan Flush All Safe untuk mengosongkan VRAM/RAM & Go Heap.",
				ActionType:  "flush_all",
				ActionLabel: "Flush All Safe",
			})
		}

		if ln.Source == "model" && (strings.Contains(low, "loop") || strings.Contains(low, "macet") || strings.Contains(low, "terjebak") || strings.Contains(low, "timeout")) {
			addOnce("model_stall", MitigationAlert{
				Severity:    "warn",
				Source:      "model",
				Title:       "Deteksi Stall / Tool Loop pada Model Chat",
				Evidence:    snippet,
				Mitigation:  "Model sempat mengalami stall/loop pada sesi sebelumnya. Flush VRAM untuk menyegarkan KV-Cache, atau gunakan model dengan akurasi tool-call lebih stabil.",
				ActionType:  "flush_vram",
				ActionLabel: "Flush Ollama VRAM",
			})
		}
	}

	// Proactive memory threshold checks
	if gpu.Online && gpu.VRAMPct >= 92 {
		addOnce("vram_high", MitigationAlert{
			Severity:    "warn",
			Source:      "memory",
			Title:       fmt.Sprintf("Tekanan VRAM Tinggi (%.1f%%)", gpu.VRAMPct),
			Mitigation:  "VRAM hampir penuh; jalankan Flush Ollama VRAM atau Free ComfyUI Cache sebelum berpindah workload LLM ↔ Image/Video.",
			ActionType:  "flush_all",
			ActionLabel: "Flush All Safe",
		})
	}
	if ram.SwapTotMiB > 0 && ram.SwapPct >= 50 {
		addOnce("swap_high", MitigationAlert{
			Severity:    "warn",
			Source:      "memory",
			Title:       fmt.Sprintf("Penggunaan Swap Tinggi (%.0f MiB / %.1f%%)", ram.SwapUsedMiB, ram.SwapPct),
			Mitigation:  "Swap tinggi dapat memperlambat loading model. Jalankan Flush All Safe atau Drop Linux Page Cache untuk melegakan memori.",
			ActionType:  "flush_all",
			ActionLabel: "Flush All Safe",
		})
	}

	if len(alerts) == 0 {
		alerts = append(alerts, MitigationAlert{
			Severity:   "ok",
			Source:     "system",
			Title:      "Semua Engine & Model Sehat • 0 Error Kritis Terdeteksi",
			Mitigation: "Log Ollama Docker, ComfyUI, dan riwayat eksekusi seluruh model dipantau aktif. Jika muncul OOM, partial offload, atau koneksi terputus, saran mitigasi 1-klik akan muncul otomatis di sini.",
		})
	}
	return alerts
}

func collectRuntimeLogsAndMitigations() MemoryLogsResponse {
	snap := telemetry.GetLatestSnapshot()
	ram := telemetry.ReadRAMMetrics()
	gpu := telemetry.ReadGPUMetrics()
	if !gpu.Online && snap.GPU.Online {
		gpu = snap.GPU
	}

	knownModels := make([]string, 0, len(snap.OllamaModels)+len(snap.Ollama.Models))
	seenKnown := map[string]bool{}
	for _, m := range snap.OllamaModels {
		if m.Name != "" && !seenKnown[m.Name] {
			seenKnown[m.Name] = true
			knownModels = append(knownModels, m.Name)
		}
	}
	for _, m := range snap.Ollama.Models {
		if m.Name != "" && !seenKnown[m.Name] {
			seenKnown[m.Name] = true
			knownModels = append(knownModels, m.Name)
		}
	}

	allLines := []RuntimeLogLine{}
	modelErrCounts := map[string]int{}
	modelLastEvent := map[string]string{}

	// 1. Ollama Docker logs (tail 100)
	if out, err := exec.Command("docker", "logs", "ollama", "--tail", "100").CombinedOutput(); err == nil {
		rawLines := strings.Split(string(out), "\n")
		for _, raw := range rawLines {
			clean := strings.TrimSpace(reANSIStrip.ReplaceAllString(raw, ""))
			if clean == "" {
				continue
			}
			lvl := classifyLogLevel(clean)
			mod := detectModelInLogLine(clean, knownModels)
			ts := extractLogTimestamp(clean)
			allLines = append(allLines, RuntimeLogLine{
				Source:    "ollama",
				Model:     mod,
				Level:     lvl,
				Timestamp: ts,
				Message:   clean,
			})
			if mod != "" {
				if lvl == "error" {
					modelErrCounts[mod]++
				}
				modelLastEvent[mod] = ts
			}
		}
	} else {
		allLines = append(allLines, RuntimeLogLine{
			Source:    "ollama",
			Level:     "warn",
			Timestamp: time.Now().In(time.Local).Format("15:04:05"),
			Message:   "Tidak dapat membaca docker logs ollama: " + strings.TrimSpace(err.Error()),
		})
	}

	// 2. ComfyUI logs (tail 60 lines)
	comfyLines := readTailLines("/home/rizal/Documents/ollama/comfyui/comfyui.log", 24*1024, 60)
	for _, clean := range comfyLines {
		lvl := classifyLogLevel(clean)
		ts := extractLogTimestamp(clean)
		allLines = append(allLines, RuntimeLogLine{
			Source:    "comfyui",
			Model:     "ComfyUI",
			Level:     lvl,
			Timestamp: ts,
			Message:   clean,
		})
		if lvl == "error" {
			modelErrCounts["ComfyUI"]++
		}
	}

	// 3. Per-Model Chat / Goal / Tool Execution Logs & Errors from chats/*.json
	dir := usageBaseDir()
	if entries, err := os.ReadDir(dir); err == nil {
		type chatLogItem struct {
			ts   int64
			line RuntimeLogLine
		}
		var chatEvents []chatLogItem
		for _, ent := range entries {
			if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".json") || ent.Name() == "usage_log.json" {
				continue
			}
			b, err := os.ReadFile(filepath.Join(dir, ent.Name()))
			if err != nil {
				continue
			}
			var sess ChatSession
			if json.Unmarshal(b, &sess) != nil {
				continue
			}
			mod := strings.TrimSpace(sess.Model)
			if mod == "" {
				mod = "local-model"
			}
			baseTS := sess.UpdatedAt
			if baseTS <= 0 {
				baseTS = sess.CreatedAt
			}
			for _, m := range sess.Messages {
				if m.Role != "assistant" {
					continue
				}
				ts := m.CreatedAt
				if ts <= 0 {
					ts = baseTS
				}
				tsStr := time.UnixMilli(ts).In(time.Local).Format("01-02 15:04")
				if strings.TrimSpace(m.Error) != "" {
					modelErrCounts[mod]++
					modelLastEvent[mod] = "Error: " + m.Error
					chatEvents = append(chatEvents, chatLogItem{
						ts: ts,
						line: RuntimeLogLine{
							Source:    "model",
							Model:     mod,
							Level:     "error",
							Timestamp: tsStr,
							Message:   fmt.Sprintf("[%s] Chat %q Error: %s", mod, sess.Title, m.Error),
						},
					})
				} else if m.Tokens > 0 {
					if modelLastEvent[mod] == "" {
						modelLastEvent[mod] = fmt.Sprintf("%d tok • %.1f tok/s", m.Tokens, m.TokPerSec)
					}
					chatEvents = append(chatEvents, chatLogItem{
						ts: ts,
						line: RuntimeLogLine{
							Source:    "model",
							Model:     mod,
							Level:     "info",
							Timestamp: tsStr,
							Message:   fmt.Sprintf("[%s] Completed (%d tok • %.1f tok/s • %.1fs) — %s", mod, m.Tokens, m.TokPerSec, m.ElapsedSec, sess.Title),
						},
					})
				}
				for _, tl := range m.Tools {
					lowOut := strings.ToLower(tl.Output)
					if strings.Contains(lowOut, "error generating") || strings.Contains(lowOut, "loop guard") || strings.Contains(lowOut, "exit status") {
						lvl := "warn"
						if strings.Contains(lowOut, "error generating") {
							lvl = "error"
							modelErrCounts[mod]++
						}
						shortOut := strings.ReplaceAll(strings.TrimSpace(tl.Output), "\n", " ")
						if len(shortOut) > 140 {
							shortOut = shortOut[:140] + "..."
						}
						chatEvents = append(chatEvents, chatLogItem{
							ts: ts,
							line: RuntimeLogLine{
								Source:    "model",
								Model:     mod,
								Level:     lvl,
								Timestamp: tsStr,
								Message:   fmt.Sprintf("[%s] Tool %s: %s", mod, tl.Name, shortOut),
							},
						})
					}
				}
			}
		}
		sort.Slice(chatEvents, func(i, j int) bool { return chatEvents[i].ts < chatEvents[j].ts })
		if len(chatEvents) > 50 {
			chatEvents = chatEvents[len(chatEvents)-50:]
		}
		for _, ce := range chatEvents {
			allLines = append(allLines, ce.line)
		}
	}

	// Build Per-Model Health Matrix (all installed Ollama models + ComfyUI)
	loadedMap := map[string]telemetry.OllamaModelInfo{}
	for _, lm := range snap.Ollama.Models {
		loadedMap[lm.Name] = lm
	}
	healthRows := []ModelHealthRow{}
	addedModel := map[string]bool{}
	for _, im := range snap.OllamaModels {
		addedModel[im.Name] = true
		lm, isLoaded := loadedMap[im.Name]
		status := "Standby"
		if isLoaded {
			status = "Active VRAM"
		}
		if modelErrCounts[im.Name] > 0 {
			status = fmt.Sprintf("%d Error", modelErrCounts[im.Name])
		}
		healthRows = append(healthRows, ModelHealthRow{
			Name:         im.Name,
			Engine:       "Ollama",
			SizeGB:       im.SizeGB,
			LoadedVRAMGB: lm.VRAMGB,
			GPUPct:       lm.GPUPct,
			Status:       status,
			ErrorCount:   modelErrCounts[im.Name],
			LastEvent:    modelLastEvent[im.Name],
		})
	}
	for name, lm := range loadedMap {
		if addedModel[name] {
			continue
		}
		healthRows = append(healthRows, ModelHealthRow{
			Name:         name,
			Engine:       "Ollama",
			SizeGB:       lm.SizeGB,
			LoadedVRAMGB: lm.VRAMGB,
			GPUPct:       lm.GPUPct,
			Status:       "Active VRAM",
			ErrorCount:   modelErrCounts[name],
			LastEvent:    modelLastEvent[name],
		})
	}
	comfyStatus := "Standby"
	if snap.Comfy.Online {
		comfyStatus = "Online"
		if snap.Comfy.QueueRunning > 0 {
			comfyStatus = "Rendering"
		}
	}
	if modelErrCounts["ComfyUI"] > 0 {
		comfyStatus = fmt.Sprintf("%d Error", modelErrCounts["ComfyUI"])
	}
	healthRows = append(healthRows, ModelHealthRow{
		Name:       "ComfyUI Studio (Image/Video)",
		Engine:     "ComfyUI",
		Status:     comfyStatus,
		ErrorCount: modelErrCounts["ComfyUI"],
		LastEvent:  snap.Comfy.StatusText,
	})

	errCount := 0
	warnCount := 0
	for _, ln := range allLines {
		if ln.Level == "error" {
			errCount++
		} else if ln.Level == "warn" {
			warnCount++
		}
	}

	mitigations := analyzeLogLinesForMitigations(allLines, ram, gpu)
	return MemoryLogsResponse{
		GeneratedAt: time.Now().In(time.Local).Format("15:04:05"),
		ErrorCount:  errCount,
		WarnCount:   warnCount,
		Lines:       allLines,
		Mitigations: mitigations,
		ModelHealth: healthRows,
	}
}

func HandleMemoryLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	resp := collectRuntimeLogsAndMitigations()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(resp)
}

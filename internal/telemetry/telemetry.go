package telemetry

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type GPUMetrics struct {
	Name        string  `json:"name"`
	TempC       float64 `json:"temp_c"`
	PowerDrawW  float64 `json:"power_draw_w"`
	PowerLimitW float64 `json:"power_limit_w"`
	UtilPct     float64 `json:"util_pct"`
	VRAMUsedMiB float64 `json:"vram_used_mib"`
	VRAMTotMiB  float64 `json:"vram_total_mib"`
	VRAMPct     float64 `json:"vram_pct"`
	FanPct      float64 `json:"fan_pct"`
	ClockMHz    int     `json:"clock_mhz"`
	PState      string  `json:"pstate"`
	Online      bool    `json:"online"`
}

type RAMMetrics struct {
	TotalMiB     float64 `json:"total_mib"`
	UsedMiB      float64 `json:"used_mib"`
	AvailMiB     float64 `json:"avail_mib"`
	BuffCacheMiB float64 `json:"buff_cache_mib"`
	UsedPct      float64 `json:"used_pct"`
	SwapTotMiB   float64 `json:"swap_total_mib"`
	SwapUsedMiB  float64 `json:"swap_used_mib"`
	SwapPct      float64 `json:"swap_pct"`
}

type CPUMetrics struct {
	UsagePct float64 `json:"usage_pct"`
	Load1    string  `json:"load_1"`
	Load5    string  `json:"load_5"`
	Load15   string  `json:"load_15"`
	Cores    int     `json:"cores"`
}

type OllamaModelInfo struct {
	Name      string  `json:"name"`
	SizeGB    float64 `json:"size_gb"`
	VRAMGB    float64 `json:"vram_gb"`
	GPUPct    int     `json:"gpu_pct"`
	Context   int     `json:"context"`
	ExpiresAt string  `json:"expires_at"`
}

type OllamaMetrics struct {
	Online        bool              `json:"online"`
	IsGenerating  bool              `json:"is_generating"`
	Models        []OllamaModelInfo `json:"models"`
	LiveTokPerSec float64           `json:"live_tok_per_sec"`
	AvgTokPerSec  float64           `json:"avg_tok_per_sec"`
	GenTokPerSec  float64           `json:"gen_tok_per_sec"`
	EvalTokPerSec float64           `json:"eval_tok_per_sec"`
	LastTokens    int               `json:"last_tokens"`
	ActiveState   string            `json:"active_state"`
	RecentLogs    []string          `json:"recent_logs"`
}

type ComfyMetrics struct {
	Online       bool   `json:"online"`
	QueueRunning int    `json:"queue_running"`
	QueuePending int    `json:"queue_pending"`
	StepProgress int    `json:"step_progress"`
	StepCurrent  int    `json:"step_current"`
	StepTotal    int    `json:"step_total"`
	ItPerSec     string `json:"it_per_sec"`
	ETA          string `json:"eta"`
	StatusText   string `json:"status_text"`
}

type DiskMetrics struct {
	TotalGB float64 `json:"total_gb"`
	FreeGB  float64 `json:"free_gb"`
	UsedPct float64 `json:"used_pct"`
}

type ProcessMemInfo struct {
	Name  string  `json:"name"`
	MemMB float64 `json:"mem_mb"`
}

type InstalledModel struct {
	Name     string  `json:"name"`
	Category string  `json:"category"`
	SizeGB   float64 `json:"size_gb"`
	Active   bool    `json:"active"`
}

type HealthAlert struct {
	Level   string `json:"level"`
	Message string `json:"message"`
}

type OpenCodeMetrics struct {
	Running bool   `json:"running"`
	PIDs    string `json:"pids"`
}

type DashboardSnapshot struct {
	Timestamp     string           `json:"timestamp"`
	UnixMS        int64            `json:"unix_ms"`
	CPU           CPUMetrics       `json:"cpu"`
	RAM           RAMMetrics       `json:"ram"`
	Disk          DiskMetrics      `json:"disk"`
	GPU           GPUMetrics       `json:"gpu"`
	Ollama        OllamaMetrics    `json:"ollama"`
	Comfy         ComfyMetrics     `json:"comfy"`
	OpenCode      OpenCodeMetrics  `json:"opencode"`
	TopProcs      []ProcessMemInfo `json:"top_procs"`
	OllamaStatus  string           `json:"ollama_container_status"`
	OllamaModels  []InstalledModel `json:"ollama_installed"`
	ComfyModels   []InstalledModel `json:"comfy_installed"`
	TotalModelsGB float64          `json:"total_models_gb"`
	Health        HealthAlert      `json:"health"`
}

var (
	cpuMu        sync.Mutex
	prevCPUTotal uint64
	prevCPUIdle  uint64

	ollamaCacheMu     sync.Mutex
	lastGenTokPerSec  float64
	lastEvalTokPerSec float64
	lastPromptTokens  int

	histMu      sync.RWMutex
	histRing    []DashboardSnapshot
	latestSnap  DashboardSnapshot
	hasSnapshot bool

	reLiveTG         = regexp.MustCompile(`n_gen\s*=\s*(\d+),\s*tg\s*=\s*([0-9.]+)\s*t/s,\s*tg_3s\s*=\s*([0-9.]+)\s*t/s`)
	reEvalTime       = regexp.MustCompile(`(?m)^\s*slot print_timing:.*?\beval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([0-9.]+)\s*tokens per second\)`)
	rePromptTime     = regexp.MustCompile(`prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([0-9.]+)\s*tokens per second\)`)
	reLivePromptProc = regexp.MustCompile(`prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([0-9.]+),\s*t\s*=\s*[0-9.]+\s*s\s*/\s*([0-9.]+)\s*tokens per second`)
	rePromptTask     = regexp.MustCompile(`new prompt,\s*n_ctx_slot\s*=\s*\d+,\s*n_keep\s*=\s*\d+,\s*task\.n_tokens\s*=\s*(\d+)`)
	reComfyStep      = regexp.MustCompile(`(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*,\s*([0-9.]+(?:it/s|s/it))`)
)

func ReadCPUMetrics() CPUMetrics {
	var m CPUMetrics
	data, err := os.ReadFile("/proc/stat")
	if err == nil {
		lines := strings.Split(string(data), "\n")
		cores := 0
		for _, line := range lines {
			if strings.HasPrefix(line, "cpu ") {
				fields := strings.Fields(line)
				if len(fields) >= 5 {
					var total uint64
					for _, f := range fields[1:] {
						v, _ := strconv.ParseUint(f, 10, 64)
						total += v
					}
					idle, _ := strconv.ParseUint(fields[4], 10, 64)
					var iowait uint64
					if len(fields) >= 6 {
						iowait, _ = strconv.ParseUint(fields[5], 10, 64)
					}
					idleAll := idle + iowait

					cpuMu.Lock()
					if prevCPUTotal > 0 && total > prevCPUTotal {
						dTotal := float64(total - prevCPUTotal)
						dIdle := float64(idleAll - prevCPUIdle)
						pct := (1.0 - dIdle/dTotal) * 100.0
						if pct < 0 {
							pct = 0
						}
						if pct > 100 {
							pct = 100
						}
						m.UsagePct = math.Round(pct*10) / 10
					}
					prevCPUTotal = total
					prevCPUIdle = idleAll
					cpuMu.Unlock()
				}
			} else if strings.HasPrefix(line, "cpu") {
				cores++
			}
		}
		m.Cores = cores
	}

	if loadBytes, err := os.ReadFile("/proc/loadavg"); err == nil {
		parts := strings.Fields(string(loadBytes))
		if len(parts) >= 3 {
			m.Load1 = parts[0]
			m.Load5 = parts[1]
			m.Load15 = parts[2]
		}
	}
	return m
}

func ReadRAMMetrics() RAMMetrics {
	var m RAMMetrics
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return m
	}
	defer f.Close()

	vals := map[string]float64{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			key := strings.TrimSuffix(parts[0], ":")
			v, _ := strconv.ParseFloat(parts[1], 64)
			vals[key] = v / 1024.0 // KiB -> MiB
		}
	}

	m.TotalMiB = math.Round(vals["MemTotal"])
	m.AvailMiB = math.Round(vals["MemAvailable"])
	m.BuffCacheMiB = math.Round(vals["Buffers"] + vals["Cached"] + vals["SReclaimable"])
	used := m.TotalMiB - m.AvailMiB
	if used < 0 {
		used = 0
	}
	m.UsedMiB = math.Round(used)
	if m.TotalMiB > 0 {
		m.UsedPct = math.Round((m.UsedMiB/m.TotalMiB)*1000) / 10
	}

	m.SwapTotMiB = math.Round(vals["SwapTotal"])
	swapFree := vals["SwapFree"]
	m.SwapUsedMiB = math.Round(math.Max(0, m.SwapTotMiB-swapFree))
	if m.SwapTotMiB > 0 {
		m.SwapPct = math.Round((m.SwapUsedMiB/m.SwapTotMiB)*1000) / 10
	}
	return m
}

func ReadGPUMetrics() GPUMetrics {
	var m GPUMetrics
	ctxCmd := exec.Command(
		"nvidia-smi",
		"--query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,fan.speed,clocks.current.graphics,pstate",
		"--format=csv,noheader,nounits",
	)
	out, err := ctxCmd.Output()
	if err != nil {
		return m
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	parts := strings.Split(line, ",")
	if len(parts) < 7 {
		return m
	}
	parseF := func(s string) float64 {
		v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
		return v
	}
	m.Online = true
	m.Name = strings.TrimSpace(parts[0])
	m.TempC = parseF(parts[1])
	m.PowerDrawW = math.Round(parseF(parts[2])*10) / 10
	m.PowerLimitW = math.Round(parseF(parts[3])*10) / 10
	m.UtilPct = parseF(parts[4])
	m.VRAMUsedMiB = parseF(parts[5])
	m.VRAMTotMiB = parseF(parts[6])
	if len(parts) >= 8 {
		m.FanPct = parseF(parts[7])
	}
	if len(parts) >= 9 {
		m.ClockMHz, _ = strconv.Atoi(strings.TrimSpace(parts[8]))
	}
	if len(parts) >= 10 {
		m.PState = strings.TrimSpace(parts[9])
	}
	if m.VRAMTotMiB > 0 {
		m.VRAMPct = math.Round((m.VRAMUsedMiB/m.VRAMTotMiB)*1000) / 10
	}
	return m
}

func readOllamaMetrics() OllamaMetrics {
	var m OllamaMetrics
	client := &http.Client{Timeout: 1200 * time.Millisecond}
	resp, err := client.Get("http://127.0.0.1:11434/api/ps")
	if err == nil {
		defer resp.Body.Close()
		m.Online = resp.StatusCode == http.StatusOK
		var psData struct {
			Models []struct {
				Name          string `json:"name"`
				Size          int64  `json:"size"`
				SizeVRAM      int64  `json:"size_vram"`
				ContextLength int    `json:"context_length"`
				ExpiresAt     string `json:"expires_at"`
			} `json:"models"`
		}
		if body, err := io.ReadAll(resp.Body); err == nil {
			if json.Unmarshal(body, &psData) == nil {
				for _, item := range psData.Models {
					sizeGB := float64(item.Size) / (1024 * 1024 * 1024)
					vramGB := float64(item.SizeVRAM) / (1024 * 1024 * 1024)
					gpuPct := 0
					if item.Size > 0 {
						gpuPct = int(math.Round((float64(item.SizeVRAM) / float64(item.Size)) * 100))
					}
					exp := item.ExpiresAt
					if len(exp) >= 19 {
						exp = strings.ReplaceAll(exp[:19], "T", " ")
					}
					m.Models = append(m.Models, OllamaModelInfo{
						Name:      item.Name,
						SizeGB:    math.Round(sizeGB*100) / 100,
						VRAMGB:    math.Round(vramGB*100) / 100,
						GPUPct:    gpuPct,
						Context:   item.ContextLength,
						ExpiresAt: exp,
					})
				}
			}
		}
	}

	// Parse recent tok/s from docker logs ollama (tail 400 + cache last non-zero metrics)
	out, err := exec.Command("docker", "logs", "ollama", "--tail", "400").CombinedOutput()
	if err == nil {
		text := string(out)
		lastPromptFinalIdx := -1
		if locs := rePromptTime.FindAllStringIndex(text, -1); len(locs) > 0 {
			lastPromptFinalIdx = locs[len(locs)-1][0]
			matches := rePromptTime.FindAllStringSubmatch(text, -1)
			last := matches[len(matches)-1]
			m.LastTokens, _ = strconv.Atoi(last[1])
			v, _ := strconv.ParseFloat(last[2], 64)
			m.EvalTokPerSec = math.Round(v*10) / 10
		}
		if locs := reLivePromptProc.FindAllStringIndex(text, -1); len(locs) > 0 {
			lastPromptLiveIdx := locs[len(locs)-1][0]
			if lastPromptLiveIdx > lastPromptFinalIdx {
				matches := reLivePromptProc.FindAllStringSubmatch(text, -1)
				last := matches[len(matches)-1]
				v, _ := strconv.ParseFloat(last[3], 64)
				m.EvalTokPerSec = math.Round(v*10) / 10
			}
		}
		if matches := rePromptTask.FindAllStringSubmatch(text, -1); len(matches) > 0 {
			last := matches[len(matches)-1]
			t, _ := strconv.Atoi(last[1])
			if t > 0 {
				m.LastTokens = t
			}
		}

		// Detect whether Ollama is actively generating right now (live tg_3s after last final eval time)
		lastLiveIdx := -1
		if locs := reLiveTG.FindAllStringIndex(text, -1); len(locs) > 0 {
			lastLiveIdx = locs[len(locs)-1][0]
			matches := reLiveTG.FindAllStringSubmatch(text, -1)
			last := matches[len(matches)-1]
			v, _ := strconv.ParseFloat(last[3], 64) // tg_3s (real-time 3s window)
			if v <= 0 {
				v, _ = strconv.ParseFloat(last[2], 64)
			}
			m.LiveTokPerSec = math.Round(v*100) / 100
		}

		lastEvalIdx := -1
		if locs := reEvalTime.FindAllStringIndex(text, -1); len(locs) > 0 {
			lastEvalIdx = locs[len(locs)-1][0]
			matches := reEvalTime.FindAllStringSubmatch(text, -1)
			last := matches[len(matches)-1]
			if !strings.Contains(last[0], "prompt eval time") {
				v, _ := strconv.ParseFloat(last[2], 64)
				if v > 0 {
					m.AvgTokPerSec = math.Round(v*100) / 100
				}
			}
		}

		if len(m.Models) > 0 && lastLiveIdx > lastEvalIdx && m.LiveTokPerSec > 0 {
			m.IsGenerating = true
			m.GenTokPerSec = m.LiveTokPerSec
		} else {
			m.IsGenerating = false
			m.LiveTokPerSec = 0
			m.GenTokPerSec = m.AvgTokPerSec
		}

		ollamaCacheMu.Lock()
		if m.AvgTokPerSec > 0 {
			lastGenTokPerSec = m.AvgTokPerSec
		} else {
			m.AvgTokPerSec = lastGenTokPerSec
			if !m.IsGenerating {
				m.GenTokPerSec = lastGenTokPerSec
			}
		}
		if m.EvalTokPerSec > 0 {
			lastEvalTokPerSec = m.EvalTokPerSec
		} else {
			m.EvalTokPerSec = lastEvalTokPerSec
		}
		if m.LastTokens > 0 {
			lastPromptTokens = m.LastTokens
		} else {
			m.LastTokens = lastPromptTokens
		}
		ollamaCacheMu.Unlock()

		lines := strings.Split(text, "\n")
		for i := len(lines) - 1; i >= 0 && len(m.RecentLogs) < 5; i-- {
			l := strings.TrimSpace(lines[i])
			if strings.Contains(l, "POST ") || strings.Contains(l, "tokens per second") || strings.Contains(l, "tg_3s =") {
				if len(l) > 110 {
					l = l[:110] + "..."
				}
				m.RecentLogs = append(m.RecentLogs, l)
			}
		}
	} else {
		ollamaCacheMu.Lock()
		m.GenTokPerSec = lastGenTokPerSec
		m.EvalTokPerSec = lastEvalTokPerSec
		m.LastTokens = lastPromptTokens
		ollamaCacheMu.Unlock()
	}

	if len(m.Models) > 0 {
		m.ActiveState = "Model Loaded in VRAM"
	} else if m.Online {
		m.ActiveState = "Online (Idle / Standby)"
	} else {
		m.ActiveState = "Offline"
	}
	return m
}

func readComfyMetrics() ComfyMetrics {
	var m ComfyMetrics
	client := &http.Client{Timeout: 900 * time.Millisecond}
	resp, err := client.Get("http://127.0.0.1:8188/queue")
	if err == nil {
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			m.Online = true
			var qData struct {
				Running []any `json:"queue_running"`
				Pending []any `json:"queue_pending"`
			}
			if body, err := io.ReadAll(resp.Body); err == nil {
				if json.Unmarshal(body, &qData) == nil {
					m.QueueRunning = len(qData.Running)
					m.QueuePending = len(qData.Pending)
				}
			}
		}
	}

	// Read last progress from comfyui.log (only after the latest "got prompt")
	if data, err := os.ReadFile("/home/rizal/Documents/ollama/comfyui/comfyui.log"); err == nil {
		tail := string(data)
		if len(tail) > 16384 {
			tail = tail[len(tail)-16384:]
		}
		if idx := strings.LastIndex(tail, "got prompt"); idx >= 0 {
			tail = tail[idx:]
		}
		if matches := reComfyStep.FindAllStringSubmatch(tail, -1); len(matches) > 0 {
			last := matches[len(matches)-1]
			m.StepProgress, _ = strconv.Atoi(last[1])
			m.StepCurrent, _ = strconv.Atoi(last[2])
			m.StepTotal, _ = strconv.Atoi(last[3])
			m.ItPerSec = last[4]
		}
	}

	if m.Online {
		if m.QueueRunning > 0 {
			if m.StepTotal > 0 {
				m.StatusText = fmt.Sprintf("Rendering (%d%% • %d/%d)", m.StepProgress, m.StepCurrent, m.StepTotal)
				rem := m.StepTotal - m.StepCurrent
				if rem > 0 && m.ItPerSec != "" {
					var secPerStep float64
					if strings.HasSuffix(m.ItPerSec, "s/it") {
						secPerStep, _ = strconv.ParseFloat(strings.TrimSuffix(m.ItPerSec, "s/it"), 64)
					} else if strings.HasSuffix(m.ItPerSec, "it/s") {
						v, _ := strconv.ParseFloat(strings.TrimSuffix(m.ItPerSec, "it/s"), 64)
						if v > 0 {
							secPerStep = 1.0 / v
						}
					}
					if secPerStep > 0 {
						etaSec := int(math.Round(float64(rem) * secPerStep))
						if etaSec >= 60 {
							m.ETA = fmt.Sprintf("ETA %dm %02ds", etaSec/60, etaSec%60)
						} else {
							m.ETA = fmt.Sprintf("ETA %ds", etaSec)
						}
					}
				}
			} else {
				m.StatusText = "Loading Model to VRAM..."
				m.ItPerSec = "Initializing"
			}
		} else {
			m.StatusText = "Online (Ready)"
		}
	} else {
		m.StatusText = "Standby (Auto-Start via MCP)"
	}
	return m
}

func readDiskMetrics() DiskMetrics {
	var st syscall.Statfs_t
	if err := syscall.Statfs("/", &st); err != nil || st.Blocks == 0 {
		return DiskMetrics{}
	}
	tot := float64(st.Blocks) * float64(st.Bsize) / (1024 * 1024 * 1024)
	free := float64(st.Bavail) * float64(st.Bsize) / (1024 * 1024 * 1024)
	usedPct := ((tot - free) / tot) * 100.0
	return DiskMetrics{
		TotalGB: math.Round(tot*10) / 10,
		FreeGB:  math.Round(free*10) / 10,
		UsedPct: math.Round(usedPct*10) / 10,
	}
}

func readTopProcesses() []ProcessMemInfo {
	out, err := exec.Command("ps", "-eo", "comm,rss", "--sort=-rss").Output()
	if err != nil {
		return nil
	}
	agg := map[string]float64{}
	lines := strings.Split(string(out), "\n")
	for _, line := range lines[1:] {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		rssKB, _ := strconv.ParseFloat(fields[1], 64)
		agg[name] += rssKB / 1024.0
	}
	var list []ProcessMemInfo
	for k, v := range agg {
		if v >= 80 {
			list = append(list, ProcessMemInfo{Name: k, MemMB: math.Round(v)})
		}
	}
	sort.Slice(list, func(i, j int) bool { return list[i].MemMB > list[j].MemMB })
	if len(list) > 5 {
		list = list[:5]
	}
	return list
}

func readOllamaContainerStatus() string {
	out, err := exec.Command("docker", "ps", "--filter", "name=^ollama$", "--format", "{{.Status}}").Output()
	if err != nil {
		return "Unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "Stopped"
	}
	return s
}

func readInstalledModels(activeOllama []OllamaModelInfo) ([]InstalledModel, []InstalledModel, float64) {
	activeSet := map[string]bool{}
	for _, a := range activeOllama {
		activeSet[a.Name] = true
	}

	var ollamaList []InstalledModel
	seenDigest := map[string]bool{}
	var totalBytes int64

	client := &http.Client{Timeout: 900 * time.Millisecond}
	if resp, err := client.Get("http://127.0.0.1:11434/api/tags"); err == nil {
		defer resp.Body.Close()
		var tagsData struct {
			Models []struct {
				Name   string `json:"name"`
				Size   int64  `json:"size"`
				Digest string `json:"digest"`
			} `json:"models"`
		}
		if body, err := io.ReadAll(resp.Body); err == nil && json.Unmarshal(body, &tagsData) == nil {
			for _, m := range tagsData.Models {
				// Profiles derived from the same base GGUF share the same blob size on disk
				sizeKey := fmt.Sprintf("%d", m.Size)
				if !seenDigest[sizeKey] {
					seenDigest[sizeKey] = true
					totalBytes += m.Size
				}
				ollamaList = append(ollamaList, InstalledModel{
					Name:     m.Name,
					Category: "LLM",
					SizeGB:   math.Round((float64(m.Size)/(1024*1024*1024))*10) / 10,
					Active:   activeSet[m.Name],
				})
			}
		}
	}

	var comfyList []InstalledModel
	comfyBase := "/home/rizal/Documents/ollama/comfyui/models"
	subDirs := []struct {
		dir string
		cat string
	}{
		{"checkpoints", "SDXL"},
		{"controlnet", "ControlNet"},
		{"unet", "Video UNet"},
		{"text_encoders", "Text Enc"},
		{"clip_vision", "Vision"},
		{"vae", "VAE"},
	}
	for _, s := range subDirs {
		entries, err := os.ReadDir(filepath.Join(comfyBase, s.dir))
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || strings.HasPrefix(e.Name(), "put_") {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if ext != ".safetensors" && ext != ".gguf" && ext != ".ckpt" {
				continue
			}
			info, err := e.Info()
			if err != nil || info.Size() < 1024*1024 {
				continue
			}
			totalBytes += info.Size()
			comfyList = append(comfyList, InstalledModel{
				Name:     e.Name(),
				Category: s.cat,
				SizeGB:   math.Round((float64(info.Size())/(1024*1024*1024))*10) / 10,
			})
		}
	}

	totalGB := math.Round((float64(totalBytes)/(1024*1024*1024))*10) / 10
	return ollamaList, comfyList, totalGB
}

func computeHealthAlert(ram RAMMetrics, ollama OllamaMetrics, comfy ComfyMetrics) HealthAlert {
	if comfy.QueueRunning > 0 && len(ollama.Models) > 0 {
		return HealthAlert{
			Level:   "warn",
			Message: "VRAM Contention: ComfyUI & Ollama aktif bersamaan — Ollama ter-offload ke CPU/RAM",
		}
	}
	for _, m := range ollama.Models {
		if m.GPUPct < 100 {
			return HealthAlert{
				Level:   "warn",
				Message: fmt.Sprintf("CPU Spillover: %s hanya %d%% di GPU (kecepatan turun)", m.Name, m.GPUPct),
			}
		}
	}
	if ram.UsedPct >= 88 {
		return HealthAlert{
			Level:   "warn",
			Message: fmt.Sprintf("High RAM Pressure: %.1f%% RAM terpakai (%.1f GiB tersedia)", ram.UsedPct, ram.AvailMiB/1024.0),
		}
	}
	return HealthAlert{
		Level:   "ok",
		Message: "Healthy • 100% GPU VRAM & Memori Stabil",
	}
}

func readOpenCodeMetrics() OpenCodeMetrics {
	var m OpenCodeMetrics
	out, err := exec.Command("pgrep", "-f", "opencode(\\.exe)? serve --service").Output()
	if err == nil {
		pids := strings.Join(strings.Fields(string(out)), ", ")
		if pids != "" {
			m.Running = true
			m.PIDs = pids
		}
	}
	return m
}

func collectSnapshot() DashboardSnapshot {
	now := time.Now()
	ram := ReadRAMMetrics()
	ollama := readOllamaMetrics()
	comfy := readComfyMetrics()
	ollamaInstalled, comfyInstalled, totalModelsGB := readInstalledModels(ollama.Models)
	return DashboardSnapshot{
		Timestamp:     now.Format("2006-01-02 15:04:05"),
		UnixMS:        now.UnixMilli(),
		CPU:           ReadCPUMetrics(),
		RAM:           ram,
		Disk:          readDiskMetrics(),
		GPU:           ReadGPUMetrics(),
		Ollama:        ollama,
		Comfy:         comfy,
		OpenCode:      readOpenCodeMetrics(),
		TopProcs:      readTopProcesses(),
		OllamaStatus:  readOllamaContainerStatus(),
		OllamaModels:  ollamaInstalled,
		ComfyModels:   comfyInstalled,
		TotalModelsGB: totalModelsGB,
		Health:        computeHealthAlert(ram, ollama, comfy),
	}
}

func StartSampler() {
	record := func() {
		snap := collectSnapshot()
		histMu.Lock()
		latestSnap = snap
		hasSnapshot = true
		histRing = append(histRing, snap)
		if len(histRing) > 60 {
			histRing = histRing[len(histRing)-60:]
		}
		histMu.Unlock()
	}
	record()
	go func() {
		t := time.NewTicker(1000 * time.Millisecond)
		defer t.Stop()
		for range t.C {
			record()
		}
	}()
}

func GetLatestSnapshot() DashboardSnapshot {
	histMu.RLock()
	ok := hasSnapshot
	snap := latestSnap
	histMu.RUnlock()
	if ok {
		return snap
	}
	return collectSnapshot()
}

func HandleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(GetLatestSnapshot())
}

func HandleHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	histMu.RLock()
	cp := make([]DashboardSnapshot, len(histRing))
	copy(cp, histRing)
	histMu.RUnlock()
	_ = json.NewEncoder(w).Encode(cp)
}

func HandleStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	send := func() error {
		b, err := json.Marshal(GetLatestSnapshot())
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(w, "data: %s\n\n", b)
		if err == nil {
			flusher.Flush()
		}
		return err
	}
	if err := send(); err != nil {
		return
	}
	ticker := time.NewTicker(1000 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			if err := send(); err != nil {
				return
			}
		}
	}
}

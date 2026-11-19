package telemetry

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

var systemPowerCheck = func() error {
	return exec.Command("sudo", "-n", "/bin/true").Run()
}

var systemPowerExec = func(subcmd string) error {
	return exec.Command("sudo", "-n", "/usr/bin/systemctl", subcmd, "--no-wall").Run()
}

func HandleSystemPowerAction(w http.ResponseWriter, r *http.Request, subcmd, msg string) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Confirm bool `json:"confirm"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&body); err != nil || !body.Confirm {
		http.Error(w, `{"ok":false,"error":"Konfirmasi power action diperlukan"}`, http.StatusBadRequest)
		return
	}
	if err := systemPowerCheck(); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":    false,
			"error": "Hak akses sudoers NOPASSWD untuk systemctl poweroff/reboot belum aktif",
		})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "message": msg})
	execFn := systemPowerExec
	go func() {
		time.Sleep(800 * time.Millisecond)
		if err := execFn(subcmd); err != nil {
			log.Printf("[POWER] %s failed: %v", subcmd, err)
		}
	}()
}

func HandleUnloadOllama(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	client := &http.Client{Timeout: 5 * time.Second}
	unloaded := 0
	if resp, err := client.Get("http://127.0.0.1:11434/api/ps"); err == nil {
		defer resp.Body.Close()
		var psData struct {
			Models []struct {
				Name string `json:"name"`
			} `json:"models"`
		}
		if body, err := io.ReadAll(resp.Body); err == nil && json.Unmarshal(body, &psData) == nil {
			for _, m := range psData.Models {
				payload, _ := json.Marshal(map[string]any{"model": m.Name, "keep_alive": 0})
				if r2, err := client.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewReader(payload)); err == nil {
					r2.Body.Close()
					unloaded++
				}
			}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "unloaded": unloaded})
}

func HandleFreeComfy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	client := &http.Client{Timeout: 5 * time.Second}
	payload := []byte(`{"unload_models":true,"free_memory":true}`)
	ok := false
	if resp, err := client.Post("http://127.0.0.1:8188/free", "application/json", bytes.NewReader(payload)); err == nil {
		resp.Body.Close()
		ok = resp.StatusCode == http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": ok})
}

func HandleRestartOllama(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	err := exec.Command("docker", "restart", "ollama").Run()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": err == nil})
}

func HandleRestartMCP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	err := exec.Command("bash", "/home/rizal/Documents/ollama/restart-mcp.sh").Run()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": err == nil})
}

func HandleShutdownPC(w http.ResponseWriter, r *http.Request) {
	HandleSystemPowerAction(w, r, "poweroff", "PC sedang melakukan shutdown...")
}

func HandleRebootPC(w http.ResponseWriter, r *http.Request) {
	HandleSystemPowerAction(w, r, "reboot", "PC sedang melakukan restart...")
}

func HandleStopComfy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	client := &http.Client{Timeout: 2 * time.Second}
	if r1, err := client.Post("http://127.0.0.1:8188/interrupt", "application/json", nil); err == nil {
		r1.Body.Close()
	}
	if r2, err := client.Post("http://127.0.0.1:8188/queue", "application/json", bytes.NewReader([]byte(`{"clear":true}`))); err == nil {
		r2.Body.Close()
	}
	if r3, err := client.Post("http://127.0.0.1:8188/free", "application/json", bytes.NewReader([]byte(`{"unload_models":true,"free_memory":true}`))); err == nil {
		r3.Body.Close()
	}
	_ = exec.Command("pkill", "-TERM", "-f", "/home/rizal/Documents/ollama/comfyui/.venv/bin/python main.py").Run()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

func HandleGPUPower(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	watts := r.URL.Query().Get("watts")
	if watts != "100" && watts != "120" && watts != "150" && watts != "170" {
		http.Error(w, "Unsupported wattage (allowed: 100, 120, 150, 170)", http.StatusBadRequest)
		return
	}
	out, err := exec.Command("sudo", "-n", "nvidia-smi", "-pl", watts).CombinedOutput()
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": strings.TrimSpace(string(out))})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "watts": watts})
}

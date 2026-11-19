package chat

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type ChatMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	Thinking  string           `json:"thinking,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
}

type OllamaToolCall struct {
	Function struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

type ChatRequest struct {
	Model      string        `json:"model"`
	Cwd        string        `json:"cwd"`
	EnableExec bool          `json:"enable_exec"`
	EnableMCP  bool          `json:"enable_mcp"`
	Messages   []ChatMessage `json:"messages"`
}

type StoredToolLog struct {
	Name     string `json:"name"`
	Args     any    `json:"args,omitempty"`
	Output   string `json:"output,omitempty"`
	MediaURL string `json:"media_url,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	FilePath string `json:"file_path,omitempty"`
}

type ToolResult struct {
	Output   string
	MediaURL string
	FileURL  string
	FilePath string
}

type StoredMessage struct {
	Role          string          `json:"role"`
	Content       string          `json:"content"`
	Thinking      string          `json:"thinking,omitempty"`
	Images        []string        `json:"images,omitempty"`
	Tools         []StoredToolLog `json:"tools,omitempty"`
	ElapsedSec    float64         `json:"elapsed_sec,omitempty"`
	TokPerSec     float64         `json:"tok_per_sec,omitempty"`
	Tokens        int             `json:"tokens,omitempty"`
	Error         string          `json:"error,omitempty"`
	Queued        bool            `json:"queued,omitempty"`
	GoalCycle     int             `json:"goal_cycle,omitempty"`
	GoalMaxCycles int             `json:"goal_max_cycles,omitempty"`
	GoalPhase     string          `json:"goal_phase,omitempty"`
	ChangedFiles  []GoalFileStat  `json:"changed_files,omitempty"`
	CreatedAt     int64           `json:"created_at,omitempty"`
}

type ChatSession struct {
	ID         string          `json:"id"`
	Title      string          `json:"title"`
	Model      string          `json:"model"`
	Cwd        string          `json:"cwd"`
	EnableExec bool            `json:"enable_exec"`
	EnableMCP  bool            `json:"enable_mcp"`
	Agent      string          `json:"agent,omitempty"`
	CreatedAt  int64           `json:"created_at"`
	UpdatedAt  int64           `json:"updated_at"`
	Running    bool            `json:"running,omitempty"`
	Goal       *GoalState      `json:"goal,omitempty"`
	Messages   []StoredMessage `json:"messages"`
}

type ChatSummary struct {
	ID         string `json:"id"`
	Title      string `json:"title"`
	Model      string `json:"model"`
	Cwd        string `json:"cwd"`
	UpdatedAt  int64  `json:"updated_at"`
	MsgCount   int    `json:"msg_count"`
	Running    bool   `json:"running"`
	GoalStatus string `json:"goal_status,omitempty"`
}

var ollamaChatEndpoint = "http://127.0.0.1:11434/api/chat"

var execTools = []map[string]any{
	{
		"type": "function",
		"function": map[string]any{
			"name":        "bash",
			"description": "Execute a bash command in the workspace directory and return stdout/stderr.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string", "description": "Shell command to run"},
				},
				"required": []string{"command"},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "read_file",
			"description": "Read the text contents of a file on disk. Supports optional 1-based start_line and end_line for large files.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":       map[string]any{"type": "string", "description": "Absolute or relative file path"},
					"start_line": map[string]any{"type": "integer", "description": "Optional 1-based starting line number"},
					"end_line":   map[string]any{"type": "integer", "description": "Optional 1-based ending line number"},
				},
				"required": []string{"path"},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "edit_file",
			"description": "Replace an exact target text block inside an existing file with replacement text.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":        map[string]any{"type": "string", "description": "File path"},
					"target":      map[string]any{"type": "string", "description": "Exact existing text to replace"},
					"replacement": map[string]any{"type": "string", "description": "New replacement text"},
				},
				"required": []string{"path", "target", "replacement"},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Create or overwrite a file with the given text content.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "File path"},
					"content": map[string]any{"type": "string", "description": "Full file content"},
				},
				"required": []string{"path", "content"},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "list_files",
			"description": "List files and folders inside the workspace directory.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "Directory path (defaults to workspace root)"},
				},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "find_files",
			"description": "Recursively find files inside the workspace matching an optional glob or filename substring (e.g. '*.go', 'chat').",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"pattern": map[string]any{"type": "string", "description": "Optional filename glob (e.g. *.go) or path substring"},
					"path":    map[string]any{"type": "string", "description": "Subdirectory to search (defaults to workspace root)"},
				},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "grep_search",
			"description": "Search text or regex pattern across workspace files and return matching file:line:content lines.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query":       map[string]any{"type": "string", "description": "Regex or text pattern to search for"},
					"path":        map[string]any{"type": "string", "description": "Directory or file path to search (defaults to workspace root)"},
					"ignore_case": map[string]any{"type": "boolean", "description": "Case-insensitive search"},
					"literal":     map[string]any{"type": "boolean", "description": "Treat query as literal substring instead of regex"},
				},
				"required": []string{"query"},
			},
		},
	},
}

var mcpTools = []map[string]any{
	{
		"type": "function",
		"function": map[string]any{
			"name":        "generate_image",
			"description": "Generate a local SDXL image (txt2img or img2img with ControlNet) via ComfyUI.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt": map[string]any{
						"type":        "string",
						"description": "Detailed English tags/phrases front-loaded in first 30 words using (detail:1.2) emphasis.",
					},
					"style": map[string]any{
						"type":        "string",
						"enum":        []string{"general", "illustration"},
						"description": "'general' = Lustify SDXL Endgame (realistic photos, cinematic); 'illustration' = WAI-Illustrious v170 (2D anime, Ghibli, manga, cartoon, digital illustration).",
					},
					"negative_prompt":      map[string]any{"type": "string"},
					"width":                map[string]any{"type": "integer"},
					"height":               map[string]any{"type": "integer"},
					"steps":                map[string]any{"type": "integer"},
					"reference_image_path": map[string]any{"type": "string"},
					"denoise": map[string]any{
						"type":        "number",
						"description": "img2img strength: 0.15-0.30 for subtle 1:1 retouch in same style; 0.55-0.70 for style transfer (e.g. photo to Ghibli/anime) or outfit change.",
					},
				},
				"required": []string{"prompt"},
			},
		},
	},
	{
		"type": "function",
		"function": map[string]any{
			"name":        "generate_video",
			"description": "Generate a local MP4 video from an image using Wan 2.1 I2V via ComfyUI.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt":               map[string]any{"type": "string"},
					"reference_image_path": map[string]any{"type": "string"},
					"negative_prompt":      map[string]any{"type": "string"},
					"seconds":              map[string]any{"type": "number"},
					"steps":                map[string]any{"type": "integer"},
				},
				"required": []string{"prompt"},
			},
		},
	},
}

var reFileOut = regexp.MustCompile(`File:\s*(/[^\r\n]+\.(?:png|jpg|jpeg|webp|mp4|webm))`)

func writeNDJSON(w io.Writer, flusher http.Flusher, payload map[string]any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "%s\n", b)
	flusher.Flush()
}

func resolvePathInCwd(cwd, p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return filepath.Clean(cwd)
	}
	if filepath.IsAbs(p) {
		clean := filepath.Clean(p)
		if strings.HasPrefix(clean, "/home/") && !strings.HasPrefix(clean, "/home/rizal/") && clean != "/home/rizal" {
			return filepath.Clean(filepath.Join(cwd, filepath.Base(clean)))
		}
		return clean
	}
	return filepath.Clean(filepath.Join(cwd, p))
}

func detectStreamLoop(s string, maxChars int) bool {
	if maxChars > 0 && len(s) > maxChars {
		return true
	}
	if len(s) < 120 {
		return false
	}
	tail := s
	if len(tail) > 900 {
		tail = tail[len(tail)-900:]
	}
	var lines []string
	for _, raw := range strings.Split(tail, "\n") {
		t := strings.TrimSpace(raw)
		if len(t) >= 12 {
			lines = append(lines, t)
		}
	}
	if n := len(lines); n >= 4 && lines[n-1] == lines[n-2] && lines[n-2] == lines[n-3] && lines[n-3] == lines[n-4] {
		return true
	}
	for pLen := 32; pLen <= 120; pLen++ {
		if len(tail) < pLen*4 {
			break
		}
		end := len(tail)
		blk := tail[end-pLen : end]
		if blk == tail[end-2*pLen:end-pLen] && blk == tail[end-3*pLen:end-2*pLen] && blk == tail[end-4*pLen:end-3*pLen] {
			return true
		}
	}
	return false
}

func randomImageSeed() int64 {
	n, err := rand.Int(rand.Reader, big.NewInt(2147483646))
	if err != nil {
		return time.Now().UnixNano()&0x7fffffff | 1
	}
	return n.Int64() + 1
}

func isDisableRefValue(v string) bool {
	s := strings.ToLower(strings.TrimSpace(v))
	return s == "" || s == "none" || s == "null" || s == "false" || s == "txt2img" || s == "no"
}

func prepareGenerateImageArgs(currentTurnPastedPath string, args map[string]any) {
	if args == nil {
		return
	}
	rawRef, _ := args["reference_image_path"].(string)
	rawRef = strings.TrimSpace(rawRef)
	rawStyle, _ := args["style"].(string)
	styleLow := strings.ToLower(strings.TrimSpace(rawStyle))
	isIllustration := styleLow == "illustration" || styleLow == "anime" || styleLow == "cartoon" || styleLow == "2d" || styleLow == "manga"

	den := 0.15
	hasExplicitDenoise := false
	if v, ok := args["denoise"].(float64); ok && v >= 0.0 && v < 0.99 {
		den = v
		hasExplicitDenoise = true
	}
	resolvedRef := "none"
	if currentTurnPastedPath != "" {
		resolvedRef = currentTurnPastedPath
	} else if !isDisableRefValue(rawRef) && !strings.EqualFold(rawRef, "auto") && !strings.Contains(strings.ToLower(rawRef), "image") {
		if info, err := os.Stat(rawRef); err == nil && !info.IsDir() {
			resolvedRef = rawRef
		}
	}
	if resolvedRef == "none" {
		args["denoise"] = 1.0
	} else {
		if !hasExplicitDenoise {
			if isIllustration {
				// Photo/image -> anime/Ghibli/illustration needs strong denoise; ControlNet keeps pose/structure locked.
				den = 0.62
			} else {
				den = 0.15
			}
		} else if den > 0.95 {
			den = 0.95
		} else if den < 0.05 {
			den = 0.05
		}
		args["denoise"] = den
	}
	args["reference_image_path"] = resolvedRef
	args["seed"] = randomImageSeed()
}

func runToolCall(ctx context.Context, cwd string, currentTurnPastedPath string, latestVideoRefPath string, name string, args map[string]any) ToolResult {
	return runToolCallWithTimeout(ctx, cwd, currentTurnPastedPath, latestVideoRefPath, name, args, 90*time.Second)
}

func runToolCallWithTimeout(ctx context.Context, cwd string, currentTurnPastedPath string, latestVideoRefPath string, name string, args map[string]any, bashTimeout time.Duration) ToolResult {
	if bashTimeout <= 0 {
		bashTimeout = 90 * time.Second
	}
	strArg := func(k, def string) string {
		if v, ok := args[k]; ok && v != nil {
			return fmt.Sprintf("%v", v)
		}
		return def
	}
	intArg := func(k string, def int) int {
		if v, ok := args[k]; ok && v != nil {
			switch n := v.(type) {
			case float64:
				return int(n)
			case int:
				return n
			}
		}
		return def
	}
	floatArg := func(k string, def float64) float64 {
		if v, ok := args[k]; ok && v != nil {
			switch n := v.(type) {
			case float64:
				return n
			case int:
				return float64(n)
			}
		}
		return def
	}

	switch strings.ToLower(name) {
	case "bash", "sh":
		cmdStr := strArg("command", strArg("cmd", ""))
		if cmdStr == "" {
			return ToolResult{Output: "Error: empty command"}
		}
		low := strings.ToLower(cmdStr)
		for _, blocked := range []string{"mkfs", "dd if=", "shutdown", "reboot", "rm -rf / "} {
			if strings.Contains(low, blocked) {
				return ToolResult{Output: "Blocked dangerous command: " + blocked}
			}
		}
		tctx, cancel := context.WithTimeout(ctx, bashTimeout)
		defer cancel()
		cmd := exec.CommandContext(tctx, "/bin/bash", "-lc", cmdStr)
		cmd.Dir = cwd
		out, err := cmd.CombinedOutput()
		res := string(out)
		if len(res) > 12000 {
			res = res[:12000] + "\n... [truncated]"
		}
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Exit error: %v\n%s", err, res)}
		}
		if strings.TrimSpace(res) == "" {
			return ToolResult{Output: "(command exited 0 with no output)"}
		}
		return ToolResult{Output: res}

	case "read_file", "read":
		p := resolvePathInCwd(cwd, strArg("path", strArg("file_path", "")))
		data, err := os.ReadFile(p)
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Error reading %s: %v", p, err)}
		}
		s := string(data)
		startLine := intArg("start_line", 0)
		endLine := intArg("end_line", 0)
		if startLine > 0 || endLine > 0 {
			allLines := strings.Split(s, "\n")
			if startLine < 1 {
				startLine = 1
			}
			if endLine <= 0 || endLine > len(allLines) {
				endLine = len(allLines)
			}
			if startLine > len(allLines) {
				return ToolResult{Output: fmt.Sprintf("File %s has only %d lines (requested start_line=%d)", p, len(allLines), startLine)}
			}
			if endLine < startLine {
				endLine = startLine
			}
			var numbered []string
			for i := startLine; i <= endLine; i++ {
				numbered = append(numbered, fmt.Sprintf("%d: %s", i, allLines[i-1]))
			}
			s = strings.Join(numbered, "\n")
		}
		if len(s) > 16000 {
			s = s[:16000] + "\n... [truncated; pass start_line/end_line to read further]"
		}
		return ToolResult{Output: s}

	case "edit_file", "edit":
		p := resolvePathInCwd(cwd, strArg("path", strArg("file_path", "")))
		target := strArg("target", "")
		replacement := strArg("replacement", "")
		if target == "" {
			return ToolResult{Output: "Error: target text cannot be empty"}
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Error reading %s: %v", p, err)}
		}
		content := string(raw)
		if !strings.Contains(content, target) {
			return ToolResult{Output: fmt.Sprintf("Error: target block not found in %s. Read the file first to match exact text.", p)}
		}
		count := strings.Count(content, target)
		updated := strings.ReplaceAll(content, target, replacement)
		if err := os.WriteFile(p, []byte(updated), 0o644); err != nil {
			return ToolResult{Output: fmt.Sprintf("Error writing %s: %v", p, err)}
		}
		return ToolResult{
			Output:   fmt.Sprintf("Edited %s (%d occurrence(s) replaced, %d bytes total)", p, count, len(updated)),
			FileURL:  "/api/chat-download?path=" + url.QueryEscape(p),
			FilePath: p,
		}

	case "write_file", "write":
		p := resolvePathInCwd(cwd, strArg("path", strArg("file_path", "output.txt")))
		content := strArg("content", "")
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return ToolResult{Output: fmt.Sprintf("Error creating parent dir: %v", err)}
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			return ToolResult{Output: fmt.Sprintf("Error writing %s: %v", p, err)}
		}
		return ToolResult{
			Output:   fmt.Sprintf("Wrote %d bytes to %s", len(content), p),
			FileURL:  "/api/chat-download?path=" + url.QueryEscape(p),
			FilePath: p,
		}

	case "list_files", "list", "ls":
		dir := resolvePathInCwd(cwd, strArg("path", strArg("file_path", ".")))
		entries, err := os.ReadDir(dir)
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Error listing %s: %v", dir, err)}
		}
		var lines []string
		for _, e := range entries {
			if e.Name() == ".git" || e.Name() == "node_modules" || e.Name() == ".venv" {
				continue
			}
			tag := "FILE"
			if e.IsDir() {
				tag = "DIR "
			}
			lines = append(lines, fmt.Sprintf("%s  %s", tag, e.Name()))
			if len(lines) >= 120 {
				lines = append(lines, "... [truncated]")
				break
			}
		}
		return ToolResult{Output: strings.Join(lines, "\n")}

	case "find_files", "find", "glob":
		baseDir := resolvePathInCwd(cwd, strArg("path", strArg("file_path", ".")))
		pat := strings.TrimSpace(strArg("pattern", strArg("glob", "")))
		patLow := strings.ToLower(pat)
		var matches []string
		err := filepath.WalkDir(baseDir, func(fpath string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			name := d.Name()
			if d.IsDir() {
				if name == ".git" || name == "node_modules" || name == ".venv" || name == "__pycache__" || name == "dist" {
					return filepath.SkipDir
				}
				return nil
			}
			rel, rerr := filepath.Rel(baseDir, fpath)
			if rerr != nil {
				rel = fpath
			}
			if pat != "" {
				matched, _ := filepath.Match(pat, name)
				if !matched && !strings.Contains(strings.ToLower(rel), patLow) {
					return nil
				}
			}
			matches = append(matches, rel)
			if len(matches) >= 150 {
				return fmt.Errorf("limit")
			}
			return nil
		})
		if err != nil && err.Error() != "limit" {
			return ToolResult{Output: fmt.Sprintf("Error finding files in %s: %v", baseDir, err)}
		}
		if len(matches) == 0 {
			return ToolResult{Output: "(no matching files found)"}
		}
		out := strings.Join(matches, "\n")
		if len(matches) >= 150 {
			out += "\n... [truncated at 150 files]"
		}
		return ToolResult{Output: out}

	case "grep_search", "grep", "search":
		query := strArg("query", strArg("pattern", ""))
		if strings.TrimSpace(query) == "" {
			return ToolResult{Output: "Error: empty query"}
		}
		targetPath := resolvePathInCwd(cwd, strArg("path", "."))
		ignoreCase := false
		if v, ok := args["ignore_case"].(bool); ok {
			ignoreCase = v
		}
		literal := false
		if v, ok := args["literal"].(bool); ok {
			literal = v
		}
		var re *regexp.Regexp
		if !literal {
			expr := query
			if ignoreCase {
				expr = "(?i)" + expr
			}
			if compiled, rerr := regexp.Compile(expr); rerr == nil {
				re = compiled
			}
		}
		queryLow := strings.ToLower(query)
		var hits []string
		totalChars := 0
		_ = filepath.WalkDir(targetPath, func(fpath string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			name := d.Name()
			if d.IsDir() {
				if name == ".git" || name == "node_modules" || name == ".venv" || name == "__pycache__" || name == "dist" {
					return filepath.SkipDir
				}
				return nil
			}
			info, ierr := d.Info()
			if ierr != nil || info.Size() > 2*1024*1024 {
				return nil
			}
			raw, rerr := os.ReadFile(fpath)
			if rerr != nil || bytes.IndexByte(raw, 0) >= 0 {
				return nil
			}
			rel, rerr := filepath.Rel(cwd, fpath)
			if rerr != nil {
				rel = fpath
			}
			lines := strings.Split(string(raw), "\n")
			for idx, ln := range lines {
				matched := false
				if re != nil {
					matched = re.MatchString(ln)
				} else if ignoreCase {
					matched = strings.Contains(strings.ToLower(ln), queryLow)
				} else {
					matched = strings.Contains(ln, query)
				}
				if matched {
					trimLn := ln
					if len(trimLn) > 220 {
						trimLn = trimLn[:220] + "..."
					}
					entry := fmt.Sprintf("%s:%d: %s", rel, idx+1, trimLn)
					hits = append(hits, entry)
					totalChars += len(entry) + 1
					if len(hits) >= 80 || totalChars >= 10000 {
						return fmt.Errorf("limit")
					}
				}
			}
			return nil
		})
		if len(hits) == 0 {
			return ToolResult{Output: "(no matches found)"}
		}
		out := strings.Join(hits, "\n")
		if len(hits) >= 80 || totalChars >= 10000 {
			out += "\n... [truncated]"
		}
		return ToolResult{Output: out}

	case "generate_image":
		rawRef := strings.TrimSpace(strArg("reference_image_path", ""))
		den := floatArg("denoise", 0.15)
		resolvedRef := "none"
		if currentTurnPastedPath != "" && den > 0.01 {
			resolvedRef = currentTurnPastedPath
		} else if !isDisableRefValue(rawRef) && !strings.EqualFold(rawRef, "auto") && !strings.Contains(strings.ToLower(rawRef), "image") && den > 0.01 {
			if info, err := os.Stat(rawRef); err == nil && !info.IsDir() {
				resolvedRef = rawRef
			}
		}
		if resolvedRef == "none" {
			den = 1.0
		}
		seedVal := int64(intArg("seed", 0))
		if seedVal <= 0 {
			seedVal = randomImageSeed()
		}
		payload, _ := json.Marshal(map[string]any{
			"prompt":               strArg("prompt", ""),
			"style":                strArg("style", "general"),
			"negative_prompt":      strArg("negative_prompt", ""),
			"width":                intArg("width", 1024),
			"height":               intArg("height", 1024),
			"steps":                intArg("steps", 28),
			"seed":                 seedVal,
			"reference_image_path": resolvedRef,
			"denoise":              den,
			"output_path":          filepath.Join(cwd, fmt.Sprintf("web_gen_%d.png", time.Now().UnixMilli())),
		})
		pyCode := `import json, sys; sys.path.insert(0, '/home/rizal/Documents/ollama/image-mcp'); import server; a=json.loads(sys.argv[1]); print(server.generate_image(**a))`
		tctx, cancel := context.WithTimeout(ctx, 360*time.Second)
		defer cancel()
		cmd := exec.CommandContext(tctx, "/home/rizal/Documents/ollama/comfyui/.venv/bin/python", "-c", pyCode, string(payload))
		cmd.Dir = cwd
		out, err := cmd.CombinedOutput()
		res := string(out)
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Error generating image: %v\n%s", err, res)}
		}
		mediaURL := ""
		if m := reFileOut.FindStringSubmatch(res); len(m) == 2 {
			mediaURL = "/api/chat-media?path=" + url.QueryEscape(m[1])
		}
		return ToolResult{Output: res, MediaURL: mediaURL}

	case "generate_video":
		ref := strings.TrimSpace(strArg("reference_image_path", ""))
		if (ref == "" || strings.EqualFold(ref, "auto") || strings.Contains(strings.ToLower(ref), "image 1")) && latestVideoRefPath != "" {
			ref = latestVideoRefPath
		}
		payload, _ := json.Marshal(map[string]any{
			"prompt":               strArg("prompt", ""),
			"reference_image_path": ref,
			"negative_prompt":      strArg("negative_prompt", ""),
			"seconds":              floatArg("seconds", 3.0),
			"steps":                intArg("steps", 0),
			"fast_mode":            true,
			"output_path":          filepath.Join(cwd, fmt.Sprintf("web_vid_%d.mp4", time.Now().Unix())),
		})
		pyCode := `import json, sys; sys.path.insert(0, '/home/rizal/Documents/ollama/image-mcp'); import server; a=json.loads(sys.argv[1]); print(server.generate_video(**a))`
		cmd := exec.CommandContext(ctx, "/home/rizal/Documents/ollama/comfyui/.venv/bin/python", "-c", pyCode, string(payload))
		cmd.Dir = cwd
		out, err := cmd.CombinedOutput()
		res := string(out)
		if err != nil {
			return ToolResult{Output: fmt.Sprintf("Error generating video: %v\n%s", err, res)}
		}
		mediaURL := ""
		if m := reFileOut.FindStringSubmatch(res); len(m) == 2 {
			mediaURL = "/api/chat-media?path=" + url.QueryEscape(m[1])
		}
		return ToolResult{Output: res, MediaURL: mediaURL}
	}
	return ToolResult{Output: "Unknown tool: " + name}
}

type ollamaTurnResult struct {
	Content                string
	Thinking               string
	ToolCalls              []OllamaToolCall
	EvalCount              int
	EvalNs                 int64
	LoopAborted            bool
	StallAbort             bool
	XMLParseErr            bool
	HallucinatedToolRescued bool
	ErrMsg                 string
	Fatal                  bool
}

var reHallucinatedMediaTool = regexp.MustCompile(`(?s)\[Executed\s+tool\s+(generate_image|generate_video)\s*\(?\s*\{`)

// extractBalancedJSONObject extracts a single balanced JSON object starting at s[0] == '{'.
func extractBalancedJSONObject(s string) string {
	if len(s) == 0 || s[0] != '{' {
		return ""
	}
	depth := 0
	inStr := false
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
		} else if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				return s[:i+1]
			}
		}
	}
	return ""
}

// parseHallucinatedMediaToolCall rescues turns where a model imitated old
// "[Executed tool generate_image {...} -> ...]" plain text instead of emitting
// a native Ollama tool_calls JSON payload.
func parseHallucinatedMediaToolCall(content string) (OllamaToolCall, bool) {
	var tc OllamaToolCall
	loc := reHallucinatedMediaTool.FindStringSubmatchIndex(content)
	if len(loc) < 4 {
		return tc, false
	}
	toolName := content[loc[2]:loc[3]]
	braceIdx := strings.IndexByte(content[loc[0]:], '{')
	if braceIdx < 0 {
		return tc, false
	}
	jsonObj := extractBalancedJSONObject(content[loc[0]+braceIdx:])
	if jsonObj == "" {
		return tc, false
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(jsonObj), &args); err != nil || len(args) == 0 {
		return tc, false
	}
	promptStr, _ := args["prompt"].(string)
	if strings.TrimSpace(promptStr) == "" {
		return tc, false
	}
	tc.Function.Name = toolName
	tc.Function.Arguments = args
	return tc, true
}

var chatWorkspaceRoot = "/home/rizal/Documents/chat"

func slugifyWorkspaceName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
		} else if !lastDash && b.Len() > 0 {
			b.WriteByte('-')
			lastDash = true
		}
		if b.Len() >= 26 {
			break
		}
	}
	return strings.Trim(b.String(), "-")
}

func resolveSessionWorkspace(chatID, existingCwd, reqCwd, promptHint string) string {
	reqTrim := strings.TrimSpace(reqCwd)
	if reqTrim == "" || reqTrim == "__auto__" {
		if strings.TrimSpace(existingCwd) != "" {
			return resolveChatWorkspace(existingCwd)
		}
		shortID := safeChatID(chatID)
		shortID = strings.TrimPrefix(shortID, "chat_")
		if len(shortID) > 10 {
			shortID = shortID[len(shortID)-10:]
		}
		if shortID == "" {
			shortID = fmt.Sprintf("%d", time.Now().UnixMilli())
		}
		slug := slugifyWorkspaceName(promptHint)
		folderName := "chat_" + shortID
		if slug != "" {
			folderName = slug + "-" + shortID
		}
		isolated := filepath.Join(chatWorkspaceRoot, folderName)
		_ = os.MkdirAll(isolated, 0o755)
		return isolated
	}
	return resolveChatWorkspace(reqTrim)
}

func resolveChatWorkspace(cwd string) string {
	cwd = strings.TrimSpace(cwd)
	if cwd == "" || cwd == "__auto__" {
		cwd = chatWorkspaceRoot
	}
	clean := filepath.Clean(cwd)
	// Auto-create custom folders safely inside /home/rizal/ or /tmp/ (or chatWorkspaceRoot in tests)
	if strings.HasPrefix(clean, "/home/rizal/") || clean == "/home/rizal" || strings.HasPrefix(clean, "/tmp/") || strings.HasPrefix(clean, filepath.Clean(chatWorkspaceRoot)) {
		_ = os.MkdirAll(clean, 0o755)
	}
	if info, err := os.Stat(clean); err == nil && info.IsDir() {
		return clean
	}
	_ = os.MkdirAll(chatWorkspaceRoot, 0o755)
	return chatWorkspaceRoot
}

func extractChatImageRefs(reqMsgs []ChatMessage) (currentTurnPastedPath, latestVideoRefPath string) {
	for i := len(reqMsgs) - 1; i >= 0; i-- {
		if reqMsgs[i].Role == "user" && len(reqMsgs[i].Images) > 0 {
			rawB64 := reqMsgs[i].Images[0]
			if rawBytes, err := base64.StdEncoding.DecodeString(rawB64); err == nil {
				p := filepath.Join("/tmp", fmt.Sprintf("web_chat_paste_%d.png", time.Now().UnixMilli()))
				if os.WriteFile(p, rawBytes, 0o644) == nil {
					if latestVideoRefPath == "" {
						latestVideoRefPath = p
					}
					isLatestUser := true
					for j := i + 1; j < len(reqMsgs); j++ {
						if reqMsgs[j].Role == "user" {
							isLatestUser = false
							break
						}
					}
					if isLatestUser {
						currentTurnPastedPath = p
					}
				}
			}
			if latestVideoRefPath != "" {
				break
			}
		}
	}
	return currentTurnPastedPath, latestVideoRefPath
}

func streamOneOllamaTurn(
	ctx context.Context,
	client *http.Client,
	model string,
	msgs []ChatMessage,
	activeTools []map[string]any,
	options map[string]any,
	firstChunkLimitMs int64,
	interChunkLimitMs int64,
	turn int,
	enableMCP bool,
	reqMsgs []ChatMessage,
	emit func(map[string]any),
) ollamaTurnResult {
	var res ollamaTurnResult
	reqMap := map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   true,
		"options":  options,
	}
	if len(activeTools) > 0 {
		reqMap["tools"] = activeTools
	}
	ollamaBody, _ := json.Marshal(reqMap)
	turnCtx, turnCancel := context.WithCancel(ctx)
	defer turnCancel()

	oReq, err := http.NewRequestWithContext(turnCtx, http.MethodPost, ollamaChatEndpoint, bytes.NewReader(ollamaBody))
	if err != nil {
		res.ErrMsg = err.Error()
		res.Fatal = true
		return res
	}
	oReq.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(oReq)
	if err != nil {
		res.ErrMsg = err.Error()
		res.Fatal = true
		return res
	}

	var lastActivityMs atomic.Int64
	var sawChunk atomic.Bool
	var stallAborted atomic.Bool
	var streamDone atomic.Bool
	lastActivityMs.Store(time.Now().UnixMilli())
	go func() {
		tk := time.NewTicker(2 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-turnCtx.Done():
				return
			case <-tk.C:
				if streamDone.Load() {
					return
				}
				idleMs := time.Now().UnixMilli() - lastActivityMs.Load()
				limitMs := firstChunkLimitMs
				if sawChunk.Load() {
					limitMs = interChunkLimitMs
				}
				if idleMs > limitMs {
					stallAborted.Store(true)
					turnCancel()
					return
				}
			}
		}
	}()

	var assistantContent strings.Builder
	var assistantThink strings.Builder
	var toolCalls []OllamaToolCall

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var chunk struct {
			Message struct {
				Content   string           `json:"content"`
				Thinking  string           `json:"thinking"`
				ToolCalls []OllamaToolCall `json:"tool_calls"`
			} `json:"message"`
			Done         bool   `json:"done"`
			Error        string `json:"error"`
			EvalCount    int    `json:"eval_count"`
			EvalDuration int64  `json:"eval_duration"`
		}
		if json.Unmarshal(line, &chunk) != nil {
			continue
		}
		sawChunk.Store(true)
		lastActivityMs.Store(time.Now().UnixMilli())
		if chunk.Done {
			res.EvalCount += chunk.EvalCount
			res.EvalNs += chunk.EvalDuration
		}
		if chunk.Error != "" {
			streamDone.Store(true)
			resp.Body.Close()
			turnCancel()
			if strings.Contains(chunk.Error, "XML syntax error") && enableMCP && turn == 0 {
				res.XMLParseErr = true
				break
			}
			res.ErrMsg = chunk.Error
			res.Fatal = true
			return res
		}
		if chunk.Message.Thinking != "" {
			assistantThink.WriteString(chunk.Message.Thinking)
			emit(map[string]any{"type": "thinking", "text": chunk.Message.Thinking})
			if detectStreamLoop(assistantThink.String(), 12000) {
				res.LoopAborted = true
				turnCancel()
				break
			}
		}
		if chunk.Message.Content != "" {
			assistantContent.WriteString(chunk.Message.Content)
			emit(map[string]any{"type": "content", "text": chunk.Message.Content})
			if detectStreamLoop(assistantContent.String(), 20000) {
				res.LoopAborted = true
				turnCancel()
				break
			}
		}
		if len(chunk.Message.ToolCalls) > 0 {
			toolCalls = append(toolCalls, chunk.Message.ToolCalls...)
		}
	}
	streamDone.Store(true)
	resp.Body.Close()

	res.StallAbort = stallAborted.Load()
	res.Content = assistantContent.String()
	res.Thinking = assistantThink.String()

	if res.XMLParseErr && len(toolCalls) == 0 {
		lastUserPrompt := ""
		prevBasePrompt := ""
		for i := len(reqMsgs) - 1; i >= 0; i-- {
			if reqMsgs[i].Role == "user" && lastUserPrompt == "" {
				lastUserPrompt = strings.TrimSpace(reqMsgs[i].Content)
			}
			if m := regexp.MustCompile(`Prompt used:\s*([^\n]+)`).FindStringSubmatch(reqMsgs[i].Content); len(m) == 2 && prevBasePrompt == "" {
				prevBasePrompt = strings.TrimSpace(m[1])
			}
		}
		finalPrompt := lastUserPrompt
		if prevBasePrompt != "" && len(lastUserPrompt) < 120 {
			finalPrompt = "(" + lastUserPrompt + ":1.3), " + prevBasePrompt
		}
		stepsVal := 30
		if sm := regexp.MustCompile(`(?i)steps?\s*[:=]?\s*(\d+)`).FindStringSubmatch(lastUserPrompt); len(sm) == 2 {
			fmt.Sscanf(sm[1], "%d", &stepsVal)
		}
		var tc OllamaToolCall
		tc.Function.Name = "generate_image"
		tc.Function.Arguments = map[string]any{
			"prompt":               finalPrompt,
			"style":                "illustration",
			"width":                832,
			"height":               1216,
			"steps":                stepsVal,
			"reference_image_path": "none",
		}
		toolCalls = append(toolCalls, tc)
	}

	if len(toolCalls) == 0 && turn == 0 && enableMCP {
		if rescuedTC, ok := parseHallucinatedMediaToolCall(res.Content); ok {
			toolCalls = append(toolCalls, rescuedTC)
			res.HallucinatedToolRescued = true
		}
	}

	if len(toolCalls) == 0 && turn == 0 && enableMCP {
		cText := res.Content
		tText := strings.ToLower(res.Thinking)
		if strings.Contains(cText, "web_gen_") || (strings.Contains(tText, "generate_image") && (strings.Contains(tText, "prompt") || strings.Contains(cText, "Prompt We Sent"))) {
			extractedPrompt := ""
			reQuote := regexp.MustCompile(`>\s*\*\*"([^"]{20,})"\*\*|>\s*"([^"]{20,})"`)
			if m := reQuote.FindStringSubmatch(cText); len(m) >= 3 {
				if m[1] != "" {
					extractedPrompt = m[1]
				} else {
					extractedPrompt = m[2]
				}
			}
			if extractedPrompt == "" {
				if m := reQuote.FindStringSubmatch(res.Thinking); len(m) >= 3 {
					if m[1] != "" {
						extractedPrompt = m[1]
					} else {
						extractedPrompt = m[2]
					}
				}
			}
			if extractedPrompt != "" {
				style := "general"
				lowAll := strings.ToLower(extractedPrompt + " " + cText + " " + tText)
				if strings.Contains(lowAll, "anime") || strings.Contains(lowAll, "illustration") || strings.Contains(lowAll, "1girl") {
					style = "illustration"
				}
				var tc OllamaToolCall
				tc.Function.Name = "generate_image"
				tc.Function.Arguments = map[string]any{
					"prompt": style + ", " + extractedPrompt,
					"style":  style,
					"width":  1024,
					"height": 1344,
					"steps":  30,
				}
				toolCalls = append(toolCalls, tc)
				res.HallucinatedToolRescued = true
			}
		}
	}

	res.ToolCalls = toolCalls
	return res
}

func emitDoneStats(reqStart time.Time, totalEvalCount int, totalEvalNs int64, emit func(map[string]any)) {
	elapsed := time.Since(reqStart).Seconds()
	tokPerSec := 0.0
	if totalEvalNs > 0 && totalEvalCount > 0 {
		tokPerSec = float64(totalEvalCount) / (float64(totalEvalNs) / 1e9)
	} else if elapsed > 0 && totalEvalCount > 0 {
		tokPerSec = float64(totalEvalCount) / elapsed
	}
	emit(map[string]any{
		"type":        "done",
		"elapsed_sec": elapsed,
		"tokens":      totalEvalCount,
		"tok_per_sec": tokPerSec,
	})
}

// toolCallLoopTracker detects consecutive or oscillating identical tool invocations
// when no file edits occur in between, preventing models from burning turns on repeated failures.
type toolCallLoopTracker struct {
	lastSig        string
	lastFailed     bool
	consec         int
	recent         []string
	lastMutatedSig string
}

func isMutatingToolName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "write_file", "write", "edit_file", "edit":
		return true
	}
	return false
}

func toolCallSignature(name string, args map[string]any) string {
	raw, _ := json.Marshal(args)
	return strings.ToLower(strings.TrimSpace(name)) + ":" + string(raw)
}

func (t *toolCallLoopTracker) check(name string, args map[string]any) (blocked bool, repeats int) {
	sig := toolCallSignature(name, args)
	if sig == t.lastSig {
		t.consec++
	} else {
		t.lastSig = sig
		t.consec = 1
		t.lastFailed = false
	}
	if isMutatingToolName(name) && t.consec == 1 {
		// Workspace changed; clear prior read-only checks so post-edit re-tests are never penalized.
		t.recent = []string{sig}
		t.lastMutatedSig = sig
		return false, 0
	}
	t.recent = append(t.recent, sig)
	if len(t.recent) > 12 {
		t.recent = t.recent[len(t.recent)-12:]
	}
	windowHits := 0
	for _, item := range t.recent {
		if item == sig {
			windowHits++
		}
	}
	// Block immediately on 2nd consecutive identical call if previous call errored; otherwise on 3rd consec or 4th in window.
	if (t.lastFailed && t.consec >= 2) || t.consec >= 3 || windowHits >= 4 {
		repeats = t.consec
		if windowHits > repeats {
			repeats = windowHits
		}
		return true, repeats
	}
	return false, 0
}

func (t *toolCallLoopTracker) recordResult(output string) {
	trimmed := strings.TrimSpace(output)
	t.lastFailed = strings.HasPrefix(trimmed, "Exit error:") || strings.HasPrefix(trimmed, "Error ")
}

func buildCycleToolsSummary(tools []StoredToolLog, cycle int) string {
	if len(tools) == 0 {
		return ""
	}
	var edited []string
	seen := map[string]bool{}
	bashCount := 0
	readCount := 0
	searchCount := 0
	for _, t := range tools {
		low := strings.ToLower(t.Name)
		switch low {
		case "write_file", "write", "edit_file", "edit":
			if m, ok := t.Args.(map[string]any); ok {
				p, _ := m["path"].(string)
				if p == "" {
					p, _ = m["file_path"].(string)
				}
				if p != "" && !seen[p] {
					seen[p] = true
					edited = append(edited, "`"+filepath.Base(p)+"`")
				}
			}
		case "bash", "sh":
			bashCount++
		case "find_files", "find", "glob", "grep_search", "grep", "search", "list_files", "list", "ls":
			searchCount++
		case "read_file", "read":
			readCount++
		}
	}
	var parts []string
	if len(edited) > 0 {
		if len(edited) > 6 {
			edited = edited[:6]
		}
		parts = append(parts, "menulis/mengubah "+strings.Join(edited, ", "))
	}
	if searchCount > 0 {
		parts = append(parts, fmt.Sprintf("%d pencarian/list", searchCount))
	}
	if readCount > 0 {
		parts = append(parts, fmt.Sprintf("%d baca file", readCount))
	}
	if bashCount > 0 {
		parts = append(parts, fmt.Sprintf("%d perintah bash", bashCount))
	}
	if len(parts) == 0 {
		parts = append(parts, fmt.Sprintf("%d tool calls", len(tools)))
	}
	return fmt.Sprintf("**Ringkasan Cycle %d:** Selesai %s. Melanjutkan ke langkah berikutnya pada Cycle %d...", cycle, strings.Join(parts, ", "), cycle+1)
}

func executeTrackedToolCall(
	ctx context.Context,
	tracker *toolCallLoopTracker,
	cwd, currentTurnPastedPath, latestVideoRefPath, tName string,
	tArgs map[string]any,
	timeout time.Duration,
	emit func(map[string]any),
) (tr ToolResult, modelContent string, blocked bool) {
	if tracker != nil {
		if isLoop, repeats := tracker.check(tName, tArgs); isLoop {
			wsTree := listWorkspaceTreeCompact(cwd, 30)
			plan := readGoalPlanMarkdown(cwd)
			planHint := ""
			if plan != "" {
				planHint = "\nCurrent .goal_plan.md:\n" + plan + "\n"
			}
			modelContent = fmt.Sprintf(
				"Notice [LOOP GUARD]: Duplicate call to '%s' skipped (%dx without file changes). Do NOT repeat this command.\nActive Working Directory: %s\nExisting project files:\n%s\n%sChoose a different file to read with read_file, or write/edit code directly with write_file / edit_file.",
				tName, repeats, cwd, wsTree, planHint,
			)
			uiOut := fmt.Sprintf("Dilewati (pemanggilan '%s' sudah dijalankan %dx tanpa perubahan file — daftar file workspace aktual dikirim ulang ke model)", tName, repeats)
			emit(map[string]any{"type": "tool_start", "name": tName, "args": tArgs})
			emit(map[string]any{
				"type":   "tool_end",
				"name":   tName,
				"output": uiOut,
			})
			return ToolResult{Output: uiOut}, modelContent, true
		}
	}
	emit(map[string]any{"type": "tool_start", "name": tName, "args": tArgs})
	tr = runToolCallWithTimeout(ctx, cwd, currentTurnPastedPath, latestVideoRefPath, tName, tArgs, timeout)
	if tracker != nil {
		tracker.recordResult(tr.Output)
	}
	emit(map[string]any{
		"type":      "tool_end",
		"name":      tName,
		"output":    tr.Output,
		"media_url": tr.MediaURL,
		"file_url":  tr.FileURL,
		"file_path": tr.FilePath,
	})
	return tr, tr.Output, false
}

func executeChatLoop(ctx context.Context, model, cwd string, enableExec, enableMCP bool, reqMsgs []ChatMessage, emit func(map[string]any)) {
	executeChatLoopWithAgent(ctx, model, cwd, enableExec, enableMCP, "", reqMsgs, emit)
}

func executeChatLoopWithAgent(ctx context.Context, model, cwd string, enableExec, enableMCP bool, agent string, reqMsgs []ChatMessage, emit func(map[string]any)) {
	if model == "" {
		model = "qwen3.5-abliterated:32k"
	}
	cwd = resolveChatWorkspace(cwd)

	var activeTools []map[string]any
	if enableExec {
		activeTools = append(activeTools, execTools...)
	}
	if enableMCP {
		activeTools = append(activeTools, mcpTools...)
	}

	currentTurnPastedPath, latestVideoRefPath := extractChatImageRefs(reqMsgs)

	choiceHint := `When you offer recommendations or ask the user to choose between approaches or next steps, end your reply with <!-- CHOICES: ["Option A", "Option B"] --> (2-5 concise clickable options in the user's language) so the UI can show choice buttons.`
	agentPersona := buildAgentPersonaPrompt(agent)
	var msgs []ChatMessage
	if len(activeTools) == 0 {
		sysContent := "You are a helpful, intelligent local AI assistant. Answer naturally in the user's language (Indonesian or English) using clean Markdown (headings, bullet lists, GFM tables, blockquotes, and fenced code blocks with language tags). " + choiceHint
		if agentPersona != "" {
			sysContent += "\n\n" + agentPersona
		}
		msgs = append(msgs, ChatMessage{
			Role:    "system",
			Content: sysContent,
		})
	} else {
		var sysParts []string
		if agentsBytes, err := os.ReadFile("/home/rizal/.config/opencode/AGENTS.md"); err == nil {
			sysParts = append(sysParts, string(agentsBytes))
		}
		if agentPersona != "" {
			sysParts = append(sysParts, agentPersona)
		}
		sysParts = append(sysParts, fmt.Sprintf("Active Working Directory: %s\nFormat all responses in clean Markdown (headings, lists, GFM tables, fenced code blocks).", cwd))
		if enableExec {
			sysParts = append(sysParts, "Terminal & File tools (`bash`, `list_files`, `find_files`, `grep_search`, `read_file`, `edit_file`, `write_file`) are enabled. Use them when the user asks to explore the workspace, search code, run commands, or generate/edit code files.")
		}
		if enableMCP {
			sysParts = append(sysParts, "MCP media tools (`generate_image`, `generate_video`) are connected via `local-image-gen` (ComfyUI). CRITICAL RULE: When the user asks to create, make, fix, or regenerate an image/picture/photo, you MUST invoke the `generate_image` tool call FIRST. NEVER pretend to generate an image by writing a fake `File Path: /home/rizal/Documents/chat/web_gen_...` or `Technical Breakdown` table in plain text without calling `generate_image`. Only pass `reference_image_path=\"auto\"` if the user attached an image in the CURRENT message (or pass an explicit file path only when editing that specific file); for a new image or regenerating from the same prompt, set `reference_image_path=\"none\"` so it runs fresh txt2img with a new random seed. For img2img: use `denoise` 0.15-0.30 for subtle 1:1 face/hand fixes in the same style, and 0.58-0.68 for style transfer (e.g. photo to Ghibli/anime with `style=\"illustration\"`) or outfit/scene edits. After `generate_image` finishes, keep your explanation brief (1-3 sentences) instead of writing a new PRD.")
		}
		sysParts = append(sysParts, choiceHint)
		msgs = append(msgs, ChatMessage{Role: "system", Content: strings.Join(sysParts, "\n\n")})
	}
	msgs = append(msgs, reqMsgs...)

	reqStart := time.Now()
	totalEvalCount := 0
	totalEvalNs := int64(0)
	client := &http.Client{Timeout: 15 * time.Minute}
	loopTracker := &toolCallLoopTracker{}
	blockedToolCount := 0
	mediaGenCount := 0
	lastUserPromptLow := ""
	for i := len(reqMsgs) - 1; i >= 0; i-- {
		if reqMsgs[i].Role == "user" {
			lastUserPromptLow = strings.ToLower(reqMsgs[i].Content)
			break
		}
	}
	allowMultiMedia := strings.Contains(lastUserPromptLow, "variasi") ||
		strings.Contains(lastUserPromptLow, "beberapa") ||
		strings.Contains(lastUserPromptLow, "multiple") ||
		strings.Contains(lastUserPromptLow, "2 gambar") ||
		strings.Contains(lastUserPromptLow, "3 gambar") ||
		strings.Contains(lastUserPromptLow, "4 gambar")
	maxMediaPerTurn := 1
	if allowMultiMedia {
		maxMediaPerTurn = 3
	}

	for turn := 0; turn < 8; turn++ {
		options := map[string]any{"num_predict": 4096}
		if len(activeTools) > 0 {
			options = map[string]any{
				"temperature":      0.6,
				"top_p":            0.95,
				"top_k":            20,
				"repeat_penalty":   1.12,
				"repeat_last_n":    128,
				"presence_penalty": 0.4,
				"num_predict":      3072,
			}
		}
		trRes := streamOneOllamaTurn(ctx, client, model, msgs, activeTools, options, 180000, 180000, turn, enableMCP, reqMsgs, emit)
		totalEvalCount += trRes.EvalCount
		totalEvalNs += trRes.EvalNs
		if trRes.Fatal {
			emit(map[string]any{"type": "error", "error": trRes.ErrMsg})
			return
		}
		if trRes.LoopAborted || trRes.StallAbort {
			if turn > 0 && len(trRes.ToolCalls) == 0 {
				break
			}
			reason := "Model terjebak loop token terlalu panjang dan dihentikan otomatis."
			if trRes.StallAbort {
				reason = "Model macet terlalu lama tanpa output dan dihentikan otomatis. Coba kirim ulang atau mulai Chat Baru."
			}
			emit(map[string]any{"type": "error", "error": reason})
			return
		}
		if len(trRes.ToolCalls) == 0 {
			break
		}

		assistantTurnContent := trRes.Content
		if trRes.HallucinatedToolRescued {
			// Clear the hallucinated "[Executed tool ...]" plain text before running the real tool
			emit(map[string]any{"type": "reset_content"})
			assistantTurnContent = ""
		}

		msgs = append(msgs, ChatMessage{
			Role:      "assistant",
			Content:   assistantTurnContent,
			Thinking:  trRes.Thinking,
			ToolCalls: trRes.ToolCalls,
		})

		for _, tc := range trRes.ToolCalls {
			tName := tc.Function.Name
			tArgs := tc.Function.Arguments
			if tArgs == nil {
				tArgs = map[string]any{}
			}
			if tName == "generate_image" || tName == "generate_video" {
				if mediaGenCount >= maxMediaPerTurn {
					msgs = append(msgs, ChatMessage{
						Role:     "tool",
						ToolName: tName,
						Content:  fmt.Sprintf("STOP: Already generated %d media output(s) for this turn. Do NOT call %s again unless the user explicitly asked for multiple variations. Summarize the generated image path and reply to the user now.", mediaGenCount, tName),
					})
					continue
				}
				mediaGenCount++
			}
			if tName == "generate_image" {
				prepareGenerateImageArgs(currentTurnPastedPath, tArgs)
			}
			_, modelOut, blocked := executeTrackedToolCall(ctx, loopTracker, cwd, currentTurnPastedPath, latestVideoRefPath, tName, tArgs, 90*time.Second, emit)
			msgs = append(msgs, ChatMessage{
				Role:     "tool",
				ToolName: tName,
				Content:  modelOut,
			})
			if blocked {
				blockedToolCount++
				if blockedToolCount >= 3 {
					emit(map[string]any{"type": "error", "error": "Model terjebak mengulang pemanggilan tool yang sama tanpa progres dan dihentikan otomatis."})
					emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
					return
				}
			}
		}
		if trRes.XMLParseErr {
			emit(map[string]any{"type": "content", "text": "Gambar selesai dirender sesuai permintaan Anda."})
			break
		}
	}

	emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
}

func executeGoalLoop(ctx context.Context, model, cwd string, enableMCP bool, goalPrompt string, reqMsgs []ChatMessage, maxCycles int, emit func(map[string]any)) {
	executeGoalLoopWithID(ctx, "", model, cwd, enableMCP, goalPrompt, reqMsgs, maxCycles, emit)
}

func executeGoalLoopWithID(ctx context.Context, chatID, model, cwd string, enableMCP bool, goalPrompt string, reqMsgs []ChatMessage, maxCycles int, emit func(map[string]any)) {
	executeGoalLoopWithSteering(ctx, chatID, model, cwd, enableMCP, goalPrompt, reqMsgs, maxCycles, nil, emit)
}

func executeGoalLoopWithSteering(ctx context.Context, chatID, model, cwd string, enableMCP bool, goalPrompt string, reqMsgs []ChatMessage, maxCycles int, drainSteering func() []string, emit func(map[string]any)) {
	executeGoalLoopWithAgent(ctx, chatID, model, cwd, enableMCP, "", goalPrompt, reqMsgs, maxCycles, drainSteering, emit)
}

func executeGoalLoopWithAgent(ctx context.Context, chatID, model, cwd string, enableMCP bool, agent, goalPrompt string, reqMsgs []ChatMessage, maxCycles int, drainSteering func() []string, emit func(map[string]any)) {
	if model == "" {
		model = "qwen3.5-abliterated:32k"
	}
	if maxCycles <= 0 {
		maxCycles = defaultGoalCycles
	}
	cwd = resolveChatWorkspace(cwd)

	var activeTools []map[string]any
	activeTools = append(activeTools, execTools...)
	if enableMCP {
		activeTools = append(activeTools, mcpTools...)
	}

	currentTurnPastedPath, latestVideoRefPath := extractChatImageRefs(reqMsgs)

	agentsMD := ""
	if agentsBytes, err := os.ReadFile("/home/rizal/.config/opencode/AGENTS.md"); err == nil {
		agentsMD = string(agentsBytes)
	}

	sysContent := buildGoalSystemPrompt(cwd, agentsMD, enableMCP)
	if persona := buildAgentPersonaPrompt(agent); persona != "" {
		sysContent += "\n\n" + persona
	}
	var msgs []ChatMessage
	msgs = append(msgs, ChatMessage{
		Role:    "system",
		Content: sysContent,
	})
	msgs = append(msgs, reqMsgs...)
	if strings.TrimSpace(goalPrompt) != "" && len(reqMsgs) > 0 {
		lastIdx := len(msgs) - 1
		if msgs[lastIdx].Role == "user" {
			msgs[lastIdx].Content = goalPrompt
		}
	}
	// Ground the model immediately in actual workspace files & .goal_plan.md (prevents blind path guessing / loops).
	if initialTree := listWorkspaceTreeCompact(cwd, 40); initialTree != "(no project files yet)" && initialTree != "(empty)" {
		planNow := readGoalPlanMarkdown(cwd)
		wsSnap := fmt.Sprintf("[CURRENT WORKSPACE STATE]\nActive Working Directory: %s\nProject files already on disk:\n%s", cwd, initialTree)
		if planNow != "" {
			wsSnap += "\n\nExisting .goal_plan.md checklist:\n" + planNow
		}
		msgs = append(msgs, ChatMessage{Role: "user", Content: wsSnap})
	}

	reqStart := time.Now()
	totalEvalCount := 0
	totalEvalNs := int64(0)
	totalToolTurns := 0
	totalToolCalls := 0
	audited := false
	consecutiveErrs := 0
	client := &http.Client{Timeout: 30 * time.Minute}
	numCtx := inferGoalNumCtx(model)
	loopTracker := &toolCallLoopTracker{}

	emitProgress := func(cycle int, status, phase string, active bool) {
		ev := map[string]any{
			"type":       "goal_progress",
			"active":     active,
			"status":     status,
			"cycle":      cycle,
			"max_cycles": maxCycles,
			"tool_turns": totalToolTurns,
			"tool_calls": totalToolCalls,
			"phase":      phase,
		}
		if pm := readGoalPlanMarkdown(cwd); pm != "" {
			ev["plan_markdown"] = pm
		}
		emit(ev)
	}

	recordCheckpoint := func(cycle int, label string) {
		if strings.TrimSpace(chatID) == "" {
			return
		}
		if cp, err := createGoalCheckpoint(chatID, cwd, cycle, label); err == nil && cp != nil {
			emit(map[string]any{
				"type":       "goal_checkpoint",
				"checkpoint": *cp,
			})
		}
	}

	recordCheckpoint(0, "Baseline before /goal")

	finalStatus := "max_cycles"
	for cycle := 1; cycle <= maxCycles; cycle++ {
		if ctx.Err() != nil {
			finalStatus = "cancelled"
			break
		}
		msgs = compactGoalMessagesWithPlan(msgs, 68000, cwd)

		phaseLabel := fmt.Sprintf("Cycle %d/%d: Implementing goal", cycle, maxCycles)
		if audited {
			phaseLabel = fmt.Sprintf("Cycle %d/%d: Final verification audit", cycle, maxCycles)
		}
		emit(map[string]any{
			"type":       "goal_cycle_start",
			"cycle":      cycle,
			"max_cycles": maxCycles,
			"phase":      phaseLabel,
		})
		emitProgress(cycle, "running", phaseLabel, true)

		var lastText string
		hitTurnLimit := false
		hadTurnFailure := false
		loopRecoveryReason := ""
		cycleBlockedCalls := 0
		cycleMutated := false
		cycleCallsBefore := totalToolCalls

		for turn := 0; turn < goalTurnsPerCycle; turn++ {
			if ctx.Err() != nil {
				finalStatus = "cancelled"
				break
			}
			if drainSteering != nil {
				if steers := drainSteering(); len(steers) > 0 {
					var sb strings.Builder
					sb.WriteString("[USER LIVE STEERING DIRECTION - HIGH PRIORITY]\n")
					sb.WriteString("The user sent live mid-run direction while this goal is executing:\n")
					for _, stTxt := range steers {
						sb.WriteString("- " + stTxt + "\n")
					}
					sb.WriteString("Acknowledge in 1 short sentence and immediately pivot your next tool calls to follow this direction.")
					msgs = append(msgs, ChatMessage{Role: "user", Content: sb.String()})
					loopTracker = &toolCallLoopTracker{}
					consecutiveErrs = 0
				}
			}
			options := map[string]any{
				"temperature":      0.4,
				"top_p":            0.95,
				"top_k":            20,
				"repeat_penalty":   1.12,
				"repeat_last_n":    128,
				"presence_penalty": 0.3,
				"num_predict":      8192,
				"num_ctx":          numCtx,
			}
			trRes := streamOneOllamaTurn(ctx, client, model, msgs, activeTools, options, 240000, 300000, turn, enableMCP, reqMsgs, emit)
			totalEvalCount += trRes.EvalCount
			totalEvalNs += trRes.EvalNs

			if ctx.Err() != nil {
				finalStatus = "cancelled"
				break
			}
			if trRes.Fatal {
				consecutiveErrs++
				hadTurnFailure = true
				emit(map[string]any{"type": "content", "text": fmt.Sprintf("\n\n> ⚠️ **[Goal Recovery %d/3]** Koneksi/model bermasalah (`%s`), mencoba melanjutkan...\n\n", consecutiveErrs, trRes.ErrMsg)})
				if consecutiveErrs >= 3 {
					finalStatus = "failed"
					emit(map[string]any{"type": "error", "error": "Goal dihentikan setelah 3x kegagalan beruntun: " + trRes.ErrMsg})
					emitProgress(cycle, finalStatus, "Failed after repeated errors", false)
					emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
					return
				}
				break
			}
			if trRes.LoopAborted || trRes.StallAbort {
				consecutiveErrs++
				hadTurnFailure = true
				why := "loop token terdeteksi"
				if trRes.StallAbort {
					why = "model macet tanpa output"
				}
				emit(map[string]any{"type": "content", "text": fmt.Sprintf("\n\n> ⚠️ **[Goal Recovery %d/3]** %s, mereset langkah dan melanjutkan...\n\n", consecutiveErrs, why)})
				if consecutiveErrs >= 3 {
					finalStatus = "failed"
					emit(map[string]any{"type": "error", "error": "Goal dihentikan setelah 3x macet/loop beruntun (" + why + ")."})
					emitProgress(cycle, finalStatus, "Failed after repeated stalls", false)
					emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
					return
				}
				msgs = append(msgs, ChatMessage{
					Role:    "user",
					Content: "[SYSTEM RECOVERY] Your previous turn timed out or looped before finishing a tool call. Immediately call a concise `write_file` (or split large files into smaller pieces and expand with `edit_file`) without long preamble.",
				})
				break
			}

			lastText = trRes.Content

			if len(trRes.ToolCalls) == 0 {
				consecutiveErrs = 0
				break
			}

			totalToolTurns++
			msgs = append(msgs, ChatMessage{
				Role:      "assistant",
				Content:   trRes.Content,
				Thinking:  trRes.Thinking,
				ToolCalls: trRes.ToolCalls,
			})

			for _, tc := range trRes.ToolCalls {
				if ctx.Err() != nil {
					finalStatus = "cancelled"
					break
				}
				tName := tc.Function.Name
				tArgs := tc.Function.Arguments
				if tArgs == nil {
					tArgs = map[string]any{}
				}
				if tName == "generate_image" {
					prepareGenerateImageArgs(currentTurnPastedPath, tArgs)
				}
				totalToolCalls++
				_, modelOut, blocked := executeTrackedToolCall(ctx, loopTracker, cwd, currentTurnPastedPath, latestVideoRefPath, tName, tArgs, 300*time.Second, emit)
				if !blocked && isMutatingToolName(tName) {
					cycleMutated = true
				}
				msgs = append(msgs, ChatMessage{
					Role:     "tool",
					ToolName: tName,
					Content:  modelOut,
				})
				if blocked {
					cycleBlockedCalls++
				}
			}
			emitProgress(cycle, "running", phaseLabel, true)

			if finalStatus == "cancelled" {
				break
			}
			if cycleBlockedCalls >= 2 {
				consecutiveErrs++
				hadTurnFailure = true
				loopRecoveryReason = "model mencoba mengulang perintah yang identik"
				emit(map[string]any{"type": "content", "text": fmt.Sprintf("\n\n> ⚠️ **[Goal Recovery %d/3]** %s, mengalihkan strategi dengan daftar file workspace aktual...\n\n", consecutiveErrs, loopRecoveryReason)})
				if consecutiveErrs >= 3 {
					finalStatus = "failed"
					emit(map[string]any{"type": "error", "error": "Goal dihentikan setelah 3x cycle terjebak mengulang tool call yang sama tanpa perubahan file."})
					emitProgress(cycle, finalStatus, "Failed after repeated tool loops", false)
					emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
					return
				}
				break
			}
			if turn == goalTurnsPerCycle-1 {
				hitTurnLimit = true
			}
		}

		if totalToolCalls > cycleCallsBefore {
			recordCheckpoint(cycle, fmt.Sprintf("Cycle %d snapshot", cycle))
		}

		if finalStatus == "cancelled" || ctx.Err() != nil {
			finalStatus = "cancelled"
			break
		}

		if !hadTurnFailure {
			if cycleMutated || (cycleBlockedCalls == 0 && strings.TrimSpace(lastText) != "") {
				consecutiveErrs = 0
			}
			if strings.Contains(lastText, goalCancelledMarker) {
				finalStatus = "cancelled"
				break
			}
			if strings.Contains(lastText, goalCompleteMarker) {
				if !audited {
					audited = true
					emitProgress(cycle, "auditing", "Self-audit: verifying PRD requirements & tests", true)
					emit(map[string]any{"type": "content", "text": "\n\n> 🔍 **[Goal Audit]** Memverifikasi seluruh checklist PRD, build, dan test sebelum menyatakan selesai...\n\n"})
					msgs = append(msgs,
						ChatMessage{Role: "assistant", Content: lastText},
						ChatMessage{Role: "user", Content: "[GOAL SYSTEM - MANDATORY AUDIT]\nBefore finalizing, strictly audit your work against the original goal/PRD:\n1. Inspect the workspace with list_files/read_file to verify all required files exist and contain complete implementations (no TODOs/stubs).\n2. Run build, typecheck, and test commands via bash and verify they pass.\n3. If anything is missing or failing, fix it now.\n4. Only when 100% verified and complete, reply with your concise final summary and include " + goalCompleteMarker + "."},
					)
					continue
				}
				finalStatus = "completed"
				break
			}
		}

		if cycle == maxCycles {
			finalStatus = "max_cycles"
			break
		}

		wsRefresh := listWorkspaceTreeCompact(cwd, 35)
		planRefresh := readGoalPlanMarkdown(cwd)
		planBlock := ""
		if planRefresh != "" {
			planBlock = "\nCurrent .goal_plan.md checklist:\n" + planRefresh + "\n"
		}
		var continuePrompt string
		if hadTurnFailure && loopRecoveryReason != "" {
			continuePrompt = fmt.Sprintf("[GOAL SYSTEM - TOOL LOOP RECOVERY]\nDo NOT repeat the previous find/cat/bash command again.\nActive Working Directory: %s\nActual project files:\n%s\n%sBegin this cycle with 1-2 brief sentences stating what exact next step you will do, then write/edit files or test directly. Include %s only when everything is built and passing.", cwd, wsRefresh, planBlock, goalCompleteMarker)
		} else if hadTurnFailure {
			continuePrompt = fmt.Sprintf("[GOAL SYSTEM - RECOVER & CONTINUE]\nThe previous step stalled or errored.\nActive Working Directory: %s\n%sFirst state in 1-2 short sentences what you are working on next, then inspect/edit/verify files. Include %s only when everything is built and passing.", cwd, planBlock, goalCompleteMarker)
		} else if hitTurnLimit {
			continuePrompt = fmt.Sprintf("[GOAL SYSTEM - CONTINUE CYCLE %d]\nFirst write 1-2 short sentences summarizing what you finished in Cycle %d and what specific task you will build/verify next in Cycle %d.%s Then immediately continue invoking tools. Only output %s after everything is built and verified.", cycle+1, cycle, cycle+1, planBlock, goalCompleteMarker)
		} else {
			if strings.TrimSpace(lastText) != "" {
				msgs = append(msgs, ChatMessage{Role: "assistant", Content: lastText})
			}
			continuePrompt = fmt.Sprintf("[GOAL SYSTEM - CONTINUE CYCLE %d]\nYou paused without calling tools and without completing the goal (%s was not emitted). First state in 1-2 short sentences what you will execute next in Cycle %d, then immediately call tools to implement and verify it.%s Only output %s once ALL requirements are implemented and verified.", cycle+1, goalCompleteMarker, cycle+1, planBlock, goalCompleteMarker)
		}
		msgs = append(msgs, ChatMessage{Role: "user", Content: continuePrompt})
	}

	switch finalStatus {
	case "completed":
		emitProgress(maxCycles, "completed", "Goal completed & verified", false)
	case "cancelled":
		emitProgress(maxCycles, "cancelled", "Goal cancelled by user", false)
	case "max_cycles":
		emit(map[string]any{"type": "content", "text": fmt.Sprintf("\n\n> ⏸️ **[Goal Paused]** Mencapai batas %d cycle otonom. Kirim `/goal lanjutkan` untuk meneruskan.\n\n", maxCycles)})
		emitProgress(maxCycles, "max_cycles", fmt.Sprintf("Paused after %d cycles", maxCycles), false)
	}
	emitDoneStats(reqStart, totalEvalCount, totalEvalNs, emit)
}

func HandleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 25*1024*1024)
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	executeChatLoop(r.Context(), req.Model, req.Cwd, req.EnableExec, req.EnableMCP, req.Messages, func(ev map[string]any) {
		writeNDJSON(w, flusher, ev)
	})
}

var allowedServeRoots = []string{
	"/home/rizal/Documents/",
	"/tmp/",
}

func isSensitivePath(cleanPath string) bool {
	base := strings.ToLower(filepath.Base(cleanPath))
	if base == "auth_config.json" || base == "initial_credentials.txt" || base == ".git-credentials" {
		return true
	}
	for _, seg := range strings.Split(filepath.ToSlash(cleanPath), "/") {
		if seg == "" || seg == "." || seg == ".." {
			continue
		}
		if strings.HasPrefix(seg, ".") {
			return true
		}
	}
	return false
}

func safeServeAllowedPath(w http.ResponseWriter, r *http.Request, rawPath string, mediaOnly bool, allowInline bool) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	clean := filepath.Clean(strings.TrimSpace(rawPath))
	if clean == "" || clean == "." || !filepath.IsAbs(clean) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	realPath, err := filepath.EvalSymlinks(clean)
	if err != nil {
		if os.IsNotExist(err) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	realPath = filepath.Clean(realPath)
	allowed := false
	for _, root := range allowedServeRoots {
		realRoot, rerr := filepath.EvalSymlinks(strings.TrimSuffix(root, "/"))
		if rerr == nil {
			root = filepath.Clean(realRoot) + "/"
		}
		if strings.HasPrefix(realPath, root) {
			allowed = true
			break
		}
	}
	if !allowed || isSensitivePath(realPath) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	info, err := os.Stat(realPath)
	if err != nil || info.IsDir() {
		http.NotFound(w, r)
		return
	}
	if mediaOnly {
		ext := strings.ToLower(filepath.Ext(realPath))
		switch ext {
		case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm":
		default:
			http.Error(w, "Forbidden media type", http.StatusForbidden)
			return
		}
	}
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if allowInline && r.URL.Query().Get("inline") == "1" {
		w.Header().Set("Content-Security-Policy", "sandbox")
	} else if !mediaOnly {
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filepath.Base(realPath)))
	}
	http.ServeFile(w, r, realPath)
}

func HandleChatMedia(w http.ResponseWriter, r *http.Request) {
	safeServeAllowedPath(w, r, r.URL.Query().Get("path"), true, true)
}

func HandleChatDownload(w http.ResponseWriter, r *http.Request) {
	safeServeAllowedPath(w, r, r.URL.Query().Get("path"), false, true)
}

func isPathUnderAllowedRoots(realPath string) bool {
	realPath = filepath.Clean(realPath)
	for _, root := range allowedServeRoots {
		realRoot, rerr := filepath.EvalSymlinks(strings.TrimSuffix(root, "/"))
		if rerr == nil {
			root = filepath.Clean(realRoot) + "/"
		}
		if strings.HasPrefix(realPath+"/", root) || strings.HasPrefix(realPath, root) {
			return true
		}
	}
	return false
}

func handleChatWorkspaceZip(w http.ResponseWriter, r *http.Request, id string) {
	var sess ChatSession
	if run := getActiveRun(id); run != nil {
		sess = snapshotRunSession(run)
	} else {
		p := filepath.Join(getChatsDir(), id+".json")
		chatStoreMu.Lock()
		b, err := os.ReadFile(p)
		chatStoreMu.Unlock()
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Sesi chat belum tersimpan atau workspace kosong."})
			return
		}
		if json.Unmarshal(b, &sess) != nil {
			http.Error(w, "Invalid session", http.StatusInternalServerError)
			return
		}
	}

	cwd := strings.TrimSpace(sess.Cwd)
	if cwd == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Workspace belum memiliki folder."})
		return
	}
	realCwd, err := filepath.EvalSymlinks(cwd)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Folder workspace belum dibuat atau kosong."})
		return
	}
	realCwd = filepath.Clean(realCwd)
	if !isPathUnderAllowedRoots(realCwd) || isSensitivePath(realCwd) {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	type zipEntryItem struct {
		diskPath string
		relName  string
		size     int64
	}
	var entries []zipEntryItem
	addedDisk := map[string]bool{}
	addedRel := map[string]bool{}
	var totalBytes int64
	const maxFileBytes = 50 * 1024 * 1024
	const maxTotalBytes = 200 * 1024 * 1024

	skipDirs := map[string]bool{
		".git": true, "node_modules": true, ".venv": true, "venv": true,
		"__pycache__": true, ".next": true, "dist": true, "build": true,
	}

	_ = filepath.WalkDir(realCwd, func(fpath string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if fpath != realCwd && (strings.HasPrefix(name, ".") || skipDirs[strings.ToLower(name)]) {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if isSensitivePath(fpath) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil || info.Size() <= 0 || info.Size() > maxFileBytes {
			return nil
		}
		if totalBytes+info.Size() > maxTotalBytes {
			return filepath.SkipAll
		}
		rel, rerr := filepath.Rel(realCwd, fpath)
		if rerr != nil || strings.HasPrefix(rel, "..") {
			return nil
		}
		rel = filepath.ToSlash(rel)
		entries = append(entries, zipEntryItem{diskPath: fpath, relName: rel, size: info.Size()})
		addedDisk[fpath] = true
		addedRel[rel] = true
		totalBytes += info.Size()
		return nil
	})

	// Also collect any tool-produced files/media from this chat session not already in realCwd
	for _, m := range sess.Messages {
		for _, tl := range m.Tools {
			cand := strings.TrimSpace(tl.FilePath)
			if cand == "" && strings.Contains(tl.MediaURL, "path=") {
				if u, uerr := url.Parse(tl.MediaURL); uerr == nil {
					cand = u.Query().Get("path")
				}
			}
			if cand == "" {
				continue
			}
			realCand, rerr := filepath.EvalSymlinks(filepath.Clean(cand))
			if rerr != nil || addedDisk[realCand] || !isPathUnderAllowedRoots(realCand) || isSensitivePath(realCand) {
				continue
			}
			info, ierr := os.Stat(realCand)
			if ierr != nil || info.IsDir() || info.Size() <= 0 || info.Size() > maxFileBytes {
				continue
			}
			if totalBytes+info.Size() > maxTotalBytes {
				continue
			}
			relName := filepath.Base(realCand)
			if addedRel[relName] {
				relName = "media/" + relName
			}
			entries = append(entries, zipEntryItem{diskPath: realCand, relName: relName, size: info.Size()})
			addedDisk[realCand] = true
			addedRel[relName] = true
			totalBytes += info.Size()
		}
	}

	if len(entries) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Workspace masih kosong — belum ada file atau gambar untuk di-zip."})
		return
	}

	slug := slugifyWorkspaceName(sess.Title)
	if slug == "" {
		slug = slugifyWorkspaceName(filepath.Base(realCwd))
	}
	if slug == "" {
		slug = id
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", slug+".zip"))
	w.Header().Set("X-Content-Type-Options", "nosniff")

	zw := zip.NewWriter(w)
	for _, ent := range entries {
		f, ferr := os.Open(ent.diskPath)
		if ferr != nil {
			continue
		}
		zf, zerr := zw.Create(ent.relName)
		if zerr == nil {
			_, _ = io.Copy(zf, f)
		}
		_ = f.Close()
	}
	_ = zw.Close()
}

var chatStoreMu sync.Mutex

func getChatsDir() string {
	dir := "/home/rizal/Documents/ollama/ai-dashboard/chats"
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), "chats")
		if os.MkdirAll(cand, 0o755) == nil {
			return cand
		}
	}
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func safeChatID(id string) string {
	id = strings.TrimSpace(id)
	var b strings.Builder
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

type QueuedTurn struct {
	Model           string
	Cwd             string
	EnableExec      bool
	EnableMCP       bool
	Agent           string
	GoalMode        bool
	GoalPrompt      string
	AssistantMsgIdx int
}

type ChatRun struct {
	mu            sync.RWMutex
	Running       bool
	Deleted       bool
	Stopped       bool
	Cancel        context.CancelFunc
	Session       ChatSession
	Queue         []QueuedTurn
	ActiveMsgIdx  int
	SteeringQueue []string
}

var (
	activeRunsMu sync.RWMutex
	activeRuns   = map[string]*ChatRun{}
)

func getActiveRun(id string) *ChatRun {
	activeRunsMu.RLock()
	defer activeRunsMu.RUnlock()
	return activeRuns[id]
}

func saveSessionFile(s ChatSession) {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return
	}
	p := filepath.Join(getChatsDir(), s.ID+".json")
	tmp := p + ".tmp"
	chatStoreMu.Lock()
	if os.WriteFile(tmp, b, 0o644) == nil {
		_ = os.Rename(tmp, p)
	}
	chatStoreMu.Unlock()
}

// deepCopySession must be called while holding run.mu (read or write).
func deepCopySession(s ChatSession) ChatSession {
	cp := s
	if s.Goal != nil {
		g := *s.Goal
		if len(s.Goal.Checkpoints) > 0 {
			g.Checkpoints = make([]GoalCheckpoint, len(s.Goal.Checkpoints))
			copy(g.Checkpoints, s.Goal.Checkpoints)
			for ci := range g.Checkpoints {
				if len(g.Checkpoints[ci].FileStats) > 0 {
					fsCp := make([]GoalFileStat, len(g.Checkpoints[ci].FileStats))
					copy(fsCp, g.Checkpoints[ci].FileStats)
					g.Checkpoints[ci].FileStats = fsCp
				}
			}
		}
		cp.Goal = &g
	}
	cp.Messages = make([]StoredMessage, len(s.Messages))
	for i, m := range s.Messages {
		mc := m
		if len(m.Tools) > 0 {
			mc.Tools = make([]StoredToolLog, len(m.Tools))
			copy(mc.Tools, m.Tools)
		}
		if len(m.Images) > 0 {
			mc.Images = make([]string, len(m.Images))
			copy(mc.Images, m.Images)
		}
		if len(m.ChangedFiles) > 0 {
			mc.ChangedFiles = make([]GoalFileStat, len(m.ChangedFiles))
			copy(mc.ChangedFiles, m.ChangedFiles)
		}
		cp.Messages[i] = mc
	}
	return cp
}

// saveRunIfAlive persists the run unless the chat was deleted meanwhile.
func saveRunIfAlive(run *ChatRun, s ChatSession) {
	run.mu.RLock()
	deleted := run.Deleted
	run.mu.RUnlock()
	if deleted {
		return
	}
	saveSessionFile(s)
}

// recoverStaleChats fixes chats left as "running" after the server was restarted mid-generation.
func recoverStaleChats() {
	dir := getChatsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var s ChatSession
		if json.Unmarshal(b, &s) != nil || !s.Running {
			continue
		}
		s.Running = false
		if s.Goal != nil && s.Goal.Active {
			s.Goal.Active = false
			s.Goal.Status = "interrupted"
			s.Goal.LastPhase = "Interrupted by dashboard restart"
			s.Goal.UpdatedAt = time.Now().UnixMilli()
		}
		if n := len(s.Messages); n > 0 && s.Messages[n-1].Role == "assistant" {
			last := &s.Messages[n-1]
			if strings.TrimSpace(last.Content) == "" && last.Error == "" {
				last.Error = "Proses terputus karena server dashboard di-restart. Silakan kirim ulang pesannya."
			}
		}
		saveSessionFile(s)
	}
}

func init() {
	recoverStaleChats()
}

func snapshotRunSession(run *ChatRun) ChatSession {
	run.mu.RLock()
	defer run.mu.RUnlock()
	cp := deepCopySession(run.Session)
	cp.Running = run.Running
	return cp
}

type BackgroundSendRequest struct {
	ChatID     string        `json:"chat_id"`
	Title      string        `json:"title"`
	Model      string        `json:"model"`
	Cwd        string        `json:"cwd"`
	EnableExec bool          `json:"enable_exec"`
	EnableMCP  bool          `json:"enable_mcp"`
	Agent      string        `json:"agent,omitempty"`
	GoalMode   bool          `json:"goal_mode,omitempty"`
	Steer      bool          `json:"steer,omitempty"`
	Messages   []ChatMessage `json:"messages"`
	UserImages []string      `json:"user_images,omitempty"`
	UserPrompt string        `json:"user_prompt"`
	// TruncateFrom (optional) rewinds the conversation to this message index
	// before appending the new user turn (used by Edit & Retry checkpoints).
	TruncateFrom *int `json:"truncate_from,omitempty"`
}

func cancelActiveChatOrGoal(id string) bool {
	now := time.Now().UnixMilli()
	if run := getActiveRun(id); run != nil {
		run.mu.Lock()
		run.Stopped = true
		if run.Session.Goal != nil && run.Session.Goal.Active {
			run.Session.Goal.Active = false
			run.Session.Goal.CancelRequested = true
			run.Session.Goal.Status = "cancelled"
			run.Session.Goal.LastPhase = "Cancelled by user"
			run.Session.Goal.UpdatedAt = now
			run.Session.Goal.FinishedAt = now
		}
		cp := deepCopySession(run.Session)
		cancelFn := run.Cancel
		run.mu.Unlock()
		saveRunIfAlive(run, cp)
		if cancelFn != nil {
			cancelFn()
		}
		return true
	}

	p := filepath.Join(getChatsDir(), id+".json")
	chatStoreMu.Lock()
	defer chatStoreMu.Unlock()
	b, err := os.ReadFile(p)
	if err != nil {
		return false
	}
	var sess ChatSession
	if json.Unmarshal(b, &sess) != nil {
		return false
	}
	if sess.Goal != nil && sess.Goal.Active {
		sess.Goal.Active = false
		sess.Goal.CancelRequested = true
		sess.Goal.Status = "cancelled"
		sess.Goal.LastPhase = "Cancelled by user"
		sess.Goal.UpdatedAt = now
		sess.Goal.FinishedAt = now
		sess.Running = false
		out, _ := json.MarshalIndent(sess, "", "  ")
		_ = os.WriteFile(p, out, 0o644)
		return true
	}
	return false
}

func handleChatSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 25*1024*1024)
	var req BackgroundSendRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	id := safeChatID(req.ChatID)
	if id == "" {
		id = fmt.Sprintf("chat_%d", time.Now().UnixMilli())
	}

	now := time.Now().UnixMilli()
	parsedGoal := parseAgentAndGoalCommand(req.UserPrompt, req.GoalMode, req.Agent)
	if parsedGoal.IsCancel {
		cancelled := cancelActiveChatOrGoal(id)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":             true,
			"id":             id,
			"goal_cancelled": true,
			"was_running":    cancelled,
		})
		return
	}
	if parsedGoal.IsGoal {
		req.EnableExec = true
		req.GoalMode = true
	}
	if parsedGoal.Agent == "ai-slop" {
		req.EnableExec = true
	}
	req.Agent = parsedGoal.Agent

	// If this chat is already running, steer active /goal immediately or enqueue new turn!
	if prev := getActiveRun(id); prev != nil {
		prev.mu.Lock()
		if prev.Running && !prev.Stopped && !prev.Deleted && req.TruncateFrom != nil {
			prev.mu.Unlock()
			http.Error(w, "Chat sedang berjalan; hentikan dulu sebelum Edit/Retry", http.StatusConflict)
			return
		}
		if prev.Running && !prev.Stopped && !prev.Deleted {
			resolvedCwd := resolveSessionWorkspace(id, prev.Session.Cwd, req.Cwd, parsedGoal.Prompt)
			prev.Session.Model = req.Model
			prev.Session.Cwd = resolvedCwd
			prev.Session.EnableExec = req.EnableExec
			prev.Session.EnableMCP = req.EnableMCP
			prev.Session.UpdatedAt = now
			isLiveSteer := prev.Session.Goal != nil && prev.Session.Goal.Active && (req.Steer || !parsedGoal.IsGoal) && strings.TrimSpace(req.UserPrompt) != ""
			if isLiveSteer {
				steerText := strings.TrimSpace(req.UserPrompt)
				prev.Session.Messages = append(prev.Session.Messages, StoredMessage{
					Role:    "user",
					Content: "🎯 " + steerText,
					Images:  req.UserImages,
				})
				prev.Session.Messages = append(prev.Session.Messages, StoredMessage{
					Role:          "assistant",
					GoalCycle:     prev.Session.Goal.Cycle,
					GoalMaxCycles: prev.Session.Goal.MaxCycles,
					GoalPhase:     "Menerapkan arahan langsung user",
				})
				prev.ActiveMsgIdx = len(prev.Session.Messages) - 1
				prev.SteeringQueue = append(prev.SteeringQueue, steerText)
				cp := deepCopySession(prev.Session)
				prev.mu.Unlock()
				saveRunIfAlive(prev, cp)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "id": id, "cwd": resolvedCwd, "steered": true})
				return
			}
			prev.Session.Messages = append(prev.Session.Messages, StoredMessage{
				Role:    "user",
				Content: req.UserPrompt,
				Images:  req.UserImages,
			})
			prev.Session.Messages = append(prev.Session.Messages, StoredMessage{
				Role:    "assistant",
				Content: "",
				Queued:  true,
			})
			aIdx := len(prev.Session.Messages) - 1
			prev.Queue = append(prev.Queue, QueuedTurn{
				Model:           req.Model,
				Cwd:             resolvedCwd,
				EnableExec:      req.EnableExec,
				EnableMCP:       req.EnableMCP,
				Agent:           parsedGoal.Agent,
				GoalMode:        req.GoalMode,
				GoalPrompt:      parsedGoal.Prompt,
				AssistantMsgIdx: aIdx,
			})
			cp := deepCopySession(prev.Session)
			prev.mu.Unlock()
			saveRunIfAlive(prev, cp)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "id": id, "cwd": resolvedCwd, "queued": true})
			return
		}
		prev.mu.Unlock()
	}

	var sess ChatSession
	p := filepath.Join(getChatsDir(), id+".json")
	chatStoreMu.Lock()
	if raw, err := os.ReadFile(p); err == nil {
		_ = json.Unmarshal(raw, &sess)
	}
	chatStoreMu.Unlock()

	if sess.ID == "" {
		sess.ID = id
		sess.CreatedAt = now
	}
	if req.TruncateFrom != nil {
		cut := *req.TruncateFrom
		if cut < 0 || cut > len(sess.Messages) {
			http.Error(w, "truncate_from out of range", http.StatusBadRequest)
			return
		}
		sess.Messages = sess.Messages[:cut]
	}
	goalPrompt := parsedGoal.Prompt
	if req.GoalMode && goalPrompt == "" && len(sess.Messages) > 0 {
		goalPrompt = "Continue and complete the engineering goal/PRD discussed in this conversation until all requirements are implemented and verified."
	}
	if parsedGoal.Agent == "ai-slop" && goalPrompt == "" {
		req.UserPrompt = "/ai-slop Audit dan bersihkan seluruh AI slop, stub/placeholder, kode mati, duplikasi, serta kelemahan kualitas pada workspace saat ini."
	} else if parsedGoal.Agent == "brainstorm" && goalPrompt == "" {
		req.UserPrompt = "/brainstorm Mari brainstorm ide, arsitektur, UX, dan trade-off terbaik untuk proyek/topik ini."
	}
	titleHint := req.UserPrompt
	if goalPrompt != "" {
		titleHint = goalPrompt
	}
	if strings.TrimSpace(sess.Title) == "" {
		t := strings.TrimSpace(req.Title)
		if t == "" {
			clean := strings.Join(strings.Fields(titleHint), " ")
			if len(clean) > 40 {
				t = clean[:40] + "..."
			} else if clean != "" {
				t = clean
			} else {
				t = "New Chat"
			}
			if req.GoalMode && !strings.HasPrefix(t, "[GOAL]") {
				t = "[GOAL] " + t
			} else if parsedGoal.Agent == "ai-slop" && !strings.HasPrefix(t, "[SLOP]") {
				t = "[SLOP] " + t
			} else if parsedGoal.Agent == "brainstorm" && !strings.HasPrefix(t, "[BRAINSTORM]") {
				t = "[BRAINSTORM] " + t
			}
		}
		sess.Title = t
	}
	resolvedCwd := resolveSessionWorkspace(id, sess.Cwd, req.Cwd, titleHint)
	sess.Model = req.Model
	sess.Cwd = resolvedCwd
	sess.EnableExec = req.EnableExec
	sess.EnableMCP = req.EnableMCP
	sess.Agent = parsedGoal.Agent
	sess.UpdatedAt = now
	sess.Running = true
	if req.GoalMode {
		var prevCP []GoalCheckpoint
		if sess.Goal != nil && len(sess.Goal.Checkpoints) > 0 {
			prevCP = append(prevCP, sess.Goal.Checkpoints...)
		}
		sess.Goal = &GoalState{
			Active:      true,
			Status:      "running",
			Prompt:      goalPrompt,
			Cycle:       1,
			MaxCycles:   defaultGoalCycles,
			LastPhase:   "Initializing autonomous /goal",
			StartedAt:   now,
			UpdatedAt:   now,
			Agent:       parsedGoal.Agent,
			Checkpoints: prevCP,
		}
	}

	// Append user message and placeholder assistant message
	sess.Messages = append(sess.Messages, StoredMessage{
		Role:    "user",
		Content: req.UserPrompt,
		Images:  req.UserImages,
	})
	sess.Messages = append(sess.Messages, StoredMessage{
		Role:    "assistant",
		Content: "",
	})
	firstAssistantIdx := len(sess.Messages) - 1
	saveSessionFile(sess)

	ctx, cancel := context.WithCancel(context.Background())
	run := &ChatRun{
		Running: true,
		Cancel:  cancel,
		Session: sess,
		Queue: []QueuedTurn{
			{
				Model:           req.Model,
				Cwd:             resolvedCwd,
				EnableExec:      req.EnableExec,
				EnableMCP:       req.EnableMCP,
				Agent:           parsedGoal.Agent,
				GoalMode:        req.GoalMode,
				GoalPrompt:      goalPrompt,
				AssistantMsgIdx: firstAssistantIdx,
			},
		},
	}
	activeRunsMu.Lock()
	activeRuns[id] = run
	activeRunsMu.Unlock()

	go func() {
		defer cancel()
		for {
			run.mu.Lock()
			if len(run.Queue) == 0 || run.Stopped || run.Deleted {
				run.mu.Unlock()
				break
			}
			turn := run.Queue[0]
			run.Queue = run.Queue[1:]
			targetIdx := turn.AssistantMsgIdx
			run.ActiveMsgIdx = targetIdx
			if targetIdx >= 0 && targetIdx < len(run.Session.Messages) {
				run.Session.Messages[targetIdx].Queued = false
			}
			if turn.GoalMode {
				tNow := time.Now().UnixMilli()
				gp := turn.GoalPrompt
				var prevCP []GoalCheckpoint
				if run.Session.Goal != nil {
					if gp == "" {
						gp = run.Session.Goal.Prompt
					}
					if len(run.Session.Goal.Checkpoints) > 0 {
						prevCP = append(prevCP, run.Session.Goal.Checkpoints...)
					}
				}
				run.Session.Goal = &GoalState{
					Active:      true,
					Status:      "running",
					Prompt:      gp,
					Cycle:       1,
					MaxCycles:   defaultGoalCycles,
					LastPhase:   "Initializing autonomous /goal",
					StartedAt:   tNow,
					UpdatedAt:   tNow,
					Checkpoints: prevCP,
				}
			}
			var turnMsgs []ChatMessage
			for i := 0; i < targetIdx && i < len(run.Session.Messages); i++ {
				m := run.Session.Messages[i]
				// Skip errored or hallucinated fake web_gen_ turns from history so they don't poison future tool calls
				if m.Role == "assistant" && m.Error != "" && strings.TrimSpace(m.Content) == "" {
					continue
				}
				if m.Role == "assistant" && len(m.Tools) > 0 {
					var histCalls []OllamaToolCall
					var histToolRes []ChatMessage
					seenSig := map[string]bool{}
					for _, tl := range m.Tools {
						outTrim := strings.TrimSpace(tl.Output)
						// Never feed old loop-guard / duplicate warnings or noisy repeated errors back into future prompts
						if strings.Contains(outTrim, "LOOP GUARD") || strings.Contains(outTrim, "Dilewati (pemanggilan") {
							continue
						}
						argsMap := map[string]any{}
						if am, ok := tl.Args.(map[string]any); ok && am != nil {
							argsMap = am
						} else if tl.Args != nil {
							if raw, err := json.Marshal(tl.Args); err == nil {
								_ = json.Unmarshal(raw, &argsMap)
							}
						}
						argsRaw, _ := json.Marshal(argsMap)
						sig := tl.Name + ":" + string(argsRaw)
						if seenSig[sig] {
							continue
						}
						seenSig[sig] = true
						if len(outTrim) > 420 {
							outTrim = outTrim[:420] + "..."
						}
						var tc OllamaToolCall
						tc.Function.Name = tl.Name
						tc.Function.Arguments = argsMap
						histCalls = append(histCalls, tc)
						histToolRes = append(histToolRes, ChatMessage{
							Role:     "tool",
							ToolName: tl.Name,
							Content:  outTrim,
						})
					}
					if len(histCalls) > 12 {
						histCalls = histCalls[len(histCalls)-12:]
						histToolRes = histToolRes[len(histToolRes)-12:]
					}
					if len(histCalls) > 0 {
						turnMsgs = append(turnMsgs, ChatMessage{
							Role:      "assistant",
							ToolCalls: histCalls,
						})
						turnMsgs = append(turnMsgs, histToolRes...)
					}
					if strings.TrimSpace(m.Content) != "" && !strings.Contains(m.Content, "[Executed tool ") {
						turnMsgs = append(turnMsgs, ChatMessage{
							Role:    "assistant",
							Content: strings.TrimSpace(m.Content),
						})
					}
					continue
				}
				if m.Role == "assistant" && len(m.Tools) == 0 {
					if strings.Contains(m.Content, "[Executed tool ") ||
						(strings.Contains(m.Content, "web_gen_") && (strings.Contains(m.Content, "Technical Breakdown") || strings.Contains(m.Content, "Image generated successfully"))) {
						continue
					}
				}
				if strings.TrimSpace(m.Content) != "" || len(m.Images) > 0 {
					turnMsgs = append(turnMsgs, ChatMessage{
						Role:    m.Role,
						Content: m.Content,
						Images:  m.Images,
					})
				}
			}
			run.mu.Unlock()

			lastFlush := time.Now()
			emitFn := func(ev map[string]any) {
				run.mu.Lock()
				idx := run.ActiveMsgIdx
				if idx >= 0 && idx < len(run.Session.Messages) {
					msg := &run.Session.Messages[idx]
					switch ev["type"] {
					case "thinking":
						if s, ok := ev["text"].(string); ok {
							msg.Thinking += s
						}
					case "content":
						if s, ok := ev["text"].(string); ok {
							msg.Content += s
						}
					case "reset_content":
						msg.Content = ""
					case "tool_start":
						tName, _ := ev["name"].(string)
						msg.Tools = append(msg.Tools, StoredToolLog{
							Name: tName,
							Args: ev["args"],
						})
					case "tool_end":
						if len(msg.Tools) > 0 {
							tLast := &msg.Tools[len(msg.Tools)-1]
							if o, ok := ev["output"].(string); ok {
								tLast.Output = o
							}
							if m, ok := ev["media_url"].(string); ok {
								tLast.MediaURL = m
							}
							if fu, ok := ev["file_url"].(string); ok {
								tLast.FileURL = fu
							}
							if fp, ok := ev["file_path"].(string); ok {
								tLast.FilePath = fp
							}
						}
					case "goal_cycle_start":
						cyc, _ := ev["cycle"].(int)
						mc, _ := ev["max_cycles"].(int)
						ph, _ := ev["phase"].(string)
						if cyc <= 1 {
							msg.GoalCycle = 1
							msg.GoalMaxCycles = mc
							msg.GoalPhase = ph
						} else {
							if strings.TrimSpace(msg.Content) == "" && len(msg.Tools) > 0 {
								msg.Content = buildCycleToolsSummary(msg.Tools, cyc-1)
							}
							run.Session.Messages = append(run.Session.Messages, StoredMessage{
								Role:          "assistant",
								GoalCycle:     cyc,
								GoalMaxCycles: mc,
								GoalPhase:     ph,
							})
							run.ActiveMsgIdx = len(run.Session.Messages) - 1
						}
					case "goal_progress":
						if run.Session.Goal == nil {
							run.Session.Goal = &GoalState{}
						}
						g := run.Session.Goal
						if act, ok := ev["active"].(bool); ok {
							g.Active = act
						}
						if st, ok := ev["status"].(string); ok && st != "" {
							g.Status = st
						}
						if cyc, ok := ev["cycle"].(int); ok && cyc > 0 {
							g.Cycle = cyc
						}
						if mc, ok := ev["max_cycles"].(int); ok && mc > 0 {
							g.MaxCycles = mc
						}
						if tt, ok := ev["tool_turns"].(int); ok {
							g.TotalToolTurns = tt
						}
						if tc, ok := ev["tool_calls"].(int); ok {
							g.TotalToolCalls = tc
						}
						if ph, ok := ev["phase"].(string); ok {
							g.LastPhase = ph
							if ph != "" && msg.GoalCycle > 0 {
								msg.GoalPhase = ph
							}
						}
						if pm, ok := ev["plan_markdown"].(string); ok && pm != "" {
							g.PlanMarkdown = pm
						}
						g.UpdatedAt = time.Now().UnixMilli()
						if !g.Active && g.FinishedAt == 0 {
							g.FinishedAt = g.UpdatedAt
						}
					case "goal_checkpoint":
						if run.Session.Goal == nil {
							run.Session.Goal = &GoalState{}
						}
						if cpItem, ok := ev["checkpoint"].(GoalCheckpoint); ok {
							if cpItem.Cycle > 0 && len(cpItem.FileStats) > 0 {
								msg.ChangedFiles = append([]GoalFileStat(nil), cpItem.FileStats...)
							}
							run.Session.Goal.Checkpoints = append(run.Session.Goal.Checkpoints, cpItem)
							if len(run.Session.Goal.Checkpoints) > 40 {
								run.Session.Goal.Checkpoints = run.Session.Goal.Checkpoints[len(run.Session.Goal.Checkpoints)-40:]
							}
						}
					case "error":
						if e, ok := ev["error"].(string); ok {
							msg.Error = e
						}
					case "done":
						if el, ok := ev["elapsed_sec"].(float64); ok {
							msg.ElapsedSec = el
						}
						if tp, ok := ev["tok_per_sec"].(float64); ok {
							msg.TokPerSec = tp
						}
						if tk, ok := ev["tokens"].(int); ok {
							msg.Tokens = tk
						}
						if msg.CreatedAt <= 0 {
							msg.CreatedAt = time.Now().UnixMilli()
						}
						if msg.Tokens > 0 {
							usageEv := UsageEvent{
								Key:         fmt.Sprintf("chat:%s:%d", run.Session.ID, idx),
								TimestampMS: msg.CreatedAt,
								Model:       turn.Model,
								ChatID:      run.Session.ID,
								Tokens:      msg.Tokens,
								TokPerSec:   msg.TokPerSec,
								ElapsedSec:  msg.ElapsedSec,
								GoalCycle:   msg.GoalCycle,
								Agent:       turn.Agent,
							}
							go recordUsageEvent(usageEv)
						}
					}
				}
				run.Session.UpdatedAt = time.Now().UnixMilli()
				needFlush := time.Since(lastFlush) > 1500*time.Millisecond || ev["type"] == "tool_end" || ev["type"] == "goal_progress" || ev["type"] == "goal_checkpoint" || ev["type"] == "done" || ev["type"] == "error"
				var cp ChatSession
				if needFlush {
					cp = deepCopySession(run.Session)
				}
				run.mu.Unlock()
				if needFlush {
					lastFlush = time.Now()
					saveRunIfAlive(run, cp)
				}
			}

			drainSteerFn := func() []string {
				run.mu.Lock()
				defer run.mu.Unlock()
				if len(run.SteeringQueue) == 0 {
					return nil
				}
				out := append([]string(nil), run.SteeringQueue...)
				run.SteeringQueue = nil
				return out
			}

			if turn.GoalMode {
				executeGoalLoopWithAgent(ctx, id, turn.Model, turn.Cwd, turn.EnableMCP, turn.Agent, turn.GoalPrompt, turnMsgs, defaultGoalCycles, drainSteerFn, emitFn)
			} else {
				executeChatLoopWithAgent(ctx, turn.Model, turn.Cwd, turn.EnableExec, turn.EnableMCP, turn.Agent, turnMsgs, emitFn)
			}

			run.mu.Lock()
			finalIdx := run.ActiveMsgIdx
			if finalIdx >= 0 && finalIdx < len(run.Session.Messages) {
				last := &run.Session.Messages[finalIdx]
				if run.Stopped {
					last.Error = ""
					if strings.TrimSpace(last.Content) == "" {
						last.Content = "_Dihentikan oleh pengguna._"
					} else {
						last.Content += "\n\n_Dihentikan oleh pengguna._"
					}
					if run.Session.Goal != nil && run.Session.Goal.Active {
						run.Session.Goal.Active = false
						run.Session.Goal.Status = "cancelled"
						run.Session.Goal.CancelRequested = true
						run.Session.Goal.FinishedAt = time.Now().UnixMilli()
					}
				} else if strings.TrimSpace(last.Content) == "" && last.Error == "" && len(last.Tools) == 0 {
					last.Error = "Model tidak mengembalikan jawaban. Coba kirim ulang atau ganti model."
				}
			}
			cpDone := deepCopySession(run.Session)
			run.mu.Unlock()
			saveRunIfAlive(run, cpDone)
		}

		run.mu.Lock()
		run.Running = false
		run.Session.Running = false
		run.Session.UpdatedAt = time.Now().UnixMilli()
		if run.Session.Goal != nil && run.Session.Goal.Active {
			run.Session.Goal.Active = false
			if run.Stopped {
				run.Session.Goal.Status = "cancelled"
			} else if run.Session.Goal.Status == "running" || run.Session.Goal.Status == "auditing" {
				run.Session.Goal.Status = "completed"
			}
			run.Session.Goal.FinishedAt = run.Session.UpdatedAt
		}
		for i := range run.Session.Messages {
			if run.Session.Messages[i].Queued {
				run.Session.Messages[i].Queued = false
				if run.Stopped && strings.TrimSpace(run.Session.Messages[i].Content) == "" {
					run.Session.Messages[i].Content = "_Antrean dibatalkan oleh pengguna._"
				}
			}
		}
		finalSess := deepCopySession(run.Session)
		run.mu.Unlock()
		saveRunIfAlive(run, finalSess)
	}()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "id": id, "cwd": resolvedCwd, "goal_mode": req.GoalMode, "agent": parsedGoal.Agent})
}

func HandleChatsCollection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	dir := getChatsDir()

	switch r.Method {
	case http.MethodGet:
		chatStoreMu.Lock()
		entries, err := os.ReadDir(dir)
		chatStoreMu.Unlock()
		if err != nil {
			_ = json.NewEncoder(w).Encode([]ChatSummary{})
			return
		}
		var list []ChatSummary
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err != nil {
				continue
			}
			var s ChatSession
			if json.Unmarshal(b, &s) != nil || s.ID == "" {
				continue
			}
			run := getActiveRun(s.ID)
			isRunning := run != nil && func() bool { run.mu.RLock(); defer run.mu.RUnlock(); return run.Running }()
			goalStatus := ""
			if run != nil {
				run.mu.RLock()
				if run.Session.Goal != nil {
					goalStatus = run.Session.Goal.Status
				}
				run.mu.RUnlock()
			} else if s.Goal != nil {
				goalStatus = s.Goal.Status
			}
			list = append(list, ChatSummary{
				ID:         s.ID,
				Title:      s.Title,
				Model:      s.Model,
				Cwd:        s.Cwd,
				UpdatedAt:  s.UpdatedAt,
				MsgCount:   len(s.Messages),
				Running:    isRunning,
				GoalStatus: goalStatus,
			})
		}
		sort.Slice(list, func(i, j int) bool { return list[i].UpdatedAt > list[j].UpdatedAt })
		if list == nil {
			list = []ChatSummary{}
		}
		_ = json.NewEncoder(w).Encode(list)

	case http.MethodPost:
		var s ChatSession
		if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}
		s.ID = safeChatID(s.ID)
		if s.ID == "" {
			s.ID = fmt.Sprintf("chat_%d", time.Now().UnixMilli())
		}
		now := time.Now().UnixMilli()
		if s.CreatedAt == 0 {
			s.CreatedAt = now
		}
		s.UpdatedAt = now
		if strings.TrimSpace(s.Title) == "" {
			s.Title = "New Chat"
		}
		b, err := json.MarshalIndent(s, "", "  ")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		chatStoreMu.Lock()
		err = os.WriteFile(filepath.Join(dir, s.ID+".json"), b, 0o644)
		chatStoreMu.Unlock()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "id": s.ID, "updated_at": s.UpdatedAt})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func HandleChatSession(w http.ResponseWriter, r *http.Request) {
	rel := strings.TrimPrefix(r.URL.Path, "/api/chats/")
	if rel == "send" {
		handleChatSend(w, r)
		return
	}
	parts := strings.SplitN(rel, "/", 2)
	id := safeChatID(parts[0])
	sub := ""
	if len(parts) == 2 {
		sub = parts[1]
	}
	if id == "" {
		http.Error(w, "Missing chat ID", http.StatusBadRequest)
		return
	}

	if (sub == "stop" || sub == "goal/cancel") && r.Method == http.MethodPost {
		cancelActiveChatOrGoal(id)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
		return
	}

	if sub == "goal/diff" && r.Method == http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		commit := strings.TrimSpace(r.URL.Query().Get("commit"))
		var cwd string
		if run := getActiveRun(id); run != nil {
			run.mu.RLock()
			cwd = run.Session.Cwd
			run.mu.RUnlock()
		} else {
			p := filepath.Join(getChatsDir(), id+".json")
			chatStoreMu.Lock()
			b, err := os.ReadFile(p)
			chatStoreMu.Unlock()
			if err == nil {
				var s ChatSession
				if json.Unmarshal(b, &s) == nil {
					cwd = s.Cwd
				}
			}
		}
		if cwd == "" {
			http.Error(w, `{"error":"Chat session not found"}`, http.StatusNotFound)
			return
		}
		diffTxt, err := getGoalCheckpointDiff(id, cwd, commit)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": err.Error()})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "commit": commit, "diff": diffTxt})
		return
	}

	if sub == "goal/rollback" && r.Method == http.MethodPost {
		w.Header().Set("Content-Type", "application/json")
		if run := getActiveRun(id); run != nil {
			run.mu.RLock()
			running := run.Running
			run.mu.RUnlock()
			if running {
				w.WriteHeader(http.StatusConflict)
				_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Hentikan /goal terlebih dahulu sebelum melakukan rollback"})
				return
			}
		}
		r.Body = http.MaxBytesReader(w, r.Body, 16384)
		var body struct {
			Commit string `json:"commit"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || strings.TrimSpace(body.Commit) == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Parameter commit wajib diisi"})
			return
		}
		p := filepath.Join(getChatsDir(), id+".json")
		chatStoreMu.Lock()
		b, err := os.ReadFile(p)
		chatStoreMu.Unlock()
		if err != nil {
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Session not found"})
			return
		}
		var s ChatSession
		if err := json.Unmarshal(b, &s); err != nil || s.Cwd == "" {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "Invalid session workspace"})
			return
		}
		restored, deleted, rerr := rollbackGoalCheckpoint(id, s.Cwd, body.Commit)
		if rerr != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": rerr.Error()})
			return
		}
		shortHash := strings.TrimSpace(body.Commit)
		if len(shortHash) > 7 {
			shortHash = shortHash[:7]
		}
		s.UpdatedAt = time.Now().UnixMilli()
		s.Messages = append(s.Messages, StoredMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("> ⏪ **[Checkpoint Rollback]** Workspace dikembalikan ke snapshot `%s` (%d file dipulihkan, %d file baru dihapus).", shortHash, restored, deleted),
		})
		saveSessionFile(s)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ok":       true,
			"commit":   body.Commit,
			"restored": restored,
			"deleted":  deleted,
		})
		return
	}

	if sub == "zip" && r.Method == http.MethodGet {
		handleChatWorkspaceZip(w, r, id)
		return
	}

	if sub == "stream" && r.Method == http.MethodGet {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		lastSig := ""
		ticks := 0
		for {
			run := getActiveRun(id)
			var snap ChatSession
			if run != nil {
				snap = snapshotRunSession(run)
			} else {
				p := filepath.Join(getChatsDir(), id+".json")
				chatStoreMu.Lock()
				b, err := os.ReadFile(p)
				chatStoreMu.Unlock()
				if err != nil {
					return
				}
				_ = json.Unmarshal(b, &snap)
				snap.Running = false
			}
			b, _ := json.Marshal(snap)
			sig := string(b)
			if sig != lastSig {
				lastSig = sig
				ticks = 0
				_, _ = fmt.Fprintf(w, "data: %s\n\n", sig)
				flusher.Flush()
			} else {
				ticks++
				if ticks >= 9 {
					ticks = 0
					_, _ = fmt.Fprintf(w, ": ping\n\n")
					flusher.Flush()
				}
			}
			if !snap.Running {
				return
			}
			select {
			case <-r.Context().Done():
				return
			case <-time.After(220 * time.Millisecond):
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	p := filepath.Join(getChatsDir(), id+".json")

	switch r.Method {
	case http.MethodGet:
		if run := getActiveRun(id); run != nil {
			_ = json.NewEncoder(w).Encode(snapshotRunSession(run))
			return
		}
		chatStoreMu.Lock()
		b, err := os.ReadFile(p)
		chatStoreMu.Unlock()
		if err != nil {
			http.Error(w, "Not found", http.StatusNotFound)
			return
		}
		var snap ChatSession
		if json.Unmarshal(b, &snap) == nil {
			snap.Running = false
			_ = json.NewEncoder(w).Encode(snap)
			return
		}
		_, _ = w.Write(b)

	case http.MethodDelete:
		if run := getActiveRun(id); run != nil {
			run.mu.Lock()
			run.Deleted = true
			run.mu.Unlock()
			if run.Cancel != nil {
				run.Cancel()
			}
		}
		activeRunsMu.Lock()
		delete(activeRuns, id)
		activeRunsMu.Unlock()
		chatStoreMu.Lock()
		err := os.Remove(p)
		chatStoreMu.Unlock()
		deleteGoalCheckpoints(id)
		if err != nil && !os.IsNotExist(err) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})

	case http.MethodPatch:
		var body struct {
			Title string `json:"title"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}
		chatStoreMu.Lock()
		defer chatStoreMu.Unlock()
		b, err := os.ReadFile(p)
		if err != nil {
			http.Error(w, "Not found", http.StatusNotFound)
			return
		}
		var s ChatSession
		if err := json.Unmarshal(b, &s); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if strings.TrimSpace(body.Title) != "" {
			s.Title = strings.TrimSpace(body.Title)
		}
		out, _ := json.MarshalIndent(s, "", "  ")
		_ = os.WriteFile(p, out, 0o644)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "title": s.Title})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func HandleMCPStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	serverPath := "/home/rizal/Documents/ollama/image-mcp/server.py"
	_, err := os.Stat(serverPath)
	scriptExists := err == nil

	comfyOnline := false
	client := &http.Client{Timeout: 1200 * time.Millisecond}
	if resp, err := client.Get("http://127.0.0.1:8188/system_stats"); err == nil {
		resp.Body.Close()
		comfyOnline = resp.StatusCode == 200
	}

	_ = json.NewEncoder(w).Encode(map[string]any{
		"connected":    scriptExists,
		"server":       "local-image-gen",
		"script":       serverPath,
		"comfy_online": comfyOnline,
		"tools":        []string{"generate_image", "generate_video"},
	})
}

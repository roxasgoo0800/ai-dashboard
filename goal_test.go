package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseGoalCommand(t *testing.T) {
	cases := []struct {
		raw      string
		flag     bool
		isGoal   bool
		isCancel bool
		prompt   string
	}{
		{"/goal Build a REST API in Go", false, true, false, "Build a REST API in Go"},
		{"  /GOAL   Implement auth and tests /goal  ", false, true, false, "Implement auth and tests"},
		{"/goal cancel", false, false, true, ""},
		{"/GOAL STOP", false, false, true, ""},
		{"/cancel", false, false, true, ""},
		{"/goal", false, true, false, ""},
		{"/goalpost is not a command", false, false, false, "/goalpost is not a command"},
		{"Build full dashboard", true, true, false, "Build full dashboard"},
		{"Hello world", false, false, false, "Hello world"},
	}
	for _, tc := range cases {
		got := parseGoalCommand(tc.raw, tc.flag)
		if got.IsGoal != tc.isGoal || got.IsCancel != tc.isCancel || got.Prompt != tc.prompt {
			t.Fatalf("parseGoalCommand(%q, %v) = %+v, want isGoal=%v isCancel=%v prompt=%q",
				tc.raw, tc.flag, got, tc.isGoal, tc.isCancel, tc.prompt)
		}
	}
}

func TestCompactGoalMessages(t *testing.T) {
	var msgs []ChatMessage
	msgs = append(msgs, ChatMessage{Role: "system", Content: "SYSTEM_PROMPT"})
	msgs = append(msgs, ChatMessage{Role: "user", Content: "LONG_PRD_REQUIREMENTS"})
	for i := 0; i < 16; i++ {
		msgs = append(msgs, ChatMessage{
			Role:    "assistant",
			Content: strings.Repeat(fmt.Sprintf("step %d ", i), 400),
			ToolCalls: []OllamaToolCall{
				{
					Function: struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments"`
					}{
						Name:      "write_file",
						Arguments: map[string]any{"path": fmt.Sprintf("file_%d.go", i)},
					},
				},
			},
		})
		msgs = append(msgs, ChatMessage{
			Role:     "tool",
			ToolName: "write_file",
			Content:  fmt.Sprintf("Wrote file_%d.go", i),
		})
	}

	compacted := compactGoalMessages(msgs, 5000)
	if len(compacted) >= len(msgs) {
		t.Fatalf("expected compacted slice smaller than %d, got %d", len(msgs), len(compacted))
	}
	if compacted[0].Content != "SYSTEM_PROMPT" {
		t.Fatalf("system prompt lost after compaction")
	}
	if compacted[1].Content != "LONG_PRD_REQUIREMENTS" {
		t.Fatalf("original PRD lost after compaction")
	}
	if !strings.Contains(compacted[2].Content, "COMPACTED GOAL HISTORY") || !strings.Contains(compacted[2].Content, "file_0.go") {
		t.Fatalf("expected compacted summary with touched files, got %q", compacted[2].Content)
	}
	if len(compacted) > 3 && compacted[3].Role == "tool" {
		t.Fatalf("compacted tail must not start with an orphaned tool message")
	}
}

func TestEditFileAndReadFileLineRange(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "sample.txt")
	if err := os.WriteFile(fpath, []byte("line1\nline2\nline3\nline4\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	readRes := runToolCall(context.Background(), dir, "", "", "read_file", map[string]any{
		"path":       "sample.txt",
		"start_line": 2,
		"end_line":   3,
	})
	if !strings.Contains(readRes.Output, "2: line2") || !strings.Contains(readRes.Output, "3: line3") || strings.Contains(readRes.Output, "line1") {
		t.Fatalf("unexpected read_file slice output: %q", readRes.Output)
	}

	badEdit := runToolCall(context.Background(), dir, "", "", "edit_file", map[string]any{
		"path":        "sample.txt",
		"target":      "nonexistent",
		"replacement": "x",
	})
	if !strings.Contains(badEdit.Output, "target block not found") {
		t.Fatalf("expected missing target error, got %q", badEdit.Output)
	}

	okEdit := runToolCall(context.Background(), dir, "", "", "edit_file", map[string]any{
		"path":        "sample.txt",
		"target":      "line2",
		"replacement": "updated_line2",
	})
	if !strings.Contains(okEdit.Output, "1 occurrence(s) replaced") {
		t.Fatalf("unexpected edit_file output: %q", okEdit.Output)
	}
	got, _ := os.ReadFile(fpath)
	if string(got) != "line1\nupdated_line2\nline3\nline4\n" {
		t.Fatalf("unexpected file content after edit_file: %q", string(got))
	}
}

func TestExecuteGoalLoopContinuationAndAudit(t *testing.T) {
	var callNum atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := callNum.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		switch n {
		case 1:
			// Cycle 1: model pauses with plain text without tool calls and without GOAL_COMPLETE -> must auto-continue!
			_ = enc.Encode(map[string]any{"message": map[string]any{"content": "I will start by creating main.txt."}})
			_ = enc.Encode(map[string]any{"done": true, "eval_count": 10})
		case 2:
			// Cycle 2 turn 0: model invokes write_file tool
			_ = enc.Encode(map[string]any{
				"message": map[string]any{
					"content": "",
					"tool_calls": []map[string]any{
						{"function": map[string]any{"name": "write_file", "arguments": map[string]any{"path": "hello.txt", "content": "ok"}}},
					},
				},
			})
			_ = enc.Encode(map[string]any{"done": true, "eval_count": 12})
		case 3:
			// Cycle 2 turn 1: model emits first GOAL_COMPLETE -> must trigger mandatory audit cycle!
			_ = enc.Encode(map[string]any{"message": map[string]any{"content": "Created hello.txt. " + goalCompleteMarker}})
			_ = enc.Encode(map[string]any{"done": true, "eval_count": 8})
		default:
			// Cycle 3 (audit): model confirms after audit and emits GOAL_COMPLETE again -> complete!
			_ = enc.Encode(map[string]any{"message": map[string]any{"content": "Verified hello.txt exists. " + goalCompleteMarker}})
			_ = enc.Encode(map[string]any{"done": true, "eval_count": 9})
		}
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	ws := t.TempDir()
	var finalStatus string
	var sawAudit bool
	var sawDone bool

	executeGoalLoop(
		context.Background(),
		"qwen3.5-abliterated:32k",
		ws,
		false,
		"Create hello.txt with ok",
		[]ChatMessage{{Role: "user", Content: "/goal Create hello.txt with ok"}},
		10,
		func(ev map[string]any) {
			if ev["type"] == "goal_progress" {
				if st, ok := ev["status"].(string); ok {
					finalStatus = st
					if st == "auditing" {
						sawAudit = true
					}
				}
			}
			if ev["type"] == "done" {
				sawDone = true
			}
		},
	)

	if callNum.Load() != 4 {
		t.Fatalf("expected 4 Ollama calls (pause -> tool -> complete -> audit), got %d", callNum.Load())
	}
	if !sawAudit {
		t.Fatalf("expected mandatory audit phase before goal completion")
	}
	if finalStatus != "completed" || !sawDone {
		t.Fatalf("expected finalStatus=completed and sawDone=true, got status=%q sawDone=%v", finalStatus, sawDone)
	}
	b, err := os.ReadFile(filepath.Join(ws, "hello.txt"))
	if err != nil || string(b) != "ok" {
		t.Fatalf("expected hello.txt='ok', got err=%v content=%q", err, string(b))
	}
}

func TestExecuteGoalLoopCancelMarkerAndContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]any{"message": map[string]any{"content": "Cancelling as requested. " + goalCancelledMarker}})
		_ = enc.Encode(map[string]any{"done": true, "eval_count": 5})
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	var finalStatus string
	executeGoalLoop(
		context.Background(),
		"qwen3.5-abliterated:32k",
		t.TempDir(),
		false,
		"Cancel test",
		[]ChatMessage{{Role: "user", Content: "/goal Cancel test"}},
		5,
		func(ev map[string]any) {
			if ev["type"] == "goal_progress" {
				if st, ok := ev["status"].(string); ok {
					finalStatus = st
				}
			}
		},
	)
	if finalStatus != "cancelled" {
		t.Fatalf("expected status=cancelled on GOAL_CANCELLED marker, got %q", finalStatus)
	}

	// Test immediate context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	finalStatus = ""
	executeGoalLoop(
		ctx,
		"qwen3.5-abliterated:32k",
		t.TempDir(),
		false,
		"Ctx cancel test",
		[]ChatMessage{{Role: "user", Content: "/goal Ctx cancel test"}},
		5,
		func(ev map[string]any) {
			if ev["type"] == "goal_progress" {
				if st, ok := ev["status"].(string); ok {
					finalStatus = st
				}
			}
		},
	)
	if finalStatus != "cancelled" {
		t.Fatalf("expected status=cancelled on cancelled ctx, got %q", finalStatus)
	}
}

func TestHandleChatSendGoalAndCancelAPI(t *testing.T) {
	// Mock slow Ollama so the run stays active while we test /goal/cancel
	unblock := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-unblock:
		}
	}))
	defer func() {
		close(unblock)
		srv.Close()
	}()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	chatID := fmt.Sprintf("test_goal_%d", time.Now().UnixNano())
	defer func() {
		// Cleanup test chat file & activeRuns entry
		delReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+chatID, nil)
		delRR := httptest.NewRecorder()
		handleChatSession(delRR, delReq)
	}()

	body := fmt.Sprintf(`{"chat_id":%q,"model":"qwen3.5-abliterated:32k","cwd":%q,"user_prompt":"/goal Build test app"}`, chatID, t.TempDir())
	req := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewBufferString(body))
	rr := httptest.NewRecorder()
	handleChatSend(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 on handleChatSend, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || resp["goal_mode"] != true {
		t.Fatalf("expected goal_mode=true in response, got %s", rr.Body.String())
	}

	// Cancel via /goal cancel command in handleChatSend
	cancelBody := fmt.Sprintf(`{"chat_id":%q,"user_prompt":"/goal cancel"}`, chatID)
	cancelReq := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewBufferString(cancelBody))
	cancelRR := httptest.NewRecorder()
	handleChatSend(cancelRR, cancelReq)
	if cancelRR.Code != http.StatusOK {
		t.Fatalf("expected 200 on /goal cancel, got %d", cancelRR.Code)
	}
	var cResp map[string]any
	if err := json.Unmarshal(cancelRR.Body.Bytes(), &cResp); err != nil || cResp["goal_cancelled"] != true {
		t.Fatalf("expected goal_cancelled=true, got %s", cancelRR.Body.String())
	}

	// Verify session goal state is cancelled
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		getReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+chatID, nil)
		getRR := httptest.NewRecorder()
		handleChatSession(getRR, getReq)
		var snap ChatSession
		if json.Unmarshal(getRR.Body.Bytes(), &snap) == nil && snap.Goal != nil && snap.Goal.Status == "cancelled" && !snap.Running {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("expected chat %s goal status to become cancelled and stop running", chatID)
}

func TestResolveChatWorkspaceAndSessionIsolation(t *testing.T) {
	origRoot := chatWorkspaceRoot
	tmpRoot := filepath.Join(t.TempDir(), "chat")
	chatWorkspaceRoot = tmpRoot
	defer func() { chatWorkspaceRoot = origRoot }()

	// 1. Auto workspace creates isolated <slug>-<shortID> folder
	ws1 := resolveSessionWorkspace("chat_1763501234567", "", "__auto__", "Build Kasir API Go")
	if !strings.HasPrefix(ws1, tmpRoot+string(filepath.Separator)) {
		t.Fatalf("expected isolated folder under %s, got %s", tmpRoot, ws1)
	}
	if !strings.Contains(filepath.Base(ws1), "build-kasir-api-go-") {
		t.Fatalf("expected readable slug in isolated folder name, got %s", filepath.Base(ws1))
	}
	if info, err := os.Stat(ws1); err != nil || !info.IsDir() {
		t.Fatalf("expected isolated workspace folder to exist on disk: err=%v", err)
	}

	// 2. Subsequent message in same session reuses existing isolated workspace
	ws1Again := resolveSessionWorkspace("chat_1763501234567", ws1, "__auto__", "lanjutkan")
	if ws1Again != ws1 {
		t.Fatalf("expected existing isolated workspace %s to be reused, got %s", ws1, ws1Again)
	}

	// 3. Nested custom folder is automatically created via os.MkdirAll
	customNested := filepath.Join(tmpRoot, "custom-proj", "backend", "v2")
	wsCustom := resolveSessionWorkspace("chat_999", "", customNested, "ignored")
	if wsCustom != customNested {
		t.Fatalf("expected custom nested workspace %s, got %s", customNested, wsCustom)
	}
	if info, err := os.Stat(customNested); err != nil || !info.IsDir() {
		t.Fatalf("expected custom nested folder to be auto-created: err=%v", err)
	}
}

func TestFindFilesAndGrepSearchTools(t *testing.T) {
	ws := t.TempDir()
	mustWrite := func(rel string, data []byte) {
		t.Helper()
		full := filepath.Join(ws, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	mustWrite("main.go", []byte("package main\n// TODO: implement handler\nfunc HelloWorld() {}\n"))
	mustWrite("sub/helper.go", []byte("package sub\n// regex test a+b literal\nfunc Calc() int { return 42 }\n"))
	mustWrite("sub/readme.md", []byte("lowercase helloWorld note\n"))
	mustWrite("node_modules/dep/index.go", []byte("package dep\nfunc HelloWorld() {}\n"))
	mustWrite(".git/config", []byte("[core]\n"))
	mustWrite("bin.dat", []byte{0x00, 0x01, 'H', 'e', 'l', 'l', 'o', 0x00})

	ctx := context.Background()

	// 1. find_files with glob skipping .git and node_modules
	res := runToolCall(ctx, ws, "", "", "find_files", map[string]any{"pattern": "*.go"})
	if strings.HasPrefix(res.Output, "Error") {
		t.Fatalf("find_files failed: %s", res.Output)
	}
	if !strings.Contains(res.Output, "main.go") || !strings.Contains(res.Output, "sub/helper.go") {
		t.Fatalf("expected main.go and sub/helper.go in find_files output, got:\n%s", res.Output)
	}
	if strings.Contains(res.Output, "node_modules") || strings.Contains(res.Output, ".git") {
		t.Fatalf("expected node_modules and .git to be skipped, got:\n%s", res.Output)
	}

	// 2. find_files scoped to subdirectory
	resSub := runToolCall(ctx, ws, "", "", "find_files", map[string]any{"path": "sub", "pattern": "*.md"})
	if !strings.Contains(resSub.Output, "readme.md") || strings.Contains(resSub.Output, "main.go") {
		t.Fatalf("unexpected subdir find_files output: out=%q", resSub.Output)
	}

	// 3. grep_search regex + binary/node_modules skip
	g1 := runToolCall(ctx, ws, "", "", "grep_search", map[string]any{"query": `func\s+Hello\w+`})
	if !strings.Contains(g1.Output, "main.go:3:") || strings.Contains(g1.Output, "node_modules") || strings.Contains(g1.Output, "bin.dat") {
		t.Fatalf("unexpected regex grep_search output: out=%q", g1.Output)
	}

	// 4. grep_search literal mode (a+b should match literally, not regex)
	gLit := runToolCall(ctx, ws, "", "", "grep_search", map[string]any{"query": "a+b", "literal": true})
	if !strings.Contains(gLit.Output, "sub/helper.go:2:") {
		t.Fatalf("expected literal grep match in sub/helper.go, got out=%q", gLit.Output)
	}

	// 5. grep_search ignore_case
	gCI := runToolCall(ctx, ws, "", "", "grep_search", map[string]any{"query": "HELLOWORLD", "ignore_case": true})
	if !strings.Contains(gCI.Output, "main.go:") || !strings.Contains(gCI.Output, "sub/readme.md:") {
		t.Fatalf("expected ignore_case matches in main.go and sub/readme.md, got:\n%s", gCI.Output)
	}

	// 6. grep_search no-match + empty query handling
	gNone := runToolCall(ctx, ws, "", "", "grep_search", map[string]any{"query": "DEFINITELY_NOT_FOUND_XYZ"})
	if gNone.Output != "(no matches found)" {
		t.Fatalf("expected '(no matches found)', got %q", gNone.Output)
	}
	gErr := runToolCall(ctx, ws, "", "", "grep_search", map[string]any{"query": ""})
	if !strings.Contains(gErr.Output, "Error: empty query") {
		t.Fatalf("expected 'Error: empty query', got %q", gErr.Output)
	}
}

func TestCompactGoalMessagesWithPlan(t *testing.T) {
	ws := t.TempDir()
	planContent := "# Goal Plan\n- [x] Setup database\n- [ ] Build REST API\n- [ ] Write unit tests\n"
	if err := os.WriteFile(filepath.Join(ws, ".goal_plan.md"), []byte(planContent), 0o644); err != nil {
		t.Fatal(err)
	}

	msgs := []ChatMessage{
		{Role: "system", Content: "System instructions"},
		{Role: "user", Content: "Build REST API"},
	}
	for i := 0; i < 16; i++ {
		var tc OllamaToolCall
		tc.Function.Name = "bash"
		tc.Function.Arguments = map[string]any{
			"command": fmt.Sprintf("echo step %d %s", i, strings.Repeat("x", 180)),
		}
		msgs = append(msgs, ChatMessage{
			Role:      "assistant",
			ToolCalls: []OllamaToolCall{tc},
		})
		msgs = append(msgs, ChatMessage{
			Role:     "tool",
			ToolName: "bash",
			Content:  fmt.Sprintf("step %d ok: %s", i, strings.Repeat("y", 220)),
		})
	}

	compacted := compactGoalMessagesWithPlan(msgs, 2400, ws)
	if len(compacted) >= len(msgs) {
		t.Fatalf("expected compaction to shrink %d messages, got %d", len(msgs), len(compacted))
	}

	joined := ""
	for _, m := range compacted {
		joined += "\n" + m.Content
	}
	if !strings.Contains(joined, "[COMPACTED GOAL HISTORY:") {
		t.Fatalf("expected [COMPACTED GOAL HISTORY:] marker after compaction, got:\n%s", joined)
	}
	if !strings.Contains(joined, ".goal_plan.md") || !strings.Contains(joined, "Build REST API") {
		t.Fatalf("expected .goal_plan.md checklist to be preserved inside compacted summary, got:\n%s", joined)
	}
	// Verify no orphaned tool result immediately without matching assistant tool_call
	for i, m := range compacted {
		if m.Role == "tool" {
			if i == 0 || compacted[i-1].Role == "system" {
				t.Fatalf("found orphaned tool message at index %d right after %v", i, compacted[i-1].Role)
			}
		}
	}
}

func TestShadowGitCheckpointsDiffAndRollback(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not installed")
	}

	ws := t.TempDir()
	chatID := fmt.Sprintf("test_cp_%d", time.Now().UnixNano())
	defer deleteGoalCheckpoints(chatID)

	// Set up an existing real git repository & node_modules inside workspace
	runGit := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = ws
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=Tester",
			"GIT_AUTHOR_EMAIL=tester@example.com",
			"GIT_COMMITTER_NAME=Tester",
			"GIT_COMMITTER_EMAIL=tester@example.com",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v\n%s", args, err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	if err := os.WriteFile(filepath.Join(ws, "app.txt"), []byte("version 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(ws, "node_modules", "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "node_modules", "pkg", "index.js"), []byte("module.exports = 1;\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	runGit("init")
	runGit("add", "app.txt")
	runGit("commit", "-m", "initial real commit")
	realHeadBefore := runGit("rev-parse", "HEAD")

	// 1. Create baseline shadow checkpoint
	cp0, err := createGoalCheckpoint(chatID, ws, 0, "Baseline snapshot")
	if err != nil {
		t.Fatalf("createGoalCheckpoint baseline failed: %v", err)
	}
	if cp0 == nil || cp0.Commit == "" {
		t.Fatalf("expected non-empty baseline checkpoint commit")
	}

	// Verify real .git HEAD was never modified
	realHeadAfter := runGit("rev-parse", "HEAD")
	if realHeadAfter != realHeadBefore {
		t.Fatalf("shadow git mutated real .git HEAD: before=%s after=%s", realHeadBefore, realHeadAfter)
	}

	// 2. Modify existing file and create new file, then take cycle 1 checkpoint
	if err := os.WriteFile(filepath.Join(ws, "app.txt"), []byte("version 2\nfeature line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "added.txt"), []byte("created in cycle 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp1, err := createGoalCheckpoint(chatID, ws, 1, "Cycle 1 feature")
	if err != nil {
		t.Fatalf("createGoalCheckpoint cycle 1 failed: %v", err)
	}
	if cp1.FilesChanged < 2 {
		t.Fatalf("expected >= 2 files changed in cp1, got %d", cp1.FilesChanged)
	}
	statMap := map[string]GoalFileStat{}
	for _, st := range cp1.FileStats {
		statMap[st.Path] = st
	}
	if sApp, ok := statMap["app.txt"]; !ok || sApp.Added != 2 || sApp.Deleted != 1 {
		t.Fatalf("expected app.txt numstat (+2/-1), got %+v (all=%+v)", sApp, cp1.FileStats)
	}
	if sAdd, ok := statMap["added.txt"]; !ok || sAdd.Added != 1 {
		t.Fatalf("expected added.txt numstat (+1/-0), got %+v", sAdd)
	}

	// 3. Check diff output for cycle 1 checkpoint
	diffOut, err := getGoalCheckpointDiff(chatID, ws, cp1.Commit)
	if err != nil {
		t.Fatalf("getGoalCheckpointDiff failed: %v", err)
	}
	if !strings.Contains(diffOut, "app.txt") || !strings.Contains(diffOut, "added.txt") || !strings.Contains(diffOut, "+version 2") {
		t.Fatalf("unexpected checkpoint diff output:\n%s", diffOut)
	}

	// 4. Introduce dirty uncommitted changes, then rollback to cp0 baseline
	if err := os.WriteFile(filepath.Join(ws, "app.txt"), []byte("broken version 3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "uncommitted.txt"), []byte("temp junk\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	restored, deleted, err := rollbackGoalCheckpoint(chatID, ws, cp0.Commit)
	if err != nil {
		t.Fatalf("rollbackGoalCheckpoint failed: %v", err)
	}
	if restored < 1 || deleted < 2 {
		t.Fatalf("expected restored>=1 and deleted>=2 (added.txt + uncommitted.txt), got restored=%d deleted=%d", restored, deleted)
	}

	gotApp, err := os.ReadFile(filepath.Join(ws, "app.txt"))
	if err != nil || string(gotApp) != "version 1\n" {
		t.Fatalf("expected app.txt restored to version 1, got %q err=%v", string(gotApp), err)
	}
	if _, err := os.Stat(filepath.Join(ws, "added.txt")); !os.IsNotExist(err) {
		t.Fatalf("expected added.txt to be removed on rollback to baseline")
	}
	if _, err := os.Stat(filepath.Join(ws, "uncommitted.txt")); !os.IsNotExist(err) {
		t.Fatalf("expected uncommitted.txt to be removed on rollback to baseline")
	}
	// Verify node_modules and real .git remain untouched
	if _, err := os.Stat(filepath.Join(ws, "node_modules", "pkg", "index.js")); err != nil {
		t.Fatalf("expected node_modules/pkg/index.js to remain intact: %v", err)
	}
	if gotHead := runGit("rev-parse", "HEAD"); gotHead != realHeadBefore {
		t.Fatalf("rollback modified real .git HEAD: got %s want %s", gotHead, realHeadBefore)
	}
}

func TestGoalDiffAndRollbackHTTPAPI(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not installed")
	}

	ws := t.TempDir()
	if err := os.WriteFile(filepath.Join(ws, "main.txt"), []byte("before\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Create session via API
	createBody, _ := json.Marshal(map[string]any{"title": "HTTP Checkpoint Test", "cwd": ws})
	createReq := httptest.NewRequest(http.MethodPost, "/api/chats", bytes.NewReader(createBody))
	createRR := httptest.NewRecorder()
	handleChatsCollection(createRR, createReq)
	if createRR.Code != http.StatusOK {
		t.Fatalf("POST /api/chats failed: %d %s", createRR.Code, createRR.Body.String())
	}
	var sess ChatSession
	if err := json.Unmarshal(createRR.Body.Bytes(), &sess); err != nil || sess.ID == "" {
		t.Fatalf("failed to decode POST /api/chats response: err=%v body=%s", err, createRR.Body.String())
	}
	sess.Cwd = ws
	chatID := sess.ID
	defer func() {
		activeRunsMu.Lock()
		delete(activeRuns, chatID)
		activeRunsMu.Unlock()
		delReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+chatID, nil)
		delRR := httptest.NewRecorder()
		handleChatSession(delRR, delReq)
		deleteGoalCheckpoints(chatID)
	}()

	cp0, err := createGoalCheckpoint(chatID, ws, 0, "Initial")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "main.txt"), []byte("after change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp1, err := createGoalCheckpoint(chatID, ws, 1, "Edited main.txt")
	if err != nil {
		t.Fatal(err)
	}

	// Register a running active run to test live diff and 409 Conflict guard on rollback
	sess.Running = true
	sess.Goal = &GoalState{
		Prompt:      "Test",
		Status:      "running",
		Active:      true,
		Checkpoints: []GoalCheckpoint{*cp0, *cp1},
	}
	saveSessionFile(sess)
	run := &ChatRun{Running: true, Session: sess}
	activeRunsMu.Lock()
	activeRuns[chatID] = run
	activeRunsMu.Unlock()

	// 1. GET diff works even while running
	diffReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+chatID+"/goal/diff?commit="+cp1.Commit, nil)
	diffRR := httptest.NewRecorder()
	handleChatSession(diffRR, diffReq)
	if diffRR.Code != http.StatusOK {
		t.Fatalf("GET /goal/diff returned %d: %s", diffRR.Code, diffRR.Body.String())
	}
	var diffResp struct {
		OK     bool   `json:"ok"`
		Commit string `json:"commit"`
		Diff   string `json:"diff"`
	}
	if err := json.Unmarshal(diffRR.Body.Bytes(), &diffResp); err != nil || !diffResp.OK || !strings.Contains(diffResp.Diff, "+after change") {
		t.Fatalf("unexpected diff response: err=%v body=%s", err, diffRR.Body.String())
	}

	// 2. Rollback while Running == true returns 409 Conflict
	rbPayload, _ := json.Marshal(map[string]string{"commit": cp0.Commit})
	rbConflictReq := httptest.NewRequest(http.MethodPost, "/api/chats/"+chatID+"/goal/rollback", bytes.NewReader(rbPayload))
	rbConflictRR := httptest.NewRecorder()
	handleChatSession(rbConflictRR, rbConflictReq)
	if rbConflictRR.Code != http.StatusConflict {
		t.Fatalf("expected 409 Conflict when rolling back active session, got %d: %s", rbConflictRR.Code, rbConflictRR.Body.String())
	}

	// 3. Mark session stopped, remove from activeRuns, and rollback to cp0 -> 200 OK & file restored
	run.mu.Lock()
	run.Running = false
	run.Session.Goal.Active = false
	run.Session.Goal.Status = "completed"
	run.mu.Unlock()
	activeRunsMu.Lock()
	delete(activeRuns, chatID)
	activeRunsMu.Unlock()

	rbReq := httptest.NewRequest(http.MethodPost, "/api/chats/"+chatID+"/goal/rollback", bytes.NewReader(rbPayload))
	rbRR := httptest.NewRecorder()
	handleChatSession(rbRR, rbReq)
	if rbRR.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on rollback, got %d: %s", rbRR.Code, rbRR.Body.String())
	}
	got, err := os.ReadFile(filepath.Join(ws, "main.txt"))
	if err != nil || string(got) != "before\n" {
		t.Fatalf("expected main.txt restored to 'before\\n', got %q err=%v", string(got), err)
	}
	// Verify rollback audit message was appended to session
	getReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+chatID, nil)
	getRR := httptest.NewRecorder()
	handleChatSession(getRR, getReq)
	var afterSnap ChatSession
	if err := json.Unmarshal(getRR.Body.Bytes(), &afterSnap); err != nil {
		t.Fatal(err)
	}
	if len(afterSnap.Messages) == 0 || !strings.Contains(afterSnap.Messages[len(afterSnap.Messages)-1].Content, "Checkpoint Rollback") {
		t.Fatalf("expected rollback note message appended to session, got %+v", afterSnap.Messages)
	}
}

func TestToolCallLoopTrackerUnit(t *testing.T) {
	tr := &toolCallLoopTracker{}
	cmdArgs := map[string]any{"command": "cat missing.json"}

	// 1-2 consecutive calls allowed
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("1st call should not be blocked")
	}
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("2nd call should not be blocked")
	}
	// 3rd consecutive call blocked
	if blocked, rep := tr.check("bash", cmdArgs); !blocked || rep != 3 {
		t.Fatalf("expected 3rd consecutive call blocked (rep=3), got blocked=%v rep=%d", blocked, rep)
	}

	// Distinct mutating tool (write_file) clears read-only window so re-running command after edits works
	if blocked, _ := tr.check("write_file", map[string]any{"path": "a.txt", "content": "1"}); blocked {
		t.Fatalf("distinct write_file should not be blocked")
	}
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("post-edit check should be allowed again")
	}

	// Oscillating A, B, A, B, A, B, A -> 4th occurrence in window triggers loop guard
	altArgs := map[string]any{"command": "ls -la"}
	if blocked, _ := tr.check("bash", altArgs); blocked {
		t.Fatalf("alt 1 should not be blocked")
	}
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("post-edit repeat 2 should not be blocked")
	}
	if blocked, _ := tr.check("bash", altArgs); blocked {
		t.Fatalf("alt 2 should not be blocked")
	}
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("post-edit repeat 3 (non-consecutive) should not be blocked")
	}
	if blocked, _ := tr.check("bash", altArgs); blocked {
		t.Fatalf("alt 3 should not be blocked")
	}
	if blocked, rep := tr.check("bash", cmdArgs); !blocked || rep < 4 {
		t.Fatalf("expected 4th identical call inside window without edits to be blocked, got blocked=%v rep=%d", blocked, rep)
	}
}

func TestExecuteGoalLoopAbortsRepeatedToolCallLoop(t *testing.T) {
	ws := t.TempDir()
	hitLog := filepath.Join(ws, "exec_hits.txt")

	var modelCalls atomic.Int32
	var sawLoopRecoveryHint atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []ChatMessage `json:"messages"`
		}
		if json.NewDecoder(r.Body).Decode(&reqBody) == nil {
			for _, m := range reqBody.Messages {
				if strings.Contains(m.Content, "[GOAL SYSTEM - TOOL LOOP RECOVERY]") {
					sawLoopRecoveryHint.Store(true)
				}
			}
		}
		modelCalls.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")

		// Always stubbornly emit the exact same failing bash tool call (worst-case stuck model)
		var tc OllamaToolCall
		tc.Function.Name = "bash"
		tc.Function.Arguments = map[string]any{
			"command": fmt.Sprintf("echo x >> %s && cat %s/nonexistent.json", hitLog, ws),
		}
		line, _ := json.Marshal(map[string]any{
			"model": "mock-loop",
			"message": map[string]any{
				"role":       "assistant",
				"content":    "",
				"tool_calls": []OllamaToolCall{tc},
			},
			"done": true,
		})
		_, _ = w.Write(append(line, '\n'))
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	var finalStatus string
	var blockedOutputs int
	var eventsText strings.Builder
	executeGoalLoop(
		context.Background(),
		"qwen3.5-abliterated:32k",
		ws,
		false,
		"Build app",
		[]ChatMessage{{Role: "user", Content: "/goal Build app"}},
		10,
		func(ev map[string]any) {
			switch ev["type"] {
			case "goal_progress":
				if st, ok := ev["status"].(string); ok {
					finalStatus = st
				}
			case "tool_end":
				if out, _ := ev["output"].(string); strings.Contains(out, "Dilewati (pemanggilan") {
					blockedOutputs++
				}
			case "content":
				if txt, _ := ev["text"].(string); txt != "" {
					eventsText.WriteString(txt)
				}
			}
		},
	)

	if finalStatus != "failed" {
		t.Fatalf("expected repeated tool-call loop to fail cleanly instead of max_cycles, got status=%q", finalStatus)
	}
	if !sawLoopRecoveryHint.Load() {
		t.Fatalf("expected [GOAL SYSTEM - TOOL LOOP RECOVERY] prompt to be injected")
	}
	if blockedOutputs < 3 {
		t.Fatalf("expected LOOP GUARD to block repeated tool executions (>=3), got %d", blockedOutputs)
	}
	rawHits, err := os.ReadFile(hitLog)
	if err != nil {
		t.Fatalf("expected initial non-blocked bash call to run, read %s err=%v", hitLog, err)
	}
	actualExecs := len(strings.Split(strings.TrimSpace(string(rawHits)), "\n"))
	if actualExecs > 2 {
		t.Fatalf("expected actual failing bash command to execute at most 2 times (blocked cleanly on repeat #2!), ran %d times", actualExecs)
	}
	if !strings.Contains(eventsText.String(), "Goal Recovery") {
		t.Fatalf("expected Goal Recovery banner text, got: %s", eventsText.String())
	}
	if modelCalls.Load() >= 20 {
		t.Fatalf("expected loop guard to abort far earlier than 1 full 25-turn cycle, made %d model calls", modelCalls.Load())
	}
}

func TestToolCallLoopTrackerFailingCommandImmediateBlock(t *testing.T) {
	tr := &toolCallLoopTracker{}
	cmdArgs := map[string]any{"command": "cat does-not-exist.txt"}
	if blocked, _ := tr.check("bash", cmdArgs); blocked {
		t.Fatalf("1st attempt should run")
	}
	tr.recordResult("Exit error: exit status 1\ncat: does-not-exist.txt: No such file or directory")
	// 2nd consecutive identical call right after a failure must be intercepted immediately
	if blocked, rep := tr.check("bash", cmdArgs); !blocked || rep != 2 {
		t.Fatalf("expected immediate block at repeat 2 after error, got blocked=%v rep=%d", blocked, rep)
	}
}

func TestGoalPerCycleSeparateMessagesAndSanitizedResume(t *testing.T) {
	ws := t.TempDir()
	// Pre-populate .goal_plan.md & a noisy prior session containing loop warnings to verify resume sanitization
	if err := os.WriteFile(filepath.Join(ws, ".goal_plan.md"), []byte("# Plan\n- [x] Init\n- [ ] API\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "existing.txt"), []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var step atomic.Int32
	var sawSanitizedHistory atomic.Bool
	var leakedOldLoopGuard atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []ChatMessage `json:"messages"`
		}
		if json.NewDecoder(r.Body).Decode(&reqBody) == nil {
			for _, m := range reqBody.Messages {
				if strings.Contains(m.Content, "existing.txt") && strings.Contains(m.Content, "[CURRENT WORKSPACE STATE]") {
					sawSanitizedHistory.Store(true)
				}
				if strings.Contains(m.Content, "OLD_LEAKED_LOOP_GUARD") {
					leakedOldLoopGuard.Store(true)
				}
			}
		}

		n := step.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")
		switch n {
		case 1:
			// Cycle 1: invoke write_file without explanatory text so Cycle 2 split should auto-generate Cycle 1 summary
			var tc OllamaToolCall
			tc.Function.Name = "write_file"
			tc.Function.Arguments = map[string]any{"path": "api.go", "content": "package main\n"}
			line, _ := json.Marshal(map[string]any{
				"model": "mock-goal",
				"message": map[string]any{
					"role":       "assistant",
					"content":    "",
					"tool_calls": []OllamaToolCall{tc},
				},
				"done": true,
			})
			_, _ = w.Write(append(line, '\n'))
		case 2:
			// Pause without marker so Cycle 1 -> Cycle 2 creates a 2nd separate assistant message
			line, _ := json.Marshal(map[string]any{
				"model":   "mock-goal",
				"message": map[string]any{"role": "assistant", "content": ""},
				"done":    true,
			})
			_, _ = w.Write(append(line, '\n'))
		case 3:
			// Cycle 2 reports completion, triggering audit Cycle 3
			line, _ := json.Marshal(map[string]any{
				"model":   "mock-goal",
				"message": map[string]any{"role": "assistant", "content": "API selesai ditulis. <!-- GOAL_COMPLETE -->"},
				"done":    true,
			})
			_, _ = w.Write(append(line, '\n'))
		default:
			// Cycle 3 audit confirms completion
			line, _ := json.Marshal(map[string]any{
				"model":   "mock-goal",
				"message": map[string]any{"role": "assistant", "content": "Semua checklist terverifikasi. <!-- GOAL_COMPLETE -->"},
				"done":    true,
			})
			_, _ = w.Write(append(line, '\n'))
		}
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	chatID := fmt.Sprintf("test_cycles_%d", time.Now().UnixNano())
	defer func() {
		delReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+chatID, nil)
		delRR := httptest.NewRecorder()
		handleChatSession(delRR, delReq)
		deleteGoalCheckpoints(chatID)
	}()

	// Seed prior crashed assistant history with duplicate/LOOP GUARD noise
	seedSess := ChatSession{
		ID:        chatID,
		Title:     "Cycle Split Test",
		Cwd:       ws,
		CreatedAt: time.Now().UnixMilli(),
		UpdatedAt: time.Now().UnixMilli(),
		Messages: []StoredMessage{
			{Role: "user", Content: "/goal Build API"},
			{
				Role: "assistant",
				Tools: []StoredToolLog{
					{Name: "bash", Args: map[string]any{"command": "cat x"}, Output: "Error [LOOP GUARD]: OLD_LEAKED_LOOP_GUARD"},
					{Name: "bash", Args: map[string]any{"command": "ls"}, Output: "existing.txt"},
				},
			},
		},
	}
	saveSessionFile(seedSess)

	body, _ := json.Marshal(map[string]any{
		"chat_id":     chatID,
		"model":       "mock-goal",
		"cwd":         ws,
		"user_prompt": "/goal Lanjutkan build API",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	handleChatSend(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("handleChatSend failed: %d %s", rr.Code, rr.Body.String())
	}

	deadline := time.Now().Add(4 * time.Second)
	var finalSess ChatSession
	for time.Now().Before(deadline) {
		getReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+chatID, nil)
		getRR := httptest.NewRecorder()
		handleChatSession(getRR, getReq)
		if json.Unmarshal(getRR.Body.Bytes(), &finalSess) == nil && finalSess.Goal != nil && !finalSess.Running && finalSess.Goal.Status == "completed" {
			break
		}
		time.Sleep(30 * time.Millisecond)
	}
	if finalSess.Goal == nil || finalSess.Goal.Status != "completed" {
		t.Fatalf("expected goal status=completed, got %+v", finalSess.Goal)
	}
	if leakedOldLoopGuard.Load() {
		t.Fatalf("old LOOP GUARD noise leaked into resumed turnMsgs")
	}
	if !sawSanitizedHistory.Load() {
		t.Fatalf("expected [CURRENT WORKSPACE STATE] with existing.txt to ground Cycle 1")
	}
	if !strings.Contains(finalSess.Goal.PlanMarkdown, "- [x] Init") {
		t.Fatalf("expected Goal.PlanMarkdown to be synced from .goal_plan.md, got %q", finalSess.Goal.PlanMarkdown)
	}

	// Verify Cycle 1 and Cycle 2/3 were stored as SEPARATE assistant messages (not piled into one!)
	var cycleNums []int
	var sawCycle1AutoSummary bool
	for _, m := range finalSess.Messages {
		if m.Role == "assistant" && m.GoalCycle > 0 {
			cycleNums = append(cycleNums, m.GoalCycle)
			if m.GoalCycle == 1 && strings.Contains(m.Content, "Ringkasan Cycle 1:") && strings.Contains(m.Content, "api.go") {
				sawCycle1AutoSummary = true
			}
		}
	}
	if len(cycleNums) < 2 {
		t.Fatalf("expected >= 2 separate per-cycle assistant messages, got cycles=%v", cycleNums)
	}
	if !sawCycle1AutoSummary {
		t.Fatalf("expected Cycle 1 without text to receive automatic tool summary mentioning api.go, msgs=%+v", finalSess.Messages)
	}
}

func TestGoalLiveSteeringMidRun(t *testing.T) {
	ws := t.TempDir()
	turn1Entered := make(chan struct{})
	turn1Release := make(chan struct{})
	var step atomic.Int32
	var sawSteeringPrompt atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []ChatMessage `json:"messages"`
		}
		if json.NewDecoder(r.Body).Decode(&reqBody) == nil {
			for _, m := range reqBody.Messages {
				if strings.Contains(m.Content, "[USER LIVE STEERING DIRECTION") && strings.Contains(m.Content, "Gunakan bahasa Go saja") {
					sawSteeringPrompt.Store(true)
				}
			}
		}

		n := step.Add(1)
		w.Header().Set("Content-Type", "application/x-ndjson")
		if n == 1 {
			// Hold first turn briefly so live steering arrives mid-run, then run a write_file
			close(turn1Entered)
			<-turn1Release
			var tc OllamaToolCall
			tc.Function.Name = "write_file"
			tc.Function.Arguments = map[string]any{"path": "main.go", "content": "package main\n"}
			line, _ := json.Marshal(map[string]any{
				"model": "mock-steer",
				"message": map[string]any{
					"role":       "assistant",
					"content":    "Mulai membangun project.",
					"tool_calls": []OllamaToolCall{tc},
				},
				"done": true,
			})
			_, _ = w.Write(append(line, '\n'))
			return
		}
		// Subsequent turns finalize (including self-audit cycle)
		line, _ := json.Marshal(map[string]any{
			"model":   "mock-steer",
			"message": map[string]any{"role": "assistant", "content": "Arahan Go diterapkan dan selesai. <!-- GOAL_COMPLETE -->"},
			"done":    true,
		})
		_, _ = w.Write(append(line, '\n'))
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	chatID := fmt.Sprintf("test_steer_%d", time.Now().UnixNano())
	defer func() {
		delReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+chatID, nil)
		delRR := httptest.NewRecorder()
		handleChatSession(delRR, delReq)
		deleteGoalCheckpoints(chatID)
	}()

	startBody, _ := json.Marshal(map[string]any{
		"chat_id":     chatID,
		"model":       "mock-steer",
		"cwd":         ws,
		"user_prompt": "/goal Buat backend API",
	})
	startReq := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(startBody))
	startRR := httptest.NewRecorder()
	handleChatSend(startRR, startReq)
	if startRR.Code != http.StatusOK {
		t.Fatalf("start handleChatSend failed: %d %s", startRR.Code, startRR.Body.String())
	}

	// Wait until Cycle 1 Turn 0 is inside the mock LLM request
	select {
	case <-turn1Entered:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for turn 1 to enter mock server")
	}

	// Send live mid-run steering instruction while /goal is actively running
	steerBody, _ := json.Marshal(map[string]any{
		"chat_id":     chatID,
		"model":       "mock-steer",
		"cwd":         ws,
		"steer":       true,
		"user_prompt": "Gunakan bahasa Go saja",
	})
	steerReq := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(steerBody))
	steerRR := httptest.NewRecorder()
	handleChatSend(steerRR, steerReq)
	close(turn1Release)

	if steerRR.Code != http.StatusOK {
		t.Fatalf("steer handleChatSend failed: %d %s", steerRR.Code, steerRR.Body.String())
	}
	var steerResp map[string]any
	if err := json.Unmarshal(steerRR.Body.Bytes(), &steerResp); err != nil || steerResp["steered"] != true {
		t.Fatalf("expected steered=true response, got %s", steerRR.Body.String())
	}

	deadline := time.Now().Add(4 * time.Second)
	var finalSess ChatSession
	for time.Now().Before(deadline) {
		getReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+chatID, nil)
		getRR := httptest.NewRecorder()
		handleChatSession(getRR, getReq)
		if json.Unmarshal(getRR.Body.Bytes(), &finalSess) == nil && finalSess.Goal != nil && !finalSess.Running && finalSess.Goal.Status == "completed" {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if finalSess.Goal == nil || finalSess.Goal.Status != "completed" {
		t.Fatalf("expected completed goal status after live steering, got %+v", finalSess.Goal)
	}
	if !sawSteeringPrompt.Load() {
		t.Fatalf("expected active /goal loop to inject [USER LIVE STEERING DIRECTION] into next turn")
	}
	var sawSteerUserMsg bool
	for _, m := range finalSess.Messages {
		if m.Role == "user" && strings.Contains(m.Content, "🎯 Gunakan bahasa Go saja") {
			sawSteerUserMsg = true
		}
	}
	if !sawSteerUserMsg {
		t.Fatalf("expected steering message saved in session timeline, msgs=%+v", finalSess.Messages)
	}
}

func TestPlusJakartaSansFontAndRPGThemeAssets(t *testing.T) {
	if len(jakartaSansWOFF2) < 10000 {
		t.Fatalf("expected embedded PlusJakartaSans-latin.woff2 >= 10KB, got %d bytes", len(jakartaSansWOFF2))
	}

	req := httptest.NewRequest(http.MethodGet, "/api/assets/fonts/plus-jakarta-sans.woff2", nil)
	rr := httptest.NewRecorder()
	handleFontAsset(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from handleFontAsset, got %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "font/woff2" {
		t.Fatalf("expected Content-Type font/woff2, got %q", ct)
	}
	if rr.Body.Len() != len(jakartaSansWOFF2) {
		t.Fatalf("expected body len %d, got %d", len(jakartaSansWOFF2), rr.Body.Len())
	}

	badReq := httptest.NewRequest(http.MethodPost, "/api/assets/fonts/plus-jakarta-sans.woff2", nil)
	badRR := httptest.NewRecorder()
	handleFontAsset(badRR, badReq)
	if badRR.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST font asset, got %d", badRR.Code)
	}

	for name, htmlDoc := range map[string]string{"indexHTML": indexHTML, "loginHTML": loginHTML} {
		if !strings.Contains(htmlDoc, "font-family: 'Plus Jakarta Sans'") || !strings.Contains(htmlDoc, "/api/assets/fonts/plus-jakarta-sans.woff2") {
			t.Fatalf("%s missing self-hosted Plus Jakarta Sans @font-face", name)
		}
		if !strings.Contains(htmlDoc, `shape-rendering="crispEdges"`) {
			t.Fatalf("%s missing 32-bit pixel-art SVG sprite assets", name)
		}
	}
	if !strings.Contains(indexHTML, `--sans: 'Plus Jakarta Sans'`) || !strings.Contains(indexHTML, `:root[data-theme="light"]`) {
		t.Fatalf("indexHTML missing Plus Jakarta Sans --sans token or light GBA parchment theme")
	}
	if !strings.Contains(indexHTML, "pixelMageAvatarSvg") || !strings.Contains(indexHTML, "rpgSpriteBob") {
		t.Fatalf("indexHTML missing RPG pixel mage avatar sprite helper/animation")
	}
	if strings.Contains(indexHTML, `<svg class="ico`) || strings.Contains(loginHTML, `<svg class="ico`) {
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
		if !strings.Contains(indexHTML, rpgSym) {
			t.Fatalf("indexHTML missing expected 32-bit RPG art direction component %q", rpgSym)
		}
	}
	for _, tok := range []string{"#e6d3bd", "#1b3b6f", "#7a1c2e", "#c24914", "#1a1310"} {
		if !strings.Contains(indexHTML, tok) {
			t.Fatalf("indexHTML missing expected theme color token %s", tok)
		}
	}
}

func TestParseAgentAndGoalCommand(t *testing.T) {
	cases := []struct {
		raw      string
		goalFlag bool
		agentIn  string
		isGoal   bool
		isCancel bool
		agentOut string
		prompt   string
	}{
		{"/ai-slop Audit this repo", false, "", false, false, "ai-slop", "Audit this repo"},
		{"/slop Clean dead code", false, "", false, false, "ai-slop", "Clean dead code"},
		{"/brainstorm New auth design", false, "", false, false, "brainstorm", "New auth design"},
		{"/brainstrom Ide tampilan RPG", false, "", false, false, "brainstorm", "Ide tampilan RPG"},
		{"/goal /ai-slop Refactor all handlers", false, "", true, false, "ai-slop", "Refactor all handlers"},
		{"/brainstorm /goal Design SaaS billing", false, "", true, false, "brainstorm", "Design SaaS billing"},
		{"Lanjutkan pembahasannya", false, "brainstorm", false, false, "brainstorm", "Lanjutkan pembahasannya"},
		{"/goal cancel", false, "ai-slop", false, true, "", ""},
	}
	for _, tc := range cases {
		got := parseAgentAndGoalCommand(tc.raw, tc.goalFlag, tc.agentIn)
		if got.IsGoal != tc.isGoal || got.IsCancel != tc.isCancel || got.Agent != tc.agentOut || got.Prompt != tc.prompt {
			t.Fatalf("parseAgentAndGoalCommand(%q, %v, %q) = %+v, want goal=%v cancel=%v agent=%q prompt=%q",
				tc.raw, tc.goalFlag, tc.agentIn, got, tc.isGoal, tc.isCancel, tc.agentOut, tc.prompt)
		}
	}
}

func TestAgentPersonasAndHTTPAPI(t *testing.T) {
	if p := buildAgentPersonaPrompt("ai-slop"); !strings.Contains(p, "ANTI-AI-SLOP") {
		t.Fatalf("expected ai-slop persona prompt, got %q", p)
	}
	if p := buildAgentPersonaPrompt("brainstorm"); !strings.Contains(p, "BRAINSTORMING PARTNER") || !strings.Contains(p, "<!-- CHOICES:") {
		t.Fatalf("expected brainstorm persona prompt, got %q", p)
	}
	for _, uiTok := range []string{"toggleSlopBtn", "toggleBrainstormBtn", "/ai-slop", "/brainstorm"} {
		if !strings.Contains(indexHTML, uiTok) {
			t.Fatalf("indexHTML missing agent UI element %q", uiTok)
		}
	}

	var sawSlopSystem atomic.Bool
	var sawSlopTools atomic.Bool
	var sawBrainSystem atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []ChatMessage    `json:"messages"`
			Tools    []map[string]any `json:"tools"`
		}
		if json.NewDecoder(r.Body).Decode(&reqBody) == nil {
			for _, m := range reqBody.Messages {
				if strings.Contains(m.Content, "ANTI-AI-SLOP AUDITOR") {
					sawSlopSystem.Store(true)
					if len(reqBody.Tools) > 0 {
						sawSlopTools.Store(true)
					}
				}
				if strings.Contains(m.Content, "BRAINSTORMING PARTNER") {
					sawBrainSystem.Store(true)
				}
			}
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		line, _ := json.Marshal(map[string]any{
			"model":   "mock-agent",
			"message": map[string]any{"role": "assistant", "content": "Respons agent terverifikasi."},
			"done":    true,
		})
		_, _ = w.Write(append(line, '\n'))
	}))
	defer srv.Close()

	origEndpoint := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = origEndpoint }()

	// 1. Test /ai-slop via API (auto-enables tools and injects ai-slop persona)
	slopID := fmt.Sprintf("test_slop_%d", time.Now().UnixNano())
	defer func() {
		dReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+slopID, nil)
		handleChatSession(httptest.NewRecorder(), dReq)
	}()
	slopBody, _ := json.Marshal(map[string]any{
		"chat_id":     slopID,
		"model":       "mock-agent",
		"cwd":         t.TempDir(),
		"user_prompt": "/ai-slop Periksa kode duplikat",
	})
	slopReq := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(slopBody))
	slopRR := httptest.NewRecorder()
	handleChatSend(slopRR, slopReq)
	if slopRR.Code != http.StatusOK {
		t.Fatalf("handleChatSend /ai-slop failed: %d %s", slopRR.Code, slopRR.Body.String())
	}
	var sResp map[string]any
	if err := json.Unmarshal(slopRR.Body.Bytes(), &sResp); err != nil || sResp["agent"] != "ai-slop" {
		t.Fatalf("expected agent=ai-slop in response, got %s", slopRR.Body.String())
	}

	// 2. Test /brainstrom typo alias via API
	brainID := fmt.Sprintf("test_brain_%d", time.Now().UnixNano())
	defer func() {
		dReq := httptest.NewRequest(http.MethodDelete, "/api/chats/"+brainID, nil)
		handleChatSession(httptest.NewRecorder(), dReq)
	}()
	brainBody, _ := json.Marshal(map[string]any{
		"chat_id":     brainID,
		"model":       "mock-agent",
		"cwd":         t.TempDir(),
		"user_prompt": "/brainstrom Ide fitur dashboard",
	})
	brainReq := httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(brainBody))
	brainRR := httptest.NewRecorder()
	handleChatSend(brainRR, brainReq)
	if brainRR.Code != http.StatusOK {
		t.Fatalf("handleChatSend /brainstrom failed: %d %s", brainRR.Code, brainRR.Body.String())
	}
	var bResp map[string]any
	if err := json.Unmarshal(brainRR.Body.Bytes(), &bResp); err != nil || bResp["agent"] != "brainstorm" {
		t.Fatalf("expected agent=brainstorm in response, got %s", brainRR.Body.String())
	}

	// Wait for both mock runs to finish cleanly
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sawSlopSystem.Load() && sawSlopTools.Load() && sawBrainSystem.Load() {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if !sawSlopSystem.Load() || !sawSlopTools.Load() {
		t.Fatalf("expected ai-slop persona & auto-enabled tools to reach model (persona=%v tools=%v)", sawSlopSystem.Load(), sawSlopTools.Load())
	}
	if !sawBrainSystem.Load() {
		t.Fatalf("expected brainstorm persona to reach model")
	}

	// Wait until both sessions finish writing so defer cleanup won't race
	for _, cid := range []string{slopID, brainID} {
		endWait := time.Now().Add(2 * time.Second)
		for time.Now().Before(endWait) {
			gReq := httptest.NewRequest(http.MethodGet, "/api/chats/"+cid, nil)
			gRR := httptest.NewRecorder()
			handleChatSession(gRR, gReq)
			var snap ChatSession
			if json.Unmarshal(gRR.Body.Bytes(), &snap) == nil && !snap.Running {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
}

func TestUsageCalendarAndPerModelAPI(t *testing.T) {
	tmpDir := t.TempDir()
	usageMu.Lock()
	prevOverride := usageDirOverride
	prevLoaded := usageLoaded
	prevEvents := usageEvents
	usageDirOverride = tmpDir
	usageLoaded = false
	usageEvents = nil
	usageMu.Unlock()
	defer func() {
		usageMu.Lock()
		usageDirOverride = prevOverride
		usageLoaded = prevLoaded
		usageEvents = prevEvents
		usageMu.Unlock()
	}()

	// Write a sample historical chat session to verify backfill + per-model aggregation
	sess := ChatSession{
		ID:        "chat_hist_usage_1",
		Title:     "Test Usage Chat",
		Model:     "mimo-v2.6:128k",
		CreatedAt: time.Date(2026, 11, 15, 10, 0, 0, 0, time.Local).UnixMilli(),
		UpdatedAt: time.Date(2026, 11, 15, 10, 5, 0, 0, time.Local).UnixMilli(),
		Messages: []StoredMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello", Tokens: 1200, TokPerSec: 40.0, GoalCycle: 1, CreatedAt: time.Date(2026, 11, 15, 10, 2, 0, 0, time.Local).UnixMilli()},
		},
	}
	sb, _ := json.Marshal(sess)
	if err := os.WriteFile(filepath.Join(tmpDir, sess.ID+".json"), sb, 0o644); err != nil {
		t.Fatalf("write sample chat: %v", err)
	}

	// Record a second event on another day & model
	recordUsageEvent(UsageEvent{
		Key:         "chat:live_2:1",
		TimestampMS: time.Date(2026, 11, 19, 9, 0, 0, 0, time.Local).UnixMilli(),
		Model:       "qwen3.5:32k",
		Tokens:      3800,
		TokPerSec:   52.0,
	})

	req := httptest.NewRequest(http.MethodGet, "/api/usage?month=2026-11", nil)
	rr := httptest.NewRecorder()
	handleUsageAPI(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from handleUsageAPI, got %d: %s", rr.Code, rr.Body.String())
	}
	var res UsageSummaryResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
		t.Fatalf("decode usage response: %v", err)
	}
	if res.TotalTokens != 5000 || res.MonthTokens != 5000 {
		t.Fatalf("expected 5000 total/month tokens, got total=%d month=%d", res.TotalTokens, res.MonthTokens)
	}
	if len(res.Daily) != 2 {
		t.Fatalf("expected 2 active days in 2026-11, got %d", len(res.Daily))
	}
	if len(res.ByModel) != 2 || res.ByModel[0].Model != "qwen3.5:32k" || res.ByModel[0].Tokens != 3800 {
		t.Fatalf("unexpected ByModel breakdown: %+v", res.ByModel)
	}
	if res.PeakDay.Date != "2026-11-19" || res.PeakDay.Tokens != 3800 {
		t.Fatalf("unexpected PeakDay: %+v", res.PeakDay)
	}
	if len(res.Recommendations) == 0 {
		t.Fatalf("expected smart usage recommendations to be populated")
	}
}

func TestMemoryStatusAndFlushAPI(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/memory", nil)
	rr := httptest.NewRecorder()
	handleMemoryStatus(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from handleMemoryStatus, got %d", rr.Code)
	}
	var st MemoryStatusResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &st); err != nil {
		t.Fatalf("decode memory status: %v", err)
	}
	if st.RAM.TotalMiB <= 0 || st.Runtime.Goroutines <= 0 {
		t.Fatalf("expected live RAM and Go runtime metrics, got RAM=%+v RT=%+v", st.RAM, st.Runtime)
	}

	// Test safe Go GC flush
	fReq := httptest.NewRequest(http.MethodPost, "/api/memory/flush", strings.NewReader(`{"target":"go_gc"}`))
	fRR := httptest.NewRecorder()
	handleMemoryFlush(fRR, fReq)
	if fRR.Code != http.StatusOK {
		t.Fatalf("expected 200 from handleMemoryFlush go_gc, got %d: %s", fRR.Code, fRR.Body.String())
	}
	var fResp map[string]any
	if err := json.Unmarshal(fRR.Body.Bytes(), &fResp); err != nil || fResp["ok"] != true {
		t.Fatalf("expected ok=true from go_gc flush, got %s", fRR.Body.String())
	}

	// Test invalid target validation
	badReq := httptest.NewRequest(http.MethodPost, "/api/memory/flush", strings.NewReader(`{"target":"invalid_target"}`))
	badRR := httptest.NewRecorder()
	handleMemoryFlush(badRR, badReq)
	if badRR.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on invalid flush target, got %d", badRR.Code)
	}
}

func TestHallucinatedMediaToolCallRescueAndNativeToolHistory(t *testing.T) {
	// 1. Exact screenshot format with parentheses around JSON
	hallucinated := `[Executed tool generate_image ({"denoise":0.18,"negative_prompt":"No text","prompt":"raw photo, candid portrait in Tokyo at golden hour, facing right","reference_image_path":"auto","seed":7391,"style":"general"}) -> Image generated successfully!
Mode: img2img
File: /home/rizal/Documents/chat/selamat-pagi-5051250534/web_gen_1795051318600.png]

Selesai! Mukanya sekarang menghadap ke kanan.`

	tc, ok := parseHallucinatedMediaToolCall(hallucinated)
	if !ok {
		t.Fatalf("expected parseHallucinatedMediaToolCall to rescue hallucinated generate_image text")
	}
	if tc.Function.Name != "generate_image" {
		t.Fatalf("expected tool name generate_image, got %q", tc.Function.Name)
	}
	if p, _ := tc.Function.Arguments["prompt"].(string); !strings.Contains(p, "facing right") {
		t.Fatalf("expected prompt to be extracted cleanly, got %q", p)
	}
	if st, _ := tc.Function.Arguments["style"].(string); st != "general" {
		t.Fatalf("expected style=general, got %q", st)
	}

	// 2. Video format without parentheses
	vidText := `[Executed tool generate_video {"prompt":"slow cinematic pan over neon Tokyo","seconds":3} -> Video generated]`
	vtc, ok2 := parseHallucinatedMediaToolCall(vidText)
	if !ok2 || vtc.Function.Name != "generate_video" {
		t.Fatalf("expected generate_video rescue, got ok=%v name=%q", ok2, vtc.Function.Name)
	}

	// 3. Normal text should NOT be falsely rescued
	if _, ok3 := parseHallucinatedMediaToolCall("Halo, ini jawaban biasa tanpa tool."); ok3 {
		t.Fatalf("normal text must not trigger hallucinated tool rescue")
	}
}

func TestRuntimeLogMitigationAnalyzerAndAPI(t *testing.T) {
	lines := []RuntimeLogLine{
		{Source: "ollama", Level: "error", Message: "CUDA out of memory while allocating kv cache"},
		{Source: "ollama", Level: "info", Message: "offloaded 22/33 layers to GPU"},
		{Source: "model", Level: "error", Message: "dial tcp 127.0.0.1:11434: connection refused"},
	}
	alerts := analyzeLogLinesForMitigations(lines, RAMMetrics{}, GPUMetrics{})
	if len(alerts) < 3 {
		t.Fatalf("expected >=3 mitigation alerts for OOM + partial offload + conn refused, got %+v", alerts)
	}
	hasOOM, hasOffload, hasConn := false, false, false
	for _, a := range alerts {
		if a.ActionType == "flush_vram" {
			hasOOM = true
		}
		if strings.Contains(a.Title, "Partial GPU Offload") {
			hasOffload = true
		}
		if a.ActionType == "restart_ollama" {
			hasConn = true
		}
	}
	if !hasOOM || !hasOffload || !hasConn {
		t.Fatalf("missing expected mitigations (oom=%v offload=%v conn=%v): %+v", hasOOM, hasOffload, hasConn, alerts)
	}

	healthy := analyzeLogLinesForMitigations([]RuntimeLogLine{{Source: "ollama", Level: "info", Message: "server listening on 11434"}}, RAMMetrics{}, GPUMetrics{})
	if len(healthy) != 1 || healthy[0].Severity != "ok" {
		t.Fatalf("expected single ok alert when healthy, got %+v", healthy)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/memory/logs", nil)
	rr := httptest.NewRecorder()
	handleMemoryLogs(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from handleMemoryLogs, got %d", rr.Code)
	}
	var resp MemoryLogsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode MemoryLogsResponse: %v", err)
	}
	if len(resp.Mitigations) == 0 {
		t.Fatalf("expected at least 1 mitigation/status entry in handleMemoryLogs response")
	}
}

func TestChatCheckpointTruncateAndDenoisePreservation(t *testing.T) {
	// 1. Explicit denoise=0.00 must clamp to 0.05 (not fall back to 0.62)
	tmpFile := filepath.Join(t.TempDir(), "ref.png")
	if err := os.WriteFile(tmpFile, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}
	aZero := map[string]any{"prompt": "1girl", "style": "illustration", "denoise": 0.0}
	prepareGenerateImageArgs(tmpFile, aZero)
	if aZero["denoise"] != 0.05 {
		t.Fatalf("expected explicit denoise=0.0 to clamp to 0.05, got %v", aZero["denoise"])
	}
	aLow := map[string]any{"prompt": "1girl", "style": "illustration", "denoise": 0.15}
	prepareGenerateImageArgs(tmpFile, aLow)
	if aLow["denoise"] != 0.15 {
		t.Fatalf("expected explicit denoise=0.15 to be preserved, got %v", aLow["denoise"])
	}

	// 2. Test handleChatSend truncate_from checkpoint rewind
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"message":{"content":"re-executed"},"done":true,"eval_count":5,"eval_duration":100000000}` + "\n"))
	}))
	defer srv.Close()

	prevEP := ollamaChatEndpoint
	ollamaChatEndpoint = srv.URL
	defer func() { ollamaChatEndpoint = prevEP }()

	cid := fmt.Sprintf("test_cp_%d", time.Now().UnixNano())
	defer func() {
		handleChatSession(httptest.NewRecorder(), httptest.NewRequest(http.MethodDelete, "/api/chats/"+cid, nil))
	}()

	// Seed session with 4 messages (turn 0 + turn 1)
	seed := ChatSession{
		ID:        cid,
		Title:     "Checkpoint Test",
		Model:     "mock",
		Cwd:       t.TempDir(),
		CreatedAt: time.Now().UnixMilli(),
		UpdatedAt: time.Now().UnixMilli(),
		Messages: []StoredMessage{
			{Role: "user", Content: "msg 0"},
			{Role: "assistant", Content: "ans 0"},
			{Role: "user", Content: "msg 1 old"},
			{Role: "assistant", Content: "ans 1 failed", Error: "interrupted"},
		},
	}
	saveSessionFile(seed)

	// Rewind to index 2 and send edited prompt
	cut := 2
	body, _ := json.Marshal(BackgroundSendRequest{
		ChatID:       cid,
		Model:        "mock",
		Cwd:          seed.Cwd,
		UserPrompt:   "msg 1 edited",
		TruncateFrom: &cut,
	})
	rr := httptest.NewRecorder()
	handleChatSend(rr, httptest.NewRequest(http.MethodPost, "/api/chats/send", bytes.NewReader(body)))
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 on checkpoint edit send, got %d: %s", rr.Code, rr.Body.String())
	}

	// Wait for mock run to finish
	deadline := time.Now().Add(3 * time.Second)
	var snap ChatSession
	for time.Now().Before(deadline) {
		gRR := httptest.NewRecorder()
		handleChatSession(gRR, httptest.NewRequest(http.MethodGet, "/api/chats/"+cid, nil))
		if json.Unmarshal(gRR.Body.Bytes(), &snap) == nil && !snap.Running {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if len(snap.Messages) != 4 || snap.Messages[2].Content != "msg 1 edited" || snap.Messages[3].Content != "re-executed" {
		t.Fatalf("unexpected truncated messages after checkpoint retry: %+v", snap.Messages)
	}
}

func TestSecurityPathTraversalAndSecretsBlocked(t *testing.T) {
	// 1. Sensitive dotfiles & auth configs must be blocked on download & media
	for _, blockedPath := range []string{
		"/home/rizal/.git-credentials",
		"/home/rizal/.ssh/id_rsa",
		"/home/rizal/Documents/ollama/ai-dashboard/auth_config.json",
		"/home/rizal/Documents/ollama/ai-dashboard/.git/config",
		"/etc/passwd",
		"/tmp/../etc/passwd",
	} {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/chat-download?path="+url.QueryEscape(blockedPath), nil)
		handleChatDownload(rr, req)
		if rr.Code != http.StatusForbidden && rr.Code != http.StatusNotFound {
			t.Fatalf("expected 403/404 for sensitive path %q, got %d", blockedPath, rr.Code)
		}
	}

	// 2. Symlink escape from /tmp -> /etc/passwd must be blocked by EvalSymlinks
	tmpDir := t.TempDir()
	symEscape := filepath.Join(tmpDir, "escape.txt")
	if err := os.Symlink("/etc/passwd", symEscape); err == nil {
		rr := httptest.NewRecorder()
		handleChatDownload(rr, httptest.NewRequest(http.MethodGet, "/api/chat-download?path="+url.QueryEscape(symEscape), nil))
		if rr.Code != http.StatusForbidden {
			t.Fatalf("expected 403 for symlink escape %q, got %d", symEscape, rr.Code)
		}
	}

	// 3. /api/chat-media must reject non-media files (e.g. .txt/.go)
	txtFile := filepath.Join(tmpDir, "notes.txt")
	_ = os.WriteFile(txtFile, []byte("hello"), 0o644)
	rrTxt := httptest.NewRecorder()
	handleChatMedia(rrTxt, httptest.NewRequest(http.MethodGet, "/api/chat-media?path="+url.QueryEscape(txtFile), nil))
	if rrTxt.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for non-media file on /api/chat-media, got %d", rrTxt.Code)
	}

	// 4. Valid PNG in /tmp serves 200 with nosniff
	pngFile := filepath.Join(tmpDir, "ok.png")
	_ = os.WriteFile(pngFile, []byte("\x89PNG\r\n\x1a\n"), 0o644)
	rrPng := httptest.NewRecorder()
	handleChatMedia(rrPng, httptest.NewRequest(http.MethodGet, "/api/chat-media?path="+url.QueryEscape(pngFile), nil))
	if rrPng.Code != http.StatusOK {
		t.Fatalf("expected 200 for valid png in /tmp, got %d", rrPng.Code)
	}
	if rrPng.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("expected X-Content-Type-Options: nosniff on served media")
	}
}

func TestWorkspaceZipExport(t *testing.T) {
	wsDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(wsDir, "index.html"), []byte("<h1>Hello Vibecode</h1>"), 0o644)
	_ = os.MkdirAll(filepath.Join(wsDir, "assets"), 0o755)
	_ = os.WriteFile(filepath.Join(wsDir, "assets", "hero.png"), []byte("\x89PNG\r\n\x1a\n"), 0o644)
	// Sensitive/ignored files that must NOT be included
	_ = os.MkdirAll(filepath.Join(wsDir, ".git"), 0o755)
	_ = os.WriteFile(filepath.Join(wsDir, ".git", "config"), []byte("[core]"), 0o644)
	_ = os.WriteFile(filepath.Join(wsDir, ".env"), []byte("SECRET=123"), 0o644)

	cid := fmt.Sprintf("test_zip_%d", time.Now().UnixNano())
	defer func() {
		handleChatSession(httptest.NewRecorder(), httptest.NewRequest(http.MethodDelete, "/api/chats/"+cid, nil))
	}()
	saveSessionFile(ChatSession{
		ID:    cid,
		Title: "Vibecode Landing Page",
		Cwd:   wsDir,
	})

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/chats/"+cid+"/zip", nil)
	handleChatSession(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from /zip, got %d: %s", rr.Code, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/zip" {
		t.Fatalf("expected application/zip, got %q", ct)
	}

	zr, err := zip.NewReader(bytes.NewReader(rr.Body.Bytes()), int64(rr.Body.Len()))
	if err != nil {
		t.Fatalf("invalid zip output: %v", err)
	}
	found := map[string]bool{}
	for _, f := range zr.File {
		found[f.Name] = true
	}
	if !found["index.html"] || !found["assets/hero.png"] {
		t.Fatalf("expected index.html and assets/hero.png in zip, got %+v", found)
	}
	if found[".git/config"] || found[".env"] {
		t.Fatalf("sensitive files leaked into zip: %+v", found)
	}
}

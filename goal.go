package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// GoalFileStat summarizes per-file line additions and deletions recorded in a checkpoint.
type GoalFileStat struct {
	Path    string `json:"path"`
	Added   int    `json:"added"`
	Deleted int    `json:"deleted"`
}

// GoalCheckpoint records a restorable shadow-git snapshot of the workspace during a /goal run.
type GoalCheckpoint struct {
	Commit       string         `json:"commit"`
	Cycle        int            `json:"cycle"`
	Label        string         `json:"label"`
	FilesChanged int            `json:"files_changed"`
	FileStats    []GoalFileStat `json:"file_stats,omitempty"`
	CreatedAt    int64          `json:"created_at"`
}

// GoalState tracks the lifecycle and progress of an autonomous /goal engineering run.
type GoalState struct {
	Active          bool             `json:"active"`
	Status          string           `json:"status"` // "running", "auditing", "completed", "cancelled", "failed", "max_cycles", "interrupted"
	Prompt          string           `json:"prompt"`
	Cycle           int              `json:"cycle"`
	MaxCycles       int              `json:"max_cycles"`
	TotalToolTurns  int              `json:"total_tool_turns"`
	TotalToolCalls  int              `json:"total_tool_calls"`
	LastPhase       string           `json:"last_phase,omitempty"`
	StartedAt       int64            `json:"started_at"`
	UpdatedAt       int64            `json:"updated_at"`
	FinishedAt      int64            `json:"finished_at,omitempty"`
	CancelRequested bool             `json:"cancel_requested,omitempty"`
	PlanMarkdown    string           `json:"plan_markdown,omitempty"`
	Agent           string           `json:"agent,omitempty"`
	Checkpoints     []GoalCheckpoint `json:"checkpoints,omitempty"`
}

// GoalParsedCommand represents the parsed intent from a user message.
type GoalParsedCommand struct {
	IsGoal   bool
	IsCancel bool
	Agent    string
	Prompt   string
}

// normalizeAgentName maps user/slash aliases to canonical agent IDs ("ai-slop", "brainstorm", or "").
func normalizeAgentName(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "ai-slop", "aislop", "slop", "anti-slop":
		return "ai-slop"
	case "brainstorm", "brainstrom", "ideation":
		return "brainstorm"
	default:
		return ""
	}
}

const (
	goalCompleteMarker  = "<!-- GOAL_COMPLETE -->"
	goalCancelledMarker = "<!-- GOAL_CANCELLED -->"
	defaultGoalCycles   = 30
	goalTurnsPerCycle   = 24
)

// parseGoalCommand detects `/goal ...` or `/goal cancel` commands from user input.
// When explicitGoalMode is true, non-cancel messages are also treated as goal tasks.
func parseGoalCommand(raw string, explicitGoalMode bool) GoalParsedCommand {
	return parseAgentAndGoalCommand(raw, explicitGoalMode, "")
}

// parseAgentAndGoalCommand detects `/goal`, `/ai-slop` (or `/slop`), and `/brainstorm` (or `/brainstrom`)
// prefixes in any order and extracts the clean prompt body plus active agent ID.
func parseAgentAndGoalCommand(raw string, explicitGoalMode bool, explicitAgent string) GoalParsedCommand {
	trimmed := strings.TrimSpace(raw)
	agent := normalizeAgentName(explicitAgent)
	isGoal := explicitGoalMode

	consumeLeadingToken := func(tok, text string) (matched bool, rest string) {
		low := strings.ToLower(text)
		if !strings.HasPrefix(low, tok) {
			return false, text
		}
		tail := text[len(tok):]
		if tail == "" || tail[0] == ' ' || tail[0] == '\t' || tail[0] == '\n' || tail[0] == '\r' {
			return true, strings.TrimSpace(tail)
		}
		return false, text
	}

	// Iteratively consume leading slash tokens so combinations like `/goal /ai-slop ...` or `/brainstorm /goal ...` work cleanly.
	cur := trimmed
	for step := 0; step < 3; step++ {
		lowCur := strings.ToLower(cur)
		if lowCur == "/cancel" || lowCur == "/stop" {
			return GoalParsedCommand{IsCancel: true}
		}
		if ok, rest := consumeLeadingToken("/goal", cur); ok {
			rLow := strings.ToLower(rest)
			if strings.HasSuffix(rLow, "/goal") {
				cand := strings.TrimSpace(rest[:len(rest)-len("/goal")])
				if cand != "" {
					rest = cand
					rLow = strings.ToLower(rest)
				}
			}
			switch rLow {
			case "cancel", "stop", "abort", "batal", "batalkan":
				return GoalParsedCommand{IsCancel: true}
			}
			isGoal = true
			cur = rest
			continue
		}
		if ok, rest := consumeLeadingToken("/ai-slop", cur); ok {
			agent = "ai-slop"
			cur = rest
			continue
		}
		if ok, rest := consumeLeadingToken("/slop", cur); ok {
			agent = "ai-slop"
			cur = rest
			continue
		}
		if ok, rest := consumeLeadingToken("/brainstorm", cur); ok {
			agent = "brainstorm"
			cur = rest
			continue
		}
		if ok, rest := consumeLeadingToken("/brainstrom", cur); ok {
			agent = "brainstorm"
			cur = rest
			continue
		}
		break
	}

	return GoalParsedCommand{
		IsGoal: isGoal,
		Agent:  agent,
		Prompt: cur,
	}
}

// buildAgentPersonaPrompt returns specialized system instructions when an agent mode is selected.
func buildAgentPersonaPrompt(agent string) string {
	switch normalizeAgentName(agent) {
	case "ai-slop":
		return `SPECIALIZED AGENT: ANTI-AI-SLOP AUDITOR & CRAFTSMANSHIP ENGINEER ("ai-slop")
Your mission is to detect, remove, and replace generic "AI slop" across code, UI, and writing so the output meets senior human-craftsmanship standards:
1. Code Slop Elimination: hunt down and eliminate over-engineered wrapper abstractions, dead/unreachable functions, unused imports/variables, placeholder TODOs/stubs/mocks left in production paths, empty catch blocks or silently swallowed errors, and duplicated logic (extract shared helpers instead).
2. UI & Copy Slop Elimination: avoid generic filler praise, buzzword bloat, stock generic templates, poor contrast, or cramped unaligned elements. Keep wording crisp, direct, and purposeful.
3. Truthful Engineering Discipline: read real files with find_files/grep_search/read_file first, cite only verified paths/symbols, make minimal surgical edits via edit_file, and run actual builds/linters/tests via bash to prove zero regressions.`
	case "brainstorm":
		return `SPECIALIZED AGENT: ARCHITECT & PRODUCT BRAINSTORMING PARTNER ("brainstorm")
Your mission is to help the user explore ideas deeply and make strong architectural, UX, and product decisions BEFORE writing code:
1. Multi-Angle Exploration: analyze the problem across user experience, technical architecture, data model, edge cases, failure modes, and MVP vs v2 scope.
2. Concrete Trade-off Options: always present 2-3 distinct, practical approaches in a concise Markdown comparison table (Option, How It Works, Pros, Cons/Risks, When to Pick) — avoid vague platitudes.
3. Interactive Convergence: ask 2-4 crisp clarifying questions to resolve key design forks, and end with <!-- CHOICES: ["Option 1", "Option 2", "Option 3"] --> so the user can click to decide or convert the chosen plan into /goal execution.
4. Grounded Context: if workspace tools are enabled, quickly check existing files with find_files/read_file so your recommendations match the real project stack.`
	default:
		return ""
	}
}

func inferGoalNumCtx(model string) int {
	low := strings.ToLower(model)
	switch {
	case strings.Contains(low, "128k") || strings.Contains(low, "256k") || strings.Contains(low, "64k"):
		return 65536
	default:
		return 32768
	}
}

func readGoalPlanMarkdown(cwd string) string {
	if strings.TrimSpace(cwd) == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(cwd, ".goal_plan.md"))
	if err != nil {
		return ""
	}
	txt := strings.TrimSpace(string(b))
	if len(txt) > 4000 {
		txt = txt[:4000] + "\n... [truncated]"
	}
	return txt
}

// listWorkspaceTreeCompact returns a deterministic, bounded file listing under cwd
// skipping heavy dependency/vcs directories so /goal always sees real project paths.
func listWorkspaceTreeCompact(cwd string, maxFiles int) string {
	if strings.TrimSpace(cwd) == "" {
		return "(empty)"
	}
	if maxFiles <= 0 {
		maxFiles = 40
	}
	var files []string
	_ = filepath.WalkDir(cwd, func(fpath string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if name == ".git" || name == "node_modules" || name == ".venv" || name == "venv" || name == "__pycache__" || name == "dist" || name == ".next" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(cwd, fpath)
		if err != nil {
			rel = fpath
		}
		files = append(files, filepath.ToSlash(rel))
		if len(files) >= maxFiles {
			return fmt.Errorf("limit")
		}
		return nil
	})
	if len(files) == 0 {
		return "(no project files yet)"
	}
	out := strings.Join(files, "\n")
	if len(files) >= maxFiles {
		out += fmt.Sprintf("\n... [showing first %d files]", maxFiles)
	}
	return out
}

func buildGoalSystemPrompt(cwd string, agentsMD string, enableMCP bool) string {
	var parts []string
	if strings.TrimSpace(agentsMD) != "" {
		parts = append(parts, agentsMD)
	}
	parts = append(parts, fmt.Sprintf("Active Working Directory: %s\nFormat all responses in clean Markdown (headings, lists, GFM tables, fenced code blocks).", cwd))
	parts = append(parts, `AUTONOMOUS /GOAL SOFTWARE ENGINEERING MODE:
You are executing a complex, long-running software engineering goal autonomously end-to-end.
Available Terminal & File tools: bash, list_files, find_files, grep_search, read_file, edit_file, write_file.

OPERATING RULES:
1. Explore fast & accurately inside Active Working Directory: ALWAYS use find_files, grep_search, or list_files instead of running repeated shell find/cat loops. Use read_file with start_line/end_line for targeted reads. Never call the same failing or identical read command repeatedly.
2. Maintain persistent plan memory: in Cycle 1, create ".goal_plan.md" in the workspace root using write_file BEFORE building files, listing concise "- [ ]" / "- [x]" checklist milestones and updating it as steps complete so live progress stays visible to the user.
3. Prefer edit_file for targeted changes to existing files, and write_file for new files.
4. Never leave TODOs, stubs, or mock placeholders unless explicitly requested. Write complete, production-quality code.
5. Verify continuously: run builds, typechecks, linters, and unit/integration tests via bash after changes, and fix any failures yourself.
6. Communicate progress clearly: briefly state (1-2 sentences) what was accomplished and what exact step you are executing next so each cycle reads cleanly for the user. Do NOT ask for confirmation; keep executing until complete.
7. COMPLETION PROTOCOL:
   - ONLY when ALL requirements are implemented AND verified with passing checks/tests, respond with a concise final report and include the exact marker: <!-- GOAL_COMPLETE -->
   - If the user explicitly asks to stop or cancel the goal, include the exact marker: <!-- GOAL_CANCELLED -->`)

	if enableMCP {
		parts = append(parts, "MCP media tools (`generate_image`, `generate_video`) are also available if the goal requires generating visual assets.")
	}
	return strings.Join(parts, "\n\n")
}

// compactGoalMessages keeps the system prompt, original PRD/goal prompt, and most recent messages
// while summarizing older middle turns when the conversation grows beyond maxChars.
func compactGoalMessages(msgs []ChatMessage, maxChars int) []ChatMessage {
	return compactGoalMessagesWithPlan(msgs, maxChars, "")
}

func compactGoalMessagesWithPlan(msgs []ChatMessage, maxChars int, cwd string) []ChatMessage {
	if len(msgs) <= 14 || maxChars <= 0 {
		return msgs
	}
	total := 0
	for _, m := range msgs {
		total += len(m.Content) + len(m.Thinking)
		for _, tc := range m.ToolCalls {
			total += len(tc.Function.Name) + 64
			for k, v := range tc.Function.Arguments {
				total += len(k) + len(fmt.Sprintf("%v", v))
			}
		}
	}
	if total <= maxChars {
		return msgs
	}

	// Preserve leading system prompt (index 0) and initial user goal/PRD messages (up to first assistant)
	headEnd := 1
	for headEnd < len(msgs) && msgs[headEnd].Role == "user" && headEnd < 3 {
		headEnd++
	}
	const keepTail = 10
	if len(msgs)-keepTail <= headEnd {
		return msgs
	}
	midStart := headEnd
	midEnd := len(msgs) - keepTail
	// Never start the kept tail on an orphaned "tool" message; advance past tool responses
	// so assistant(tool_calls) -> tool(output) pairs stay intact.
	for midEnd < len(msgs) && msgs[midEnd].Role == "tool" {
		midEnd++
	}
	if midEnd <= midStart || len(msgs)-midEnd < 2 {
		return msgs
	}

	var summaryLines []string
	var touchedFiles []string
	seenFiles := map[string]bool{}
	var commands []string

	for i := midStart; i < midEnd; i++ {
		m := msgs[i]
		for _, tc := range m.ToolCalls {
			name := strings.ToLower(tc.Function.Name)
			switch name {
			case "write_file", "write", "edit_file", "edit":
				p, _ := tc.Function.Arguments["path"].(string)
				if p == "" {
					p, _ = tc.Function.Arguments["file_path"].(string)
				}
				if p != "" && !seenFiles[p] {
					seenFiles[p] = true
					touchedFiles = append(touchedFiles, p)
				}
			case "bash", "sh":
				cmd, _ := tc.Function.Arguments["command"].(string)
				if cmd == "" {
					cmd, _ = tc.Function.Arguments["cmd"].(string)
				}
				cmd = strings.TrimSpace(cmd)
				if cmd != "" {
					if len(cmd) > 90 {
						cmd = cmd[:90] + "..."
					}
					commands = append(commands, cmd)
				}
			}
		}
		if m.Role == "assistant" && strings.TrimSpace(m.Content) != "" && len(summaryLines) < 6 {
			snippet := strings.Join(strings.Fields(m.Content), " ")
			if len(snippet) > 160 {
				snippet = snippet[:160] + "..."
			}
			summaryLines = append(summaryLines, "- "+snippet)
		}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("[COMPACTED GOAL HISTORY: %d earlier messages summarized to preserve context window]\n", midEnd-midStart))
	if strings.TrimSpace(cwd) != "" {
		if planBytes, err := os.ReadFile(filepath.Join(cwd, ".goal_plan.md")); err == nil {
			planTxt := strings.TrimSpace(string(planBytes))
			if planTxt != "" {
				if len(planTxt) > 4000 {
					planTxt = planTxt[:4000] + "\n... [truncated]"
				}
				b.WriteString("Current .goal_plan.md checklist:\n" + planTxt + "\n\n")
			}
		}
	}
	if len(touchedFiles) > 0 {
		if len(touchedFiles) > 25 {
			touchedFiles = touchedFiles[len(touchedFiles)-25:]
		}
		b.WriteString("Files written/edited so far:\n- " + strings.Join(touchedFiles, "\n- ") + "\n")
	}
	if len(commands) > 0 {
		if len(commands) > 12 {
			commands = commands[len(commands)-12:]
		}
		b.WriteString("Recent bash commands executed:\n- " + strings.Join(commands, "\n- ") + "\n")
	}
	if len(summaryLines) > 0 {
		b.WriteString("Key notes:\n" + strings.Join(summaryLines, "\n"))
	}

	out := make([]ChatMessage, 0, headEnd+1+len(msgs)-midEnd)
	out = append(out, msgs[:headEnd]...)
	out = append(out, ChatMessage{
		Role:    "user",
		Content: b.String(),
	})
	out = append(out, msgs[midEnd:]...)
	return out
}

var reHexCommit = regexp.MustCompile(`^[0-9a-fA-F]{7,40}$`)

func getCheckpointRepoDir(chatID string) string {
	id := safeChatID(chatID)
	if id == "" {
		return ""
	}
	return filepath.Join(getChatsDir(), "checkpoints", id+".git")
}

func deleteGoalCheckpoints(chatID string) {
	dir := getCheckpointRepoDir(chatID)
	if dir != "" {
		_ = os.RemoveAll(dir)
	}
}

func runShadowGit(shadowDir, cwd string, args ...string) (string, error) {
	fullArgs := append([]string{"--git-dir=" + shadowDir, "--work-tree=" + cwd}, args...)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", fullArgs...)
	cmd.Dir = cwd
	cmd.Env = append(os.Environ(),
		"GIT_TERMINAL_PROMPT=0",
		"GIT_AUTHOR_NAME=ai-dashboard",
		"GIT_AUTHOR_EMAIL=agent@local",
		"GIT_COMMITTER_NAME=ai-dashboard",
		"GIT_COMMITTER_EMAIL=agent@local",
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	return out.String(), err
}

func ensureShadowGitRepo(shadowDir, cwd string) error {
	if _, err := os.Stat(filepath.Join(shadowDir, "HEAD")); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(shadowDir), 0o755); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	initCmd := exec.CommandContext(ctx, "git", "init", "--bare", shadowDir)
	if out, err := initCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git init bare failed: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	excludeContent := strings.Join([]string{
		".git/",
		".git",
		"node_modules/",
		".venv/",
		"venv/",
		"__pycache__/",
		"dist/",
		"build/",
		".next/",
		"target/",
		"*.log",
	}, "\n") + "\n"
	infoDir := filepath.Join(shadowDir, "info")
	_ = os.MkdirAll(infoDir, 0o755)
	_ = os.WriteFile(filepath.Join(infoDir, "exclude"), []byte(excludeContent), 0o644)
	return nil
}

// createGoalCheckpoint takes a non-intrusive snapshot of cwd in an isolated shadow git repo
// so existing .git repositories, branches, and staging areas are never touched.
func createGoalCheckpoint(chatID, cwd string, cycle int, label string) (*GoalCheckpoint, error) {
	if _, err := exec.LookPath("git"); err != nil {
		return nil, nil
	}
	shadowDir := getCheckpointRepoDir(chatID)
	if shadowDir == "" || strings.TrimSpace(cwd) == "" {
		return nil, nil
	}
	if info, err := os.Stat(cwd); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("workspace dir unavailable: %s", cwd)
	}
	if err := ensureShadowGitRepo(shadowDir, cwd); err != nil {
		return nil, err
	}

	if out, err := runShadowGit(shadowDir, cwd, "add", "-A"); err != nil {
		return nil, fmt.Errorf("shadow git add failed: %v (%s)", err, strings.TrimSpace(out))
	}
	stOut, _ := runShadowGit(shadowDir, cwd, "status", "--porcelain")
	changed := 0
	for _, ln := range strings.Split(strings.TrimSpace(stOut), "\n") {
		if strings.TrimSpace(ln) != "" {
			changed++
		}
	}
	_, headErr := runShadowGit(shadowDir, cwd, "rev-parse", "--verify", "HEAD")
	if headErr == nil && changed == 0 && cycle > 0 {
		return nil, nil
	}

	if strings.TrimSpace(label) == "" {
		label = fmt.Sprintf("Cycle %d checkpoint", cycle)
	}
	if out, err := runShadowGit(shadowDir, cwd, "commit", "--allow-empty", "-m", label); err != nil {
		return nil, fmt.Errorf("shadow git commit failed: %v (%s)", err, strings.TrimSpace(out))
	}
	hashOut, err := runShadowGit(shadowDir, cwd, "rev-parse", "HEAD")
	if err != nil {
		return nil, err
	}
	var fileStats []GoalFileStat
	if cycle > 0 {
		if numOut, nerr := runShadowGit(shadowDir, cwd, "show", "--numstat", "--format=", "HEAD"); nerr == nil {
			for _, ln := range strings.Split(strings.TrimSpace(numOut), "\n") {
				parts := strings.SplitN(ln, "\t", 3)
				if len(parts) != 3 {
					continue
				}
				p := strings.TrimSpace(parts[2])
				if p == "" {
					continue
				}
				addN, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
				delN, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
				fileStats = append(fileStats, GoalFileStat{
					Path:    p,
					Added:   addN,
					Deleted: delN,
				})
				if len(fileStats) >= 24 {
					break
				}
			}
		}
	}
	return &GoalCheckpoint{
		Commit:       strings.TrimSpace(hashOut),
		Cycle:        cycle,
		Label:        label,
		FilesChanged: changed,
		FileStats:    fileStats,
		CreatedAt:    time.Now().UnixMilli(),
	}, nil
}

func getGoalCheckpointDiff(chatID, cwd, commit string) (string, error) {
	if !reHexCommit.MatchString(strings.TrimSpace(commit)) {
		return "", fmt.Errorf("invalid commit hash")
	}
	shadowDir := getCheckpointRepoDir(chatID)
	if shadowDir == "" {
		return "", fmt.Errorf("invalid chat id")
	}
	out, err := runShadowGit(shadowDir, cwd, "show", "--stat", "--patch", "--no-color", strings.TrimSpace(commit))
	if err != nil {
		return "", fmt.Errorf("diff failed: %v (%s)", err, strings.TrimSpace(out))
	}
	if len(out) > 20000 {
		out = out[:20000] + "\n... [diff truncated]"
	}
	return out, nil
}

// rollbackGoalCheckpoint surgically restores the workspace to targetCommit while first saving
// a safety snapshot of any uncommitted work and never touching ignored dirs like .git or node_modules.
func rollbackGoalCheckpoint(chatID, cwd, targetCommit string) (restored int, deleted int, err error) {
	targetCommit = strings.TrimSpace(targetCommit)
	if !reHexCommit.MatchString(targetCommit) {
		return 0, 0, fmt.Errorf("invalid checkpoint commit hash")
	}
	shadowDir := getCheckpointRepoDir(chatID)
	if shadowDir == "" {
		return 0, 0, fmt.Errorf("missing chat id")
	}
	if _, err := runShadowGit(shadowDir, cwd, "cat-file", "-e", targetCommit+"^{commit}"); err != nil {
		return 0, 0, fmt.Errorf("checkpoint commit not found: %s", targetCommit)
	}

	// Capture any uncommitted work first so even rollback itself is reversible
	_, _ = createGoalCheckpoint(chatID, cwd, 0, "Pre-rollback safety snapshot")

	headOut, err := runShadowGit(shadowDir, cwd, "rev-parse", "HEAD")
	if err != nil {
		return 0, 0, err
	}
	headCommit := strings.TrimSpace(headOut)

	diffOut, err := runShadowGit(shadowDir, cwd, "diff", "--name-status", targetCommit, headCommit)
	if err != nil {
		return 0, 0, fmt.Errorf("diff failed: %v (%s)", err, strings.TrimSpace(diffOut))
	}

	for _, rawLine := range strings.Split(strings.TrimSpace(diffOut), "\n") {
		ln := strings.TrimSpace(rawLine)
		if ln == "" {
			continue
		}
		fields := strings.Split(ln, "\t")
		if len(fields) < 2 {
			continue
		}
		status := fields[0]
		relPath := fields[len(fields)-1]
		if strings.HasPrefix(status, "A") {
			absPath := resolvePathInCwd(cwd, relPath)
			if absPath != filepath.Clean(cwd) && strings.HasPrefix(absPath, filepath.Clean(cwd)+string(filepath.Separator)) {
				if os.Remove(absPath) == nil {
					deleted++
				}
			}
		} else {
			restored++
		}
	}

	if out, err := runShadowGit(shadowDir, cwd, "checkout", targetCommit, "--", "."); err != nil {
		return restored, deleted, fmt.Errorf("checkout failed: %v (%s)", err, strings.TrimSpace(out))
	}
	return restored, deleted, nil
}

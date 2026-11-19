package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrepareGenerateImageArgsRandomSeedAndRef(t *testing.T) {
	// 1. No attachment this turn + LLM passes "auto" -> must force "none" (txt2img) + random distinct seeds
	a1 := map[string]any{"prompt": "same prompt", "reference_image_path": "auto"}
	a2 := map[string]any{"prompt": "same prompt", "reference_image_path": ""}
	prepareGenerateImageArgs("", a1)
	prepareGenerateImageArgs("", a2)

	if a1["reference_image_path"] != "none" || a1["denoise"] != 1.0 {
		t.Fatalf("expected txt2img ref=none denoise=1.0, got %+v", a1)
	}
	if a2["reference_image_path"] != "none" || a2["denoise"] != 1.0 {
		t.Fatalf("expected txt2img ref=none denoise=1.0, got %+v", a2)
	}
	s1, ok1 := a1["seed"].(int64)
	s2, ok2 := a2["seed"].(int64)
	if !ok1 || !ok2 || s1 <= 0 || s2 <= 0 || s1 == s2 {
		t.Fatalf("expected distinct positive random seeds, got %v vs %v", s1, s2)
	}

	// 2. Current turn HAS attached image -> must resolve to currentTurnPastedPath (img2img)
	tmpFile := filepath.Join(t.TempDir(), "ref.png")
	if err := os.WriteFile(tmpFile, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}
	a3 := map[string]any{"prompt": "fix hands", "reference_image_path": ""}
	prepareGenerateImageArgs(tmpFile, a3)
	if a3["reference_image_path"] != tmpFile || a3["denoise"] != 0.15 {
		t.Fatalf("expected img2img ref=%s denoise=0.15, got %+v", tmpFile, a3)
	}

	// 3. Current turn HAS attached image + illustration style transfer -> defaults to 0.62 when denoise omitted
	a4 := map[string]any{"prompt": "1girl, ghibli anime style", "style": "illustration", "reference_image_path": "auto"}
	prepareGenerateImageArgs(tmpFile, a4)
	if a4["reference_image_path"] != tmpFile || a4["denoise"] != 0.62 {
		t.Fatalf("expected illustration img2img ref=%s denoise=0.62, got %+v", tmpFile, a4)
	}

	// 4. Explicit higher denoise (e.g. 0.68) must be preserved
	a5 := map[string]any{"prompt": "ghibli style", "style": "illustration", "denoise": 0.68}
	prepareGenerateImageArgs(tmpFile, a5)
	if a5["reference_image_path"] != tmpFile || a5["denoise"] != 0.68 {
		t.Fatalf("expected explicit img2img denoise=0.68 preserved, got %+v", a5)
	}

	// 5. Accidental denoise=1.0 or 0.0 on img2img illustration falls back safely to 0.62
	a6 := map[string]any{"prompt": "ghibli style", "style": "illustration", "denoise": 1.0}
	prepareGenerateImageArgs(tmpFile, a6)
	if a6["reference_image_path"] != tmpFile || a6["denoise"] != 0.62 {
		t.Fatalf("expected accidental denoise=1.0 on img2img illustration -> 0.62, got %+v", a6)
	}
}


func TestDetectStreamLoop(t *testing.T) {
	if detectStreamLoop("1girl, solo, studio ghibli style, warm sunlight, soft watercolor", 12000) {
		t.Fatal("normal concise prompt should not trigger loop detection")
	}
	repLine := strings.Repeat("16. Natural pose and gentle facial expression\n", 4)
	if !detectStreamLoop(repLine, 12000) {
		t.Fatal("4 identical consecutive lines should trigger loop detection")
	}
	repChunk := strings.Repeat("detailed natural skin texture with shallow depth of field ", 4)
	if !detectStreamLoop(repChunk, 12000) {
		t.Fatal("4 identical consecutive 40+ char blocks should trigger loop detection")
	}
	if !detectStreamLoop(strings.Repeat("abc ", 3500), 12000) {
		t.Fatal("oversized stream beyond maxChars should trigger loop detection")
	}
}

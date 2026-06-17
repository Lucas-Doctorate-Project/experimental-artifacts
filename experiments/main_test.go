package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestValidateCampaignRejectsDuplicateNames(t *testing.T) {
	campaign := Campaign{
		Experiments: []Experiment{
			{Name: "exp-a"},
			{Name: "exp-a"},
		},
	}

	err := validateCampaign(campaign)
	if err == nil {
		t.Fatal("expected duplicate name validation error")
	}
	if !strings.Contains(err.Error(), `duplicate experiment name "exp-a"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCreateSocketEndpointCleansUpTempDir(t *testing.T) {
	endpoint, cleanup, err := createSocketEndpoint()
	if err != nil {
		t.Fatalf("createSocketEndpoint returned error: %v", err)
	}

	socketPath := strings.TrimPrefix(endpoint, "ipc://")
	socketDir := filepath.Dir(socketPath)

	if _, err := os.Stat(socketDir); err != nil {
		t.Fatalf("expected socket dir to exist: %v", err)
	}

	cleanup()

	if _, err := os.Stat(socketDir); !os.IsNotExist(err) {
		t.Fatalf("expected socket dir to be removed, got err=%v", err)
	}
}

func TestFormatLinesCarryCounterAndName(t *testing.T) {
	if got := formatOK(2, 10, "exp-a", 83*time.Second); got != "[2/10] OK   exp-a  (1m23s)" {
		t.Fatalf("formatOK: %q", got)
	}
	if got := formatSkip(1, 10, "exp-b"); got != "[1/10] SKIP exp-b  (already complete)" {
		t.Fatalf("formatSkip: %q", got)
	}
	// Sub-second durations round to 0s and the reason is appended after the colon.
	if got := formatFail(3, 10, "exp-c", 400*time.Millisecond, errSentinel{}); got != "[3/10] FAIL exp-c  (0s): boom" {
		t.Fatalf("formatFail: %q", got)
	}
}

func TestFormatSummary(t *testing.T) {
	got := formatSummary(10, 6, 3, 1, 90*time.Second)
	want := "Summary: 10 total, 6 succeeded, 3 skipped, 1 failed, elapsed 1m30s"
	if got != want {
		t.Fatalf("formatSummary: got %q want %q", got, want)
	}
}

// errSentinel is a fixed-message error for deterministic line formatting tests.
type errSentinel struct{}

func (errSentinel) Error() string { return "boom" }

func TestWaitForResultsDoesNotTreatBufferedExitAsGraceTimeout(t *testing.T) {
	opts := runOptions{
		failureTimeout: 0,
		successTimeout: 0,
	}

	for i := 0; i < 128; i++ {
		results := make(chan processResult, 2)
		results <- processResult{name: "batsched", err: nil}
		results <- processResult{name: "batsim", err: nil}

		result, err := waitForResults(results, opts, func() {})
		if err != nil {
			t.Fatalf("iteration %d: waitForResults returned error: %v", i, err)
		}
		if result.batschedErr != nil || result.batsimErr != nil {
			t.Fatalf("iteration %d: unexpected process errors: batsched=%v batsim=%v", i, result.batschedErr, result.batsimErr)
		}
	}
}

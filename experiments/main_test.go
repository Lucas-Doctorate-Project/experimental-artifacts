package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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

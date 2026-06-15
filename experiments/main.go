// Package main runs a campaign of Batsim and Batsched simulations defined
// in a TOML file. Each experiment runs batsim and batsched as co-running
// processes and writes artifacts into its own directory.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/BurntSushi/toml"
)

const (
	// outputRootDir is the parent directory holding per-experiment output.
	outputRootDir = "out"

	// outputDirMode and logFileMode are the permissions used for created
	// experiment directories and their append-only log files.
	outputDirMode = 0o755
	logFileMode   = 0o644

	// killGracePeriod is the delay killGroup waits between SIGTERM and
	// SIGKILL when terminating a process group.
	killGracePeriod = 500 * time.Millisecond

	// coRunningProcesses is the number of processes (batsched and batsim)
	// launched together for each experiment.
	coRunningProcesses = 2

	// maxConcurrentExperiments bounds how many experiments run at once. Each
	// experiment is a batsim+batsched pair locked in ZMQ step, so the live
	// process count is twice this. Kept well under the core count on purpose:
	// oversubscribing these lock-step pairs collapses throughput, since every
	// protocol round-trip then waits on the OS to reschedule the counterpart.
	maxConcurrentExperiments = 4

	// Default grace periods applied when the corresponding flag is unset.
	defaultFailureTimeout = 30 * time.Second
	defaultSuccessTimeout = 30 * time.Second
)

// batschedProcess and batsimProcess name the two co-running executables.
// They double as the result tags carried on processResult.
const (
	batschedProcess = "batsched"
	batsimProcess   = "batsim"
)

// runOptions carries the execution policy shared by all experiments.
// failureTimeout and successTimeout are the grace periods granted to the
// surviving process after the other exits with a non-zero or zero status.
// There is no cap on overall experiment runtime: simulations run to
// completion however long they take.
type runOptions struct {
	failureTimeout time.Duration
	successTimeout time.Duration
}

// Experiment describes one run decoded from the campaign TOML.
// Name is the output directory. Workload, Platform and EnvironmentalTrace
// are input paths resolved relative to the working directory. VariantName
// selects the Batsched variant. VariantOptions is the JSON file passed
// as --variant_options_filepath.
type Experiment struct {
	Name               string `toml:"name"`
	Workload           string `toml:"workload"`
	Platform           string `toml:"platform"`
	EnvironmentalTrace string `toml:"environmental_trace"`
	VariantName        string `toml:"variant_name"`
	VariantOptions     string `toml:"variant_options"`
}

// Campaign is the top-level TOML structure, an ordered list of
// experiments.
type Campaign struct {
	Experiments []Experiment `toml:"experiment"`
}

// openAppendFile opens path for appending, creating it with logFileMode
// if absent. The caller closes the file.
func openAppendFile(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, logFileMode)
}

// openProcessLog opens the combined stdout and stderr destination (.log)
// for the named process inside dir. Both streams share one file so their
// output is interleaved in submission order. The caller closes the file.
func openProcessLog(dir, name string) (*os.File, error) {
	logFile, err := openAppendFile(filepath.Join(dir, name+".log"))
	if err != nil {
		return nil, fmt.Errorf("opening %s.log: %w", name, err)
	}
	return logFile, nil
}

// killGroup terminates the process group led by cmd and any children
// it spawned. It is a no-op when cmd is nil or unstarted. It sends
// SIGTERM, waits 500 ms, then sends SIGKILL. cmd must have been started
// with SysProcAttr.Setpgid set.
func killGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid := cmd.Process.Pid
	_ = syscall.Kill(-pgid, syscall.SIGTERM)
	time.Sleep(killGracePeriod)
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
}

type processResult struct {
	name string
	err  error
}

// experimentResult holds the exit error of each co-running process. A
// nil field means that process exited cleanly.
type experimentResult struct {
	batschedErr error
	batsimErr   error
}

// waitForResults waits for both process results emitted by the wait
// goroutines. Once the first process exits, a grace timer bounds the wait for
// the second; kill is called if it overruns. The returned error is non-nil
// only for that grace-period overrun, not for a process exit error.
func waitForResults(results <-chan processResult, opts runOptions, kill func()) (experimentResult, error) {
	var result experimentResult
	remaining := coRunningProcesses

	// graceTimer arms once a single process is left, bounding how long we
	// wait for it before killing both groups.
	var graceTimer *time.Timer
	var graceTimerC <-chan time.Time

	recordResult := func(r processResult) {
		switch r.name {
		case batschedProcess:
			result.batschedErr = r.err
		case batsimProcess:
			result.batsimErr = r.err
		}

		remaining--
		if remaining == coRunningProcesses-1 && graceTimer == nil {
			graceDuration := opts.successTimeout
			if r.err != nil {
				graceDuration = opts.failureTimeout
			}
			graceTimer = time.NewTimer(graceDuration)
			graceTimerC = graceTimer.C
		}
	}

	drainResults := func() {
		for remaining > 0 {
			recordResult(<-results)
		}
	}

	stopGraceTimer := func() {
		if graceTimer != nil {
			graceTimer.Stop()
		}
	}

	for remaining > 0 {
		select {
		case r := <-results:
			recordResult(r)
		case <-graceTimerC:
			// A buffered final result can race the grace timer; prefer it.
			select {
			case r := <-results:
				recordResult(r)
				continue
			default:
			}

			kill()
			drainResults()
			return experimentResult{}, fmt.Errorf("other process did not finish within grace period")
		}
	}

	stopGraceTimer()
	return result, nil
}

// waitWithTimeouts waits for both processes to exit under the policy in
// opts. When the first process exits, a timer arms for successTimeout or
// failureTimeout depending on its exit status. That timer firing kills both
// groups and drains the pending Wait calls. The function returns nil only
// when both processes exit cleanly.
func waitWithTimeouts(batsched, batsim *exec.Cmd, opts runOptions) error {
	results := make(chan processResult, coRunningProcesses)
	go func() { results <- processResult{name: batschedProcess, err: batsched.Wait()} }()
	go func() { results <- processResult{name: batsimProcess, err: batsim.Wait()} }()

	result, err := waitForResults(results, opts, func() {
		killGroup(batsched)
		killGroup(batsim)
	})
	if err != nil {
		return err
	}
	if result.batschedErr != nil || result.batsimErr != nil {
		return fmt.Errorf("batsched=%v batsim=%v", result.batschedErr, result.batsimErr)
	}
	return nil
}

// createSocketEndpoint allocates a unique IPC endpoint for one
// experiment run and returns a cleanup function for the temporary
// directory that contains it.
func createSocketEndpoint() (string, func(), error) {
	socketDir, err := os.MkdirTemp("", "experimental-campaigns-")
	if err != nil {
		return "", nil, fmt.Errorf("creating socket dir: %w", err)
	}

	socketPath := filepath.Join(socketDir, "socket")
	cleanup := func() {
		_ = os.Remove(socketPath)
		_ = os.RemoveAll(socketDir)
	}

	return "ipc://" + socketPath, cleanup, nil
}

// validateCampaign rejects duplicate experiment names because the
// runner uses them as output directory names.
func validateCampaign(campaign Campaign) error {
	names := make(map[string]struct{}, len(campaign.Experiments))
	for i, exp := range campaign.Experiments {
		if exp.Name == "" {
			return fmt.Errorf("experiment %d has an empty name", i+1)
		}

		if _, exists := names[exp.Name]; exists {
			return fmt.Errorf("duplicate experiment name %q", exp.Name)
		}

		names[exp.Name] = struct{}{}
	}

	return nil
}

// experimentComplete reports whether the experiment's output already holds a
// finished schedule summary. Batsim writes out/<name>/out_schedule.csv (a
// header plus a value row) only at a clean end of simulation, so its presence
// marks a run that need not be repeated. Delete the output directory to force
// a re-run.
func experimentComplete(name string) bool {
	path := filepath.Join(outputRootDir, name, "out_schedule.csv")
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return bytes.Count(data, []byte("\n")) >= 2
}

// runExperiment executes one experiment under opts. It creates
// out/<name>, opens the two combined log files, allocates a unique IPC
// endpoint, then starts batsched and batsim in their own process
// groups and delegates to waitWithTimeouts. Returns nil only when both
// processes exit cleanly.
func runExperiment(exp Experiment, opts runOptions) error {
	outputDir := filepath.Join(outputRootDir, exp.Name)
	if err := os.MkdirAll(outputDir, outputDirMode); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	batschedLog, err := openProcessLog(outputDir, batschedProcess)
	if err != nil {
		return err
	}
	defer batschedLog.Close()

	batsimLog, err := openProcessLog(outputDir, batsimProcess)
	if err != nil {
		return err
	}
	defer batsimLog.Close()

	socketEndpoint, cleanupSocket, err := createSocketEndpoint()
	if err != nil {
		return err
	}
	defer cleanupSocket()

	batschedArgs := []string{
		"-v", exp.VariantName,
		"--socket-endpoint", socketEndpoint,
	}
	if exp.VariantOptions != "" {
		batschedArgs = append(batschedArgs, "--variant_options_filepath", exp.VariantOptions)
	}
	batschedCmd := exec.Command(batschedProcess, batschedArgs...)
	batschedCmd.Stdout = batschedLog
	batschedCmd.Stderr = batschedLog
	batschedCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	batsimCmd := exec.Command(batsimProcess,
		"-p", exp.Platform,
		"-w", exp.Workload,
		"-e", filepath.Join(outputDir, "out"),
		"--socket-endpoint", socketEndpoint,
		"--energy",
		"--environmental-footprint-dynamic", exp.EnvironmentalTrace,
	)
	batsimCmd.Stdout = batsimLog
	batsimCmd.Stderr = batsimLog
	batsimCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := batschedCmd.Start(); err != nil {
		return fmt.Errorf("starting batsched: %w", err)
	}
	if err := batsimCmd.Start(); err != nil {
		killGroup(batschedCmd)
		_ = batschedCmd.Wait()
		return fmt.Errorf("starting batsim: %w", err)
	}

	return waitWithTimeouts(batschedCmd, batsimCmd, opts)
}

// main parses flags, decodes the campaign TOML, validates experiment
// names, and runs the experiments with bounded parallelism. A failed
// experiment does not abort the campaign. Exits 0 only when every
// experiment succeeds, 1 otherwise.
func main() {
	campaignPath := flag.String("campaign", "experiments.toml", "Path to the campaign TOML file")
	failureTimeout := flag.Duration("failure-timeout", defaultFailureTimeout, "Grace period for the surviving process after the other fails")
	successTimeout := flag.Duration("success-timeout", defaultSuccessTimeout, "Grace period for the surviving process after the other succeeds")
	flag.Parse()

	opts := runOptions{
		failureTimeout: *failureTimeout,
		successTimeout: *successTimeout,
	}

	var campaign Campaign
	if _, err := toml.DecodeFile(*campaignPath, &campaign); err != nil {
		log.Fatal(err)
	}
	if err := validateCampaign(campaign); err != nil {
		log.Fatal(err)
	}

	sem := make(chan struct{}, maxConcurrentExperiments)

	var wg sync.WaitGroup
	var mu sync.Mutex
	failedCount := 0

	for _, exp := range campaign.Experiments {
		if experimentComplete(exp.Name) {
			fmt.Printf("Skipping experiment (already complete): %s\n", exp.Name)
			continue
		}

		wg.Add(1)
		sem <- struct{}{}

		go func(e Experiment) {
			defer wg.Done()
			defer func() { <-sem }()

			fmt.Printf("Running experiment: %s\n", e.Name)
			if err := runExperiment(e, opts); err != nil {
				fmt.Fprintf(os.Stderr, "experiment %q failed: %v\n", e.Name, err)

				mu.Lock()
				failedCount++
				mu.Unlock()
			}
		}(exp)
	}

	wg.Wait()

	if failedCount > 0 {
		os.Exit(1)
	}
}

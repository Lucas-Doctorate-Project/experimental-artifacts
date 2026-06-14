// Package main runs a campaign of Batsim and Batsched simulations defined
// in a TOML file. Each experiment runs batsim and batsched as co-running
// processes and writes artifacts into its own directory.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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

	// Default timeout policy applied when the corresponding flag is unset.
	defaultSimulationTimeout = time.Hour
	defaultFailureTimeout    = 30 * time.Second
	defaultSuccessTimeout    = 30 * time.Second
)

// batschedProcess and batsimProcess name the two co-running executables.
// They double as the result tags carried on processResult.
const (
	batschedProcess = "batsched"
	batsimProcess   = "batsim"
)

// runOptions carries the execution policy shared by all experiments.
// simulationTimeout caps the runtime of one experiment. failureTimeout
// and successTimeout are the grace periods granted to the surviving
// process after the other exits with a non-zero or zero status.
type runOptions struct {
	simulationTimeout time.Duration
	failureTimeout    time.Duration
	successTimeout    time.Duration
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

// openProcessLogs opens the stdout (.log) and stderr (.err) destinations
// for the named process inside dir. On failure it closes any file it
// already opened so the caller never leaks a descriptor.
func openProcessLogs(dir, name string) (stdout, stderr *os.File, err error) {
	stdout, err = openAppendFile(filepath.Join(dir, name+".log"))
	if err != nil {
		return nil, nil, fmt.Errorf("opening %s.log: %w", name, err)
	}

	stderr, err = openAppendFile(filepath.Join(dir, name+".err"))
	if err != nil {
		stdout.Close()
		return nil, nil, fmt.Errorf("opening %s.err: %w", name, err)
	}

	return stdout, stderr, nil
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

// waitForResults applies the timeout policy to process results emitted
// by wait goroutines. kill is called when a timeout fires. The returned
// error is non-nil only for a timeout, not for a process exit error.
func waitForResults(results <-chan processResult, opts runOptions, kill func()) (experimentResult, error) {
	simTimer := time.NewTimer(opts.simulationTimeout)
	defer simTimer.Stop()

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
		case <-simTimer.C:
			kill()
			drainResults()
			stopGraceTimer()
			return experimentResult{}, fmt.Errorf("simulation timeout exceeded (%s)", opts.simulationTimeout)
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
// opts. A simulation timer runs from entry. When the first process
// exits, a second timer arms for successTimeout or failureTimeout
// depending on its exit status. Any timer firing kills both groups and
// drains the pending Wait calls. The function returns nil only when
// both processes exit cleanly.
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

// runExperiment executes one experiment under opts. It creates
// out/<name>, opens the four log files, allocates a unique IPC
// endpoint, then starts batsched and batsim in their own process
// groups and delegates to waitWithTimeouts. Returns nil only when both
// processes exit cleanly.
func runExperiment(exp Experiment, opts runOptions) error {
	outputDir := filepath.Join(outputRootDir, exp.Name)
	if err := os.MkdirAll(outputDir, outputDirMode); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	batschedOut, batschedErrLog, err := openProcessLogs(outputDir, batschedProcess)
	if err != nil {
		return err
	}
	defer batschedOut.Close()
	defer batschedErrLog.Close()

	batsimOut, batsimErrLog, err := openProcessLogs(outputDir, batsimProcess)
	if err != nil {
		return err
	}
	defer batsimOut.Close()
	defer batsimErrLog.Close()

	socketEndpoint, cleanupSocket, err := createSocketEndpoint()
	if err != nil {
		return err
	}
	defer cleanupSocket()

	batschedCmd := exec.Command(batschedProcess,
		"-v", exp.VariantName,
		"--variant_options_filepath", exp.VariantOptions,
		"--socket-endpoint", socketEndpoint,
	)
	batschedCmd.Stdout = batschedOut
	batschedCmd.Stderr = batschedErrLog
	batschedCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	batsimCmd := exec.Command(batsimProcess,
		"-p", exp.Platform,
		"-w", exp.Workload,
		"-e", outputDir,
		"--socket-endpoint", socketEndpoint,
		"--energy",
		"--environmental-footprint-dynamic", exp.EnvironmentalTrace,
	)
	batsimCmd.Stdout = batsimOut
	batsimCmd.Stderr = batsimErrLog
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
	simulationTimeout := flag.Duration("simulation-timeout", defaultSimulationTimeout, "Maximum runtime for a single experiment")
	failureTimeout := flag.Duration("failure-timeout", defaultFailureTimeout, "Grace period for the surviving process after the other fails")
	successTimeout := flag.Duration("success-timeout", defaultSuccessTimeout, "Grace period for the surviving process after the other succeeds")
	flag.Parse()

	opts := runOptions{
		simulationTimeout: *simulationTimeout,
		failureTimeout:    *failureTimeout,
		successTimeout:    *successTimeout,
	}

	var campaign Campaign
	if _, err := toml.DecodeFile(*campaignPath, &campaign); err != nil {
		log.Fatal(err)
	}
	if err := validateCampaign(campaign); err != nil {
		log.Fatal(err)
	}

	maxConcurrent := runtime.NumCPU()
	sem := make(chan struct{}, maxConcurrent)

	var wg sync.WaitGroup
	var mu sync.Mutex
	failedCount := 0

	for _, exp := range campaign.Experiments {
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

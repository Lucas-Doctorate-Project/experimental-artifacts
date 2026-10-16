// Run Batsim and Batsched pairs from a TOML file.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/BurntSushi/toml"
)

const (
	batschedProcess    = "batsched"
	batsimProcess      = "batsim"
	coRunningProcesses = 2
	killGracePeriod    = 500 * time.Millisecond
)

type Experiment struct {
	Name               string `toml:"name"`
	Workload           string `toml:"workload"`
	Platform           string `toml:"platform"`
	EnvironmentalTrace string `toml:"environmental_trace"`
	VariantName        string `toml:"variant_name"`
	VariantOptions     string `toml:"variant_options"`
}

type Campaign struct {
	OutputDir   string       `toml:"output_dir"`
	Experiments []Experiment `toml:"experiment"`
}

type runOptions struct {
	outputDir      string
	failureTimeout time.Duration
	successTimeout time.Duration
}

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

type experimentResult struct {
	batschedErr error
	batsimErr   error
}

func waitForResults(results <-chan processResult, opts runOptions, kill func()) (experimentResult, error) {
	var result experimentResult
	remaining := coRunningProcesses

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

func validateCampaign(campaign Campaign) error {
	names := make(map[string]struct{}, len(campaign.Experiments))
	for i, exp := range campaign.Experiments {
		if exp.Name == "" {
			return fmt.Errorf("experiment %d has an empty name", i+1)
		}

		if filepath.Base(exp.Name) != exp.Name || exp.Name == "." || exp.Name == ".." {
			return fmt.Errorf("experiment name %q must be a directory name", exp.Name)
		}

		if _, exists := names[exp.Name]; exists {
			return fmt.Errorf("duplicate experiment name %q", exp.Name)
		}

		names[exp.Name] = struct{}{}
	}

	return nil
}

func runExperiment(exp Experiment, opts runOptions) error {
	outputDir := opts.outputDir
	if err := os.Mkdir(outputDir, 0o755); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	batschedLog, err := os.Create(filepath.Join(outputDir, "batsched.log"))
	if err != nil {
		return err
	}
	defer batschedLog.Close()

	batsimLog, err := os.Create(filepath.Join(outputDir, "batsim.log"))
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

func main() {
	configPath := flag.String("campaign", "original.toml", "Experiment TOML file")
	outDir := flag.String("out-dir", "", "New output directory (overrides TOML output_dir)")
	concurrency := flag.Int("concurrency", 4, "Simultaneous simulation pairs")
	timeout := flag.Duration("timeout", 30*time.Second, "Grace period after one process exits")
	flag.Parse()
	if *concurrency < 1 {
		log.Fatal("--concurrency must be positive")
	}
	var config Campaign
	if _, err := toml.DecodeFile(*configPath, &config); err != nil {
		log.Fatal(err)
	}
	if err := validateCampaign(config); err != nil {
		log.Fatal(err)
	}
	output := config.OutputDir
	if *outDir != "" {
		output = *outDir
	}
	if output == "" {
		log.Fatal("set --out-dir or output_dir in the TOML")
	}
	// Require a new directory so saved results can never be overwritten.
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.Mkdir(output, 0o755); err != nil {
		log.Fatalf("choose a new output directory: %v", err)
	}
	sem := make(chan struct{}, *concurrency)
	var wg sync.WaitGroup
	var failed atomic.Int32
	for _, exp := range config.Experiments {
		sem <- struct{}{}
		wg.Add(1)
		go func(exp Experiment) {
			defer wg.Done()
			defer func() { <-sem }()
			fmt.Println("RUN", exp.Name)
			opts := runOptions{filepath.Join(output, exp.Name), *timeout, *timeout}
			if err := runExperiment(exp, opts); err != nil {
				fmt.Fprintln(os.Stderr, "FAIL", exp.Name, err)
				failed.Add(1)
			} else {
				fmt.Println("OK", exp.Name)
			}
		}(exp)
	}
	wg.Wait()
	fmt.Printf("%d experiments, %d failed\n", len(config.Experiments), failed.Load())
	if failed.Load() > 0 {
		os.Exit(1)
	}
}

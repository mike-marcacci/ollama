package llm

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

func newFitGPUScenario() (discover.SystemInfo, discover.GpuInfoList, *ollamaServer) {
	var systemInfo discover.SystemInfo
	systemInfo.System.TotalMemory = format.GibiByte
	systemInfo.System.FreeMemory = 512 * format.MebiByte
	systemInfo.System.FreeSwap = 256 * format.MebiByte

	gpus := discover.GpuInfoList{
		{
			ID:            "gpu0",
			Library:       "cuda",
			MemInfo:       discover.MemInfo{FreeMemory: 128 * format.MebiByte},
			MinimumMemory: 0,
		},
		{
			ID:            "gpu1",
			Library:       "cuda",
			MemInfo:       discover.MemInfo{FreeMemory: 256 * format.MebiByte},
			MinimumMemory: 0,
		},
	}

	// BackendMemory with 2 layers, each 100MB
	mem := &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: []ml.Memory{
				{Size: 100 * format.MebiByte},
				{Size: 100 * format.MebiByte},
			},
			Cache: []ml.Memory{
				{Size: 0},
				{Size: 0},
			},
		},
		GPUs: []ml.DeviceMemory{
			{
				ID: "gpu0",
				Weights: []ml.Memory{
					{Size: 0},
					{Size: 0},
				},
				Cache: []ml.Memory{
					{Size: 0},
					{Size: 0},
				},
			},
			{
				ID: "gpu1",
				Weights: []ml.Memory{
					{Size: 0},
					{Size: 0},
				},
				Cache: []ml.Memory{
					{Size: 0},
					{Size: 0},
				},
			},
		},
	}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 2,
			options: api.Options{
				Runner: api.Runner{
					NumGPU: -1,
				},
			},
		},
		mem: mem,
	}

	return systemInfo, gpus, s
}

func checkFitGPU(t *testing.T, gpuLayers ml.GPULayersList, err error, want ml.GPULayersList) {
	if err != nil {
		t.Fatalf("fitGPU returned error: %v", err)
	}
	if !reflect.DeepEqual(gpuLayers, want) {
		t.Errorf("fitGPU assigned %v, want %v", gpuLayers, want)
	}
}

func TestLLMServerFitGPU(t *testing.T) {
	systemInfo, gpus, s := newFitGPUScenario()

	// Should fit both layers on 1 GPUs
	t.Run("Single GPU", func(t *testing.T) {
		gpuLayers, err := s.fitGPU(systemInfo, gpus, s.mem, false, 0)
		checkFitGPU(t, gpuLayers, err, ml.GPULayersList{{ID: "gpu1", Layers: []int{0, 1}}})
	})

	// Split across GPUs
	t.Run("Split GPU", func(t *testing.T) {
		s.mem.CPU.Weights[0].Size = 256 * format.MebiByte
		gpuLayers, err := s.fitGPU(systemInfo, gpus, s.mem, false, 0)
		checkFitGPU(t, gpuLayers, err, ml.GPULayersList{{ID: "gpu1", Layers: []int{0}}, {ID: "gpu0", Layers: []int{1}}})
	})

	// Partial fit
	t.Run("Partial fit", func(t *testing.T) {
		s.mem.CPU.Weights[0].Size = 256 * format.MebiByte
		s.mem.CPU.Weights[1].Size = 256 * format.MebiByte
		gpuLayers, err := s.fitGPU(systemInfo, gpus, s.mem, false, 0)
		checkFitGPU(t, gpuLayers, err, ml.GPULayersList{{ID: "gpu1", Layers: []int{0}}})
	})

	// Should fail if requireFull and not enough memory
	t.Run("requireFull", func(t *testing.T) {
		gpuLayers, err := s.fitGPU(systemInfo, gpus, s.mem, true, 0)
		if err == nil {
			t.Errorf("fitGPU should fail when requireFull and not enough memory: %v", gpuLayers)
		}
	})
}

func TestLLMServerCompletionFormat(t *testing.T) {
	// This test was written to fix an already deployed issue. It is a bit
	// of a mess, and but it's good enough, until we can refactoring the
	// Completion method to be more testable.

	ctx, cancel := context.WithCancel(t.Context())
	s := &llmServer{
		sem: semaphore.NewWeighted(1), // required to prevent nil panic
	}

	checkInvalid := func(format string) {
		t.Helper()
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(format),
		}, nil)

		want := fmt.Sprintf("invalid format: %q; expected \"json\" or a valid JSON Schema", format)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v; want %q", err, want)
		}
	}

	checkInvalid("X")   // invalid format
	checkInvalid(`"X"`) // invalid JSON Schema

	cancel() // prevent further processing if request makes it past the format check

	checkValid := func(err error) {
		t.Helper()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Completion: err = %v; expected context.Canceled", err)
		}
	}

	valids := []string{
		// "missing"
		``,
		`""`,
		`null`,

		// JSON
		`"json"`,
		`{"type":"object"}`,
	}
	for _, valid := range valids {
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(valid),
		}, nil)
		checkValid(err)
	}

	err := s.Completion(ctx, CompletionRequest{
		Options: new(api.Options),
		Format:  nil, // missing format
	}, nil)
	checkValid(err)
}

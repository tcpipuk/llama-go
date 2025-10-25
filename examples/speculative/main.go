// Speculative example demonstrates speculative decoding for faster generation.
//
// This program loads two models - a large target model and a smaller draft
// model - and uses speculative decoding to accelerate text generation. The
// draft model generates candidate tokens which the target model verifies in
// parallel, reducing overall latency whilst maintaining the target model's
// quality.
//
// Usage:
//
//	speculative -target large-model.gguf -draft small-model.gguf -p "prompt"
//
// Speculative decoding works best when:
//   - The draft model is significantly smaller than the target (e.g. 1-3B vs 70B)
//   - Both models share similar vocabularies and tokenisation
//   - Generation requires many tokens (speedup compounds over longer outputs)
//
// The example demonstrates:
//   - Loading and managing multiple models simultaneously
//   - Configuring speculative decoding with draft token count
//   - Measuring performance improvements from speculation
//   - Proper resource cleanup for multiple model instances
//
// Typical speedups range from 1.5× to 3× depending on model sizes and acceptance
// rates. The technique is particularly effective for large models where inference
// latency is high.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		targetModel = flag.String("target", "./Qwen3-0.6B-Q8_0.gguf", "path to target (main) model")
		draftModel  = flag.String("draft", "./Qwen3-0.6B-Q8_0.gguf", "path to draft model")
		prompt      = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens   = flag.Int("n", 100, "maximum number of tokens to generate")
		context     = flag.Int("c", 2048, "context size")
		gpuLayers   = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp        = flag.Float64("temp", 0.7, "temperature for sampling")
		draftTokens = flag.Int("draft-tokens", 16, "number of draft tokens per iteration")
		debug       = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load target model
	fmt.Printf("Loading target model: %s\n", *targetModel)
	target, err := llama.LoadModel(*targetModel,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading target model: %v\n", err)
		os.Exit(1)
	}
	defer target.Close()

	// Create target context
	targetCtx, err := target.NewContext(
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating target context: %v\n", err)
		os.Exit(1)
	}
	defer targetCtx.Close()

	// Load draft model
	fmt.Printf("Loading draft model: %s\n", *draftModel)
	draft, err := llama.LoadModel(*draftModel,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading draft model: %v\n", err)
		os.Exit(1)
	}
	defer draft.Close()

	// Create draft context
	draftCtx, err := draft.NewContext(
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating draft context: %v\n", err)
		os.Exit(1)
	}
	defer draftCtx.Close()

	fmt.Printf("Models loaded successfully.\n")
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Using speculative sampling with %d draft tokens per iteration\n", *draftTokens)

	// Generate with speculative sampling
	var opts []llama.GenerateOption
	opts = append(opts, llama.WithMaxTokens(*maxTokens))
	opts = append(opts, llama.WithTemperature(float32(*temp)))
	opts = append(opts, llama.WithDraftTokens(*draftTokens))
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	response, err := targetCtx.GenerateWithDraft(*prompt, draftCtx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nResponse: %s\n", response)
}

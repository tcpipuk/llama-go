// Simple example demonstrates basic text generation using llama-go.
//
// This program loads a GGUF model and generates text from a single prompt,
// showcasing the core functionality of the library. It supports customising
// generation parameters via command-line flags including temperature, top-p/top-k
// sampling, seed, and GPU acceleration.
//
// Usage:
//
//	simple -m model.gguf -p "prompt text" -n 100
//
// The example demonstrates:
//   - Loading a GGUF model with configuration options
//   - Creating an execution context
//   - Performing synchronous text generation
//   - Customising sampling parameters
//   - GPU layer offloading for acceleration
//   - Proper resource cleanup with defer
//
// This is the recommended starting point for new users of llama-go.
package main

import (
	"flag"
	"fmt"
	"os"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		modelPath = flag.String("m", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
		prompt    = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens = flag.Int("n", 50, "maximum number of tokens to generate")
		context   = flag.Int("c", 2048, "context size")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp      = flag.Float64("temp", 0.7, "temperature for sampling")
		topP      = flag.Float64("top-p", 0.95, "top-p for sampling")
		topK      = flag.Int("top-k", 40, "top-k for sampling")
		seed      = flag.Int("s", -1, "random seed")
		debug     = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load model weights
	fmt.Printf("Loading model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithGPULayers(*gpuLayers),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded successfully.\n")

	// Create execution context
	ctx, err := model.NewContext(
		llama.WithContext(*context),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Prompt: %s\n", *prompt)

	// Generate text
	var opts []llama.GenerateOption
	opts = append(opts, llama.WithMaxTokens(*maxTokens))
	opts = append(opts, llama.WithTemperature(float32(*temp)))
	opts = append(opts, llama.WithTopP(float32(*topP)))
	opts = append(opts, llama.WithTopK(*topK))
	opts = append(opts, llama.WithSeed(*seed))
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	response, err := ctx.Generate(*prompt, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nResponse: %s\n", response)
}

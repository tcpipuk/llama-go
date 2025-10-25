// Streaming example demonstrates both callback and channel-based token streaming.
//
// This example shows two approaches to streaming generation:
//   - Callback-based: Direct function callbacks for each token (default)
//   - Channel-based: Go channels with context support for cancellation
//
// Usage:
//
//	# Callback-based streaming (default)
//	streaming -model model.gguf -prompt "Hello world" -max-tokens 50
//
//	# Channel-based streaming with timeout
//	streaming -model model.gguf -prompt "Hello world" -max-tokens 50 -channel -timeout 30
//
// The callback approach is simpler for basic use cases, whilst the channel
// approach provides better integration with Go's concurrency patterns and
// supports cancellation via context.Context.
//
// The example demonstrates:
//   - Both streaming approaches with identical output
//   - Context-based cancellation and timeout handling
//   - Real-time token output as the model generates text
//   - Error handling during generation
//   - Configuration via command-line flags
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"runtime"
	"strings"
	"time"

	llama "github.com/tcpipuk/llama-go"
)

var (
	modelPath  = flag.String("model", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
	prompt     = flag.String("prompt", "Once upon a time", "prompt text")
	maxTokens  = flag.Int("max-tokens", 100, "maximum tokens to generate")
	contextLen = flag.Int("context", 2048, "context size")
	gpuLayers  = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
	temp       = flag.Float64("temperature", 0.8, "temperature")
	topP       = flag.Float64("top-p", 0.9, "top-p for sampling")
	topK       = flag.Int("top-k", 40, "top-k for sampling")
	useChannel = flag.Bool("channel", false, "use channel-based streaming instead of callbacks")
	timeout    = flag.Int("timeout", 30, "timeout in seconds (channel mode only)")
)

func main() {
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide --model path")
	}

	// Load model
	fmt.Printf("Loading model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Create context
	ctx, err := model.NewContext(
		llama.WithContext(*contextLen),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
	)
	if err != nil {
		log.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	fmt.Printf("Model loaded successfully.\n\n")

	if *useChannel {
		streamWithChannels(ctx)
	} else {
		streamWithCallbacks(ctx)
	}
}

func streamWithCallbacks(ctx *llama.Context) {
	fmt.Println("=== Callback-based streaming ===")
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Max tokens: %d\n\n", *maxTokens)

	var result strings.Builder
	err := ctx.GenerateStream(*prompt,
		func(token string) bool {
			fmt.Print(token)
			result.WriteString(token)
			return true
		},
		llama.WithMaxTokens(*maxTokens),
		llama.WithTemperature(float32(*temp)),
		llama.WithTopP(float32(*topP)),
		llama.WithTopK(*topK),
	)

	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("\n\n=== Complete (%d chars) ===\n", result.Len())
}

func streamWithChannels(ctx *llama.Context) {
	fmt.Println("=== Channel-based streaming ===")
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Max tokens: %d\n", *maxTokens)
	fmt.Printf("Timeout: %d seconds\n\n", *timeout)

	ctxTimeout, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
	defer cancel()

	tokenCh, errCh := ctx.GenerateChannel(ctxTimeout, *prompt,
		llama.WithMaxTokens(*maxTokens),
		llama.WithTemperature(float32(*temp)),
		llama.WithTopP(float32(*topP)),
		llama.WithTopK(*topK),
	)

	var result strings.Builder
	tokenCount := 0

	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				// Channel closed, generation complete
				fmt.Printf("\n\n=== Complete (%d tokens, %d chars) ===\n",
					tokenCount, result.Len())
				return
			}
			fmt.Print(token)
			result.WriteString(token)
			tokenCount++

		case err := <-errCh:
			if err != nil {
				log.Fatalf("Generation error: %v", err)
			}

		case <-ctxTimeout.Done():
			fmt.Printf("\n\n=== Cancelled (%v) after %d tokens ===\n",
				ctxTimeout.Err(), tokenCount)
			return
		}
	}
}

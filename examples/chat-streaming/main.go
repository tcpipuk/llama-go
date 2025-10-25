// Chat streaming example demonstrates real-time chat completion with visual distinction
// between regular content and reasoning output.
//
// This example shows:
//   - Streaming chat deltas via channels
//   - Visual distinction between content (normal) and reasoning (dimmed)
//   - Context timeout and cancellation
//   - Proper error handling for streaming
//
// The streaming API uses channels to deliver tokens as they're generated, providing
// a more responsive user experience. Content appears immediately rather than waiting
// for the entire response to complete.
//
// For reasoning models (like DeepSeek-R1), this example demonstrates how to handle
// both regular content and reasoning/thinking tokens with different visual styles.
//
// Usage:
//
//	chat-streaming -model model.gguf -message "Explain quantum computing"
//	chat-streaming -model model.gguf -message "Think through: 2+2*3" -reasoning
//	chat-streaming -model model.gguf -message "Tell me a story" -max-tokens 300
//	chat-streaming -model model.gguf  # Interactive mode - no message flag enters chat loop
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	llama "github.com/tcpipuk/llama-go"
	"github.com/tcpipuk/llama-go/internal/exampleui"
)

// Note: This file uses exampleui.ColouriseField (UK spelling) for rainbow gradient colours

var (
	modelPath   = flag.String("model", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
	system      = flag.String("system", "You are a helpful assistant called George.", "system prompt")
	message     = flag.String("message", "", "user message (leave empty for interactive mode)")
	contextSize = flag.Int("context", 0, "context size (0 = use model's native maximum)")
	maxTokens   = flag.Int("max-tokens", 1024, "maximum tokens to generate")
	temp        = flag.Float64("temperature", 0.7, "temperature (0.0-2.0)")
	topP        = flag.Float64("top-p", 0.9, "nucleus sampling threshold")
	topK        = flag.Int("top-k", 40, "top-K sampling")
	timeout     = flag.Int("timeout", 60, "timeout in seconds")
	reasoning   = flag.Bool("reasoning", false, "enable reasoning output (for reasoning models)")
	debug       = flag.Bool("debug", false, "show conversation history before each generation")
)

func main() {
	flag.Parse()

	// Configure logging verbosity based on debug flag
	if *debug {
		os.Setenv("LLAMA_LOG", "info") // Full llama.cpp output for debugging
	} else {
		os.Setenv("LLAMA_LOG", "error") // Quiet mode - only warnings and errors
	}
	llama.InitLogging()

	// Load model with sensible defaults
	fmt.Printf("Loading model: %s\n", *modelPath)

	// Build model options
	modelOpts := []llama.ModelOption{
		llama.WithGPULayers(-1), // Use all available GPU layers
	}

	model, err := llama.LoadModel(*modelPath, modelOpts...)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// Build context options
	contextOpts := []llama.ContextOption{
		llama.WithThreads(runtime.NumCPU()),
	}

	// Add context size if specified (0 = use model's native maximum)
	if *contextSize != 0 {
		contextOpts = append(contextOpts, llama.WithContext(*contextSize))
	}

	ctx, err := model.NewContext(contextOpts...)
	if err != nil {
		log.Fatalf("Failed to create context: %v", err)
	}
	defer ctx.Close()

	// Display model statistics
	stats, err := model.Stats()
	if err != nil {
		log.Printf("Warning: Could not retrieve model stats: %v", err)
	} else {
		exampleui.DisplayModelStats(*stats, *maxTokens, *timeout, *temp, *topP, *topK)
	}

	// Choose between single-message mode or interactive mode
	if *message != "" {
		runSingleMessage(model, ctx)
	} else {
		runInteractive(model, ctx)
	}
}

// runSingleMessage handles single message streaming completion.
func runSingleMessage(model *llama.Model, ctx *llama.Context) {
	messages := []llama.ChatMessage{
		{Role: "system", Content: *system},
		{Role: "user", Content: *message},
	}

	exampleui.DisplaySystemPrompt(*system)
	fmt.Printf("\nUser: %s\n", *message)

	ctxTimeout, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
	defer cancel()

	opts := llama.ChatOptions{
		MaxTokens:   llama.Int(*maxTokens),
		Temperature: llama.Float32(float32(*temp)),
		TopP:        llama.Float32(float32(*topP)),
		TopK:        llama.Int(*topK),
	}

	if *reasoning {
		opts.EnableThinking = llama.Bool(true)
		opts.ReasoningFormat = llama.ReasoningFormatAuto
	}

	streamResponse(ctxTimeout, ctx, messages, opts)
}

// runInteractive handles interactive chat loop with streaming.
func runInteractive(model *llama.Model, ctx *llama.Context) {
	exampleui.DisplaySystemPrompt(*system)
	fmt.Println("\nType your messages and press Enter. Press Ctrl-C to quit.")
	fmt.Println()

	// Initialize conversation with system prompt
	messages := []llama.ChatMessage{
		{Role: "system", Content: *system},
	}

	scanner := bufio.NewScanner(os.Stdin)

	for {
		// Prompt for user input
		userInput, err := exampleui.PromptUser(scanner)
		if err != nil {
			log.Printf("Error reading input: %v", err)
			return
		}
		if userInput == "" {
			fmt.Println("\nGoodbye!")
			return
		}

		// Add user message to conversation
		messages = append(messages, llama.ChatMessage{
			Role:    "user",
			Content: userInput,
		})

		ctxTimeout, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)

		opts := llama.ChatOptions{
			MaxTokens:   llama.Int(*maxTokens),
			Temperature: llama.Float32(float32(*temp)),
			TopP:        llama.Float32(float32(*topP)),
			TopK:        llama.Int(*topK),
		}

		if *reasoning {
			opts.EnableThinking = llama.Bool(true)
			opts.ReasoningFormat = llama.ReasoningFormatAuto
		}

		// Debug: Show conversation history and formatted prompt
		if *debug {
			formattedPrompt, err := model.FormatChatPrompt(messages, opts)
			exampleui.DisplayDebugInfo(messages, formattedPrompt, err)
		}

		// Stream assistant response (streamResponse handles all formatting)
		assistantContent := streamResponse(ctxTimeout, ctx, messages, opts)
		cancel()

		// Add assistant response to conversation history
		messages = append(messages, llama.ChatMessage{
			Role:    "assistant",
			Content: assistantContent,
		})
	}
}

// streamResponse handles the streaming of a single response and returns the content.
func streamResponse(ctxTimeout context.Context, ctx *llama.Context, messages []llama.ChatMessage, opts llama.ChatOptions) string {
	deltaCh, errCh := ctx.ChatStream(ctxTimeout, messages, opts)
	renderer := exampleui.NewStreamRenderer()

	for {
		select {
		case delta, ok := <-deltaCh:
			if !ok {
				return renderer.Finish()
			}
			renderer.ProcessDelta(delta)

		case err := <-errCh:
			if err != nil {
				return renderer.HandleError(err)
			}

		case <-ctxTimeout.Done():
			return renderer.HandleTimeout(ctxTimeout.Err())
		}
	}
}

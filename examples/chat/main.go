// Chat example demonstrates non-streaming chat completion.
//
// This example shows the simplest way to use the Chat API - send messages
// and get a complete response back. The Chat() method handles all token
// collection internally and returns the full response when generation finishes.
//
// For real-time streaming output where tokens appear as they're generated,
// see the chat-streaming example instead.
//
// Usage:
//
//	chat -model model.gguf -system "You are a helpful assistant" -message "What is Go?"
//	chat -model model.gguf -message "Explain quantum computing in simple terms"
//	chat -model model.gguf -message "Think through: 2+2*3" -reasoning
//	chat -model model.gguf -max-tokens 200 -temperature 0.9 -timeout 120 -message "Tell me a joke"
//	chat -model model.gguf  # Interactive mode - no message flag enters chat loop
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

// runSingleMessage handles single message completion.
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

	response, err := ctx.Chat(ctxTimeout, messages, opts)
	if err != nil {
		log.Fatalf("Chat completion failed: %v", err)
	}

	fmt.Printf("Assistant: %s\n", response.Content)

	if response.ReasoningContent != "" {
		fmt.Printf("\n=== Reasoning ===\n%s\n", response.ReasoningContent)
	}
}

// runInteractive handles interactive chat loop.
func runInteractive(model *llama.Model, ctx *llama.Context) {
	fmt.Println("=== Interactive Chat Mode ===")
	exampleui.DisplaySystemPrompt(*system)
	fmt.Printf("Parameters: max_tokens=%d, temperature=%.2f, top_p=%.2f, top_k=%d\n",
		*maxTokens, *temp, *topP, *topK)
	fmt.Println("\nType your messages and press Enter. Press Ctrl-C to quit.")
	fmt.Println()

	// Initialize conversation with system prompt
	messages := []llama.ChatMessage{
		{Role: "system", Content: *system},
	}

	scanner := bufio.NewScanner(os.Stdin)

	for {
		// Prompt for user input
		userInput, err := exampleui.PromptUserSimple(scanner)
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

		// Debug: Show conversation history and formatted prompt
		if *debug {
			formattedPrompt, err := model.FormatChatPrompt(messages, opts)
			exampleui.DisplayDebugInfo(messages, formattedPrompt, err)
		}

		// Get assistant response
		response, err := ctx.Chat(ctxTimeout, messages, opts)
		if err != nil {
			log.Printf("Chat completion failed: %v\n", err)
			continue
		}

		// Display response with clean formatting
		fmt.Println()
		if response.ReasoningContent != "" {
			fmt.Printf("(%s)\n\n", response.ReasoningContent)
		}
		fmt.Printf("A: %s\n\n", response.Content)

		// Add assistant response to conversation history
		messages = append(messages, llama.ChatMessage{
			Role:    "assistant",
			Content: response.Content,
		})
	}
}

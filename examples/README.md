# Examples

These examples cover everything from interactive chat with streaming output to basic building blocks
for custom integrations. If you want to see what llama-go can do, start with chat-streaming - it's
the most complete demonstration of the API.

**API pattern**: All examples use the Model/Context separation pattern:

1. Load model weights with `llama.LoadModel()` (thread-safe, create once)
2. Create execution context(s) with `model.NewContext()` (one per goroutine)
3. Use context for inference: `ctx.Generate()`, `ctx.Chat()`, `ctx.GetEmbeddings()`, etc.
4. Clean up with `defer model.Close()` and `defer ctx.Close()`

See the [API guide](../docs/api-guide.md) for detailed explanations of Model vs Context, option
types (ModelOption vs ContextOption), thread safety, and resource management.

## Interactive streaming chat

The chat-streaming example shows the full API in action: structured chat messages, streaming deltas,
context management, and proper conversation handling. It displays GPU details, model metadata, and
session configuration at startup, then streams tokens as they're generated. Reasoning content
appears dimmed for reasoning models.

```bash
# Interactive streaming chat - the full experience
docker run --rm -it -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat-streaming -model Qwen3-0.6B-Q8_0.gguf"

# Or try a single message
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat-streaming -model Qwen3-0.6B-Q8_0.gguf -message 'Explain quantum computing'"
```

## Building your own integration

The examples below are organised from highest-level (chat) to lowest-level (simple). Start with
whichever pattern matches your use case, then look at simpler examples if you need to understand the
underlying primitives.

### 1. Chat (`./examples/chat`)

Complete chat API without streaming - returns full responses instead of token-by-token deltas.
Simpler to integrate than chat-streaming when you don't need real-time output, useful for batch
processing or when you want the entire response before displaying it. Uses the same message
structure and conversation management as chat-streaming.

**Usage:**

```bash
# Single message mode
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat -model Qwen3-0.6B-Q8_0.gguf -message 'What is Go?'"

# Interactive chat mode (no -message flag)
docker run --rm -it -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat -model Qwen3-0.6B-Q8_0.gguf"

# With reasoning output (for reasoning models)
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat -model model.gguf -message 'Think through: 2+2*3' -reasoning"
```

This example shows the chat API with system/user/assistant messages, conversation history
management, and both single-shot and interactive modes. For reasoning models like DeepSeek-R1, use
`-reasoning` to see the model's internal reasoning process. The `internal/exampleui` package handles
stats display.

**Options:**

- `-message` - User message (leave empty for interactive mode)
- `-system` - System prompt (default: "You are a helpful assistant called George.")
- `-reasoning` - Enable reasoning output for reasoning models
- `-context` - Context size (0 = use model's native maximum)
- `-max-tokens` - Maximum tokens to generate (default: 1024)
- `-timeout` - Timeout in seconds (default: 60)
- `-debug` - Show conversation history before each generation

### 2. Chat streaming (`./examples/chat-streaming`)

The full chat API with streaming deltas via channels - tokens appear as they're generated rather
than waiting for complete responses. Content and reasoning output are visually distinct (reasoning
appears dimmed). Same message structure and conversation management as the non-streaming chat
example.

**Usage:**

```bash
# Single message with streaming
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat-streaming -model Qwen3-0.6B-Q8_0.gguf -message 'Explain quantum computing'"

# Interactive streaming chat (no -message flag)
docker run --rm -it -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat-streaming -model Qwen3-0.6B-Q8_0.gguf"

# With reasoning visualisation
docker run --rm -it -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/chat-streaming -model model.gguf -message 'Think step by step: 2+2*3' -reasoning"
```

Supports the same options as the non-streaming chat example. Tokens appear immediately as they're
generated, and reasoning content appears dimmed for reasoning models. The `internal/exampleui`
package handles error reporting for streaming and terminal output formatting.

### 3. Streaming (`./examples/streaming`)

Raw streaming without chat templates - you provide prompts directly instead of structured messages.
Useful for custom prompt formats or when you need direct control over what gets sent to the model.
Supports both callback-based streaming (simple function callbacks) and channel-based streaming (Go
channels with context support for cancellation and timeouts).

**Usage:**

```bash
# Callback-based streaming (default)
echo "Tell me a story about robots" | docker run --rm -i -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/streaming -model Qwen3-0.6B-Q8_0.gguf"

# Channel-based streaming with timeout
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/streaming -model Qwen3-0.6B-Q8_0.gguf -prompt 'Hello' -channel -timeout 30"
```

**Options:**

- `-channel` - Use channel-based streaming instead of callbacks
- `-timeout` - Timeout in seconds (channel mode only)
- `-max-tokens` - Maximum tokens to generate (default: 100)
- `-temperature` - Temperature for sampling (default: 0.8)

### 4. Simple (`./examples/simple`)

Synchronous generation from a raw prompt - no streaming, no chat features, just load a model and
generate text. This is the simplest possible integration pattern.

**Usage:**

```bash
# Single-shot generation
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/simple -m Qwen3-0.6B-Q8_0.gguf -p 'The capital of France is' -n 50"
```

Shows how to load GGUF models, customise sampling parameters (temperature, top-p, top-k), and handle
GPU layer offloading for acceleration.

**Options:**

| Flag     | Description                                  | Default                   |
| -------- | -------------------------------------------- | ------------------------- |
| `-m`     | Path to GGUF model file                      | `./Qwen3-0.6B-Q8_0.gguf`  |
| `-p`     | Prompt for generation                        | `"The capital of France"` |
| `-c`     | Context length                               | `2048`                    |
| `-ngl`   | Number of GPU layers to utilise (-1 for all) | `-1`                      |
| `-n`     | Number of tokens to predict                  | `50`                      |
| `-s`     | Predict RNG seed (-1 for random)             | `-1`                      |
| `-temp`  | Temperature for sampling                     | `0.7`                     |
| `-top-p` | Top-p for sampling                           | `0.95`                    |
| `-top-k` | Top-k for sampling                           | `40`                      |
| `-debug` | Enable debug output                          | `false`                   |

## Advanced features

### Speculative decoding (`./examples/speculative`)

Faster generation using a draft model to propose candidate tokens that the target model verifies in
parallel. This reduces latency whilst maintaining the target model's quality - typical speedups
range from 1.5× to 3×.

**Usage:**

```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/speculative -target large-model.gguf -draft small-model.gguf -p 'Write a story'"
```

This example loads and manages two models simultaneously, with proper resource cleanup for both
instances. Works best when the draft model is significantly smaller than the target (e.g. 1-3B vs
70B), both models share similar vocabularies, and you're generating many tokens (speedup compounds
over longer outputs).

**Options:**

- `-target` - Path to target (main) model
- `-draft` - Path to draft model
- `-draft-tokens` - Number of draft tokens per iteration (default: 16)

### Embeddings (`./examples/embedding`)

Generate text embeddings for semantic search, clustering, classification, or similarity scoring.
Loads models in embedding mode and produces vector representations from text.

```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  "go run ./examples/embedding -m embedding-model.gguf -t 'Hello world'"
```

Not all models support embeddings - check model documentation before use. Typical embedding models
include sentence transformers and specialised embedding variants.

## internal/exampleui package

The `internal/exampleui` package provides shared utilities for the chat examples: stats display with
gradient formatting, text wrapping for terminal width, visual distinction between content and
reasoning output, interactive input handling, and colour utilities. The chat and chat-streaming
examples both use it for consistent output formatting and model statistics display.

This package is kept internal rather than exported as part of the main library API - it's useful for
examples but not necessary for library integration.

## Getting started

See [docs/getting-started.md](../docs/getting-started.md) for setup instructions and
[docs/building.md](../docs/building.md) for build options including GPU acceleration (CUDA, ROCm,
Metal, Vulkan, OpenCL, SYCL, RPC).

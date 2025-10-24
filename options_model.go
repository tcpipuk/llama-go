package llama

import (
	"time"
)

// Model loading options

// WithContext sets the context window size in tokens.
//
// The context size determines how many tokens (prompt + generation) the model
// can process. By default, the library uses the model's native maximum context
// length (e.g. 32768 for Qwen3, 128000 for Gemma 3 models >4B).
//
// Override this if you need to limit memory usage or have specific requirements.
//
// IMPORTANT: Very small context sizes (< 64 tokens) may cause llama.cpp to
// crash internally. The library provides defensive checks but cannot prevent
// all edge cases with absurdly small contexts.
//
// Default: 0 (uses model's native maximum from GGUF metadata)
//
// Examples:
//
//	// Use model's full capability (default)
//	model, err := llama.LoadModel("model.gguf")
//
//	// Limit to 8K for memory savings
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	)
func WithContext(size int) ModelOption {
	return func(c *modelConfig) {
		c.contextSize = size
	}
}

// WithBatch sets the batch size for prompt processing.
//
// Larger batch sizes improve throughput for long prompts but increase memory
// usage. The batch size determines how many tokens are processed in parallel
// during the prompt evaluation phase.
//
// Default: 512
//
// Example:
//
//	// Process 1024 tokens at once for faster prompt handling
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithBatch(1024),
//	)
func WithBatch(size int) ModelOption {
	return func(c *modelConfig) {
		c.batchSize = size
	}
}

// WithGPULayers sets the number of model layers to offload to GPU.
//
// By default, all layers are offloaded to GPU (-1). If GPU acceleration is
// unavailable, the library automatically falls back to CPU execution. Set to 0
// to force CPU-only execution, or specify a positive number to partially
// offload layers (useful for models larger than GPU memory).
//
// Default: -1 (offload all layers, with CPU fallback)
//
// Examples:
//
//	// Force CPU execution
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithGPULayers(0),
//	)
//
//	// Offload 35 layers to GPU, rest on CPU
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithGPULayers(35),
//	)
func WithGPULayers(n int) ModelOption {
	return func(c *modelConfig) {
		c.gpuLayers = n
	}
}

// WithThreads sets the number of threads for token generation.
// If not specified, defaults to runtime.NumCPU().
// This also sets threadsBatch to the same value unless WithThreadsBatch is used.
func WithThreads(n int) ModelOption {
	return func(c *modelConfig) {
		c.threads = n
	}
}

// WithThreadsBatch sets the number of threads for batch/prompt processing.
// If not specified, defaults to the same value as threads.
// For most use cases, leaving this unset is recommended.
func WithThreadsBatch(n int) ModelOption {
	return func(c *modelConfig) {
		c.threadsBatch = n
	}
}

// WithF16Memory enables 16-bit floating point memory mode.
//
// When enabled, the model uses FP16 precision for KV cache storage, reducing
// memory usage at the cost of slight accuracy loss. Most useful when working
// with very long contexts or memory-constrained environments.
//
// Default: false (uses FP32 for KV cache)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithF16Memory(),
//	)
func WithF16Memory() ModelOption {
	return func(c *modelConfig) {
		c.f16Memory = true
	}
}

// WithMLock forces the model to stay in RAM using mlock().
//
// When enabled, prevents the operating system from swapping model data to disk.
// Useful for production environments where consistent latency is critical, but
// requires sufficient physical RAM and may require elevated privileges.
//
// Default: false (allows OS to manage memory)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMLock(),
//	)
func WithMLock() ModelOption {
	return func(c *modelConfig) {
		c.mlock = true
	}
}

// WithMMap enables or disables memory-mapped file I/O for model loading.
//
// Memory mapping (mmap) allows the OS to load model data on-demand rather than
// reading the entire file upfront. This significantly reduces startup time and
// memory usage. Disable only if you encounter platform-specific issues.
//
// Default: true (enabled)
//
// Example:
//
//	// Disable mmap for compatibility
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMMap(false),
//	)
func WithMMap(enabled bool) ModelOption {
	return func(c *modelConfig) {
		c.mmap = enabled
	}
}

// WithEmbeddings enables embedding extraction mode.
//
// When enabled, the model can compute text embeddings via GetEmbeddings().
// This mode is required for semantic search, clustering, or similarity tasks.
// Note that not all models support embeddings - check model documentation.
//
// Default: false (text generation mode)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	)
//	embeddings, err := model.GetEmbeddings("Hello world")
func WithEmbeddings() ModelOption {
	return func(c *modelConfig) {
		c.embeddings = true
	}
}

// WithMainGPU sets the primary GPU device for model execution.
//
// Use this option to select a specific GPU in multi-GPU systems. The device
// string format depends on the backend (e.g. "0" for CUDA device 0). Most
// users with single-GPU systems don't need this option.
//
// Default: "" (uses default GPU)
//
// Example:
//
//	// Use second GPU
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithMainGPU("1"),
//	)
func WithMainGPU(gpu string) ModelOption {
	return func(c *modelConfig) {
		c.mainGPU = gpu
	}
}

// WithTensorSplit configures tensor distribution across multiple GPUs.
//
// Allows manual control of how model layers are distributed across GPUs in
// multi-GPU setups. The split string format is backend-specific (e.g.
// "0.7,0.3" for CUDA to use 70% on GPU 0, 30% on GPU 1). Most users should
// rely on automatic distribution instead.
//
// Default: "" (automatic distribution)
//
// Example:
//
//	// Distribute 60/40 across two GPUs
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithTensorSplit("0.6,0.4"),
//	)
func WithTensorSplit(split string) ModelOption {
	return func(c *modelConfig) {
		c.tensorSplit = split
	}
}

// WithPoolSize configures the context pool for concurrent inference.
//
// The pool maintains between min and max execution contexts, allowing parallel
// inference operations. Set max > 1 to enable concurrent generations. Each
// context requires memory (~contextSize × 4 bytes for KV cache), so balance
// concurrency needs with available RAM.
//
// Default: min=1, max=1 (no concurrency)
//
// Examples:
//
//	// Allow up to 4 concurrent generations
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithPoolSize(1, 4),
//	)
//
//	// Pre-allocate 2 contexts, allow up to 8
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithPoolSize(2, 8),
//	)
func WithPoolSize(min, max int) ModelOption {
	return func(c *modelConfig) {
		if min < 1 {
			min = 1
		}
		if max < min {
			max = min
		}
		c.minContexts = min
		c.maxContexts = max
	}
}

// WithIdleTimeout sets the duration before idle contexts are freed.
//
// Contexts unused for longer than this duration may be freed to reclaim memory.
// Currently, idle context cleanup is disabled, but this option reserves the
// API for future implementation.
//
// Default: 1 minute
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithIdleTimeout(5 * time.Minute),
//	)
func WithIdleTimeout(d time.Duration) ModelOption {
	return func(c *modelConfig) {
		c.idleTimeout = d
	}
}

// WithKVCacheType sets the quantization type for KV cache storage.
//
// The KV (key-value) cache stores attention states during generation and grows
// with context length. Quantizing this cache dramatically reduces VRAM usage
// with minimal quality impact:
//
//   - "q8_0" (default): 50% VRAM savings, ~0.1% quality loss (imperceptible)
//   - "f16": Full precision, no savings, maximum quality
//   - "q4_0": 75% VRAM savings, noticeable quality loss (models become forgetful)
//
// Memory scaling example for 131K context (DeepSeek-R1 trained capacity):
//   - f16:  18 GB
//   - q8_0:  9 GB (recommended)
//   - q4_0:  4.5 GB (use only for extreme VRAM constraints)
//
// Default: "q8_0" (best balance of memory and quality)
//
// Examples:
//
//	// Use default Q8 quantization (recommended)
//	model, err := llama.LoadModel("model.gguf")
//
//	// Maximum quality for VRAM-rich systems
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithKVCacheType("f16"),
//	)
//
//	// Extreme memory savings (accept quality loss)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithKVCacheType("q4_0"),
//	)
func WithKVCacheType(cacheType string) ModelOption {
	return func(c *modelConfig) {
		// Validate cache type
		switch cacheType {
		case "f16", "q8_0", "q4_0":
			c.kvCacheType = cacheType
		default:
			// Silently ignore invalid types and keep default
			// This prevents hard failures from typos while maintaining sensible behaviour
		}
	}
}

// WithFlashAttn controls Flash Attention kernel usage for attention computation.
//
// Flash Attention is a GPU-optimized attention implementation that significantly
// reduces VRAM usage and improves performance, especially for longer contexts.
// It's required when using quantized KV cache types (q8_0, q4_0).
//
// Available modes:
//   - "auto" (default): llama.cpp decides based on hardware and model config
//   - "enabled": Force Flash Attention on (fails if hardware doesn't support it)
//   - "disabled": Use traditional attention (incompatible with quantized KV cache)
//
// Technical details:
//   - Requires CUDA compute capability 7.0+ (Volta/Turing or newer)
//   - With GGML_CUDA_FA_ALL_QUANTS: Supports all KV cache quantization types
//   - Without flag: Only supports f16, q4_0, and q8_0 (matching K/V types)
//   - AUTO mode detects if backend scheduler supports the Flash Attention ops
//
// Default: "auto" (llama.cpp chooses optimal path)
//
// Examples:
//
//	// Use default auto-detection (recommended)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithKVCacheType("q8_0"),
//	)
//
//	// Force Flash Attention on (errors if unsupported)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithFlashAttn("enabled"),
//	)
//
//	// Disable Flash Attention (requires f16 KV cache)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithKVCacheType("f16"),
//	    llama.WithFlashAttn("disabled"),
//	)
func WithFlashAttn(mode string) ModelOption {
	return func(c *modelConfig) {
		// Validate flash attention mode
		switch mode {
		case "auto", "enabled", "disabled":
			c.flashAttn = mode
		default:
			// Silently ignore invalid modes and keep default
			// This prevents hard failures from typos while maintaining sensible behaviour
		}
	}
}

// WithParallel sets the number of parallel sequences for batch processing.
//
// This option controls how many independent sequences can be processed
// simultaneously in a single batch. Higher values enable larger batch sizes
// for operations like GetEmbeddingsBatch() but consume more VRAM.
//
// For embedding models, the library defaults to n_parallel=8 if not explicitly
// set. This option allows tuning this value for your specific VRAM constraints
// and batch sizes.
//
// VRAM usage scales approximately as:
//
//	base_model_size + (n_parallel × context_size × kv_cache_bytes)
//
// For example, a 4B Q8 embedding model with 8192 context and q8_0 cache:
//   - n_parallel=8: ~12 GB VRAM
//   - n_parallel=4: ~8 GB VRAM
//   - n_parallel=2: ~6 GB VRAM
//   - n_parallel=1: ~5 GB VRAM (disables batch processing)
//
// Trade-offs:
//   - Lower values: Less VRAM usage, slower batch processing, smaller max batch size
//   - Higher values: More VRAM usage, faster batch processing, larger max batch size
//
// Default: 1 for generation models, 8 for embedding models (auto-set)
//
// Examples:
//
//	// Use default (8 for embeddings, 1 for generation)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	)
//
//	// Tune down for large embedding model with limited VRAM
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	    llama.WithParallel(4),
//	)
//
//	// Single sequence (minimal VRAM, no batching)
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	    llama.WithParallel(1),
//	)
func WithParallel(n int) ModelOption {
	return func(c *modelConfig) {
		if n < 1 {
			n = 1
		}
		c.nParallel = n
	}
}

// WithSilentLoading disables progress output during model loading.
//
// By default, llama.cpp prints dots to stderr to indicate loading progress.
// This option suppresses that output completely, useful for clean logs in
// production environments or when progress output interferes with other
// output formatting.
//
// Note: The LLAMA_LOG environment variable controls general logging but
// does not suppress progress dots. Use this option for truly silent loading.
//
// Default: false (shows progress dots)
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithSilentLoading(),
//	)
func WithSilentLoading() ModelOption {
	return func(c *modelConfig) {
		c.disableProgressCallback = true
	}
}

// ProgressCallback is called during model loading with progress 0.0-1.0.
// Return false to cancel loading, true to continue.
type ProgressCallback func(progress float32) bool

// WithProgressCallback sets a custom progress callback for model loading.
//
// The callback is invoked periodically during model loading with progress
// values from 0.0 (start) to 1.0 (complete). This allows implementing
// custom progress indicators, logging, or loading cancellation.
//
// The callback receives:
//   - progress: float32 from 0.0 to 1.0 indicating loading progress
//
// The callback must return:
//   - true: continue loading
//   - false: cancel loading (LoadModel will return an error)
//
// IMPORTANT: The callback is invoked from a C thread during model loading.
// Ensure any operations are thread-safe. The callback should complete
// quickly to avoid blocking the loading process.
//
// Default: nil (uses llama.cpp default dot printing)
//
// Examples:
//
//	// Simple progress indicator
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithProgressCallback(func(progress float32) bool {
//	        fmt.Printf("\rLoading: %.0f%%", progress*100)
//	        return true
//	    }),
//	)
//
//	// Cancel loading after 50%
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithProgressCallback(func(progress float32) bool {
//	        if progress > 0.5 {
//	            return false // Cancel
//	        }
//	        return true
//	    }),
//	)
func WithProgressCallback(cb ProgressCallback) ModelOption {
	return func(c *modelConfig) {
		c.progressCallback = cb
	}
}

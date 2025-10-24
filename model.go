package llama

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo CPPFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo LDFLAGS: -L./ -lbinding -lcommon -lllama -lggml-cpu -lggml-base -lggml -lstdc++ -lm
#include "wrapper.h"
#include <stdlib.h>

// Helper function to get the address of the Go progress callback
extern bool goProgressCallback(float progress, void* user_data);

static inline llama_progress_callback_wrapper get_go_progress_callback() {
	return (llama_progress_callback_wrapper)goProgressCallback;
}
*/
import "C"

func init() {
	// Initialise llama.cpp logging based on LLAMA_LOG environment variable
	C.llama_wrapper_init_logging()
}

// Progress callback registry for Go callbacks
var (
	progressCallbackRegistry sync.Map
	progressCallbackCounter  uintptr
	progressCallbackMutex    sync.Mutex
)

// InitLogging (re)initializes llama.cpp logging system based on LLAMA_LOG environment variable.
//
// This function is called automatically when the package loads, but can be called again
// to reconfigure logging after changing the LLAMA_LOG environment variable.
//
// Supported LLAMA_LOG values:
//   - "none" - No logging
//   - "error" - Only errors
//   - "warn" - Warnings and errors (recommended for production)
//   - "info" - Informational messages (default)
//   - "debug" - Verbose debug output
//
// Example:
//
//	os.Setenv("LLAMA_LOG", "warn")  // Quiet mode
//	llama.InitLogging()             // Apply the change
func InitLogging() {
	C.llama_wrapper_init_logging()
}

// context represents a single execution context with usage tracking
type context struct {
	ptr     unsafe.Pointer // llama_wrapper_context_t*
	lastUse time.Time
	mu      sync.Mutex
}

// contextPool manages a dynamic pool of contexts for a model
type contextPool struct {
	model     unsafe.Pointer // llama_wrapper_model_t* (weights only)
	config    modelConfig
	contexts  []*context
	available chan *context
	done      chan struct{}
	mu        sync.Mutex
	wg        sync.WaitGroup
}

// Model represents a loaded LLAMA model with concurrent inference support.
//
// Model instances are thread-safe for concurrent inference operations but NOT
// for concurrent Close() calls. The internal context pool manages multiple
// execution contexts, allowing parallel generations when configured with
// WithPoolSize.
//
// Resources are automatically freed via finaliser, but explicit Close() is
// recommended for deterministic cleanup:
//
//	model, _ := llama.LoadModel("model.gguf")
//	defer model.Close()
//
// Note: Calling methods after Close() returns an error.
type Model struct {
	modelPtr           unsafe.Pointer // llama_wrapper_model_t* (weights only)
	pool               *contextPool
	mu                 sync.RWMutex
	closed             bool
	chatTemplates      unsafe.Pointer // cached common_chat_templates*
	ProgressCallbackID uintptr        // Internal ID for progress callback cleanup (for testing)
}

// modelConfig holds configuration for model loading
type modelConfig struct {
	contextSize            int
	batchSize              int
	gpuLayers              int
	threads                int
	threadsBatch           int
	nParallel              int // Number of parallel sequences (for batch embeddings)
	f16Memory              bool
	mlock                  bool
	mmap                   bool
	embeddings             bool
	mainGPU                string
	tensorSplit            string
	minContexts            int
	maxContexts            int
	idleTimeout            time.Duration
	prefixCaching          bool   // Enable KV cache prefix reuse (default: true)
	kvCacheType            string // KV cache quantization type: "f16", "q8_0", "q4_0" (default: "q8_0")
	flashAttn              string // Flash Attention mode: "auto", "enabled", "disabled" (default: "auto")
	disableProgressCallback bool
	progressCallback       ProgressCallback
}

// generateConfig holds configuration for text generation
type generateConfig struct {
	// Basic generation
	maxTokens     int
	temperature   float32
	seed          int
	stopWords     []string
	draftTokens   int
	debug         bool
	prefixCaching bool // Per-generation prefix caching control

	// Basic sampling parameters
	topK      int
	topP      float32
	minP      float32
	typP      float32
	topNSigma float32
	minKeep   int

	// Repetition penalties
	penaltyLastN   int
	penaltyRepeat  float32
	penaltyFreq    float32
	penaltyPresent float32

	// DRY (Don't Repeat Yourself) sampling
	dryMultiplier       float32
	dryBase             float32
	dryAllowedLength    int
	dryPenaltyLastN     int
	drySequenceBreakers []string

	// Dynamic temperature
	dynatempRange    float32
	dynatempExponent float32

	// XTC (eXclude Top Choices) sampling
	xtcProbability float32
	xtcThreshold   float32

	// Mirostat sampling
	mirostat    int
	mirostatTau float32
	mirostatEta float32

	// Other parameters
	nPrev     int
	nProbs    int
	ignoreEOS bool
}

// Default configurations
var defaultModelConfig = modelConfig{
	contextSize:   0, // 0 = use model's native maximum (queried after load)
	batchSize:     512,
	gpuLayers:     -1, // Offload all layers to GPU by default (falls back to CPU if unavailable)
	threads:       runtime.NumCPU(),
	threadsBatch:  0, // 0 means use same as threads (set in wrapper)
	nParallel:     1, // 1 for generation, auto-set higher for embeddings
	f16Memory:     false,
	mlock:         false,
	mmap:          true,
	embeddings:    false,
	minContexts:   1,
	maxContexts:   1,
	idleTimeout:   1 * time.Minute,
	prefixCaching: true,   // Enable by default for performance
	kvCacheType:   "q8_0", // 50% VRAM savings with ~0.1% quality loss
	flashAttn:     "auto", // Let llama.cpp choose optimal path
}

var defaultGenerateConfig = generateConfig{
	// Basic generation
	maxTokens:     128,
	temperature:   0.8,
	seed:          -1,
	draftTokens:   16,
	debug:         false,
	prefixCaching: true, // Inherit from model default

	// Basic sampling parameters
	topK:      40,
	topP:      0.95,
	minP:      0.05,
	typP:      1.0,  // 1.0 = disabled
	topNSigma: -1.0, // -1.0 = disabled
	minKeep:   0,

	// Repetition penalties
	penaltyLastN:   64,
	penaltyRepeat:  1.0, // 1.0 = disabled
	penaltyFreq:    0.0, // 0.0 = disabled
	penaltyPresent: 0.0, // 0.0 = disabled

	// DRY sampling
	dryMultiplier:       0.0, // 0.0 = disabled
	dryBase:             1.75,
	dryAllowedLength:    2,
	dryPenaltyLastN:     -1, // -1 = context size
	drySequenceBreakers: []string{"\n", ":", "\"", "*"},

	// Dynamic temperature
	dynatempRange:    0.0, // 0.0 = disabled
	dynatempExponent: 1.0,

	// XTC sampling
	xtcProbability: 0.0, // 0.0 = disabled
	xtcThreshold:   0.1,

	// Mirostat sampling
	mirostat:    0, // 0 = disabled
	mirostatTau: 5.0,
	mirostatEta: 0.1,

	// Other parameters
	nPrev:     64,
	nProbs:    0, // 0 = disabled
	ignoreEOS: false,
}

// ModelOption configures model loading behaviour.
//
// Options are applied using the functional options pattern. Available options
// include WithContext, WithGPULayers, WithThreads, WithPoolSize, and others.
// See individual With* functions for details.
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	    llama.WithGPULayers(35),
//	    llama.WithThreads(8),
//	)
type ModelOption func(*modelConfig)

// GenerateOption configures text generation behaviour.
//
// Options are applied using the functional options pattern. Available options
// include WithMaxTokens, WithTemperature, WithTopP, WithTopK, WithSeed,
// WithStopWords, WithDraftTokens, WithDebug, and WithPrefixCaching.
// See individual With* functions for details.
//
// Example:
//
//	text, err := model.Generate("Write a story",
//	    llama.WithMaxTokens(256),
//	    llama.WithTemperature(0.7),
//	)
type GenerateOption func(*generateConfig)

// LoadModel loads a GGUF model from the specified path.
//
// The path must point to a valid GGUF format model file. Legacy GGML formats
// are not supported. The function applies the provided options using the
// functional options pattern, with sensible defaults if none are specified.
//
// Resources are managed automatically via finaliser, but explicit cleanup with
// Close() is recommended for deterministic resource management:
//
//	model, err := llama.LoadModel("model.gguf")
//	if err != nil {
//	    return err
//	}
//	defer model.Close()
//
// Returns an error if the file doesn't exist, is not a valid GGUF model, or
// if context pool initialisation fails.
//
// Examples:
//
//	// Load with defaults
//	model, err := llama.LoadModel("model.gguf")
//
//	// Load with custom configuration
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	    llama.WithGPULayers(35),
//	    llama.WithPoolSize(1, 4),
//	)
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	if path == "" {
		return nil, fmt.Errorf("Model path cannot be null")
	}

	config := defaultModelConfig
	for _, opt := range opts {
		opt(&config)
	}

	// Auto-set nParallel for embeddings if not explicitly configured
	// Use 8 parallel sequences for efficient batch embedding processing
	if config.embeddings && config.nParallel == 1 {
		config.nParallel = 8
	}

	// Convert Go config to C struct for model loading
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cMainGPU *C.char
	if config.mainGPU != "" {
		cMainGPU = C.CString(config.mainGPU)
		defer C.free(unsafe.Pointer(cMainGPU))
	}

	var cTensorSplit *C.char
	if config.tensorSplit != "" {
		cTensorSplit = C.CString(config.tensorSplit)
		defer C.free(unsafe.Pointer(cTensorSplit))
	}

	var cKVCacheType *C.char
	if config.kvCacheType != "" {
		cKVCacheType = C.CString(config.kvCacheType)
		defer C.free(unsafe.Pointer(cKVCacheType))
	}

	var cFlashAttn *C.char
	if config.flashAttn != "" {
		cFlashAttn = C.CString(config.flashAttn)
		defer C.free(unsafe.Pointer(cFlashAttn))
	}

	params := C.llama_wrapper_model_params{
		n_ctx:           C.int(config.contextSize),
		n_batch:         C.int(config.batchSize),
		n_gpu_layers:    C.int(config.gpuLayers),
		n_threads:       C.int(config.threads),
		n_threads_batch: C.int(config.threadsBatch),
		n_parallel:      C.int(config.nParallel),
		f16_memory:      C.bool(config.f16Memory),
		mlock:           C.bool(config.mlock),
		mmap:            C.bool(config.mmap),
		embeddings:      C.bool(config.embeddings),
		main_gpu:        cMainGPU,
		tensor_split:    cTensorSplit,
		kv_cache_type:   cKVCacheType,
		flash_attn:      cFlashAttn,
	}

	// Configure progress callback if requested
	var callbackID uintptr
	if config.progressCallback != nil {
		progressCallbackMutex.Lock()
		progressCallbackCounter++
		callbackID = progressCallbackCounter
		progressCallbackMutex.Unlock()

		progressCallbackRegistry.Store(callbackID, config.progressCallback)

		// Set C callback (using helper function to get the function pointer)
		params.progress_callback = C.get_go_progress_callback()
		params.progress_callback_user_data = unsafe.Pointer(callbackID)
	} else if config.disableProgressCallback {
		params.disable_progress_callback = C.bool(true)
	}

	// Load model (weights only)
	modelPtr := C.llama_wrapper_model_load(cPath, params)
	if modelPtr == nil {
		// Clean up callback registry on failure
		if callbackID != 0 {
			progressCallbackRegistry.Delete(callbackID)
		}
		return nil, fmt.Errorf("failed to load model: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// Query model's native context if user didn't specify
	if config.contextSize == 0 {
		nativeContext := int(C.llama_wrapper_get_model_context_length(modelPtr))
		config.contextSize = nativeContext
	}

	// Optimisation: clamp batch size to context size
	// You can never process more tokens per batch than fit in total context
	if config.batchSize > config.contextSize {
		config.batchSize = config.contextSize
	}

	// Create context pool
	pool, err := newContextPool(modelPtr, config)
	if err != nil {
		C.llama_wrapper_model_free(modelPtr)
		return nil, fmt.Errorf("failed to create context pool: %w", err)
	}

	model := &Model{
		modelPtr:           modelPtr,
		pool:               pool,
		ProgressCallbackID: callbackID,
	}

	// Set finaliser to ensure cleanup
	runtime.SetFinalizer(model, (*Model).Close)

	return model, nil
}

// Close frees the model and its associated resources.
//
// This method is idempotent - multiple calls are safe and subsequent calls
// return immediately without error. Close() blocks until all active generation
// requests complete, ensuring clean shutdown.
//
// After Close() is called, all other methods return an error. The method uses
// a write lock to prevent concurrent operations during cleanup.
//
// Example:
//
//	model, _ := llama.LoadModel("model.gguf")
//	defer model.Close()
func (m *Model) Close() error {
	m.mu.Lock() // Write lock to block all operations
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}

	// Remove finaliser FIRST to prevent race with GC
	runtime.SetFinalizer(m, nil)

	// Clean up progress callback registry
	if m.ProgressCallbackID != 0 {
		progressCallbackRegistry.Delete(m.ProgressCallbackID)
		m.ProgressCallbackID = 0
	}

	// Free chat templates if cached
	if m.chatTemplates != nil {
		C.llama_wrapper_chat_templates_free(m.chatTemplates)
		m.chatTemplates = nil
	}

	// Close pool (frees all contexts)
	if m.pool != nil {
		m.pool.close()
		m.pool = nil
	}

	// Free model
	if m.modelPtr != nil {
		C.llama_wrapper_model_free(m.modelPtr)
		m.modelPtr = nil
	}

	m.closed = true
	return nil
}

// ChatTemplate returns the chat template from the model's GGUF metadata.
//
// Returns an empty string if the model has no embedded chat template.
// Most modern instruction-tuned models include a template in their GGUF metadata
// that specifies how to format messages for that specific model.
//
// Example:
//
//	template := model.ChatTemplate()
//	if template == "" {
//	    // Model has no template - user must provide one or use Generate()
//	}
func (m *Model) ChatTemplate() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return ""
	}

	// Call C function to get template from model metadata
	cTemplate := C.llama_wrapper_get_chat_template(m.modelPtr)
	if cTemplate == nil {
		return ""
	}

	return C.GoString(cTemplate)
}

// FormatChatPrompt formats chat messages using the model's chat template.
//
// This method applies the chat template to the provided messages and returns
// the resulting prompt string without performing generation. Useful for:
//   - Debugging what will be sent to the model
//   - Pre-computing prompts for caching
//   - Understanding how the template formats conversations
//
// The template priority is: opts.ChatTemplate > model's GGUF template > error.
//
// Example:
//
//	messages := []llama.ChatMessage{
//	    {Role: "system", Content: "You are helpful."},
//	    {Role: "user", Content: "Hello"},
//	}
//	prompt, err := model.FormatChatPrompt(messages, llama.ChatOptions{})
//	fmt.Println("Formatted prompt:", prompt)
func (m *Model) FormatChatPrompt(messages []ChatMessage, opts ChatOptions) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return "", fmt.Errorf("model is closed")
	}

	// Use the same template resolution logic as Chat/ChatStream
	template := opts.ChatTemplate
	if template == "" {
		template = m.ChatTemplate()
	}
	if template == "" {
		return "", fmt.Errorf("no chat template available: provide ChatOptions.ChatTemplate or use a model with embedded template")
	}

	// Apply template with addAssistant=true (same as generation)
	return applyChatTemplate(template, messages, true)
}

// getChatFormat gets the auto-detected chat format for reasoning parsing.
// This is cached on the model to avoid repeated detection.
func (m *Model) getChatFormat() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Initialize templates if not cached
	if m.chatTemplates == nil {
		m.chatTemplates = C.llama_wrapper_chat_templates_init(m.modelPtr, nil)
		if m.chatTemplates == nil {
			// Fallback to CONTENT_ONLY if init fails
			return int(C.LLAMA_CHAT_FORMAT_CONTENT_ONLY)
		}
	}

	return int(C.llama_wrapper_chat_templates_get_format(m.chatTemplates))
}

// applyChatTemplate applies a Jinja2 chat template to messages.
//
// This is an internal helper that wraps llama.cpp's native chat template system.
// The template can be from GGUF metadata or a custom Jinja2 template string.
//
// Returns the formatted prompt string ready for generation, or an error if
// template application fails.
func applyChatTemplate(template string, messages []ChatMessage, addAssistant bool) (string, error) {
	if template == "" {
		return "", fmt.Errorf("template cannot be empty")
	}
	if len(messages) == 0 {
		return "", fmt.Errorf("messages cannot be empty")
	}

	// Convert template to C string
	cTemplate := C.CString(template)
	defer C.free(unsafe.Pointer(cTemplate))

	// Build C arrays for roles and contents
	cRoles := make([]*C.char, len(messages))
	cContents := make([]*C.char, len(messages))

	// Allocate C strings and set up defer cleanup
	for i, msg := range messages {
		cRoles[i] = C.CString(msg.Role)
		cContents[i] = C.CString(msg.Content)
	}

	// Defer cleanup of all C strings
	defer func() {
		for i := range messages {
			C.free(unsafe.Pointer(cRoles[i]))
			C.free(unsafe.Pointer(cContents[i]))
		}
	}()

	// Call C function to apply template
	cResult := C.llama_wrapper_apply_chat_template(
		cTemplate,
		(**C.char)(unsafe.Pointer(&cRoles[0])),
		(**C.char)(unsafe.Pointer(&cContents[0])),
		C.int(len(messages)),
		C.bool(addAssistant),
	)

	if cResult == nil {
		return "", fmt.Errorf("failed to apply chat template: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// Convert result and free
	result := C.GoString(cResult)
	C.llama_wrapper_free_result(cResult)

	return result, nil
}

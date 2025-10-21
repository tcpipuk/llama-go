package llama

import (
	"fmt"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// Tokenize converts text to tokens.
//
// Tokens are integer IDs representing subword units in the model's vocabulary.
// This method is useful for advanced use cases like manual prompt construction,
// token counting, or analysis. For normal generation, use Generate() which
// handles tokenisation automatically.
//
// The tokeniser returns all tokens without truncation - there is no artificial
// limit on the number of tokens that can be returned.
//
// Examples:
//
//	// Count tokens in a prompt
//	tokens, _ := model.Tokenize("Hello world")
//	fmt.Printf("Token count: %d\n", len(tokens))
//
//	// Manual generation with tokens
//	tokens, _ := model.Tokenize("Once upon a time")
//	result, _ := model.GenerateWithTokens(tokens)
func (m *Model) Tokenize(text string) ([]int32, error) {
	m.mu.RLock() // Read lock to check closed state
	defer m.mu.RUnlock()

	if m.closed {
		return nil, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Use dynamic allocation approach - C++ wrapper manages memory
	var tokensPtr *C.int
	var count C.int

	C.llama_wrapper_tokenize_alloc(ctx.ptr, cText, &tokensPtr, &count)

	// Ensure tokens are freed even if we error out
	if tokensPtr != nil {
		defer C.llama_wrapper_free_tokens(tokensPtr)
	}

	// Check for errors
	if count < 0 || tokensPtr == nil {
		return nil, fmt.Errorf("tokenisation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// Copy from C array to Go slice
	// Create a Go slice header pointing to C memory (without copying)
	tokens := (*[1 << 30]C.int)(unsafe.Pointer(tokensPtr))[:count:count]

	// Now copy to a proper Go-managed slice
	result := make([]int32, count)
	for i := 0; i < int(count); i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// GetCachedTokenCount returns the number of cached tokens in a context (for debugging/metrics).
//
// This method provides insight into prefix caching behaviour, showing how many
// tokens from previous prompts are cached. Useful for diagnostics, performance
// monitoring, or validating prefix caching is working as expected.
//
// Note: This acquires a context from the pool, so the count may vary between
// calls depending on which context is available (each context maintains its
// own cache state).
//
// Example:
//
//	model.Generate("System prompt: You are helpful.\n\nUser: Hello")
//	cached, _ := model.GetCachedTokenCount()
//	fmt.Printf("Cached tokens: %d\n", cached)
//
//	model.Generate("System prompt: You are helpful.\n\nUser: Goodbye")
//	cached, _ = model.GetCachedTokenCount()
//	fmt.Printf("Cached tokens after reuse: %d\n", cached)  // Should be higher
func (m *Model) GetCachedTokenCount() (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return 0, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return 0, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	count := int(C.llama_wrapper_get_cached_token_count(ctx.ptr))
	if count < 0 {
		return 0, fmt.Errorf("failed to get cached token count: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	return count, nil
}

// GetEmbeddings computes embeddings for the given text.
//
// Embeddings are vector representations useful for semantic search, clustering,
// or similarity tasks. The model must be loaded with WithEmbeddings() to use
// this method, otherwise it returns an error. Not all models support embeddings -
// check model documentation.
//
// Example:
//
//	model, _ := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	)
//	defer model.Close()
//
//	emb1, _ := model.GetEmbeddings("Hello world")
//	emb2, _ := model.GetEmbeddings("Hi there")
//	similarity := cosineSimilarity(emb1, emb2)
func (m *Model) GetEmbeddings(text string) ([]float32, error) {
	m.mu.RLock() // Read lock to check closed state
	defer m.mu.RUnlock()

	if m.closed {
		return nil, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxEmbeddings := 4096 // Reasonable upper bound
	embeddings := make([]C.float, maxEmbeddings)

	count := C.llama_wrapper_embeddings(ctx.ptr, cText, &embeddings[0], C.int(maxEmbeddings))
	if count < 0 {
		return nil, fmt.Errorf("embedding generation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	result := make([]float32, count)
	for i := 0; i < int(count); i++ {
		result[i] = float32(embeddings[i])
	}

	return result, nil
}

// GetEmbeddingsBatch computes embeddings for multiple texts efficiently.
//
// This method processes multiple texts in a single batch operation, which is
// significantly more efficient than calling GetEmbeddings repeatedly. The model
// must be loaded with WithEmbeddings() to use this method.
//
// Batch processing works by:
//   - Tokenising all texts upfront
//   - Processing multiple sequences in parallel where possible
//   - Automatically splitting into sub-batches if needed to respect memory limits
//
// Example:
//
//	model, _ := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	    llama.WithBatch(256),  // Smaller batch for memory control
//	)
//	defer model.Close()
//
//	texts := []string{
//	    "First document",
//	    "Second document",
//	    "Third document",
//	}
//	embeddings, _ := model.GetEmbeddingsBatch(texts)
//	// embeddings[i] contains the embedding for texts[i]
func (m *Model) GetEmbeddingsBatch(texts []string) ([][]float32, error) {
	m.mu.RLock() // Read lock to check closed state
	defer m.mu.RUnlock()

	if m.closed {
		return nil, fmt.Errorf("model is closed")
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	// Get embedding dimension from model
	nEmbd := int(C.llama_wrapper_model_n_embd(m.modelPtr))
	if nEmbd <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", nEmbd)
	}

	// Convert Go strings to C strings
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}
	defer func() {
		for i := range cTexts {
			C.free(unsafe.Pointer(cTexts[i]))
		}
	}()

	// Allocate output buffer: n_texts * n_embd floats
	outputSize := len(texts) * nEmbd
	cEmbeddings := make([]C.float, outputSize)

	// Call C batch function
	count := C.llama_wrapper_embeddings_batch(
		ctx.ptr,
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.int(len(texts)),
		&cEmbeddings[0],
		C.int(nEmbd),
	)

	if count < 0 {
		return nil, fmt.Errorf("batch embedding generation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	if int(count) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: expected %d, got %d", len(texts), count)
	}

	// Convert C float array to Go [][]float32
	result := make([][]float32, len(texts))
	for i := 0; i < len(texts); i++ {
		result[i] = make([]float32, nEmbd)
		for j := 0; j < nEmbd; j++ {
			result[i][j] = float32(cEmbeddings[i*nEmbd+j])
		}
	}

	return result, nil
}

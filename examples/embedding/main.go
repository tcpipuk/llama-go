// Embedding example demonstrates generating text embeddings for semantic tasks.
//
// This program loads a GGUF embedding model and computes vector representations
// of input text. Embeddings are useful for semantic search, clustering, similarity
// comparison, and other machine learning tasks that require numerical representations
// of text.
//
// Usage:
//
//	embedding -m embedding-model.gguf -t "text to embed"
//
// The model must be loaded with embedding support enabled (WithEmbeddings option).
// Not all models support embeddings - check model documentation before use. Typical
// embedding models include sentence transformers and specialised embedding variants.
//
// The example demonstrates:
//   - Loading models in embedding mode
//   - Generating embeddings from text
//   - Inspecting embedding vector properties
//   - Computing basic embedding statistics
//
// Embeddings can be used for:
//   - Semantic search (finding similar documents)
//   - Clustering (grouping related texts)
//   - Classification (ML training features)
//   - Similarity scoring (comparing text meaning)
//
// Output includes the embedding dimension, sample values, and basic magnitude
// statistics to verify the embedding generation succeeded.
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
		modelPath = flag.String("m", "./Qwen3-Embedding-0.6B-Q8_0.gguf", "path to GGUF embedding model file")
		text      = flag.String("t", "Hello world", "text to get embeddings for")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		context   = flag.Int("c", 2048, "context size")
	)
	flag.Parse()

	// Load model with embeddings enabled
	fmt.Printf("Loading embedding model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	// Create context with embedding support
	ctx, err := model.NewContext(
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithEmbeddings(),
		llama.WithF16Memory(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Model loaded successfully.\n")
	fmt.Printf("Getting embeddings for: %s\n", *text)

	// Generate embeddings
	embeddings, err := ctx.GetEmbeddings(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating embeddings: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nEmbeddings generated successfully!\n")
	fmt.Printf("Vector dimension: %d\n", len(embeddings))
	fmt.Printf("First 10 values: ")
	for i := 0; i < 10 && i < len(embeddings); i++ {
		fmt.Printf("%.4f ", embeddings[i])
	}
	fmt.Printf("\n")

	// Calculate magnitude for demonstration
	magnitude := float32(0.0)
	for _, val := range embeddings {
		magnitude += val * val
	}
	magnitude = float32(1.0) / float32(len(embeddings)) * magnitude // Mean squared
	fmt.Printf("Mean squared magnitude: %.6f\n", magnitude)
}

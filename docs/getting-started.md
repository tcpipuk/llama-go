# Getting started guide

This guide walks you through setting up llama-go from scratch to running your first inference
example. By the end, you'll have a working installation and understand the basic workflow.

## Prerequisites

Before starting, you'll need:

- Git with submodule support
- Docker (recommended) or a C++ compiler with CMake
- About 1GB of disk space for the build and test model

That's it - we'll handle everything else through containers to avoid dependency issues.

## Step 1: Clone the repository

Clone the repository with its llama.cpp submodule:

```bash
git clone --recurse-submodules https://github.com/tcpipuk/llama-go
cd llama-go
```

If you've already cloned without submodules, initialise them:

```bash
git submodule update --init --recursive
```

## Step 2: Build the library

We'll use the project's build containers which include all necessary build tools, including CMake:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"
```

This creates several files:

- `libbinding.a` - The main library for Go
- `libllama.so`, `libggml.so`, etc. - Shared libraries needed at runtime
- `libcommon.a` - Common utilities

The build process typically takes 2-5 minutes depending on your system.

## Step 3: Download a test model

For testing, we'll use Qwen3 0.6B - it's small enough to download quickly but capable enough to
demonstrate the library:

```bash
wget -q https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

This downloads approximately 600MB. The model uses the GGUF format, which is the current standard
for llama.cpp.

## Step 4: Run your first inference

Now test the installation with a simple question:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples -m Qwen3-0.6B-Q8_0.gguf \
           -p 'What is the capital of France?' -n 50"
```

If everything works correctly, you'll see:

1. Model loading messages
2. Your prompt: "What is the capital of France?"
3. Generated text completing your prompt

The inference should complete in under a minute on most systems.

## Understanding the environment variables

The library requires three environment variables:

- `LIBRARY_PATH`: Tells the Go compiler where to find the static library
- `C_INCLUDE_PATH`: Tells the compiler where to find header files
- `LD_LIBRARY_PATH`: Tells the runtime where to find shared libraries

Without these, you'll see "undefined symbol" or "library not found" errors.

## Interactive mode

You can also run the example in interactive mode by omitting the `-p` parameter:

```bash
docker run --rm -it -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples -m Qwen3-0.6B-Q8_0.gguf"
```

This starts an interactive session where you can type prompts and see responses in real-time.

## Troubleshooting

### Build issues

**"cmake: command not found"**

- Use the build container as shown above, or install CMake on your system

**"No such file or directory: wrapper.cpp"**

- Make sure you're in the correct directory and the submodules are initialised

**Missing `.so` files after build**

- Check that the build completed successfully and didn't exit early due to errors

### Runtime issues

**"undefined symbol" errors**

- Ensure `LD_LIBRARY_PATH` includes the directory containing the `.so` files
- Verify the shared libraries exist in your project directory

**"failed to load model"**

- Check the model file path is correct
- Confirm the file is a valid GGUF format (not GGML or corrupted)
- Ensure you have enough RAM for the model

**"context size" warnings**

- These are normal for small models and don't affect basic functionality

### Performance considerations

If inference seems slow:

- The CPU-only build is functional but not optimised for speed
- Consider hardware acceleration options (see [building guide](building.md))
- Smaller models like Qwen3-0.6B prioritise compatibility over performance

## Next steps

Now that you have a working installation:

- **[API guide](api-guide.md)** - Complete guide to Model/Context separation, thread safety,
  streaming, embeddings, and advanced patterns
- **[Building guide](building.md)** - Hardware acceleration options (CUDA, Metal, Vulkan, etc.)
- **[Examples](../examples/README.md)** - Working code for chat, streaming, embeddings, and
  speculative decoding
- **[Hugging Face GGUF models](https://huggingface.co/models?library=gguf)** - Try different models

## What you've accomplished

You've successfully:

- Built the llama-go library with all dependencies
- Downloaded and tested with a working language model
- Verified the complete inference pipeline works
- Understood the basic environment setup
- Seen the Model/Context separation pattern in action

## Using in your own project

Now that the library works, here's how to integrate it into your Go application:

1. **Import the package** in your Go code:

   ```go
   import llama "github.com/tcpipuk/llama-go"
   ```

2. **Use the Model/Context API pattern**:

   ```go
   // Load model weights (ModelOption: WithGPULayers, WithMLock, etc.)
   model, err := llama.LoadModel(
       "model.gguf",
       llama.WithGPULayers(-1), // Offload all layers to GPU
   )
   if err != nil {
       return err
   }
   defer model.Close()

   // Create execution context (ContextOption: WithContext, WithBatch, etc.)
   ctx, err := model.NewContext(
       llama.WithContext(2048),
       llama.WithF16Memory(),
   )
   if err != nil {
       return err
   }
   defer ctx.Close()

   // Generate text
   response, err := ctx.Generate("Hello world", llama.WithMaxTokens(50))
   if err != nil {
       return err
   }
   fmt.Println(response)
   ```

3. **Build your application** with the same environment variables:

   ```bash
   export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD
   go build -o myapp
   ```

4. **Distribute the shared libraries** (`.so` files) alongside your binary - see the
   [building guide](building.md#distributing-your-application) for deployment details.

The [API guide](api-guide.md) shows common patterns like streaming, chat completion, embeddings,
concurrent inference, and speculative decoding. For hardware acceleration, see the
[building guide](building.md#hardware-acceleration).

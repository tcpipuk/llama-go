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

This creates several static archive files (default linkage):

- `libbinding.a` - The main library for Go
- `libllama.a`, `libllama-common.a`, `libllama-common-base.a` - llama.cpp libraries
- `libggml.a`, `libggml-base.a`, `libggml-cpu.a` - ggml libraries
- For GPU backends (e.g. `BUILD_TYPE=cublas`): a backend-specific archive like `libggml-cuda.a`

The build process typically takes 2-5 minutes depending on your system. To build with shared
libraries instead (`.so` files alongside the binary), pass `BUILD_LINKAGE=shared` to make.

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
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace \
           go run ./examples -m Qwen3-0.6B-Q8_0.gguf \
           -p 'What is the capital of France?' -n 50"
```

If everything works correctly, you'll see:

1. Model loading messages
2. Your prompt: "What is the capital of France?"
3. Generated text completing your prompt

The inference should complete in under a minute on most systems.

## Understanding the environment variables

The library requires two environment variables at build time:

- `LIBRARY_PATH`: Tells the Go compiler where to find the static archives
- `C_INCLUDE_PATH`: Tells the compiler where to find header files

Without these, you'll see "undefined symbol" or "library not found" errors.

`LD_LIBRARY_PATH` is **not** required:

- In the default static linkage mode, everything is linked into the binary at build time
- In shared linkage mode (`BUILD_LINKAGE=shared`), `-Wl,-rpath,$ORIGIN` is baked into the
  binary so it finds `.so` files relative to the executable

## Interactive mode

You can also run the example in interactive mode by omitting the `-p` parameter:

```bash
docker run --rm -it -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace \
           go run ./examples -m Qwen3-0.6B-Q8_0.gguf"
```

This starts an interactive session where you can type prompts and see responses in real-time.

## Troubleshooting

### Build issues

**"cmake: command not found"**

- Use the build container as shown above, or install CMake on your system

**"No such file or directory: wrapper.cpp"**

- Make sure you're in the correct directory and the submodules are initialised

**Missing artefacts after build**

- Default static mode produces `.a` files (`libbinding.a`, `libllama-common.a`, `libggml*.a` etc.)
- Shared mode (`BUILD_LINKAGE=shared`) produces `.so` files instead
- Check that the build completed successfully and didn't exit early due to errors

### Runtime issues

**"undefined symbol" errors at link time**

- Ensure you set `LIBRARY_PATH` and `C_INCLUDE_PATH` to the workspace directory
- Verify the static archives (or `.so` files in shared mode) exist in your project directory
- For shared mode, ensure you pass `-tags shared_lib` to `go build`

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
   export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD
   go build -o myapp
   ```

4. **Distribute your application**: in the default static mode, the binary is self-contained
   for the llama.cpp side — no `.so` files needed alongside it. (Backend runtimes like the
   CUDA driver or NCCL still need to be installed on the target system, of course.) If you
   built with `BUILD_LINKAGE=shared`, the `.so` files need to sit next to the binary. See
   the [building guide](building.md#distributing-your-application) for deployment details.

The [API guide](api-guide.md) shows common patterns like streaming, chat completion, embeddings,
concurrent inference, and speculative decoding. For hardware acceleration, see the
[building guide](building.md#hardware-acceleration).

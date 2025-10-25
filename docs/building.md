# Building guide

This guide covers all build options for llama-go, from basic CPU-only builds to
hardware-accelerated configurations for maximum performance.

## Build requirements

The library requires:

- C++ compiler (GCC/Clang)
- CMake (required for llama.cpp b6603+)
- Git with submodule support

For containerised builds (recommended), you only need Docker.

## Build methods

### Recommended: Using project build containers

We recommend using the project's build containers which include:

- Complete C/C++ build toolchain
- CMake (required for modern llama.cpp)
- CUDA development tools
- All required dependencies

The build container includes CUDA support: `git.tomfos.tr/tom/llama-go:build-cuda`

```bash
# Standard build (includes CUDA support)
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"

# CUDA-accelerated build
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"
```

### Alternative: Generic Ubuntu containers

If you prefer not to use the project containers:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace ubuntu:24.04 \
  bash -c "apt-get update && apt-get install -y build-essential cmake && \
           LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"
```

### Local builds

For local builds, ensure you have CMake installed:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake

# Build the library
make libbinding.a
```

## Build output

The build process creates several files:

**Static libraries:**

- `libbinding.a` - Main static library for Go linking (contains wrapper.o)
- `libcommon.a` - Common utilities

**Shared libraries:**

- `libllama.so` - LLaMA model operations
- `libggml.so` - GGML tensor operations
- `libggml-base.so` - Base GGML functionality
- `libggml-cpu.so` - CPU-specific operations

## Linking options

You need to decide how to link the llama.cpp libraries into your application. Both approaches are
equally valid - choose based on your deployment needs.

### Dynamic linking (default)

The default build creates shared libraries that load at runtime. Your Go binary includes the
static wrapper libraries (`libbinding.a`, `libcommon.a`) whilst the llama.cpp components remain
as separate `.so` files.

**When to use:**

- You want smaller binaries
- You're comfortable managing library paths
- You might update llama.cpp independently

**Building:** Use the standard build commands shown above.

**What you ship:**

- Your application binary (~10-20MB)
- The `.so` files listed above
- Model files

### Static linking

Build llama.cpp as static libraries to create a single self-contained binary. Modify the CMake
build to pass `-DBUILD_SHARED_LIBS=OFF` in the Makefile.

**When to use:**

- You want single-binary deployment
- You're deploying to containers
- You want to avoid library path configuration

**Building:** Modify the Makefile's CMake configuration to include `-DBUILD_SHARED_LIBS=OFF`.

**What you ship:**

- Your application binary (30-100MB depending on acceleration)
- Model files

## Distributing your application

### With dynamic linking

Users need the `.so` files accessible at runtime:

```bash
# Option 1: Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
./your-app

# Option 2: Install to system library directory
sudo cp *.so /usr/local/lib/
sudo ldconfig
./your-app
```

**Docker deployment:**

```dockerfile
FROM golang:1.25 as builder
WORKDIR /build
# ... build your application ...

FROM debian:stable-slim
COPY --from=builder /build/your-app /app/
COPY --from=builder /build/*.so /usr/local/lib/
RUN ldconfig
CMD ["/app/your-app"]
```

### With static linking

No runtime configuration needed - just run your binary:

```bash
./your-app
```

## Hardware acceleration

Different backends provide hardware-accelerated inference. Build with the appropriate `BUILD_TYPE`
and link with the required libraries.

Once built, use `llama.WithGPULayers(-1)` when loading models to offload computation to the GPU:

```go
// Load model with GPU acceleration (ModelOption)
model, _ := llama.LoadModel(
    "model.gguf",
    llama.WithGPULayers(-1), // -1 = offload all layers
)
defer model.Close()

// Create context (ContextOption)
ctx, _ := model.NewContext(llama.WithContext(2048))
defer ctx.Close()
```

| Backend | Build command | CGO flags | Notes |
|---------|---------------|-----------|-------|
| CUDA | `BUILD_TYPE=cublas make libbinding.a` | `-lcublas -lcudart -L/usr/local/cuda/lib64/` | NVIDIA GPUs |
| Metal | `BUILD_TYPE=metal make libbinding.a` | `-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders` | Apple Silicon |
| OpenBLAS | `BUILD_TYPE=openblas make libbinding.a` | `-lopenblas` | CPU acceleration |
| OpenCL | `BUILD_TYPE=opencl make libbinding.a` | `-lOpenCL` (Linux), `-framework OpenCL` (macOS) | Broad compatibility including mobile GPUs |
| RPC | `BUILD_TYPE=rpc make libbinding.a` | `-lpthread` | Distributed inference across machines |
| ROCm | `BUILD_TYPE=hipblas make libbinding.a` | `-O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc -lrocblas -lhipblas` | AMD GPUs, requires ROCm compilers |
| SYCL | `BUILD_TYPE=sycl make libbinding.a` | `-lsycl -L/opt/intel/oneapi/compiler/latest/linux/lib` | Intel Arc/Xe GPUs, optional NVIDIA/AMD |
| Vulkan | `BUILD_TYPE=vulkan make libbinding.a` | `-lvulkan -L/usr/lib/x86_64-linux-gnu` | Cross-platform GPU (NVIDIA, AMD, Intel, ARM) |

### CUDA acceleration example

Build with CUDA support:

```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"
```

Run with CUDA libraries:

```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples -m /path/to/model.gguf -p 'Hello world' -n 50"
```

### OpenBLAS acceleration example

For CPU acceleration without GPU hardware:

```bash
# Build with OpenBLAS
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "BUILD_TYPE=openblas LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"

# Run with OpenBLAS
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "CGO_LDFLAGS='-lopenblas' \
           LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples -m /path/to/model.gguf -p 'Hello world' -n 50"
```

### Metal acceleration (Apple Silicon)

For Apple Silicon Macs:

```bash
BUILD_TYPE=metal make libbinding.a

# Copy the Metal shader (required)
cp build/bin/ggml-metal.metal .

# Run with Metal frameworks
CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD \
  go run ./examples -m /path/to/model.gguf -p "Hello world" -n 50
```

## Multi-architecture builds

For release builds targeting multiple architectures:

```bash
# Build for linux/amd64
docker buildx build --platform linux/amd64 -f Dockerfile.build -o type=local,dest=. .

# Build for linux/arm64
docker buildx build --platform linux/arm64 -f Dockerfile.build -o type=local,dest=. .
```

This creates architecture-specific static libraries:

- `libbinding_linux_amd64.a` - For x86_64 systems
- `libbinding_linux_arm64.a` - For ARM64 systems

Note: ARM64 builds take significantly longer due to emulation (5-15 minutes vs 1-2 minutes for
amd64).

## Environment variables

```bash
export LIBRARY_PATH=$PWD        # Build-time: linking static libraries
export C_INCLUDE_PATH=$PWD      # Build-time: locating header files
export LD_LIBRARY_PATH=$PWD     # Runtime: loading shared libraries
```

## Build troubleshooting

### CMake not found

- Use the project build containers which include CMake
- Or install CMake: `apt-get install cmake` (Ubuntu/Debian)

### Submodule not initialised

```bash
git submodule update --init --recursive
```

### Build fails with "undefined reference"

- Ensure you're using a complete build environment (act-runner recommended)
- Check that all required development packages are installed

### Shared libraries not created

- Verify the build completed successfully without errors
- Check that CMake found all required dependencies

### Cross-compilation issues

- Use the provided Dockerfile.build for reliable multi-architecture builds
- Ensure you have buildx enabled: `docker buildx create --use`

## Configuration options

The API separates options into **ModelOption** (affects how weights are loaded) and
**ContextOption** (affects execution behaviour):

### ModelOption (for LoadModel)

```go
model, _ := llama.LoadModel(
    "model.gguf",
    llama.WithGPULayers(-1),              // Offload all layers to GPU
    llama.WithMLock(),                    // Lock model in RAM (prevents swapping)
    llama.WithMMap(true),                 // Enable memory mapping
    llama.WithMainGPU(0),                 // Primary GPU for multi-GPU
    llama.WithTensorSplit([]float32{0.7, 0.3}), // Split across GPUs
    llama.WithSilentLoading(),            // Suppress progress output
)
```

### ContextOption (for NewContext)

```go
ctx, _ := model.NewContext(
    llama.WithContext(8192),        // Context window size
    llama.WithBatch(512),           // Batch size for parallel processing
    llama.WithThreads(16),          // CPU threads for inference
    llama.WithThreadsBatch(16),     // CPU threads for batch processing
    llama.WithF16Memory(),          // Use float16 for KV cache (saves VRAM)
    llama.WithEmbeddings(),         // Enable embedding mode
    llama.WithKVCacheType("q8_0"),  // Quantized KV cache (saves VRAM)
    llama.WithFlashAttn(),          // Enable Flash Attention (faster)
    llama.WithParallel(4),          // Number of parallel sequences
    llama.WithPrefixCaching(),      // Cache common prompt prefixes
)
```

See the [API guide](api-guide.md) for detailed option explanations and usage patterns.

## Performance considerations

Choose your build type based on your hardware:

- **CPU only**: Use OpenBLAS for better performance than plain CPU
- **NVIDIA GPU**: CUDA provides the best performance for supported hardware
- **AMD GPU**: ROCm/HIP support varies by GPU generation
- **Apple Silicon**: Metal provides excellent performance on M-series Macs
- **General GPU**: OpenCL works across platforms but with varying performance

After building with acceleration, use `llama.WithGPULayers(-1)` in LoadModel to offload
computation to the GPU.

## Clean builds

To ensure a fresh build:

```bash
make clean
# Remove any cached Docker layers if needed
docker system prune -f
```

This removes all build artifacts and forces a complete rebuild on the next `make` command.

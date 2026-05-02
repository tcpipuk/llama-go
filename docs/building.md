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

The build process creates a set of artefacts in the workspace root. Which type depends on
the linkage mode (see next section).

**Static linkage (default, `BUILD_LINKAGE=static`)** — produces `.a` archives:

- `libbinding.a` - Main static library for Go linking (contains wrapper.o)
- `libllama-common.a`, `libllama-common-base.a` - llama.cpp common utilities
- `libllama.a` - llama.cpp core
- `libggml.a`, `libggml-base.a`, `libggml-cpu.a` - ggml libraries
- For GPU backends: a backend-specific archive (e.g. `libggml-cuda.a` for CUDA)

**Shared linkage (`BUILD_LINKAGE=shared`)** — produces `.so` files instead, with the same
naming pattern (e.g. `libllama-common.so`, `libllama.so`, `libggml-cuda.so`).

## Linkage modes

The build supports two linkage modes, controlled by the `BUILD_LINKAGE` Makefile variable.
Both are first-class options — pick based on your deployment needs.

### Static (default)

Static archives get linked into your Go binary at build time. The result is a single
self-contained executable for the llama.cpp side; nothing needs to ship alongside it.

**When to use:**

- You want single-binary deployment (containers, embedded, distribution)
- You don't want to manage library paths or rpath at runtime
- You're integrating llama-go into another Go binary that needs to stay portable

**Building:**

```bash
make libbinding.a   # BUILD_LINKAGE=static is implicit
```

**Building Go consumers:**

```bash
go build              # default — pulls in linkage_static.go automatically
go build -tags cublas # for CUDA-enabled builds (or other backends)
```

**What you ship:**

- Your application binary (50-200MB depending on backend; CUDA archives are large)
- Model files
- Backend system libraries (CUDA toolkit, NCCL etc.) if your backend needs them — these
  are dynamically loaded from the system, not bundled

### Shared

Shared libraries (`.so` files on Linux, `.dylib` on macOS) sit next to the binary and are
loaded at runtime. The linker bakes `-Wl,-rpath,$ORIGIN` into the binary so the `.so` files
only need to be in the same directory — no `LD_LIBRARY_PATH` required.

**When to use:**

- You want a smaller binary at the cost of shipping multiple files
- You want to swap out specific backend libraries without recompiling everything
- You're updating llama.cpp components independently of your application

**Building:**

```bash
BUILD_LINKAGE=shared make libbinding.a
```

**Building Go consumers:**

```bash
go build -tags shared_lib                # default backend
go build -tags 'cublas shared_lib'       # CUDA + shared
```

The `shared_lib` tag pulls in the shared-mode CGO LDFLAGS (which include the rpath directive).

**What you ship:**

- Your application binary
- All `.so` files from the workspace root, in the same directory as the binary
- Model files
- Backend system libraries (as for static mode)

## Distributing your application

### Static linkage

No runtime configuration needed — just run your binary. The only external dependencies are
backend system libraries (e.g. CUDA toolkit), which the user must have installed.

```bash
./your-app
```

**Docker deployment (static):**

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04
COPY your-app /app/
COPY model.gguf /app/
CMD ["/app/your-app"]
```

### Shared linkage

Distribute the `.so` files alongside your binary. They must end up in the same directory
as the executable so `$ORIGIN` rpath resolution finds them.

```bash
./your-app   # finds libllama-common.so etc next to itself, no env vars needed
```

**Docker deployment (shared):**

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu24.04
COPY your-app /app/
COPY *.so* /app/
COPY model.gguf /app/
CMD ["/app/your-app"]
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

Each backend is enabled by setting `BUILD_TYPE` at build time and passing the corresponding
Go build tag to `go build`. The CGO LDFLAGS for each backend are now wired up inside
per-backend Go files (`llama_<backend>.go` and, for CUDA, `llama_cublas_static.go` /
`llama_cublas_shared.go`) so consumers don't need to pass them via `CGO_LDFLAGS`.

| Backend | Build command | Go build tag | Notes |
| --- | --- | --- | --- |
| CUDA | `BUILD_TYPE=cublas CUDA_ARCHITECTURES=86 make libbinding.a` | `-tags cublas` | NVIDIA GPUs. Static archive ~280MB; shared `.so` is much smaller. Both modes link `libcudart`, `libcublas`, `libcuda`, `libnccl` from the system. |
| Metal | `BUILD_TYPE=metal make libbinding.a` | `-tags metal` | Apple Silicon. Build links Metal/MetalKit/Foundation frameworks via `CGO_LDFLAGS` set in the Makefile. |
| OpenBLAS | `BUILD_TYPE=openblas make libbinding.a` | `-tags openblas` | CPU acceleration. Always uses system shared `libopenblas`. |
| OpenCL | `BUILD_TYPE=opencl make libbinding.a` | `-tags opencl` | Broad compatibility including mobile GPUs. |
| RPC | `BUILD_TYPE=rpc make libbinding.a` | `-tags rpc` | Distributed inference across machines. |
| ROCm | `BUILD_TYPE=hipblas make libbinding.a` | `-tags hipblas` | AMD GPUs, requires ROCm compilers. |
| SYCL | `BUILD_TYPE=sycl make libbinding.a` | `-tags sycl` | Intel Arc/Xe GPUs, optional NVIDIA/AMD. |
| Vulkan | `BUILD_TYPE=vulkan make libbinding.a` | `-tags vulkan` | Cross-platform GPU (NVIDIA, AMD, Intel, ARM). |

For shared linkage, append `shared_lib` to the build tag list (e.g. `-tags 'cublas shared_lib'`).
The CUDA backend's static-mode CGO LDFLAGS are explicit (`-lcudart -lcublas -lcuda -lnccl`)
because static archives don't carry their own dependency information; shared mode gets these
transitively via `libggml-cuda.so`'s NEEDED entries.

### CUDA acceleration example

Build with CUDA support (default static linkage):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace BUILD_TYPE=cublas CUDA_ARCHITECTURES=86 make libbinding.a"
```

Run, passing `-tags cublas` to enable the CUDA build tag:

```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace \
           go run -tags cublas ./examples -m /path/to/model.gguf -p 'Hello world' -n 50"
```

### OpenBLAS acceleration example

For CPU acceleration without GPU hardware:

```bash
# Build with OpenBLAS
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "BUILD_TYPE=openblas LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"

# Run with OpenBLAS — the openblas build tag activates `#cgo LDFLAGS: -lopenblas`
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace \
           go run -tags openblas ./examples -m /path/to/model.gguf -p 'Hello world' -n 50"
```

### Metal acceleration (Apple Silicon)

For Apple Silicon Macs:

```bash
BUILD_TYPE=metal make libbinding.a

# Copy the Metal shader (required)
cp build/bin/ggml-metal.metal .

# Run with Metal frameworks
CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run -tags metal ./examples -m /path/to/model.gguf -p "Hello world" -n 50
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
export LIBRARY_PATH=$PWD        # Build-time: linking against archives in $PWD
export C_INCLUDE_PATH=$PWD      # Build-time: locating header files
```

`LD_LIBRARY_PATH` is **not** required:

- Default static linkage links everything into the binary at build time
- Shared linkage (`BUILD_LINKAGE=shared`) bakes `-Wl,-rpath,$ORIGIN` into the binary so
  `.so` files are found relative to the executable

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

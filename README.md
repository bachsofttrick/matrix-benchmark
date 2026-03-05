# Matrix Multiplication Benchmark

Benchmarks five NxN (default 8192×8192) matrix multiplication methods:

| # | Method | Description |
|---|--------|-------------|
| 1 | **Naive** | Triple nested loop, O(n³) |
| 2 | **AVX2** | Transposed B + 256-bit FMA SIMD (8 floats/cycle) |
| 3 | **Vulkan** | GPU compute shader (16×16 work groups) |
| 4 | **pthread** | Row-partitioned parallel naive loop, one thread per CPU core |
| 5 | **pthread+AVX2** | Row-partitioned parallel loop with AVX2+FMA inside each thread |

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| `g++` / `clang++` (C++17) | Compile the program |
| `libvulkan-dev` | Vulkan headers + loader |
| `glslang-tools` / `glslangValidator` | Compile GLSL → SPIR-V |

Install on Ubuntu/Debian:
```sh
sudo apt install g++ libvulkan-dev glslang-tools
```

---

## Build & Run

```sh
make        # compiles shader + binary
./test      # run the benchmark
```

`make clean` removes the binary and SPIR-V file.

### Manual build (without make)

#### 1. Compile the GLSL compute shader → SPIR-V
```sh
glslangValidator -V matmul.comp -o matmul.spv
```

#### 2. Compile the C++ program
```sh
g++ -O2 -mavx2 -mfma -std=c++17 test.cpp -o test -lvulkan -lpthread
```

> **Note:** `-mavx2 -mfma` enables AVX2 and FMA intrinsics. These are supported
> on Intel Haswell (2013+) and AMD Ryzen (2017+) and most CPUs since.

The `matmul.spv` file must be in the same directory as the executable.

---

## Expected Output

```
Matrix size: 8192 x 8192
Threads (pthread): 8

Naive:        XXXX ms
AVX2:         XXXX ms
pthread:      XXXX ms
pthread+AVX2: XXXX ms
  [Vulkan] GPU: <your GPU name>
Vulkan:       XXXX ms  (upload + dispatch + download)
```

- **Naive** is the slowest — purely serial scalar code.
- **AVX2** is typically 4–10× faster than Naive thanks to SIMD vectorisation and
  cache-friendly memory access (transposed B).
- **Vulkan** timing covers buffer upload + GPU dispatch + result download.
  For small matrices the PCIe transfer may dominate; for large matrices
  the GPU compute dominates and this method tends to be fastest.

---

## Changing the Matrix Size

Edit `test.cpp` line:
```cpp
static const uint32_t N = 8192;
```
Then recompile. Values must be divisible by 16 for the Vulkan shader's work-group
size to divide evenly (the shader clamps out-of-bounds threads otherwise).

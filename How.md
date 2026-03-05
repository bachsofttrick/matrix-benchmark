# How the Matrix Benchmark Works

This program benchmarks five different ways to multiply two large square matrices (NxN, default N=8192).
It measures wall-clock time for each method and prints the results.

---

## Table of Contents

1. [Data Setup](#1-data-setup)
2. [Naive Triple-Loop](#2-naive-triple-loop)
3. [AVX2 + FMA](#3-avx2--fma)
4. [pthread (parallel naive)](#4-pthread-parallel-naive)
5. [pthread + AVX2](#5-pthread--avx2)
6. [Vulkan Compute Shader](#6-vulkan-compute-shader)
7. [main() — Putting It Together](#7-main----putting-it-together)

---

## Background: What Is Matrix Multiplication?

Given two NxN matrices A and B, we want to compute C = A × B where each element is:

```
C[row][col] = sum over k of ( A[row][k] * B[k][col] )
```

For N=8192, that's 8192 × 8192 = ~67 million output elements,
each requiring 8192 multiply-add operations = ~1.1 trillion FLOPs total.

---

## 1. Data Setup

### `fill_random()`

```cpp
static void fill_random(std::vector<float>& v) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}
```

**Step-by-step:**

1. Create a Mersenne Twister random number generator seeded with `42` (fixed seed = reproducible results).
2. Create a distribution that produces floats uniformly between 0.0 and 1.0.
3. Walk through every element in the vector and assign a random float.

This is called twice in `main()` to fill matrices A and B before any benchmark runs.
Matrix C is not pre-filled — each benchmark function is responsible for initializing it.

---

## 2. Naive Triple-Loop

```cpp
static void matmul_naive(const float* A, const float* B, float* C, uint32_t n)
```

This is the textbook algorithm, nothing clever.

**Step-by-step:**

1. Zero out C entirely with `std::fill`.
2. Loop `i` from 0 to N-1 (rows of A / rows of C).
3. Loop `k` from 0 to N-1 (shared dimension).
   - Read `A[i][k]` once into a local variable `aik`.
4. Loop `j` from 0 to N-1 (columns of B / columns of C).
   - Add `aik * B[k][j]` into `C[i][j]`.

**Why the loop order `i, k, j` instead of `i, j, k`?**

- `i, j, k` (textbook order) reads `B[k][j]` in an inner loop — each step jumps N floats = 32 KB, causing cache misses.
- `i, k, j` reads `B[k][j]` sequentially (j steps by 1), which is cache-friendly.
- `aik` is hoisted outside the `j` loop so it is read from a register, not memory.

**Memory layout:** all matrices are stored row-major (flat 1D array), so `A[i][k]` = `A[i*n + k]`.

**Performance:** very slow for large N because the CPU can only do one multiply-add per cycle here.

---

## 3. AVX2 + FMA

```cpp
static void matmul_avx2(const float* A, const float* B, float* C, uint32_t n)
```

Uses Intel SIMD intrinsics to multiply 8 floats at once per instruction.

**Step-by-step:**

### Phase 1: Transpose B

```cpp
BT[j * n + i] = B[i * n + j];
```

Instead of reading column `col` of B (which is strided/cache-unfriendly), we first create `BT`
where `BT[row]` = original column `row` of B. Now reading a column is a sequential row access.

### Phase 2: Compute with SIMD

For each output element `C[i][j]`:

1. Point `rowA` to row `i` of A. Point `rowBT` to row `j` of BT (= column `j` of B).
2. Initialize an AVX register `vsum` to all zeros. This register holds **8 floats** simultaneously.
3. Loop `k` in steps of 8:
   - Load 8 floats from `rowA+k` into `va` (256-bit register).
   - Load 8 floats from `rowBT+k` into `vb` (256-bit register).
   - `vsum = vsum + va * vb` using a single FMA (Fused Multiply-Add) instruction — one CPU instruction that does 8 multiply-adds at once.
4. When fewer than 8 elements remain (tail), fall back to scalar multiply-add.
5. Collapse `vsum` (8 floats) into a single scalar with a horizontal sum:
   - Split the 256-bit register into two 128-bit halves.
   - Add them together.
   - Repeatedly apply `hadd` (horizontal add) to sum adjacent pairs.
   - Extract the final scalar.
6. Store the scalar result into `C[i][j]`.

**Why FMA?** Fused Multiply-Add (`a * b + c`) is done in one instruction instead of two, reducing rounding error and instruction count.

**Throughput gain:** 8 floats per FMA instruction vs 1 per scalar = up to 8× speedup (in practice, 4–6× after overhead).

---

## 4. pthread (Parallel Naive)

```cpp
static void matmul_pthread(const float* A, const float* B, float* C, int n)
```

Splits the row range across OS threads so multiple CPU cores work in parallel.

**Step-by-step:**

### In `matmul_pthread()`:

1. Zero out C.
2. Detect number of hardware threads via `std::thread::hardware_concurrency()`.
3. Divide N rows evenly across threads. If N is not divisible, the first `remainder` threads each get one extra row.
4. For each thread `t`:
   - Fill a `PthreadArgs` struct with pointers to A, B, C, and its assigned row range `[row_start, row_end)`.
   - Call `pthread_create()` to launch the thread, passing the args struct.
5. Call `pthread_join()` for each thread to wait until all are done.

### In `pthread_worker()` (runs in each thread):

Each thread independently computes the rows assigned to it using the same `i, k, j` loop order as the naive version.
No locking is needed because each thread writes to a completely separate slice of C (row ranges do not overlap).

**Performance:** nearly linear scaling with core count for this workload, since there is no shared mutable state.

---

## 5. pthread + AVX2

```cpp
static void matmul_pthread_avx2(const float* A, const float* B, float* C, int n)
```

Combines parallelism (pthread) with SIMD (AVX2+FMA). Best CPU performance.

**Step-by-step:**

1. **On the main thread:** transpose B into `BT` once. All worker threads will share this read-only `BT`.
   - Doing this once on the main thread avoids N threads each duplicating the work.
2. Divide rows and launch threads exactly as in the pthread version.
3. Each thread runs `pthread_avx2_worker()`, which is the AVX2 kernel (same as `matmul_avx2()`) applied only to its assigned rows.
4. Join all threads.

**Why transpose on the main thread?**

If each thread transposed its own slice, threads with overlapping `k` ranges would write to overlapping parts of BT, causing data races.
Transposing the full BT once before spawning avoids this.

---

## 6. Vulkan Compute Shader

This is the most complex method. It runs the multiplication on the GPU using the Vulkan graphics API.

### 6a. The Shader (`matmul.comp`)

```glsl
layout(local_size_x = 16, local_size_y = 16) in;
```

The GPU launches thousands of tiny programs (invocations) in parallel. They are grouped into **workgroups** of 16×16=256 threads.

Each invocation is assigned one output element `C[row][col]`:

```glsl
uint col = gl_GlobalInvocationID.x;
uint row = gl_GlobalInvocationID.y;
```

Each invocation independently computes its dot product:

```glsl
for (uint k = 0; k < pc.N; k++) {
    sum += A.data[row * pc.N + k] * B.data[k * pc.N + col];
}
C.data[row * pc.N + col] = sum;
```

`pc.N` is passed from the CPU via a **push constant** (a small fast-path for passing values to shaders).

### 6b. `VulkanMatMul` Constructor

The constructor creates all Vulkan objects that are reused across calls. In order:

| Step | What It Creates | Why |
|---|---|---|
| `vkCreateInstance` | Vulkan instance | Entry point to the Vulkan API |
| `vkEnumeratePhysicalDevices` | Physical device handle | Selects the GPU (index = `deviceNo`) |
| `vkCreateDevice` | Logical device | Software handle to the GPU for issuing commands |
| `vkGetDeviceQueue` | Command queue | Channel for submitting work to the GPU |
| `makeBuffer` × 3 (DEVICE_LOCAL) | bufA, bufB, bufC | Buffers in GPU VRAM for the shader to read/write |
| `makeBuffer` × 3 (HOST_VISIBLE) | stagBufA, stagBufB, stagBufC | Staging buffers in CPU-accessible RAM for upload/download |
| `vkCreateShaderModule` | Shader module | Loads the compiled SPIR-V shader binary |
| `vkCreateDescriptorSetLayout` | Descriptor set layout | Declares "this shader uses 3 storage buffers" |
| `vkCreatePipelineLayout` | Pipeline layout | Combines descriptor layout + push constants |
| `vkCreateComputePipelines` | Compute pipeline | Compiled shader + layout, ready to dispatch |
| `vkCreateDescriptorPool` / `vkAllocateDescriptorSets` | Descriptor set | Actual binding of bufA/bufB/bufC to shader slots |
| `vkUpdateDescriptorSets` | (updates descriptor set) | Tells the shader which buffers are A, B, C |
| `vkCreateCommandPool` / `vkAllocateCommandBuffers` | Command buffer | Recording buffer for GPU commands |
| `vkCreateFence` | Fence | CPU-side synchronization primitive to wait for GPU |

### 6c. `compute()` — The Timed Region

This is what gets benchmarked. It does 5 phases:

**Phase 1: CPU → Staging (memcpy)**

```
CPU RAM (A, B vectors)  →  stagBufA, stagBufB  (HOST_VISIBLE memory)
```

The CPU maps the staging buffer memory, memcpy's the data in, then unmaps.
This is a regular CPU memcpy — no GPU involved yet.

**Phase 2: Record GPU Commands**

The CPU records a sequence of commands into the command buffer (nothing executes yet):

1. `vkCmdCopyBuffer(stagBufA → bufA)` — DMA transfer A into VRAM
2. `vkCmdCopyBuffer(stagBufB → bufB)` — DMA transfer B into VRAM
3. `vkCmdPipelineBarrier` — wait for DMA to finish before the shader reads
4. `vkCmdBindPipeline` — select the compute shader
5. `vkCmdBindDescriptorSets` — connect bufA/bufB/bufC to the shader
6. `vkCmdPushConstants` — send N to the shader
7. `vkCmdDispatch(groups, groups, 1)` — launch `(N/16)²` workgroups
8. `vkCmdPipelineBarrier` — wait for shader to finish before reading back C
9. `vkCmdCopyBuffer(bufC → stagBufC)` — DMA transfer C out of VRAM

**Phase 3: Submit and Wait**

```cpp
vkQueueSubmit(queue_, 1, &si, fence_);
vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX);
```

The recorded commands are sent to the GPU. The CPU blocks at `vkWaitForFences` until the GPU signals the fence (all commands done).

**Phase 4: Staging → CPU (memcpy)**

```
stagBufC  →  C (output float* pointer)
```

The CPU maps the staging buffer and memcpy's the result out.

### 6d. Why Two Buffer Tiers?

Vulkan distinguishes between:

- `DEVICE_LOCAL` — physically in GPU VRAM. Fast for the GPU shader to access. CPU cannot directly read/write it.
- `HOST_VISIBLE` — accessible by both CPU and GPU via memory mapping. Slower for GPU shader access.

The staging pattern (`HOST_VISIBLE` → DMA → `DEVICE_LOCAL`) is used to maximize GPU shader performance:
the shader always reads from fast VRAM, never from slow host memory.

---

## 7. `main()` — Putting It Together

```
1. Allocate A, B, C as flat float vectors of size N*N
2. Fill A and B with random floats (same seed → same data every run)
3. Print matrix size and thread count

For each benchmark:
    record start time
    call the function
    record end time
    print elapsed milliseconds

Order: Naive → AVX2 → pthread → pthread+AVX2 → Vulkan
```

Each benchmark writes its result into the same `C` vector (overwriting the previous result). No correctness check is done — the benchmark only measures speed.

The Vulkan benchmark is wrapped in a try/catch because GPU initialization can fail (no Vulkan driver, wrong `deviceNo`, missing SPIR-V file, etc.).

> **Note:** At the time of writing, the Naive, AVX2, pthread, and pthread+AVX2 calls are commented out in `main()`. Only the Vulkan path runs. Uncomment the calls to run the CPU benchmarks.

---

## Summary Table

| Method | Parallelism | Key Idea | Bottleneck |
|---|---|---|---|
| Naive | 1 CPU thread | Simple loops, cache-friendly order | Single core speed |
| AVX2 | 1 CPU thread, 8-wide SIMD | FMA on 8 floats at once | Memory bandwidth |
| pthread | N CPU threads | Row partitioning, no locks | Memory bandwidth |
| pthread+AVX2 | N threads × 8-wide SIMD | Best of both above | Memory bandwidth |
| Vulkan | Thousands of GPU threads | Massively parallel GPU dispatch | PCIe transfer + memory access pattern |

<div align="center">
  <h1>PagedAttentionMetal 🚀</h1>
  <p><b>State-of-the-Art, Hardware-Accelerated Paged Attention for Apple Silicon.</b></p>
  
  [![Swift 6.0](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
  [![Metal 3](https://img.shields.io/badge/Metal-3-blue.svg)](https://developer.apple.com/metal/)
  [![macOS 14.0+](https://img.shields.io/badge/macOS-14.0+-success.svg)]()
  [![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0-blueviolet.svg)]()
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

<br/>

`PagedAttentionMetal` is a blazing-fast, bare-metal implementation of the memory-efficient [PagedAttention](https://arxiv.org/abs/2309.06180) algorithm (as popularized by **vLLM**). 

This library was built from the ground up natively for **Apple Silicon (M1/M2/M3/M4)**. It completely bypasses the CPU and executes long-context LLM sequence attention entirely on the GPU utilizing advanced **Metal 3** capabilities, achieving massive throughput for modern LLM inference.

---

## ✨ Core Features

- **🚀 Multi-Head & Grouped-Query Attention (MHA/MQA/GQA):** Full native support for standard attention, LLaMA-style GQA (e.g. 8Q heads / 2KV heads), and Multi-Query Attention. Dispatch is performed on a 3D GPU grid, executing all attention heads simultaneously in parallel.
- **⚡️ FP16 Mixed-Precision:** Cuts memory bandwidth in half. The engine natively reads 16-bit `Float16` Key-Value caches from VRAM, performs high-precision Softmax/dot products in 32-bit registers, and writes back 16-bit outputs, yielding near 2.0x latency speedups.
- **🧠 Two-Pass (Split-K) Scaling:** For exceptionally long sequences (e.g., 100k+ tokens), the library automatically switches from standard execution to a massive Map-Reduce split-pass execution layout to avoid threadgroup memory exhaustion.
- **✨ SIMD Vectorization:** Fully leverages native Apple ALU boundaries using `float4` vector execution and custom `simd_dot_product` helpers to minimize instruction overhead.
- **📦 JIT Compiled:** Safely JIT-compiles Metal shaders at runtime via `MTLDevice.makeLibrary(source:)`. This guarantees 100% bug-free behavior across all Swift Package Manager setups (Xcode, CLI, and CI/CD), sidestepping notorious SPM `.metallib` compilation bugs.
- **🔧 Runtime-Configurable Block Size:** No longer hardcoded to 16 — pass any block size (8, 16, 32, 64) at dispatch time without shader recompilation.
- **🎯 Causal Masking:** Built-in autoregressive masking for LLM generation (prevents tokens from attending to future positions).
- **⚙️ Dynamic head_dim Support:** Supports any head dimension (64, 128, 256, etc.) via dynamic threadgroup memory allocation.
- **🔄 Batch Processing:** `BatchKVCacheManager` handles multiple concurrent sequences with 2D block tables.
- **💾 GPU-Side KV Cache Writes:** Zero-copy `appendToCache()` kernel writes new K/V tokens directly to paged pool on GPU.
- **🎮 Optimized Decode Kernel:** Dedicated `paged_decode_single` kernel for single-token generation (the LLM hot path).

---

## 💻 Installation

`PagedAttentionMetal` is distributed exclusively as a **Swift Package**. 

### Xcode
1. Select **File** > **Add Package Dependencies...**
2. Paste the URL of this repository.
3. Add it to your app target.

### Swift Package Manager
Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-username/PagedAttentionMetal.git", from: "1.0.0")
],
targets: [
    .executableTarget(
        name: "YourApp",
        dependencies: ["PagedAttentionMetal"]
    )
]
```

---

## 🛠 Quick Start

Using `PagedAttentionMetal` requires two components: the **Memory Manager** (to handle virtual-to-physical block mapping) and the **Compute Engine** (to execute the Metal shaders). 

Here is how you execute a complete LLM generation step:

```swift
import PagedAttentionMetal
import Metal

// 1. Initialize the Hardware Engine
let device = MTLCreateSystemDefaultDevice()!
let engine = try PagedAttentionEngine()

// 2. Initialize the KV Cache Memory Manager
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// 3. Register a new sequence
let sequenceID = 1
try cacheManager.allocateSequence(id: sequenceID)

// 4. Process a 25-token prompt (prefill)
try cacheManager.appendTokens(toSequence: sequenceID, count: 25)

engine.prefill(
    q: queryBuffer,              // [seqLen, numHeads, headDim]
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
    seqLen: 25,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    causal: true,                // Autoregressive masking
    output: outputBuffer,
    dataType: .float16
)

// 5. Generate next token (decode)
try cacheManager.appendTokens(toSequence: sequenceID, count: 1)

engine.decode(
    q: nextTokenQuery,           // [1, numHeads, headDim]
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTables: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
    seqLengths: seqLengthsBuffer,
    batchSize: 1,
    maxNumBlocks: 64,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    output: decodeOutput,
    dataType: .float16
)

// 6. Append new K/V to cache (GPU-side)
engine.appendToCache(
    keys: newKeysBuffer,
    values: newValuesBuffer,
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
    tokenOffset: 25,
    numNewTokens: 1,
    numKVHeads: 2,
    headDim: 128,
    blockSize: 16,
    dataType: .float16
)

// 7. Free sequence when done
cacheManager.freeSequence(id: sequenceID)
```

---

## 🔬 Benchmarking & Verification

This library is rigorously tested against a mathematically pure CPU reference implementation. The test suite aggressively validates standard attention, GQA scaling, and F16 memory precision (verifying maximum rounding error deviations are within safe LLM boundaries).

To verify your specific Apple Silicon chip's capability, clone the repository and run:

```bash
swift test
```

To run the raw memory bandwidth benchmark comparing FP32 vs FP16 speeds, run the example app:

```bash
cd ExampleApp
swift run -c release
```

**Expected Example Benchmark Output:**
```text
=== Benchmark Configuration ===
Sequence Length: 4096
Head Dim:        128
Num Heads:       8
Tokens in Cache: 4096
FP32 KV Cache Size: 16 MB
FP16 KV Cache Size: 8 MB

Benchmarking FP32 (100 iterations)...
Benchmarking FP16 (100 iterations)...

=== Results ===
FP32 Latency: 897.685 ms
FP16 Latency: 476.083 ms
Speedup:      1.89x
```

---

## 📊 Performance Benchmarks

Measured on **Apple M4** (11 GB unified memory):

| Operation | Configuration | FP32 | FP16 | Speedup |
|-----------|--------------|------|------|---------|
| **Prefill** | 1024 tokens | 0.05ms | 0.03ms | **1.36x** |
| **Decode** | batch=8, ctx=1024 | 8.24ms | 6.60ms | **1.25x** |

**Throughput:**
- Prefill FP16: **30.7M tokens/sec**
- Decode FP16 (batch=8): **1,212 tokens/sec**

Run benchmarks yourself:
```bash
swift run QuickBench -c release  # Fast (30 seconds)
swift run Benchmarks -c release  # Comprehensive (2-3 minutes)
```

---

## 📚 Documentation

- **[Quick Start](#-quick-start)** - Get started in 5 minutes
- **[MLX Integration Guide](docs/MLX_INTEGRATION.md)** - Integrate with MLX-Swift for real LLM inference
- **[Advanced Usage Guide](docs/ADVANCED_USAGE.md)** - Batch processing, memory management, performance tuning
- **[Framework Integration](docs/FRAMEWORK_INTEGRATION.md)** - PyTorch, TensorFlow, ONNX, Hugging Face
- **[Architecture Deep-Dive](ARCHITECTURE.md)** - Kernel implementation, memory layout, optimization techniques
- **[Synthetic LLM Demo](Sources/MinimalLLM/main.swift)** - Working example of prefill→decode loop

---

## 📝 Requirements

- **Platform:** macOS 14.0+, iOS 16.0+, iPadOS 16.0+
- **Hardware:** Apple Silicon (M1 or newer) or A13 Bionic (or newer). *Legacy Intel Macs are strictly unsupported due to missing Metal 3 atomic float capabilities.*
- **Toolchain:** Swift 6.0+

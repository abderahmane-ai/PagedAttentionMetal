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
// Automatically compiles the Apple Silicon Mixed-Precision kernels.
let device = MTLCreateSystemDefaultDevice()!
let engine = try PagedAttentionEngine()

// 2. Initialize the KV Cache Memory Manager
// This acts as the GPU OS, pre-allocating a massive pool of VRAM.
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,             // Pre-allocate 1024 physical KV blocks
    blockSize: 16,               // 16 tokens per block
    headDim: 128,                // Dimension per attention head
    numKVHeads: 2,               // 2 KV Heads (GQA)
    dataType: .float16           // ⚡️ FP16 Mixed Precision for 2x Bandwidth!
)

// --- A New User Connects! ---

// 3. Register their Sequence
let sequenceID = 1
try cacheManager.allocateSequence(id: sequenceID)

// 4. They submit a 25-token prompt. 
// The Manager automatically reserves 2 physical blocks (32 token capacity) from the free list.
try cacheManager.appendTokens(toSequence: sequenceID, count: 25)

// 5. Dispatch the GPU Forward Pass
try engine.forward(
    q: queryBuffer,              // Your Q sequence [seqLen, numHeads, headDim]
    kPool: cacheManager.kPoolBuffer,  // Managed dynamically!
    vPool: cacheManager.vPoolBuffer,  // Managed dynamically!
    blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
    seqLen: 25,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    numBlocks: cacheManager.maxBlocks,
    blockSize: cacheManager.blockSize,
    output: outputBuffer,
    dataType: cacheManager.dataType
)

print("Attention calculated natively on Apple Silicon!")

// 6. When the user disconnects, recycle their memory instantly in O(1)
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

## 📝 Requirements

- **Platform:** macOS 14.0+, iOS 16.0+, iPadOS 16.0+
- **Hardware:** Apple Silicon (M1 or newer) or A13 Bionic (or newer). *Legacy Intel Macs are strictly unsupported due to missing Metal 3 atomic float capabilities.*
- **Toolchain:** Swift 6.0+

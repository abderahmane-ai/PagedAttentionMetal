<div align="center">
  <h1>PagedAttentionMetal 🚀</h1>
  <p><b>State-of-the-Art, Hardware-Accelerated Paged Attention for Apple Silicon.</b></p>
  
  [![Swift 6.0](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org)
  [![Metal 3](https://img.shields.io/badge/Metal-3-blue.svg)](https://developer.apple.com/metal/)
  [![macOS 13.0+](https://img.shields.io/badge/macOS-13.0+-success.svg)]()
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

<br/>

`PagedAttentionMetal` is a blazing-fast, bare-metal implementation of the memory-efficient [PagedAttention](https://arxiv.org/abs/2309.06180) algorithm (as popularized by **vLLM**). 

This library was built from the ground up natively for **Apple Silicon (M1/M2/M3/M4)**. It completely bypasses the CPU and executes long-context LLM sequence attention entirely on the GPU utilizing advanced **Metal 3** capabilities, achieving typical speedups of **> 5,500x** over perfect CPU references.

---

## ✨ Core Features

- **🚀 Extreme Performance:** Achieves ~3ms execution time for 1000+ token context windows over scattered KV block tables.
- **🧠 Two-Pass (Split-K) Scaling:** For exceptionally long sequences (e.g., 100k+ tokens), the library automatically switches from standard execution to a massive Map-Reduce split-pass execution layout to avoid threadgroup memory exhaustion.
- **⚡️ SIMD Vectorization:** Fully leverages native Apple ALU boundaries using `simd_float4` vector execution, quartering memory boundary loads.
- **🛡 Hardware Atomics:** Built using Metal 3's native `atomic_fetch_add_explicit` for zero-stall gradient and backward accumulation. No more slow CAS spinlocks.
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
    .package(url: "https://github.com/your-username/PagedAttentionMetal.git", branch: "main")
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

The library exposes a clean, high-level Swift wrapper (`PagedAttentionEngine`). You do not need to manually configure Metal devices, manage compute pipelines, or calculate threadgroup grids—the engine dynamically handles the math based on your input tensors.

```swift
import PagedAttentionMetal
import Metal

// 1. Initialize the Engine
// This automatically discovers your Apple Silicon GPU and JIT-compiles the V2 kernels.
let engine = try PagedAttentionEngine()

// Set your context window threshold for Split-K dispatch (Default is 1024)
engine.splitThreshold = 1024 

// 2. Dispatch the GPU Forward Pass
try engine.forward(
    q: queryBuffer,              // Your Q sequence
    kPool: keyCachePool,         // Pre-allocated Key Cache
    vPool: valueCachePool,       // Pre-allocated Value Cache
    blockTable: blockMapBuffer,  // Paged memory mapping [Int32]
    seqLen: 1024,                // Sequence Length
    headDim: 64,                 // Dimension per attention head
    numBlocks: 64,               // Total KV blocks in your memory pool
    blockSize: 16,               // Tokens mapped per physical block
    output: outputBuffer         // Destination buffer for the calculated attention
)

print("Attention calculated natively on Apple Silicon!")
```

---

## 🔬 Running the Benchmarks

This library is thoroughly tested against a mathematically pure CPU reference implementation. The test suite aggressively validates standard attention, tiled attention, vector math, backward passes, and the V2 Split-K architecture.

To verify your specific Apple Silicon chip's capability, clone the repository and run:

```bash
swift test
```

**Expected Sample Output (M-Series Chip):**
```text
=== Paged Attention V2 (Longer Sequence Test) ===
Running CPU reference...
CPU Time: 17811.67 ms (17.8 seconds)

Running GPU benchmark...
PagedAttention V2 (GPU): 3.135ms ± 0.470ms
Speedup vs CPU: 5682.1x
Max difference from CPU: 1.4528632e-07
✓ Paged attention V2 (long seq) verified!
```

---

## 📝 Requirements

- **Platform:** macOS 13.0+, iOS 16.0+, iPadOS 16.0+
- **Hardware:** Apple Silicon (M1 or newer) or A13 Bionic (or newer). *Legacy Intel Macs are strictly unsupported due to missing Metal 3 atomic float capabilities.*
- **Toolchain:** Swift 6.0+

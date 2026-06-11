# PagedAttentionMetal

High-performance PagedAttention implementation for Apple Silicon GPUs using Metal.

## Features

- **PagedAttention** — Efficient non-contiguous KV cache management for LLM inference
- **Apple Silicon Native** — Optimized for M1-M5 series GPUs using Metal 3.1+
- **Multi-Data Type** — FP32, FP16, and FP8 KV cache quantization
- **Prefix Caching** — Content-addressable block sharing with copy-on-write
- **Sliding Window** — Windowed attention for efficient long-context inference
- **Chunked Prefill** — Split long prefill into chunks to interleave with decode
- **Continuous Batching** — Dynamic scheduling for optimal GPU utilization
- **GQA/MQA Support** — Grouped-query and multi-query attention
- **Zero-Copy** — Shared memory buffers for CPU-GPU coherence

## Requirements

- macOS 14+ / iOS 17+
- Apple Silicon (M1 or later)
- Swift 6.0+

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/abderahmane-ai/PagedAttentionMetal", from: "3.0.0")
]
```

## Usage

### Basic Usage

```swift
import PagedAttentionMetal
import Metal

// Setup
let device = MTLCreateSystemDefaultDevice()!
let engine = try PagedAttentionEngine(device: device)
let cache = try KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 8
)

// Allocate sequence
try cache.allocateSequence(id: 0)
try cache.appendTokens(toSequence: 0, count: 128)

// Run attention
let qBuffer = device.makeBuffer(length: ...)!
let outputBuffer = device.makeBuffer(length: ...)!
try engine.prefill(q: qBuffer, ..., output: outputBuffer)
try engine.decode(q: qBuffer, ..., output: outputBuffer)

// Cleanup
cache.freeSequence(id: 0)
```

## Architecture

- `PagedAttentionEngine` — Metal compute pipeline management and kernel dispatch
- `KVCacheManager` — Paged KV cache allocation with prefix caching
- `ContinuousBatchingScheduler` — Sequence lifecycle management
- `PagedAttentionInference` — End-to-end inference loop
- `Scheduler.swift` — Request scheduling and preemption

## Benchmarks

```bash
swift run -c release Benchmarks all
```

## License

MIT

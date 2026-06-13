# Quickstart Guide

Get up and running with PagedAttentionMetal in minutes.

## Prerequisites

- macOS 14.0+ (Sonoma) or iOS 17.0+
- Xcode 16.0+ with Swift 6.0
- Apple Silicon Mac (M1, M2, M3, M4, M5 series)

## Installation

### Swift Package Manager

Add PagedAttentionMetal to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/PagedAttentionMetal.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "PagedAttentionMetal", package: "PagedAttentionMetal"),
    ]
)
```

## Basic Usage

### 1. Initialize the Engine

```swift
import PagedAttentionMetal

// Create the engine (compiles Metal kernels on launch)
let engine = try PagedAttentionEngine()
```

### 2. Create a KV Cache Manager

```swift
let layerSpec = PagedLayerSpec(
    headDim: 128,
    numHeads: 32,
    numKVHeads: 8,
    blockSize: 16,
    dataType: .float16
)

let cacheManager = try KVCacheManager(
    device: engine.device,
    maxBlocks: 1024,
    blockSize: layerSpec.blockSize,
    headDim: layerSpec.headDim,
    numKVHeads: layerSpec.numKVHeads,
    dataType: layerSpec.dataType
)
```

### 3. Run Prefill

```swift
let prefillRequest = PagedAttentionPrefillRequest(
    q: queryBuffer,            // [1, seqLen, numHeads, headDim]
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTable: blockTableBuffer,
    output: outputBuffer,      // [1, seqLen, numHeads, headDim]
    seqLen: seqLen,
    layer: layerSpec,
    causal: true
)

try engine.prefill(prefillRequest)
```

### 4. Run Decode

```swift
let decodeRequest = PagedAttentionDecodeRequest(
    q: queryBuffer,            // [batchSize, 1, numHeads, headDim]
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTables: blockTablesBuffer,
    seqLengths: seqLengthsBuffer,
    output: outputBuffer,      // [batchSize, 1, numHeads, headDim]
    batchSize: batchSize,
    maxNumBlocks: maxNumBlocks,
    layer: layerSpec
)

try engine.decode(decodeRequest)
```

## Next Steps

- <doc:ModelDeployment> - Deploy models in production
- <doc:Benchmarking> - Measure and optimize performance
- <doc:API-Reference> - Complete API documentation
# PagedAttentionMetal Documentation

Complete documentation for integrating and using PagedAttentionMetal in your projects.

## 📖 Table of Contents

### Getting Started
- **[README](../README.md)** - Overview, installation, and quick start
- **[Quick Start Example](../README.md#-quick-start)** - 5-minute integration guide
- **[Synthetic LLM Demo](../Sources/MinimalLLM/main.swift)** - Working prefill→decode example

### Integration Guides
- **[MLX Integration](MLX_INTEGRATION.md)** - Integrate with MLX-Swift for real LLM inference
  - Tensor conversion bridge
  - Custom attention layers
  - Model integration patterns
  - Performance optimization
  - Troubleshooting

- **[Framework Integration](FRAMEWORK_INTEGRATION.md)** - Integrate with other ML frameworks
  - PyTorch via Swift-Python bridge
  - TensorFlow Lite custom delegates
  - ONNX Runtime execution providers
  - Custom inference engines
  - Hugging Face Transformers

### Advanced Topics
- **[Advanced Usage](ADVANCED_USAGE.md)** - Production patterns and optimization
  - Batch processing strategies
  - Memory management and monitoring
  - Performance tuning (block size, FP16 vs FP32)
  - Long context handling (100k+ tokens)
  - Multi-sequence patterns
  - Error handling and retry logic
  - Profiling and debugging

- **[Architecture Deep-Dive](../ARCHITECTURE.md)** - Implementation details
  - Virtual memory paradigm for LLMs
  - Kernel evolution (single-pass to split-K)
  - Hardware acceleration and vectorization
  - Memory layout and data structures
  - Numerical stability and precision
  - Debugging and validation
  - Performance benchmarks
  - Future optimization opportunities

### API Reference

#### Core Classes

**PagedAttentionEngine**
```swift
class PagedAttentionEngine {
    var splitThreshold: Int  // Default: 1024
    
    func prefill(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        causal: Bool,
        output: MTLBuffer,
        dataType: PagedAttentionDataType
    )
    
    func decode(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTables: MTLBuffer,
        seqLengths: MTLBuffer,
        batchSize: Int,
        maxNumBlocks: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        output: MTLBuffer,
        dataType: PagedAttentionDataType
    )
    
    func appendToCache(
        keys: MTLBuffer,
        values: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        numKVHeads: Int,
        headDim: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType
    )
}
```

**KVCacheManager**
```swift
class KVCacheManager {
    let device: MTLDevice
    let blockSize: Int
    let headDim: Int
    let numKVHeads: Int
    let dataType: PagedAttentionDataType
    let maxBlocks: Int
    
    let kPoolBuffer: MTLBuffer
    let vPoolBuffer: MTLBuffer
    
    func allocateSequence(id: Int) throws
    func appendTokens(toSequence id: Int, count: Int) throws
    func freeSequence(id: Int)
    func getBlockTable(forSequence id: Int) throws -> [Int32]
    func getBlockTableBuffer(forSequence id: Int) throws -> MTLBuffer
    func getSequence(id: Int) throws -> LogicalSequence
}
```

**BatchKVCacheManager**
```swift
class BatchKVCacheManager {
    // Same as KVCacheManager, plus:
    
    func getBlockTablesBuffer() throws -> MTLBuffer
    func getSeqLengthsBuffer() throws -> MTLBuffer
}
```

#### Data Types

**PagedAttentionDataType**
```swift
enum PagedAttentionDataType {
    case float32  // Higher precision, 2x memory
    case float16  // Recommended: 2x faster, minimal accuracy loss
}
```

**KVCacheError**
```swift
enum KVCacheError: Error {
    case sequenceNotFound
    case sequenceAlreadyExists
    case outOfMemory
}
```

### Examples

#### Basic Usage
```swift
import PagedAttentionMetal

let device = MTLCreateSystemDefaultDevice()!
let engine = try PagedAttentionEngine()
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Allocate sequence
try cacheManager.allocateSequence(id: 1)

// Prefill
try cacheManager.appendTokens(toSequence: 1, count: 25)
engine.prefill(
    q: queryBuffer,
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
    seqLen: 25,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    causal: true,
    output: outputBuffer,
    dataType: .float16
)

// Decode
try cacheManager.appendTokens(toSequence: 1, count: 1)
engine.decode(
    q: nextTokenQuery,
    kPool: cacheManager.kPoolBuffer,
    vPool: cacheManager.vPoolBuffer,
    blockTables: try cacheManager.getBlockTableBuffer(forSequence: 1),
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

// Cleanup
cacheManager.freeSequence(id: 1)
```

#### Batch Processing
```swift
let batchManager = BatchKVCacheManager(
    device: device,
    maxBlocks: 2048,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Allocate multiple sequences
for id in 1...8 {
    try batchManager.allocateSequence(id: id)
    try batchManager.appendTokens(toSequence: id, count: 1)
}

// Batch decode
engine.decode(
    q: batchQueriesBuffer,
    kPool: batchManager.kPoolBuffer,
    vPool: batchManager.vPoolBuffer,
    blockTables: try batchManager.getBlockTablesBuffer(),
    seqLengths: try batchManager.getSeqLengthsBuffer(),
    batchSize: 8,
    maxNumBlocks: 64,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    output: batchOutputBuffer,
    dataType: .float16
)
```

### Performance Guidelines

#### Memory Sizing
```swift
// Calculate memory requirements
let bytesPerToken = numKVHeads * headDim * (dataType == .float16 ? 2 : 4)
let bytesPerBlock = blockSize * bytesPerToken
let totalMemory = maxBlocks * bytesPerBlock * 2  // K + V

// Example: 1024 blocks, 16 tokens/block, 2 KV heads, 128 dim, FP16
// = 1024 * 16 * 2 * 128 * 2 = 8 MB per cache (16 MB total)
```

#### Block Size Selection
- **8 tokens/block:** Best memory utilization, more overhead
- **16 tokens/block:** Recommended default, good balance
- **32 tokens/block:** Better for long contexts, potential waste
- **64 tokens/block:** Minimal overhead, high waste for short sequences

#### FP16 vs FP32
- **FP16:** 2x faster, 2x less memory, ~0.01% accuracy loss (recommended)
- **FP32:** Higher precision, 2x slower, 2x more memory

### Troubleshooting

#### Common Issues

**Out of Memory**
```swift
// Reduce maxBlocks or increase blockSize
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 512,  // Reduced from 1024
    blockSize: 32,   // Increased from 16
    ...
)
```

**Slow Prefill**
```swift
// Increase split threshold for shorter sequences
engine.splitThreshold = 2048  // Default: 1024
```

**Numerical Differences**
```swift
// Use FP32 for higher precision
let cacheManager = KVCacheManager(
    device: device,
    dataType: .float32,  // Instead of .float16
    ...
)
```

### Testing

Run the test suite:
```bash
swift test
```

Run the synthetic LLM demo:
```bash
swift run MinimalLLM
```

### Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### License

MIT License - see [LICENSE](../LICENSE) for details.

---

## Quick Links

- [GitHub Repository](https://github.com/yourusername/PagedAttentionMetal)
- [Report Issues](https://github.com/yourusername/PagedAttentionMetal/issues)
- [Discussions](https://github.com/yourusername/PagedAttentionMetal/discussions)

# MLX-Swift Integration Guide

This guide shows how to integrate PagedAttentionMetal with [MLX-Swift](https://github.com/ml-explore/mlx-swift) to accelerate LLM inference on Apple Silicon.

## Overview

PagedAttentionMetal handles the memory-intensive attention computation on GPU, while MLX manages model weights, tokenization, and other operations. The integration requires:

1. **Tensor conversion** between MLX arrays and Metal buffers
2. **Attention layer replacement** in your MLX model
3. **KV cache management** using PagedAttention's memory system

## Installation

Add both dependencies to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/PagedAttentionMetal.git", from: "1.0.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
]
```

## Tensor Conversion Bridge

Create a bridge to convert between MLX arrays and Metal buffers:

```swift
import MLX
import Metal
import PagedAttentionMetal

enum MLXBridge {
    /// Convert MLX array to Metal buffer (copies data)
    static func mlxToMetal(_ array: MLXArray, device: MTLDevice) -> MTLBuffer? {
        let data = array.asData(Array.self)
        return device.makeBuffer(
            bytes: data,
            length: data.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
    }
    
    /// Convert Metal buffer to MLX array
    static func metalToMLX(_ buffer: MTLBuffer, shape: [Int], dtype: DType = .float32) -> MLXArray {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        let count = shape.reduce(1, *)
        let data = Array(UnsafeBufferPointer(start: ptr, count: count))
        return MLXArray(data, shape)
    }
    
    /// Convert MLX array to FP16 Metal buffer
    static func mlxToMetalFloat16(_ array: MLXArray, device: MTLDevice) -> MTLBuffer? {
        let fp32Data = array.asData(Array.self)
        let fp16Data = fp32Data.map { Float16($0) }
        return device.makeBuffer(
            bytes: fp16Data,
            length: fp16Data.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )
    }
    
    /// Convert FP16 Metal buffer to MLX array
    static func metalFloat16ToMLX(_ buffer: MTLBuffer, shape: [Int]) -> MLXArray {
        let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
        let count = shape.reduce(1, *)
        let fp16Data = Array(UnsafeBufferPointer(start: ptr, count: count))
        let fp32Data = fp16Data.map { Float($0) }
        return MLXArray(fp32Data, shape)
    }
}
```

## Custom Attention Layer

Replace MLX's standard attention with PagedAttention:

```swift
import MLXNN

class PagedAttentionLayer: Module {
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let device: MTLDevice
    
    init(numHeads: Int, numKVHeads: Int, headDim: Int, maxBlocks: Int, blockSize: Int) throws {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.device = MTLCreateSystemDefaultDevice()!
        
        self.engine = try PagedAttentionEngine()
        self.cacheManager = KVCacheManager(
            device: device,
            maxBlocks: maxBlocks,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: .float16
        )
    }
    
    func callAsFunction(
        _ q: MLXArray,      // [seqLen, numHeads, headDim]
        _ k: MLXArray,      // [seqLen, numKVHeads, headDim]
        _ v: MLXArray,      // [seqLen, numKVHeads, headDim]
        sequenceID: Int,
        isPrefill: Bool
    ) throws -> MLXArray {
        let seqLen = q.shape[0]
        
        // Convert Q to Metal
        guard let qBuffer = MLXBridge.mlxToMetal(q, device: device) else {
            throw PagedAttentionError.bufferCreationFailed
        }
        
        // Append K/V to cache
        guard let kBuffer = MLXBridge.mlxToMetalFloat16(k, device: device),
              let vBuffer = MLXBridge.mlxToMetalFloat16(v, device: device) else {
            throw PagedAttentionError.bufferCreationFailed
        }
        
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        let blockTable = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)
        
        engine.appendToCache(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: blockTable,
            tokenOffset: cacheManager.getSequenceLength(sequenceID) - seqLen,
            numNewTokens: seqLen,
            numKVHeads: numKVHeads,
            headDim: headDim,
            blockSize: cacheManager.blockSize,
            dataType: .float16
        )
        
        // Run attention
        let outputBuffer = device.makeBuffer(
            length: seqLen * numHeads * headDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        if isPrefill {
            engine.prefill(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: blockTable,
                seqLen: seqLen,
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: cacheManager.blockSize,
                causal: true,
                output: outputBuffer,
                dataType: .float16
            )
        } else {
            let seqLengthsBuffer = device.makeBuffer(
                bytes: [Int32(cacheManager.getSequenceLength(sequenceID))],
                length: MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )!
            
            engine.decode(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTables: blockTable,
                seqLengths: seqLengthsBuffer,
                batchSize: 1,
                maxNumBlocks: cacheManager.getNumBlocks(forSequence: sequenceID),
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: cacheManager.blockSize,
                output: outputBuffer,
                dataType: .float16
            )
        }
        
        // Convert output back to MLX
        return MLXBridge.metalToMLX(outputBuffer, shape: [seqLen, numHeads, headDim])
    }
}
```

## Model Integration

Replace attention in your MLX model:

```swift
import MLXLLM

class OptimizedLLaMAModel: Module {
    let embeddings: Embedding
    let layers: [TransformerBlock]
    let norm: RMSNorm
    let output: Linear
    
    // Replace standard attention with PagedAttention
    init(config: LLaMAConfiguration) throws {
        self.embeddings = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        
        self.layers = try (0..<config.numLayers).map { _ in
            TransformerBlock(
                config: config,
                attentionLayer: try PagedAttentionLayer(
                    numHeads: config.attentionHeads,
                    numKVHeads: config.kvHeads,
                    headDim: config.hiddenSize / config.attentionHeads,
                    maxBlocks: 1024,
                    blockSize: 16
                )
            )
        }
        
        self.norm = RMSNorm(dimensions: config.hiddenSize)
        self.output = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
}
```

## Complete Generation Example

```swift
import MLX
import MLXLLM
import PagedAttentionMetal

func generateText(prompt: String, maxTokens: Int = 100) throws -> String {
    // Load model and tokenizer
    let modelConfig = LLaMAConfiguration.smolLM_1_7B
    let model = try OptimizedLLaMAModel(config: modelConfig)
    let tokenizer = try AutoTokenizer.from(pretrained: "HuggingFaceTB/SmolLM-1.7B")
    
    // Tokenize prompt
    let inputIDs = tokenizer.encode(text: prompt)
    let sequenceID = 1
    
    // Allocate sequence in cache
    for layer in model.layers {
        if let pagedLayer = layer.attention as? PagedAttentionLayer {
            try pagedLayer.cacheManager.allocateSequence(id: sequenceID)
        }
    }
    
    var tokens = inputIDs
    var generatedText = prompt
    
    // Prefill phase
    var logits = model(MLXArray(inputIDs))
    var nextToken = logits[logits.shape[0] - 1].argMax().item(Int.self)
    tokens.append(nextToken)
    
    // Decode phase
    for _ in 0..<maxTokens {
        logits = model(MLXArray([nextToken]))
        nextToken = logits[0].argMax().item(Int.self)
        
        if nextToken == tokenizer.eosTokenID {
            break
        }
        
        tokens.append(nextToken)
        generatedText += tokenizer.decode(tokens: [nextToken])
    }
    
    // Cleanup
    for layer in model.layers {
        if let pagedLayer = layer.attention as? PagedAttentionLayer {
            pagedLayer.cacheManager.freeSequence(id: sequenceID)
        }
    }
    
    return generatedText
}

// Usage
let output = try generateText(prompt: "Once upon a time", maxTokens: 50)
print(output)
```

## Performance Optimization

### 1. Use FP16 for KV Cache

```swift
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16  // 2x memory reduction
)
```

### 2. Tune Block Size

```swift
// Smaller blocks = better memory utilization, more overhead
// Larger blocks = less overhead, potential waste
// Recommended: 16 for most models, 32 for long contexts
let blockSize = 16
```

### 3. Batch Multiple Sequences

```swift
let batchManager = BatchKVCacheManager(
    device: device,
    maxBlocks: 2048,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Process multiple prompts simultaneously
for (idx, prompt) in prompts.enumerated() {
    try batchManager.allocateSequence(id: idx)
}

engine.decode(
    q: batchQueriesBuffer,
    kPool: batchManager.kPoolBuffer,
    vPool: batchManager.vPoolBuffer,
    blockTables: batchManager.blockTablesBuffer,
    seqLengths: seqLengthsBuffer,
    batchSize: prompts.count,
    maxNumBlocks: 64,
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    output: outputBuffer,
    dataType: .float16
)
```

### 4. Minimize Data Copies

```swift
// Bad: Multiple conversions
let mlxArray = someComputation()
let buffer = MLXBridge.mlxToMetal(mlxArray, device: device)

// Good: Keep data in Metal as long as possible
// Only convert at model boundaries
```

## Memory Management

### Sequence Lifecycle

```swift
// 1. Allocate before generation
try cacheManager.allocateSequence(id: sequenceID)

// 2. Append tokens as you generate
try cacheManager.appendTokens(toSequence: sequenceID, count: 1)

// 3. Free when done
cacheManager.freeSequence(id: sequenceID)
```

### Monitor Memory Usage

```swift
let stats = cacheManager.getMemoryStats()
print("Used blocks: \(stats.usedBlocks)/\(stats.totalBlocks)")
print("Fragmentation: \(stats.fragmentationRatio)")
print("Memory: \(stats.usedMemoryMB) MB / \(stats.totalMemoryMB) MB")
```

## Troubleshooting

### Issue: Numerical Differences

**Cause:** FP16 precision loss in KV cache  
**Solution:** Use FP32 for cache if accuracy is critical:

```swift
let cacheManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float32  // Higher precision
)
```

### Issue: Out of Memory

**Cause:** Too many allocated blocks  
**Solution:** Reduce `maxBlocks` or increase `blockSize`:

```swift
// Option 1: Fewer blocks
let cacheManager = KVCacheManager(maxBlocks: 512, blockSize: 16, ...)

// Option 2: Larger blocks (less waste)
let cacheManager = KVCacheManager(maxBlocks: 1024, blockSize: 32, ...)
```

### Issue: Slow Prefill

**Cause:** Very long sequences trigger split-K algorithm  
**Solution:** Tune split threshold:

```swift
engine.splitThreshold = 2048  // Default: 1024
```

## Next Steps

- See [ADVANCED_USAGE.md](ADVANCED_USAGE.md) for batch processing patterns
- See [ARCHITECTURE.md](ARCHITECTURE.md) for kernel implementation details
- See [Examples/SyntheticLLM](../Examples/SyntheticLLM) for a complete working demo

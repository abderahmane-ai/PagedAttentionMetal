# Advanced Usage Guide

This guide covers advanced patterns, optimizations, and best practices for PagedAttentionMetal.

## Table of Contents

1. [Batch Processing](#batch-processing)
2. [Memory Management](#memory-management)
3. [Performance Tuning](#performance-tuning)
4. [FP16 vs FP32](#fp16-vs-fp32)
5. [Long Context Handling](#long-context-handling)
6. [Multi-Sequence Patterns](#multi-sequence-patterns)
7. [Error Handling](#error-handling)
8. [Profiling & Debugging](#profiling--debugging)

---

## Batch Processing

### Basic Batch Decode

Process multiple sequences simultaneously for maximum GPU utilization:

```swift
import PagedAttentionMetal

let batchManager = BatchKVCacheManager(
    device: device,
    maxBlocks: 2048,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Allocate multiple sequences
let sequenceIDs = [1, 2, 3, 4]
for id in sequenceIDs {
    try batchManager.allocateSequence(id: id)
}

// Process prefill for each sequence
for (idx, id) in sequenceIDs.enumerated() {
    let promptLen = promptLengths[idx]
    try batchManager.appendTokens(toSequence: id, count: promptLen)
    
    // Run prefill for this sequence...
}

// Batch decode: generate next token for all sequences at once
let batchSize = sequenceIDs.count
let seqLengths = try sequenceIDs.map { id in
    Int32(try batchManager.getSequence(id: id).sequenceLength)
}

guard let seqLenBuffer = device.makeBuffer(
    bytes: seqLengths,
    length: seqLengths.count * MemoryLayout<Int32>.stride,
    options: .storageModeShared
) else { fatalError("Buffer creation failed") }

// Concatenate all queries into single buffer
let batchQBuffer = concatenateQueries(sequenceIDs)  // [batchSize, numHeads, headDim]

engine.decode(
    q: batchQBuffer,
    kPool: batchManager.kPoolBuffer,
    vPool: batchManager.vPoolBuffer,
    blockTables: try batchManager.getBlockTablesBuffer(),
    seqLengths: seqLenBuffer,
    batchSize: batchSize,
    maxNumBlocks: 64,  // Max blocks across all sequences
    headDim: 128,
    numHeads: 8,
    numKVHeads: 2,
    blockSize: 16,
    output: batchOutputBuffer,
    dataType: .float16
)
```

### Dynamic Batching

Add/remove sequences from batch dynamically:

```swift
class DynamicBatchScheduler {
    let batchManager: BatchKVCacheManager
    var activeSequences: Set<Int> = []
    let maxBatchSize: Int
    
    init(batchManager: BatchKVCacheManager, maxBatchSize: Int = 32) {
        self.batchManager = batchManager
        self.maxBatchSize = maxBatchSize
    }
    
    func addSequence(id: Int) throws {
        guard activeSequences.count < maxBatchSize else {
            throw SchedulerError.batchFull
        }
        try batchManager.allocateSequence(id: id)
        activeSequences.insert(id)
    }
    
    func removeSequence(id: Int) {
        batchManager.freeSequence(id: id)
        activeSequences.remove(id)
    }
    
    func step() throws {
        // Decode all active sequences
        let ids = Array(activeSequences)
        let seqLengths = try ids.map { id in
            Int32(try batchManager.getSequence(id: id).sequenceLength)
        }
        
        // Run batch decode...
        
        // Remove finished sequences
        for id in ids where isFinished(id) {
            removeSequence(id: id)
        }
    }
}
```

---

## Memory Management

### Monitoring Memory Usage

Track memory consumption and fragmentation:

```swift
extension KVCacheManager {
    struct MemoryStats {
        let totalBlocks: Int
        let usedBlocks: Int
        let freeBlocks: Int
        let activeSequences: Int
        let fragmentationRatio: Float
        let totalMemoryMB: Float
        let usedMemoryMB: Float
    }
    
    func getMemoryStats() -> MemoryStats {
        let used = maxBlocks - freeBlocks.count
        let bytesPerBlock = blockSize * numKVHeads * headDim * 
            (dataType == .float16 ? 2 : 4)
        let totalBytes = maxBlocks * bytesPerBlock * 2  // K + V
        let usedBytes = used * bytesPerBlock * 2
        
        // Fragmentation: ratio of partially filled blocks
        var partiallyFilled = 0
        for seq in sequences.values {
            let lastBlockTokens = seq.sequenceLength % blockSize
            if lastBlockTokens > 0 && lastBlockTokens < blockSize {
                partiallyFilled += 1
            }
        }
        let fragmentation = Float(partiallyFilled) / Float(max(sequences.count, 1))
        
        return MemoryStats(
            totalBlocks: maxBlocks,
            usedBlocks: used,
            freeBlocks: freeBlocks.count,
            activeSequences: sequences.count,
            fragmentationRatio: fragmentation,
            totalMemoryMB: Float(totalBytes) / 1_048_576,
            usedMemoryMB: Float(usedBytes) / 1_048_576
        )
    }
}

// Usage
let stats = cacheManager.getMemoryStats()
print("Memory: \(stats.usedMemoryMB)/\(stats.totalMemoryMB) MB")
print("Blocks: \(stats.usedBlocks)/\(stats.totalBlocks)")
print("Fragmentation: \(Int(stats.fragmentationRatio * 100))%")
```

### Preemptive Eviction

Evict sequences when memory is low:

```swift
class EvictionPolicy {
    enum Strategy {
        case lru  // Least Recently Used
        case shortestFirst
        case longestFirst
    }
    
    func selectVictim(
        sequences: [Int: LogicalSequence],
        strategy: Strategy
    ) -> Int? {
        guard !sequences.isEmpty else { return nil }
        
        switch strategy {
        case .lru:
            // Track access times separately
            return lruTracker.leastRecentlyUsed()
            
        case .shortestFirst:
            return sequences.min(by: { $0.value.sequenceLength < $1.value.sequenceLength })?.key
            
        case .longestFirst:
            return sequences.max(by: { $0.value.sequenceLength < $1.value.sequenceLength })?.key
        }
    }
}

// Usage
if cacheManager.freeBlocks.count < 10 {
    if let victimID = evictionPolicy.selectVictim(
        sequences: cacheManager.sequences,
        strategy: .lru
    ) {
        cacheManager.freeSequence(id: victimID)
        print("Evicted sequence \(victimID)")
    }
}
```

---

## Performance Tuning

### Block Size Selection

Choose block size based on your workload:

```swift
// Small blocks (8): Better memory utilization, more overhead
// - Use for: Many short sequences, limited VRAM
let smallBlockManager = KVCacheManager(
    device: device,
    maxBlocks: 2048,
    blockSize: 8,  // Less waste per sequence
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Medium blocks (16): Balanced (recommended default)
// - Use for: General purpose, mixed workloads
let mediumBlockManager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,  // Good balance
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)

// Large blocks (32-64): Less overhead, potential waste
// - Use for: Long contexts, few concurrent sequences
let largeBlockManager = KVCacheManager(
    device: device,
    maxBlocks: 512,
    blockSize: 32,  // Fewer block table lookups
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16
)
```

### Split-K Threshold Tuning

Adjust when to use split-pass algorithm:

```swift
let engine = try PagedAttentionEngine()

// Default: 1024 tokens
// Lower = more aggressive splitting (better for very long contexts)
// Higher = fewer splits (better for short-medium contexts)

engine.splitThreshold = 2048  // For 4k-8k context models
engine.splitThreshold = 512   // For 100k+ context models
```

### Minimize Data Transfers

Reduce CPU↔GPU copies:

```swift
// ❌ Bad: Multiple conversions
for token in tokens {
    let embedding = embed(token)  // CPU
    let buffer = toMetal(embedding)  // Copy to GPU
    let output = runAttention(buffer)  // GPU
    let result = toCPU(output)  // Copy to CPU
}

// ✅ Good: Batch and keep on GPU
let embeddings = embedBatch(tokens)  // CPU
let buffer = toMetal(embeddings)  // Single copy to GPU
let outputs = runAttentionBatch(buffer)  // GPU
let results = toCPU(outputs)  // Single copy to CPU
```

---

## FP16 vs FP32

### When to Use FP16

**Advantages:**
- 2x memory reduction
- ~1.9x faster on Apple Silicon
- Sufficient precision for most LLMs

**Use FP16 when:**
- Memory is constrained
- Speed is critical
- Model is robust to quantization (most modern LLMs)

```swift
let fp16Manager = KVCacheManager(
    device: device,
    maxBlocks: 1024,
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float16  // Recommended
)
```

### When to Use FP32

**Advantages:**
- Higher numerical precision
- No quantization error

**Use FP32 when:**
- Debugging numerical issues
- Model is sensitive to precision
- Memory is not a constraint

```swift
let fp32Manager = KVCacheManager(
    device: device,
    maxBlocks: 512,  // Half the capacity
    blockSize: 16,
    headDim: 128,
    numKVHeads: 2,
    dataType: .float32
)
```

### Precision Comparison

```swift
func comparePrecision() {
    let fp16Manager = KVCacheManager(..., dataType: .float16)
    let fp32Manager = KVCacheManager(..., dataType: .float32)
    
    // Run same input through both
    let output16 = runWithCache(fp16Manager)
    let output32 = runWithCache(fp32Manager)
    
    // Measure difference
    let maxDiff = zip(output16, output32).map { abs($0 - $1) }.max()!
    let avgDiff = zip(output16, output32).map { abs($0 - $1) }.reduce(0, +) / Float(output16.count)
    
    print("Max difference: \(maxDiff)")
    print("Avg difference: \(avgDiff)")
    // Typical: max ~0.001, avg ~0.0001 (negligible for LLMs)
}
```

---

## Long Context Handling

### Chunked Prefill

Process very long prompts in chunks:

```swift
func chunkedPrefill(
    tokens: [Int],
    sequenceID: Int,
    chunkSize: Int = 512
) throws {
    try cacheManager.allocateSequence(id: sequenceID)
    
    for chunkStart in stride(from: 0, to: tokens.count, by: chunkSize) {
        let chunkEnd = min(chunkStart + chunkSize, tokens.count)
        let chunk = Array(tokens[chunkStart..<chunkEnd])
        
        try cacheManager.appendTokens(toSequence: sequenceID, count: chunk.count)
        
        // Process chunk
        let (q, k, v) = computeQKV(chunk)
        engine.prefill(
            q: q, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            seqLen: chunk.count, headDim: 128, numHeads: 8, numKVHeads: 2,
            blockSize: 16, causal: true, output: outputBuffer, dataType: .float16
        )
        
        print("Processed chunk \(chunkStart/chunkSize + 1)/\(tokens.count/chunkSize)")
    }
}
```

### Sliding Window Attention

Implement sliding window for ultra-long contexts:

```swift
class SlidingWindowCache {
    let cacheManager: KVCacheManager
    let windowSize: Int
    
    func appendWithWindow(sequenceID: Int, newTokens: Int) throws {
        let seq = try cacheManager.getSequence(id: sequenceID)
        let newLen = seq.sequenceLength + newTokens
        
        if newLen > windowSize {
            // Evict oldest blocks
            let tokensToEvict = newLen - windowSize
            let blocksToEvict = tokensToEvict / cacheManager.blockSize
            
            // Free oldest blocks (implementation depends on your needs)
            // This is a simplified example
            print("Would evict \(blocksToEvict) blocks")
        }
        
        try cacheManager.appendTokens(toSequence: sequenceID, count: newTokens)
    }
}
```

---

## Multi-Sequence Patterns

### Request Batching

Batch incoming requests for efficiency:

```swift
class RequestBatcher {
    var pendingRequests: [(id: Int, tokens: [Int])] = []
    let batchTimeout: TimeInterval = 0.01  // 10ms
    let maxBatchSize: Int = 32
    
    func addRequest(id: Int, tokens: [Int]) {
        pendingRequests.append((id, tokens))
        
        if pendingRequests.count >= maxBatchSize {
            flush()
        }
    }
    
    func flush() {
        guard !pendingRequests.isEmpty else { return }
        
        // Process batch
        for (id, tokens) in pendingRequests {
            try? processPrefill(id: id, tokens: tokens)
        }
        
        pendingRequests.removeAll()
    }
}
```

### Continuous Batching

Continuously add/remove sequences from batch:

```swift
class ContinuousBatcher {
    var runningBatch: [Int: GenerationState] = [:]
    
    func step() throws {
        // Add new requests
        while let newReq = requestQueue.dequeue(),
              runningBatch.count < maxBatchSize {
            runningBatch[newReq.id] = GenerationState(tokens: newReq.prompt)
        }
        
        // Batch decode
        let ids = Array(runningBatch.keys)
        let outputs = try batchDecode(ids)
        
        // Update states and remove finished
        for (id, output) in zip(ids, outputs) {
            runningBatch[id]?.append(output)
            
            if runningBatch[id]?.isFinished == true {
                runningBatch.removeValue(forKey: id)
                cacheManager.freeSequence(id: id)
            }
        }
    }
}
```

---

## Error Handling

### Graceful Degradation

Handle errors without crashing:

```swift
func safeGenerate(prompt: [Int], maxTokens: Int) -> [Int]? {
    let sequenceID = UUID().hashValue
    
    do {
        try cacheManager.allocateSequence(id: sequenceID)
        defer { cacheManager.freeSequence(id: sequenceID) }
        
        // Prefill
        try cacheManager.appendTokens(toSequence: sequenceID, count: prompt.count)
        var tokens = prompt
        
        // Decode
        for _ in 0..<maxTokens {
            guard cacheManager.freeBlocks.count > 0 else {
                print("Warning: Out of memory, returning partial result")
                return tokens
            }
            
            try cacheManager.appendTokens(toSequence: sequenceID, count: 1)
            let nextToken = try generateNextToken(sequenceID: sequenceID)
            tokens.append(nextToken)
        }
        
        return tokens
        
    } catch KVCacheError.outOfMemory {
        print("Error: Out of memory")
        return nil
    } catch {
        print("Error: \(error)")
        return nil
    }
}
```

### Retry Logic

Retry with smaller batch on OOM:

```swift
func batchGenerateWithRetry(prompts: [[Int]]) throws -> [[Int]] {
    var batchSize = prompts.count
    
    while batchSize > 0 {
        do {
            return try batchGenerate(Array(prompts.prefix(batchSize)))
        } catch KVCacheError.outOfMemory {
            batchSize /= 2
            print("OOM: Retrying with batch size \(batchSize)")
        }
    }
    
    throw GenerationError.allRetriesFailed
}
```

---

## Profiling & Debugging

### Metal Performance HUD

Enable Metal HUD for real-time GPU stats:

```bash
# Run with Metal HUD
MTL_HUD_ENABLED=1 swift run YourApp
```

### Custom Timing

Measure kernel performance:

```swift
import QuartzCore

class PerformanceMonitor {
    var timings: [String: [Double]] = [:]
    
    func measure<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        let start = CACurrentMediaTime()
        let result = try block()
        let elapsed = CACurrentMediaTime() - start
        
        timings[label, default: []].append(elapsed * 1000)  // ms
        return result
    }
    
    func report() {
        for (label, times) in timings {
            let avg = times.reduce(0, +) / Double(times.count)
            let min = times.min() ?? 0
            let max = times.max() ?? 0
            print("\(label): avg=\(String(format: "%.2f", avg))ms, min=\(String(format: "%.2f", min))ms, max=\(String(format: "%.2f", max))ms")
        }
    }
}

// Usage
let monitor = PerformanceMonitor()

monitor.measure("prefill") {
    engine.prefill(...)
}

monitor.measure("decode") {
    engine.decode(...)
}

monitor.report()
```

### Validation Mode

Add assertions for debugging:

```swift
#if DEBUG
extension KVCacheManager {
    func validate() {
        // Check no duplicate blocks
        var usedBlocks = Set<Int32>()
        for seq in sequences.values {
            for block in seq.blockTable {
                assert(!usedBlocks.contains(block), "Duplicate block \(block)")
                usedBlocks.insert(block)
            }
        }
        
        // Check free list consistency
        let totalUsed = sequences.values.reduce(0) { $0 + $1.blockTable.count }
        assert(totalUsed + freeBlocks.count == maxBlocks, "Block accounting mismatch")
    }
}
#endif
```

---

## Best Practices Summary

1. **Use FP16** for production (2x speedup, minimal accuracy loss)
2. **Batch aggressively** (8-32 sequences for optimal GPU utilization)
3. **Monitor memory** (track fragmentation, implement eviction)
4. **Tune block size** (16 for general use, 32 for long contexts)
5. **Profile regularly** (identify bottlenecks with Metal HUD)
6. **Handle errors gracefully** (OOM is common, implement retry logic)
7. **Minimize CPU↔GPU transfers** (batch operations, keep data on GPU)
8. **Use continuous batching** for serving (maximize throughput)

---

## Next Steps

- See [MLX_INTEGRATION.md](MLX_INTEGRATION.md) for framework integration
- See [ARCHITECTURE.md](ARCHITECTURE.md) for kernel implementation details
- See [Examples/](../Examples/) for complete working examples

import XCTest
import Metal
@testable import PagedAttentionMetal

final class PagedAttentionMetalTests: XCTestCase {
    
    var device: MTLDevice!
    var engine: PagedAttentionEngine!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()!
        engine = try! PagedAttentionEngine()
    }
    
    // MARK: - Helpers
    
    func makeBuffer<T>(from data: [T]) -> MTLBuffer {
        data.withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared)!
        }
    }
    
    func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
    
    func readFloat16s(from buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }
    
    func cpuAttention(q: [Float], k: [Float], v: [Float], seqLen: Int, headDim: Int, causal: Bool) -> [Float] {
        var output = [Float](repeating: 0, count: seqLen * headDim)
        for i in 0..<seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            for j in 0..<seqLen {
                if causal && j > i { continue }
                var dot: Float = 0
                for d in 0..<headDim { dot += q[i * headDim + d] * k[j * headDim + d] }
                scores[j] = dot / sqrtf(Float(headDim))
            }
            let maxScore = scores.max() ?? 0
            var expSum: Float = 0
            for j in 0..<seqLen {
                if causal && j > i { scores[j] = 0 } else {
                    scores[j] = expf(scores[j] - maxScore)
                    expSum += scores[j]
                }
            }
            for j in 0..<seqLen { scores[j] /= expSum }
            for d in 0..<headDim {
                var sum: Float = 0
                for j in 0..<seqLen { sum += scores[j] * v[j * headDim + d] }
                output[i * headDim + d] = sum
            }
        }
        return output
    }
    
    func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }
    
    // MARK: - Test 1: Correctness vs CPU Reference
    
    func testCorrectnessVsCPU() throws {
        print("\n=== Test 1: Correctness vs CPU Reference ===")
        
        let seqLen = 16
        let headDim = 64
        let numHeads = 1
        let numKVHeads = 1
        let blockSize = 16
        
        // Random input
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        
        // CPU reference
        let cpuOutput = cpuAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: false)
        
        // GPU execution
        let cacheManager = KVCacheManager(device: device, maxBlocks: 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheManager.allocateSequence(id: 1)
        try cacheManager.appendTokens(toSequence: 1, count: seqLen)
        
        // Write K/V to cache
        let kPtr = cacheManager.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim * 4)
        let vPtr = cacheManager.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim * 4)
        for i in 0..<seqLen {
            let blockIdx = i / blockSize
            let offsetInBlock = i % blockSize
            for d in 0..<headDim {
                let cacheIdx = blockIdx * blockSize * headDim + offsetInBlock * headDim + d
                kPtr[cacheIdx] = k[i * headDim + d]
                vPtr[cacheIdx] = v[i * headDim + d]
            }
        }
        
        let bufQ = makeBuffer(from: q)
        let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))
        
        engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO, dataType: .float32)
        
        let gpuOutput = readFloats(from: bufO, count: seqLen * headDim)
        
        let maxErr = maxAbsError(cpuOutput, gpuOutput)
        print("  Max error: \(maxErr)")
        XCTAssertLessThan(maxErr, 1e-3, "GPU output differs from CPU reference")
        print("  ✓ Correctness verified")
    }

    
    // MARK: - Test 2: Causal Masking Correctness
    
    func testCausalMasking() throws {
        print("\n=== Test 2: Causal Masking ===")
        
        let seqLen = 8
        let headDim = 32
        let numHeads = 1
        let numKVHeads = 1
        let blockSize = 16
        
        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        
        let cpuCausal = cpuAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: true)
        
        let cacheManager = KVCacheManager(device: device, maxBlocks: 2, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheManager.allocateSequence(id: 1)
        try cacheManager.appendTokens(toSequence: 1, count: seqLen)
        
        let kPtr = cacheManager.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim * 4)
        let vPtr = cacheManager.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim * 4)
        for i in 0..<seqLen {
            let blockIdx = i / blockSize
            let offsetInBlock = i % blockSize
            for d in 0..<headDim {
                let cacheIdx = blockIdx * blockSize * headDim + offsetInBlock * headDim + d
                kPtr[cacheIdx] = k[i * headDim + d]
                vPtr[cacheIdx] = v[i * headDim + d]
            }
        }
        
        let bufQ = makeBuffer(from: q)
        let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))
        
        engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: true, output: bufO, dataType: .float32)
        
        let gpuCausal = readFloats(from: bufO, count: seqLen * headDim)
        
        let maxErr = maxAbsError(cpuCausal, gpuCausal)
        print("  Max error: \(maxErr)")
        XCTAssertLessThan(maxErr, 1e-3, "Causal masking incorrect")
        print("  ✓ Causal masking verified")
    }

    
    // MARK: - Test 3: Grouped-Query Attention (GQA)
    
    func testGQA() throws {
        print("\n=== Test 3: Grouped-Query Attention ===")
        
        let seqLen = 16
        let headDim = 64
        let numHeads = 8
        let numKVHeads = 2  // 4:1 ratio
        let blockSize = 16
        
        let cacheManager = KVCacheManager(device: device, maxBlocks: 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheManager.allocateSequence(id: 1)
        try cacheManager.appendTokens(toSequence: 1, count: seqLen)
        
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        
        // Write K/V
        let kPtr = cacheManager.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 4)
        let vPtr = cacheManager.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 4)
        for kvHead in 0..<numKVHeads {
            for i in 0..<seqLen {
                let blockIdx = i / blockSize
                let offsetInBlock = i % blockSize
                for d in 0..<headDim {
                    let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    kPtr[cacheIdx] = k[kvHead * seqLen * headDim + i * headDim + d]
                    vPtr[cacheIdx] = v[kvHead * seqLen * headDim + i * headDim + d]
                }
            }
        }
        
        let bufQ = makeBuffer(from: q)
        let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))
        
        engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO, dataType: .float32)
        
        let output = readFloats(from: bufO, count: seqLen * numHeads * headDim)
        
        // Verify output is reasonable (not NaN, not all zeros, values in expected range)
        XCTAssertFalse(output.contains { $0.isNaN }, "GQA produced NaN")
        XCTAssertFalse(output.allSatisfy { $0 == 0 }, "GQA produced all zeros")
        XCTAssertTrue(output.allSatisfy { abs($0) < 10 }, "GQA values out of range")
        
        print("  ✓ GQA verified (8 Q heads → 2 KV heads)")
    }

    
    // MARK: - Test 4: FP16 Correctness & Performance
    
    func testFP16Performance() throws {
        print("\n=== Test 4: FP16 vs FP32 ===")
        
        let seqLen = 512
        let headDim = 128
        let numHeads = 8
        let numKVHeads = 2
        let blockSize = 16
        let iterations = 50
        
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        
        // FP32
        let cache32 = KVCacheManager(device: device, maxBlocks: 64, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cache32.allocateSequence(id: 1)
        try cache32.appendTokens(toSequence: 1, count: seqLen)
        
        let kPtr32 = cache32.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 64)
        let vPtr32 = cache32.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 64)
        for kvHead in 0..<numKVHeads {
            for i in 0..<seqLen {
                let blockIdx = i / blockSize
                let offsetInBlock = i % blockSize
                for d in 0..<headDim {
                    let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    kPtr32[cacheIdx] = k[kvHead * seqLen * headDim + i * headDim + d]
                    vPtr32[cacheIdx] = v[kvHead * seqLen * headDim + i * headDim + d]
                }
            }
        }
        
        let bufQ32 = makeBuffer(from: q)
        let bufO32 = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))
        
        let start32 = Date()
        for _ in 0..<iterations {
            engine.prefill(q: bufQ32, kPool: cache32.kPoolBuffer, vPool: cache32.vPoolBuffer,
                          blockTable: try cache32.getBlockTableBuffer(forSequence: 1),
                          seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                          blockSize: blockSize, causal: false, output: bufO32, dataType: .float32)
        }
        let time32 = Date().timeIntervalSince(start32) * 1000 / Double(iterations)
        let output32 = readFloats(from: bufO32, count: seqLen * numHeads * headDim)
        
        // FP16
        let cache16 = KVCacheManager(device: device, maxBlocks: 64, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float16)
        try cache16.allocateSequence(id: 1)
        try cache16.appendTokens(toSequence: 1, count: seqLen)
        
        let kPtr16 = cache16.kPoolBuffer.contents().bindMemory(to: Float16.self, capacity: seqLen * numKVHeads * headDim * 64)
        let vPtr16 = cache16.vPoolBuffer.contents().bindMemory(to: Float16.self, capacity: seqLen * numKVHeads * headDim * 64)
        for kvHead in 0..<numKVHeads {
            for i in 0..<seqLen {
                let blockIdx = i / blockSize
                let offsetInBlock = i % blockSize
                for d in 0..<headDim {
                    let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    kPtr16[cacheIdx] = Float16(k[kvHead * seqLen * headDim + i * headDim + d])
                    vPtr16[cacheIdx] = Float16(v[kvHead * seqLen * headDim + i * headDim + d])
                }
            }
        }
        
        let bufQ16 = makeBuffer(from: q.map { Float16($0) })
        let bufO16 = makeBuffer(from: [Float16](repeating: 0, count: seqLen * numHeads * headDim))
        
        let start16 = Date()
        for _ in 0..<iterations {
            engine.prefill(q: bufQ16, kPool: cache16.kPoolBuffer, vPool: cache16.vPoolBuffer,
                          blockTable: try cache16.getBlockTableBuffer(forSequence: 1),
                          seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                          blockSize: blockSize, causal: false, output: bufO16, dataType: .float16)
        }
        let time16 = Date().timeIntervalSince(start16) * 1000 / Double(iterations)
        let output16 = readFloat16s(from: bufO16, count: seqLen * numHeads * headDim)
        
        let maxErr = maxAbsError(output32, output16)
        let speedup = time32 / time16
        let memoryRatio = Float(cache32.kPoolBuffer.length + cache32.vPoolBuffer.length) / Float(cache16.kPoolBuffer.length + cache16.vPoolBuffer.length)
        
        print("  FP32: \(String(format: "%.2f", time32)) ms")
        print("  FP16: \(String(format: "%.2f", time16)) ms")
        print("  Speedup: \(String(format: "%.2fx", speedup))")
        print("  Memory reduction: \(String(format: "%.2fx", memoryRatio))")
        print("  Max error: \(maxErr)")
        
        XCTAssertLessThan(maxErr, 0.01, "FP16 error too large")
        XCTAssertEqual(memoryRatio, 2.0, accuracy: 0.01, "FP16 memory not 2x smaller")
        print("  ✓ FP16 uses 2x less memory with acceptable precision")
    }

    
    // MARK: - Test 5: Batch Decode
    
    func testBatchDecode() throws {
        print("\n=== Test 5: Batch Decode ===")
        
        let batchSize = 4
        let headDim = 64
        let numHeads = 4
        let numKVHeads = 2
        let blockSize = 16
        let seqLengths = [32, 48, 64, 80]
        
        let batchCache = BatchKVCacheManager(device: device, maxBatchSize: batchSize, maxSequenceBlocks: 16, maxBlocks: 64, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        
        let batchIDs = Array(0..<batchSize)
        for i in 0..<batchSize {
            try batchCache.allocateSequence(id: batchIDs[i])
            try batchCache.appendTokens(toSequence: batchIDs[i], count: seqLengths[i])
        }
        
        let q = (0..<(batchSize * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let bufQ = makeBuffer(from: q)
        let bufO = makeBuffer(from: [Float](repeating: 0, count: batchSize * numHeads * headDim))
        
        engine.decode(q: bufQ, kPool: batchCache.kPoolBuffer, vPool: batchCache.vPoolBuffer,
                     blockTables: try batchCache.getBatchBlockTableBuffer(forBatch: batchIDs),
                     seqLengths: try batchCache.getSeqLengthsBuffer(forBatch: batchIDs),
                     batchSize: batchSize, maxNumBlocks: 16, headDim: headDim,
                     numHeads: numHeads, numKVHeads: numKVHeads, blockSize: blockSize,
                     output: bufO, dataType: .float32)
        
        let output = readFloats(from: bufO, count: batchSize * numHeads * headDim)
        
        XCTAssertFalse(output.contains { $0.isNaN }, "Batch decode produced NaN")
        print("  ✓ Batch decode completed for \(batchSize) sequences")
    }
    
    // MARK: - Test 6: Variable Block Sizes
    
    func testVariableBlockSizes() throws {
        print("\n=== Test 6: Variable Block Sizes ===")
        
        let seqLen = 32
        let headDim = 64
        let numHeads = 2
        let numKVHeads = 2
        let blockSizes = [8, 16, 32]
        
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        
        var outputs: [[Float]] = []
        
        for blockSize in blockSizes {
            let cache = KVCacheManager(device: device, maxBlocks: 8, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
            try cache.allocateSequence(id: 1)
            try cache.appendTokens(toSequence: 1, count: seqLen)
            
            let kPtr = cache.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 8)
            let vPtr = cache.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 8)
            for kvHead in 0..<numKVHeads {
                for i in 0..<seqLen {
                    let blockIdx = i / blockSize
                    let offsetInBlock = i % blockSize
                    for d in 0..<headDim {
                        let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                        kPtr[cacheIdx] = k[kvHead * seqLen * headDim + i * headDim + d]
                        vPtr[cacheIdx] = v[kvHead * seqLen * headDim + i * headDim + d]
                    }
                }
            }
            
            let bufQ = makeBuffer(from: q)
            let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))
            
            engine.prefill(q: bufQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                          blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                          seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                          blockSize: blockSize, causal: false, output: bufO, dataType: .float32)
            
            outputs.append(readFloats(from: bufO, count: seqLen * numHeads * headDim))
        }
        
        // All block sizes should produce same result
        for i in 1..<blockSizes.count {
            let err = maxAbsError(outputs[0], outputs[i])
            XCTAssertLessThan(err, 0.5, "Block size \(blockSizes[i]) differs from \(blockSizes[0])")
        }
        
        print("  ✓ Block sizes 8, 16, 32 all produce identical results")
    }
}

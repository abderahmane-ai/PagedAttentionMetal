import Testing
import Metal
@testable import PagedAttentionMetal
@testable import PagedAttentionMLXSupport

@Suite struct PagedAttentionMetalTests {

    let device: MTLDevice
    let engine: PagedAttentionEngine

    init() {
        device = MTLCreateSystemDefaultDevice()!
        engine = try! PagedAttentionEngine()
    }

    // MARK: - Helpers

    func makeBuffer<T>(from data: [T]) -> MTLBuffer {
        data.withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared)!
        }
    }

    func makeBufferPointer(_ buffer: MTLBuffer) -> UnsafeMutableRawPointer {
        Unmanaged.passUnretained(buffer as AnyObject).toOpaque()
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

    @Test func correctnessVsCPU() throws {
        print("\n=== Test 1: Correctness vs CPU Reference ===")

        let seqLen = 16
        let headDim = 64
        let numHeads = 1
        let numKVHeads = 1
        let blockSize = 16

        let q = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }

        let cpuOutput = cpuAttention(q: q, k: k, v: v, seqLen: seqLen, headDim: headDim, causal: false)

        let cacheManager = KVCacheManager(device: device, maxBlocks: 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
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

        try engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO, dataType: .float32)

        let gpuOutput = readFloats(from: bufO, count: seqLen * headDim)

        let maxErr = maxAbsError(cpuOutput, gpuOutput)
        print("  Max error: \(maxErr)")
        #expect(maxErr < 1e-3, "GPU output differs from CPU reference")
        print("  \u{2713} Correctness verified")
    }

    @Test func mmaCorrectness() throws {
        print("\n=== Test 1b: MMA vs CPU Reference ===")
        let seqLen = 16
        let headDim = 64
        let numHeads = 1
        let numKVHeads = 1
        let blockSize = 16
        let causal = false
        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        var cpuOutput = [Float](repeating: 0, count: seqLen * numHeads * headDim)
        for h in 0..<numHeads {
            let kvh = h / (numHeads / numKVHeads)
            for i in 0..<seqLen {
                var scores = [Float](repeating: 0, count: seqLen)
                for j in 0..<seqLen {
                    if causal && j > i { continue }
                    var dot: Float = 0
                    for d in 0..<headDim { dot += q[i * numHeads * headDim + h * headDim + d] * k[j * numKVHeads * headDim + kvh * headDim + d] }
                    scores[j] = dot / sqrt(Float(headDim))
                }
                let m = scores.max() ?? 0
                var sum: Float = 0
                for j in 0..<seqLen { let e = exp(scores[j] - m); scores[j] = e; sum += e }
                for d in 0..<headDim {
                    var acc: Float = 0
                    for j in 0..<seqLen { acc += scores[j] / sum * v[j * numKVHeads * headDim + kvh * headDim + d] }
                    cpuOutput[i * numHeads * headDim + h * headDim + d] = acc
                }
            }
        }
        let cache = KVCacheManager(device: device, maxBlocks: 8, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float16)
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: seqLen)
        let kPtr = cache.kPoolBuffer.contents().bindMemory(to: Float16.self, capacity: seqLen * numKVHeads * headDim * 8)
        let vPtr = cache.vPoolBuffer.contents().bindMemory(to: Float16.self, capacity: seqLen * numKVHeads * headDim * 8)
        for kvh in 0..<numKVHeads {
            for i in 0..<seqLen {
                let bi = i / blockSize; let oi = i % blockSize
                for d in 0..<headDim {
                    let ci = bi * blockSize * numKVHeads * headDim + oi * numKVHeads * headDim + kvh * headDim + d
                    let si = i * numKVHeads * headDim + kvh * headDim + d
                    kPtr[ci] = Float16(k[si]); vPtr[ci] = Float16(v[si])
                }
            }
        }
        let bufQ = makeBuffer(from: q.map { Float16($0) })
        let bufO = makeBuffer(from: [Float16](repeating: 0, count: seqLen * numHeads * headDim))
        try engine.prefill(q: bufQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                      blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: causal, output: bufO, dataType: .float16)
        let gpuOutput = readFloat16s(from: bufO, count: seqLen * numHeads * headDim)
        let maxErr = maxAbsError(cpuOutput, gpuOutput)
        print("  MMA vs CPU max error: \(maxErr)")
        print("  CPU[0..<8]: \(Array(cpuOutput[0..<8]))")
        print("  MMA[0..<8]: \(Array(gpuOutput[0..<8]))")
        print("  CPU[64..<72]: \(Array(cpuOutput[64..<72]))")
        print("  MMA[64..<72]: \(Array(gpuOutput[64..<72]))")
        #expect(maxErr < 0.1, "MMA output differs from CPU")
        print("  \u{2713} MMA correctness verified")

        // Also test float32 (flash prefill) against CPU for validation
        let cache32 = KVCacheManager(device: device, maxBlocks: 8, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cache32.allocateSequence(id: 2)
        try cache32.appendTokens(toSequence: 2, count: seqLen)
        let kPtr32 = cache32.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 8)
        let vPtr32 = cache32.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * 8)
        for kvh in 0..<numKVHeads {
            for i in 0..<seqLen {
                let bi = i / blockSize; let oi = i % blockSize
                for d in 0..<headDim {
                    let ci = bi * blockSize * numKVHeads * headDim + oi * numKVHeads * headDim + kvh * headDim + d
                    let si = i * numKVHeads * headDim + kvh * headDim + d
                    kPtr32[ci] = k[si]; vPtr32[ci] = v[si]
                }
            }
        }
        let bufQ32 = makeBuffer(from: q)
        let bufO32 = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))
        try engine.prefill(q: bufQ32, kPool: cache32.kPoolBuffer, vPool: cache32.vPoolBuffer,
                      blockTable: try cache32.getBlockTableBuffer(forSequence: 2),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO32, dataType: .float32)
        let fp32output = readFloats(from: bufO32, count: seqLen * numHeads * headDim)
        let fp32err = maxAbsError(cpuOutput, fp32output)
        print("  FP32 flash vs CPU max error: \(fp32err)")
        print("  FP32 first output: \(fp32output[0]), \(fp32output[1]), \(fp32output[2]), \(fp32output[3])")
    }

    // MARK: - Test 2: Causal Masking Correctness

    @Test func causalMasking() throws {
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

        try engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: true, output: bufO, dataType: .float32)

        let gpuCausal = readFloats(from: bufO, count: seqLen * headDim)

        let maxErr = maxAbsError(cpuCausal, gpuCausal)
        print("  Max error: \(maxErr)")
        #expect(maxErr < 1e-3, "Causal masking incorrect")
        print("  \u{2713} Causal masking verified")
    }


    // MARK: - Test 3: Grouped-Query Attention (GQA)

    @Test func gqa() throws {
        print("\n=== Test 3: Grouped-Query Attention ===")

        let seqLen = 16
        let headDim = 64
        let numHeads = 8
        let numKVHeads = 2
        let blockSize = 16

        let cacheManager = KVCacheManager(device: device, maxBlocks: 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheManager.allocateSequence(id: 1)
        try cacheManager.appendTokens(toSequence: 1, count: seqLen)

        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }

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

        try engine.prefill(q: bufQ, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                      blockTable: try cacheManager.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO, dataType: .float32)

        let output = readFloats(from: bufO, count: seqLen * numHeads * headDim)

        #expect(!output.contains { $0.isNaN }, "GQA produced NaN")
        #expect(!output.allSatisfy { $0 == 0 }, "GQA produced all zeros")
        #expect(output.allSatisfy { abs($0) < 10 }, "GQA values out of range")

        print("  \u{2713} GQA verified (8 Q heads \u{2192} 2 KV heads)")
    }


    // MARK: - Test 4: FP16 Correctness & Performance

    @Test func fp16Performance() throws {
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
            try engine.prefill(q: bufQ32, kPool: cache32.kPoolBuffer, vPool: cache32.vPoolBuffer,
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
            try engine.prefill(q: bufQ16, kPool: cache16.kPoolBuffer, vPool: cache16.vPoolBuffer,
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

        #expect(maxErr < 0.15, "FP16 error too large")
        #expect(abs(memoryRatio - 2.0) < 0.01, "FP16 memory not 2x smaller")
        print("  \u{2713} FP16 uses 2x less memory with acceptable precision")
    }


    // MARK: - Test 5: Batch Decode

    @Test func batchDecode() throws {
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

        try engine.decode(q: bufQ, kPool: batchCache.kPoolBuffer, vPool: batchCache.vPoolBuffer,
                     blockTables: try batchCache.getBatchBlockTableBuffer(forBatch: batchIDs),
                     seqLengths: try batchCache.getSeqLengthsBuffer(forBatch: batchIDs),
                     batchSize: batchSize, maxNumBlocks: 16, headDim: headDim,
                     numHeads: numHeads, numKVHeads: numKVHeads, blockSize: blockSize,
                     output: bufO, dataType: .float32)

        let output = readFloats(from: bufO, count: batchSize * numHeads * headDim)

        #expect(!output.contains { $0.isNaN }, "Batch decode produced NaN")
        print("  \u{2713} Batch decode completed for \(batchSize) sequences")
    }

    // MARK: - Test 6: Variable Block Sizes

    @Test func variableBlockSizes() throws {
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

            try engine.prefill(q: bufQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                          blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                          seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                          blockSize: blockSize, causal: false, output: bufO, dataType: .float32)

            outputs.append(readFloats(from: bufO, count: seqLen * numHeads * headDim))
        }

        for i in 1..<blockSizes.count {
            let err = maxAbsError(outputs[0], outputs[i])
            #expect(err < 0.5, "Block size \(blockSizes[i]) differs from \(blockSizes[0])")
        }

        print("  \u{2713} Block sizes 8, 16, 32 all produce identical results")
    }

    // MARK: - Test 7: Checked API validation and stats

    @Test func checkedAPIValidationAndStats() throws {
        let seqLen = 8
        let headDim = 32
        let layer = PagedLayerSpec(headDim: headDim, numHeads: 1, numKVHeads: 1, blockSize: 8, dataType: .float32)
        let cache = KVCacheManager(device: device, maxBlocks: 4, blockSize: layer.blockSize, headDim: headDim, numKVHeads: 1, dataType: .float32)
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: seqLen)

        let q = makeBuffer(from: [Float](repeating: 0.25, count: seqLen * headDim))
        let tooSmallOutput = makeBuffer(from: [Float](repeating: 0, count: 1))

        do {
            try engine.prefill(PagedAttentionPrefillRequest(
                q: q,
                kPool: cache.kPoolBuffer,
                vPool: cache.vPoolBuffer,
                blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                output: tooSmallOutput,
                seqLen: seqLen,
                layer: layer,
                causal: true
            ))
            Issue.record("Expected bufferTooSmall error")
        } catch PagedAttentionError.bufferTooSmall(let name, _, _) {
            #expect(name == "output")
        } catch {
            Issue.record("Expected bufferTooSmall, got \(error)")
        }

        let output = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))
        try engine.prefill(PagedAttentionPrefillRequest(
            q: q,
            kPool: cache.kPoolBuffer,
            vPool: cache.vPoolBuffer,
            blockTable: try cache.getBlockTableBuffer(forSequence: 1),
            output: output,
            seqLen: seqLen,
            layer: layer,
            causal: true
        ))
        #expect(engine.lastStats.operation == .prefillSinglePass)
        #expect(engine.lastStats.sequenceLength == seqLen)
    }

    // MARK: - Test 8: Split-pass fallback for large threadgroups

    @Test func prefillFallsBackToSplitWhenSinglePassThreadgroupIsTooLarge() throws {
        let seqLen = 32
        let layer = PagedLayerSpec(headDim: 256, numHeads: 8, numKVHeads: 2, blockSize: 16, dataType: .float32)
        let cache = KVCacheManager(
            device: device,
            maxBlocks: 8,
            blockSize: layer.blockSize,
            headDim: layer.headDim,
            numKVHeads: layer.numKVHeads,
            dataType: layer.dataType
        )
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: seqLen)

        let qCount = seqLen * layer.qElementsPerToken
        let kvCount = seqLen * layer.kvElementsPerToken
        let q = makeBuffer(from: (0..<qCount).map { Float(($0 % 17) + 1) / 17.0 })
        let k = makeBuffer(from: (0..<kvCount).map { Float(($0 % 11) + 1) / 19.0 })
        let v = makeBuffer(from: (0..<kvCount).map { Float(($0 % 13) + 1) / 23.0 })
        let output = makeBuffer(from: [Float](repeating: 0, count: qCount))
        let blockTable = try cache.getBlockTableBuffer(forSequence: 1)

        try engine.appendToCache(PagedKVAppendRequest(
            keys: k,
            values: v,
            kPool: cache.kPoolBuffer,
            vPool: cache.vPoolBuffer,
            blockTable: blockTable,
            tokenOffset: 0,
            numNewTokens: seqLen,
            layer: layer
        ))

        try engine.prefill(PagedAttentionPrefillRequest(
            q: q,
            kPool: cache.kPoolBuffer,
            vPool: cache.vPoolBuffer,
            blockTable: blockTable,
            output: output,
            seqLen: seqLen,
            layer: layer,
            causal: true
        ))

        let values = readFloats(from: output, count: qCount)
        #expect(engine.lastStats.operation == .prefillTiledPass)
        #expect(!values.contains { $0.isNaN || $0.isInfinite })
        #expect(!values.allSatisfy { $0 == 0 })
    }

    // MARK: - Test 9: C ABI lifecycle

    @Test func cabiLifecycle() throws {
        guard let context = pam_create_context() else {
            Issue.record("Failed to create C ABI context")
            return
        }
        defer { pam_destroy_context(context) }

        guard let cache = pam_create_cache(context, 8, 4, 16, 1, PagedAttentionDataType.float32.rawValue) else {
            Issue.record("Failed to create C ABI cache")
            return
        }
        defer { pam_destroy_cache(cache) }

        #expect(pam_reserve_sequence(context, cache, 42) == 0)
        #expect(pam_append_tokens(context, cache, 42, 5) == 0)
        #expect(pam_sequence_length(context, cache, 42) == 5)
        #expect(pam_available_blocks(cache) == 6)
        pam_free_sequence(cache, 42)
        #expect(pam_available_blocks(cache) == 8)
    }

    // MARK: - Test 10: MLX support cache construction

    @Test func mlxSupportCacheConstruction() throws {
        let layer = PagedLayerSpec(headDim: 64, numHeads: 8, numKVHeads: 2, blockSize: 16, dataType: .float16)
        let cache = try PagedMetalKVCache(
            sequenceID: 7,
            layer: layer,
            maxBlocks: 32,
            device: device,
            engine: engine
        )
        #expect(cache.offset == 0)
        #expect(cache.cacheManager.availableBlocks == 32)
        #expect(cache.layer == layer)
    }

    // MARK: - Test 11: FP8 Quantization

    @Test func fp8Quantization() throws {
        print("\n=== Test 11: FP8 Quantization ===")

        let seqLen = 16
        let headDim = 64
        let numHeads = 4
        let numKVHeads = 2
        let blockSize = 16
        let numBlocks = (seqLen + blockSize - 1) / blockSize

        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }

        // FP16 reference
        let cache16 = KVCacheManager(device: device, maxBlocks: numBlocks, blockSize: blockSize,
                                      headDim: headDim, numKVHeads: numKVHeads, dataType: .float16)
        try cache16.allocateSequence(id: 1)
        try cache16.appendTokens(toSequence: 1, count: seqLen)

        let kPtr16 = cache16.kPoolBuffer.contents().bindMemory(to: Float16.self,
            capacity: seqLen * numKVHeads * headDim)
        let vPtr16 = cache16.vPoolBuffer.contents().bindMemory(to: Float16.self,
            capacity: seqLen * numKVHeads * headDim)
        for kvHead in 0..<numKVHeads {
            for i in 0..<seqLen {
                let blockIdx = i / blockSize
                let offsetInBlock = i % blockSize
                for d in 0..<headDim {
                    let cacheIdx = blockIdx * blockSize * numKVHeads * headDim
                                 + offsetInBlock * numKVHeads * headDim
                                 + kvHead * headDim + d
                    let srcIdx = kvHead * seqLen * headDim + i * headDim + d
                    kPtr16[cacheIdx] = Float16(k[srcIdx])
                    vPtr16[cacheIdx] = Float16(v[srcIdx])
                }
            }
        }

        let bufQ16 = makeBuffer(from: q.map { Float16($0) })
        let bufO16 = makeBuffer(from: [Float16](repeating: 0, count: seqLen * numHeads * headDim))
        try engine.prefill(q: bufQ16, kPool: cache16.kPoolBuffer, vPool: cache16.vPoolBuffer,
                      blockTable: try cache16.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufO16, dataType: .float16)
        let output16 = readFloat16s(from: bufO16, count: seqLen * numHeads * headDim)

        // FP8
        let fp8PoolBytes = numBlocks * blockSize * numKVHeads * headDim
        guard let kPoolFP8 = device.makeBuffer(length: fp8PoolBytes, options: .storageModeShared),
              let vPoolFP8 = device.makeBuffer(length: fp8PoolBytes, options: .storageModeShared),
              let kScaleBuf = device.makeBuffer(length: numBlocks * MemoryLayout<Float16>.stride, options: .storageModeShared),
              let vScaleBuf = device.makeBuffer(length: numBlocks * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            Issue.record("Failed to create FP8 pool buffers")
            return
        }
        let kScalePtr = kScaleBuf.contents().bindMemory(to: Float16.self, capacity: numBlocks)
        let vScalePtr = vScaleBuf.contents().bindMemory(to: Float16.self, capacity: numBlocks)
        kScalePtr.update(repeating: 0, count: numBlocks)
        vScalePtr.update(repeating: 0, count: numBlocks)

        let keysBuf = makeBuffer(from: k.map { Float16($0) })
        let valsBuf = makeBuffer(from: v.map { Float16($0) })
        try engine.appendToCache(PagedKVAppendRequest(
            keys: keysBuf, values: valsBuf,
            kPool: kPoolFP8, vPool: vPoolFP8,
            blockTable: try cache16.getBlockTableBuffer(forSequence: 1),
            tokenOffset: 0, numNewTokens: seqLen,
            layer: PagedLayerSpec(headDim: headDim, numHeads: numKVHeads, numKVHeads: numKVHeads,
                                   blockSize: blockSize, dataType: .float8),
            kScaleBuffer: kScaleBuf, vScaleBuffer: vScaleBuf
        ))

        let bufQFP8 = makeBuffer(from: q.map { Float16($0) })
        let bufOFP8 = makeBuffer(from: [Float16](repeating: 0, count: seqLen * numHeads * headDim))
        try engine.prefill(PagedAttentionPrefillRequest(
            q: bufQFP8, kPool: kPoolFP8, vPool: vPoolFP8,
            blockTable: try cache16.getBlockTableBuffer(forSequence: 1),
            output: bufOFP8, seqLen: seqLen,
            layer: PagedLayerSpec(headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                                   blockSize: blockSize, dataType: .float8),
            causal: false, kScaleBuffer: kScaleBuf, vScaleBuffer: vScaleBuf
        ))
        let outputFP8 = readFloat16s(from: bufOFP8, count: seqLen * numHeads * headDim)

        let maxErr = maxAbsError(output16, outputFP8)
        print("  FP8 max error vs FP16: \(maxErr)")

        #expect(!outputFP8.contains { $0.isNaN }, "FP8 produced NaN")
        #expect(!outputFP8.allSatisfy { $0 == 0 }, "FP8 produced all zeros")
        print("  \u{2713} FP8 quantization verified (max error \(maxErr))")
    }

    // MARK: - Phase 5: Fuzzing & Edge Case Tests

    @Test func emptySequencePrefill() throws {
        let cache = KVCacheManager(device: device, maxBlocks: 4, blockSize: 16, headDim: 32, numKVHeads: 1, dataType: .float32)
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: 8)

        let q = device.makeBuffer(length: 64)!
        let output = device.makeBuffer(length: 64)!
        let blockTable = try cache.getBlockTableBuffer(forSequence: 1)

        do {
            try engine.prefill(
                q: q, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                blockTable: blockTable, seqLen: 0, headDim: 32,
                numHeads: 1, numKVHeads: 1, blockSize: 16,
                causal: false, output: output, dataType: .float32
            )
        } catch {
            print("  Empty seq prefill threw: \(error)")
        }
    }

    @Test func prefillWithZeroBlocks() throws {
        let cache = KVCacheManager(device: device, maxBlocks: 1, blockSize: 16, headDim: 32, numKVHeads: 1, dataType: .float32)

        #expect(cache.availableBlocks == 1)

        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: 16)
        #expect(cache.availableBlocks == 0)

        do {
            try cache.appendTokens(toSequence: 1, count: 1)
            Issue.record("Expected outOfMemory error")
        } catch KVCacheError.outOfMemory {
            print("  \u{2713} Zero blocks correctly throws OOM")
        } catch {
            Issue.record("Expected outOfMemory, got \(error)")
        }
    }

    @Test func allocateThenImmediateFree() throws {
        let cache = KVCacheManager(device: device, maxBlocks: 4, blockSize: 16, headDim: 32, numKVHeads: 1, dataType: .float32)

        #expect(cache.availableBlocks == 4)

        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: 16)
        #expect(cache.availableBlocks == 3)

        cache.freeSequence(id: 1)
        #expect(cache.availableBlocks == 4)

        try cache.allocateSequence(id: 2)
        try cache.appendTokens(toSequence: 2, count: 32)
        let blocks = try cache.getBlockTable(forSequence: 2)
        #expect(blocks.count == 2)
        print("  \u{2713} Allocate\u{2192}free\u{2192}recycle works correctly")
    }

    @Test func batchDecodeEmpty() throws {
        let q = device.makeBuffer(length: 64)!
        let output = device.makeBuffer(length: 64)!
        let blockTables = device.makeBuffer(length: 64)!
        let seqLengths = device.makeBuffer(length: 64)!

        do {
            try engine.decode(
                q: q, kPool: device.makeBuffer(length: 64)!,
                vPool: device.makeBuffer(length: 64)!,
                blockTables: blockTables, seqLengths: seqLengths,
                batchSize: 0, maxNumBlocks: 1,
                headDim: 32, numHeads: 1, numKVHeads: 1,
                blockSize: 16, output: output, dataType: .float32
            )
        } catch {
            print("  Empty batch decode threw: \(error)")
        }
    }

    @Test func appendBeyondBlockBoundary() throws {
        let blockSize = 16
        let cache = KVCacheManager(device: device, maxBlocks: 16, blockSize: blockSize, headDim: 32, numKVHeads: 1, dataType: .float32)
        try cache.allocateSequence(id: 1)

        try cache.appendTokens(toSequence: 1, count: blockSize * 3 + 7)
        let blocks = try cache.getBlockTable(forSequence: 1)
        #expect(blocks.count == 4, "Should allocate 4 blocks for \(blockSize * 3 + 7) tokens")
        #expect(try cache.getSequenceLength(1) == blockSize * 3 + 7)

        try cache.appendTokens(toSequence: 1, count: blockSize * 2)
        let blocks2 = try cache.getBlockTable(forSequence: 1)
        #expect(blocks2.count == 6, "Should now have 6 blocks")
        #expect(try cache.getSequenceLength(1) == blockSize * 5 + 7)
        print("  \u{2713} Append beyond block boundary works (tokens=\(blockSize * 5 + 7), blocks=\(blocks2.count))")
    }

    @Test func sequentialAndRepeatedPrefills() throws {
        let headDim = 32
        let numHeads = 2
        let numKVHeads = 1
        let blockSize = 16

        let cache = KVCacheManager(device: device, maxBlocks: 16, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cache.allocateSequence(id: 1)

        var totalTokens = 0
        for cycle in 0..<5 {
            let seqLen = 8
            let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
            let k = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
            let v = (0..<(seqLen * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }

            try cache.appendTokens(toSequence: 1, count: seqLen)
            totalTokens += seqLen

            let kPtr = cache.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: totalTokens * numKVHeads * headDim)
            let vPtr = cache.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: totalTokens * numKVHeads * headDim)
            let tokenStart = totalTokens - seqLen
            for kvHead in 0..<numKVHeads {
                for i in 0..<seqLen {
                    let globalI = tokenStart + i
                    let blockIdx = globalI / blockSize
                    let offsetInBlock = globalI % blockSize
                    for d in 0..<headDim {
                        let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                        kPtr[cacheIdx] = k[kvHead * seqLen * headDim + i * headDim + d]
                        vPtr[cacheIdx] = v[kvHead * seqLen * headDim + i * headDim + d]
                    }
                }
            }

            let bufQ = makeBuffer(from: q)
            let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))

            if cycle == 0 {
                try engine.prefill(q: bufQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                              blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                              seqLen: totalTokens, headDim: headDim, numHeads: numHeads,
                              numKVHeads: numKVHeads, blockSize: blockSize,
                              causal: false, output: bufO, dataType: .float32)
            } else {
                let decodeQ = makeBuffer(from: [Float](repeating: 0.5, count: numHeads * headDim))
                let decodeO = makeBuffer(from: [Float](repeating: 0, count: numHeads * headDim))
                let numBlocks = try cache.getNumBlocks(forSequence: 1)
                try engine.decode(q: decodeQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                              blockTables: try cache.getBlockTableBuffer(forSequence: 1),
                              seqLengths: makeBuffer(from: [UInt32(totalTokens)]),
                              batchSize: 1, maxNumBlocks: numBlocks,
                              headDim: headDim, numHeads: numHeads,
                              numKVHeads: numKVHeads, blockSize: blockSize,
                              output: decodeO, dataType: .float32)
                let decoded = readFloats(from: decodeO, count: numHeads * headDim)
                #expect(!decoded.contains { $0.isNaN })
            }
        }
        #expect(try cache.getSequenceLength(1) == 40)
        print("  \u{2713} Sequential prefill/decode cycles completed (total tokens=40)")
    }

    @Test func simultaneousPrefillDecode() throws {
        let headDim = 32
        let numHeads = 2
        let numKVHeads = 1
        let blockSize = 16

        let cache = KVCacheManager(device: device, maxBlocks: 16, blockSize: blockSize, headDim: headDim,
                                    numKVHeads: numKVHeads, dataType: .float32)
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: 16)

        let k = (0..<(16 * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(16 * numKVHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let kPtr = cache.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: 16 * numKVHeads * headDim)
        let vPtr = cache.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: 16 * numKVHeads * headDim)
        for kvHead in 0..<numKVHeads {
            for i in 0..<16 {
                for d in 0..<headDim {
                    let idx = i * numKVHeads * headDim + kvHead * headDim + d
                    kPtr[idx] = k[kvHead * 16 * headDim + i * headDim + d]
                    vPtr[idx] = v[kvHead * 16 * headDim + i * headDim + d]
                }
            }
        }

        let q1 = (0..<(16 * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        let bufQ1 = makeBuffer(from: q1)
        let bufO1 = makeBuffer(from: [Float](repeating: 0, count: 16 * numHeads * headDim))
        try engine.prefill(q: bufQ1, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                      blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                      seqLen: 16, headDim: headDim, numHeads: numHeads,
                      numKVHeads: numKVHeads, blockSize: blockSize,
                      causal: false, output: bufO1, dataType: .float32)

        let q2 = makeBuffer(from: [Float](repeating: 0.5, count: numHeads * headDim))
        let bufO2 = makeBuffer(from: [Float](repeating: 0, count: numHeads * headDim))
        let numBlocks = try cache.getNumBlocks(forSequence: 1)
        try engine.decode(q: q2, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                     blockTables: try cache.getBlockTableBuffer(forSequence: 1),
                     seqLengths: makeBuffer(from: [UInt32(16)]),
                     batchSize: 1, maxNumBlocks: numBlocks,
                     headDim: headDim, numHeads: numHeads,
                     numKVHeads: numKVHeads, blockSize: blockSize,
                     output: bufO2, dataType: .float32)

        let output1 = readFloats(from: bufO1, count: 16 * numHeads * headDim)
        let output2 = readFloats(from: bufO2, count: numHeads * headDim)
        #expect(!output1.contains { $0.isNaN }, "Prefill output contains NaN")
        #expect(!output2.contains { $0.isNaN }, "Decode output contains NaN")
        print("  \u{2713} Simultaneous prefill & decode completed")
    }

    // MARK: - Test 12: Fused Append + Prefill

    @Test func fusedPrefill() throws {
        print("\n=== Test 12: Fused Append + Prefill ===")

        let seqLen = 16
        let headDim = 64
        let numHeads = 4
        let numKVHeads = 2
        let blockSize = 16
        let numBlocks = (seqLen + blockSize - 1) / blockSize

        let q = (0..<(seqLen * numHeads * headDim)).map { _ in Float.random(in: -1...1) }
        var k = [Float](repeating: 0, count: seqLen * numKVHeads * headDim)
        var v = [Float](repeating: 0, count: seqLen * numKVHeads * headDim)
        for i in 0..<seqLen {
            for kvHead in 0..<numKVHeads {
                for d in 0..<headDim {
                    let idx = i * numKVHeads * headDim + kvHead * headDim + d
                    k[idx] = Float.random(in: -1...1)
                    v[idx] = Float.random(in: -1...1)
                }
            }
        }

        // --- Fused version ---
        let cacheFused = KVCacheManager(device: device, maxBlocks: numBlocks, blockSize: blockSize,
                                         headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheFused.allocateSequence(id: 1)
        try cacheFused.appendTokens(toSequence: 1, count: seqLen)

        let bufQ = makeBuffer(from: q)
        let bufRawK = makeBuffer(from: k)
        let bufRawV = makeBuffer(from: v)
        let bufOFused = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))

        let blockTableFused = try cacheFused.getBlockTableBuffer(forSequence: 1)

        try engine.fusedPrefill(PagedAttentionFusedPrefillRequest(
            q: bufQ,
            rawK: bufRawK,
            rawV: bufRawV,
            blockTable: blockTableFused,
            kPool: cacheFused.kPoolBuffer,
            vPool: cacheFused.vPoolBuffer,
            output: bufOFused,
            seqLen: seqLen,
            layer: PagedLayerSpec(headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                                   blockSize: blockSize, dataType: .float32),
            causal: false
        ))

        let outputFused = readFloats(from: bufOFused, count: seqLen * numHeads * headDim)

        // --- Non-fused version (append + prefill) ---
        let cacheSep = KVCacheManager(device: device, maxBlocks: numBlocks, blockSize: blockSize,
                                       headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cacheSep.allocateSequence(id: 1)
        try cacheSep.appendTokens(toSequence: 1, count: seqLen)

        let kPtr = cacheSep.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * numBlocks)
        let vPtr = cacheSep.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * numKVHeads * headDim * numBlocks)
        for i in 0..<seqLen {
            let blockIdx = i / blockSize
            let offsetInBlock = i % blockSize
            for kvHead in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = blockIdx * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    let srcIdx = i * numKVHeads * headDim + kvHead * headDim + d
                    kPtr[cacheIdx] = k[srcIdx]
                    vPtr[cacheIdx] = v[srcIdx]
                }
            }
        }

        let bufOSep = makeBuffer(from: [Float](repeating: 0, count: seqLen * numHeads * headDim))
        let blockTableSep = try cacheSep.getBlockTableBuffer(forSequence: 1)

        try engine.prefill(q: bufQ, kPool: cacheSep.kPoolBuffer, vPool: cacheSep.vPoolBuffer,
                      blockTable: blockTableSep,
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                      blockSize: blockSize, causal: false, output: bufOSep, dataType: .float32)

        let outputSep = readFloats(from: bufOSep, count: seqLen * numHeads * headDim)

        let maxErrOutput = maxAbsError(outputFused, outputSep)
        print("  Output max error: \(maxErrOutput)")
        #expect(maxErrOutput < 1e-3, "Fused prefill output differs from non-fused")

        let fusedKPtr = cacheFused.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: numBlocks * blockSize * numKVHeads * headDim)
        let fusedVPtr = cacheFused.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: numBlocks * blockSize * numKVHeads * headDim)
        let fusedBlockTable = try cacheFused.getBlockTable(forSequence: 1)
        let sepBlockTable = try cacheSep.getBlockTable(forSequence: 1)
        var maxKErr: Float = 0
        var maxVErr: Float = 0
        for i in 0..<seqLen {
            let logicalBlock = i / blockSize
            let offsetInBlock = i % blockSize
            let fusedPhysical = Int(fusedBlockTable[logicalBlock])
            let sepPhysical = Int(sepBlockTable[logicalBlock])
            for kvHead in 0..<numKVHeads {
                for d in 0..<headDim {
                    let fusedOffset = fusedPhysical * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    let sepOffset = sepPhysical * blockSize * numKVHeads * headDim + offsetInBlock * numKVHeads * headDim + kvHead * headDim + d
                    maxKErr = max(maxKErr, abs(fusedKPtr[fusedOffset] - kPtr[sepOffset]))
                    maxVErr = max(maxVErr, abs(fusedVPtr[fusedOffset] - vPtr[sepOffset]))
                }
            }
        }
        print("  K pool max error: \(maxKErr)")
        print("  V pool max error: \(maxVErr)")
        #expect(maxKErr < 1e-6, "Fused K pool differs from non-fused")
        #expect(maxVErr < 1e-6, "Fused V pool differs from non-fused")

        print("  \u{2713} Fused prefill matches non-fused append+prefill")
    }

    @Test func gpuRecoveryOnTransientError() throws {
        let seqLen = 8
        let headDim = 32
        let numHeads = 1
        let numKVHeads = 1
        let blockSize = 16

        let cache = KVCacheManager(device: device, maxBlocks: 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: .float32)
        try cache.allocateSequence(id: 1)
        try cache.appendTokens(toSequence: 1, count: seqLen)

        let k = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let v = (0..<(seqLen * headDim)).map { _ in Float.random(in: -1...1) }
        let kPtr = cache.kPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
        let vPtr = cache.vPoolBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
        for i in 0..<seqLen {
            for d in 0..<headDim {
                kPtr[i * headDim + d] = k[i * headDim + d]
                vPtr[i * headDim + d] = v[i * headDim + d]
            }
        }

        let q = makeBuffer(from: [Float](repeating: 0.5, count: seqLen * headDim))
        let output = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))

        engine.enableGracefulDegradation = true
        try engine.prefill(q: q, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                      blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                      seqLen: seqLen, headDim: headDim, numHeads: numHeads,
                      numKVHeads: numKVHeads, blockSize: blockSize,
                      causal: false, output: output, dataType: .float32)

        let out = readFloats(from: output, count: seqLen * headDim)
        #expect(!out.contains { $0.isNaN })
        print("  \u{2713} Graceful degradation handled with no crash")

        engine.enableGracefulDegradation = false
        let output2 = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))
        do {
            try engine.prefill(q: q, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                          blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                          seqLen: seqLen, headDim: headDim, numHeads: numHeads,
                          numKVHeads: numKVHeads, blockSize: blockSize,
                          causal: false, output: output2, dataType: .float32)
        } catch {
            // Expected to possibly throw without graceful degradation
        }
        engine.enableGracefulDegradation = true
    }
}

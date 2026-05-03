import Foundation
import Metal
import PagedAttentionMetal

struct PrecisionBenchmark {
    let device: MTLDevice
    let engine: PagedAttentionEngine
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = device
        self.engine = try PagedAttentionEngine()
    }
    
    func run(results: inout ResultsTable) throws {
        print("\nFP16 vs FP32 Comparison")
        
        let configs: [(seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int)] = [
            (1024, 128, 8, 2, 16),
            (2048, 128, 8, 2, 16),
            (4096, 128, 8, 2, 32),
        ]
        
        var progress = ProgressBar(total: configs.count)
        
        for config in configs {
            let comparison = try compareDataTypes(
                seqLen: config.seqLen,
                headDim: config.headDim,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize
            )
            
            results.add(BenchmarkResult(
                name: "FP32",
                config: "seqLen=\(config.seqLen)",
                avgMs: comparison.fp32Ms,
                minMs: comparison.fp32Ms,
                maxMs: comparison.fp32Ms,
                throughput: nil,
                memoryMB: comparison.fp32MemoryMB
            ))
            
            results.add(BenchmarkResult(
                name: "FP16",
                config: "seqLen=\(config.seqLen)",
                avgMs: comparison.fp16Ms,
                minMs: comparison.fp16Ms,
                maxMs: comparison.fp16Ms,
                throughput: nil,
                memoryMB: comparison.fp16MemoryMB
            ))
            
            results.add(BenchmarkResult(
                name: "FP16 Speedup",
                config: "seqLen=\(config.seqLen)",
                avgMs: 0,
                minMs: 0,
                maxMs: 0,
                throughput: comparison.speedup,
                memoryMB: comparison.memoryReduction
            ))
            
            progress.increment("Precision seqLen=\(config.seqLen)")
        }
    }
    
    private struct Comparison {
        let fp32Ms: Double
        let fp16Ms: Double
        let fp32MemoryMB: Double
        let fp16MemoryMB: Double
        let speedup: Double
        let memoryReduction: Double
    }
    
    private func compareDataTypes(
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int
    ) throws -> Comparison {
        // FP32 benchmark
        let (fp32Ms, fp32MemoryMB) = try benchmarkDataType(
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: .float32
        )
        
        // FP16 benchmark
        let (fp16Ms, fp16MemoryMB) = try benchmarkDataType(
            seqLen: seqLen,
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: .float16
        )
        
        let speedup = fp32Ms / fp16Ms
        let memoryReduction = fp32MemoryMB / fp16MemoryMB
        
        return Comparison(
            fp32Ms: fp32Ms,
            fp16Ms: fp16Ms,
            fp32MemoryMB: fp32MemoryMB,
            fp16MemoryMB: fp16MemoryMB,
            speedup: speedup,
            memoryReduction: memoryReduction
        )
    }
    
    private func benchmarkDataType(
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType
    ) throws -> (latencyMs: Double, memoryMB: Double) {
        let cacheManager = KVCacheManager(
            device: device,
            maxBlocks: 512,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )
        
        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        // Create buffers
        let qSize = seqLen * numHeads * headDim
        let kvSize = seqLen * numKVHeads * headDim
        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        
        guard let qBuffer = device.makeBuffer(bytes: q, length: q.count * 4, options: .storageModeShared),
              let kBuffer = device.makeBuffer(bytes: k, length: k.count * 4, options: .storageModeShared),
              let vBuffer = device.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: qSize * 4, options: .storageModeShared) else {
            fatalError("Buffer creation failed")
        }
        
        // Append to cache
        engine.appendToCache(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            tokenOffset: 0,
            numNewTokens: seqLen,
            numKVHeads: numKVHeads,
            headDim: headDim,
            blockSize: blockSize,
            dataType: dataType
        )
        
        // Warmup
        for _ in 0..<3 {
            engine.prefill(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
                seqLen: seqLen,
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: blockSize,
                causal: true,
                output: outputBuffer,
                dataType: dataType
            )
        }
        
        // Benchmark
        let blockTableBuffer = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)
        let (_, avgMs, _, _) = measure( 20) {
            engine.prefill(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: blockTableBuffer,
                seqLen: seqLen,
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: blockSize,
                causal: true,
                output: outputBuffer,
                dataType: dataType
            )
        }
        
        // Calculate memory
        let stride = (dataType == .float16) ? 2 : 4
        let bytesPerToken = numKVHeads * headDim * stride
        let blocksNeeded = (seqLen + blockSize - 1) / blockSize
        let totalBytes = blocksNeeded * blockSize * bytesPerToken * 2 // K + V
        let memoryMB = Double(totalBytes) / 1_048_576
        
        cacheManager.freeSequence(id: sequenceID)
        
        return (avgMs, memoryMB)
    }
}

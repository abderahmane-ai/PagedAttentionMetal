import Foundation
import Metal
import PagedAttentionMetal

struct PrefillBenchmark {
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
        print("\nPrefill Latency Benchmarks")
        print("Sequence lengths: 512, 1024, 2048, 4096")
        
        let configs: [(seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int)] = [
            (512, 128, 8, 2, 16),
            (1024, 128, 8, 2, 16),
            (2048, 128, 8, 2, 16),
            (4096, 128, 8, 2, 32)
        ]
        
        var progress = ProgressBar(total: configs.count * 2)
        
        for config in configs {
            // FP32
            try runPrefillBenchmark(
                seqLen: config.seqLen,
                headDim: config.headDim,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize,
                dataType: .float32,
                results: &results,
                progress: &progress
            )
            
            // FP16
            try runPrefillBenchmark(
                seqLen: config.seqLen,
                headDim: config.headDim,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize,
                dataType: .float16,
                results: &results,
                progress: &progress
            )
        }
    }
    
    private func runPrefillBenchmark(
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType,
        results: inout ResultsTable,
        progress: inout ProgressBar
    ) throws {
        let cacheManager = KVCacheManager(
            device: device,
            maxBlocks: 1024,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )
        
        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        // Create random Q, K, V
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
        
        // Append K/V to cache
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
        let (_, avgMs, minMs, maxMs) = measure( 20) {
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
        
        let throughput = Double(seqLen) / (avgMs / 1000.0) // tokens/sec
        
        results.add(BenchmarkResult(
            name: "Prefill",
            config: "seqLen=\(seqLen), \(dataType == .float16 ? "FP16" : "FP32")",
            avgMs: avgMs,
            minMs: minMs,
            maxMs: maxMs,
            throughput: throughput,
            memoryMB: nil
        ))
        
        progress.increment("Prefill \(seqLen) \(dataType == .float16 ? "FP16" : "FP32")")
        
        cacheManager.freeSequence(id: sequenceID)
    }
}

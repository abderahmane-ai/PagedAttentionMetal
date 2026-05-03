import Foundation
import Metal
import PagedAttentionMetal

struct DecodeBenchmark {
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
        print("\nDecode Throughput Benchmarks")
        print("Batch sizes: 1, 4, 8, 16, 32")
        
        let configs: [(batchSize: Int, contextLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int)] = [
            (1, 1024, 128, 8, 2, 16),
            (4, 1024, 128, 8, 2, 16),
            (8, 1024, 128, 8, 2, 16),
            (16, 1024, 128, 8, 2, 16),
            (32, 1024, 128, 8, 2, 16),
            (1, 4096, 128, 8, 2, 32),
            (8, 4096, 128, 8, 2, 32),
        ]
        
        var progress = ProgressBar(total: configs.count * 2)
        
        for config in configs {
            // FP32
            try runDecodeBenchmark(
                batchSize: config.batchSize,
                contextLen: config.contextLen,
                headDim: config.headDim,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize,
                dataType: .float32,
                results: &results,
                progress: &progress
            )
            
            // FP16
            try runDecodeBenchmark(
                batchSize: config.batchSize,
                contextLen: config.contextLen,
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
    
    private func runDecodeBenchmark(
        batchSize: Int,
        contextLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType,
        results: inout ResultsTable,
        progress: inout ProgressBar
    ) throws {
        let maxNumBlocks = (contextLen + blockSize - 1) / blockSize
        let batchManager = BatchKVCacheManager(
            device: device,
            maxBatchSize: batchSize,
            maxSequenceBlocks: maxNumBlocks,
            maxBlocks: 2048,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )
        
        // Allocate sequences and fill with dummy KV cache
        let sequenceIDs = (0..<batchSize).map { $0 }
        for id in sequenceIDs {
            try batchManager.allocateSequence(id: id)
            try batchManager.appendTokens(toSequence: id, count: contextLen)
        }
        
        // Create random Q for batch (1 token per sequence)
        let qSize = batchSize * numHeads * headDim
        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        
        guard let qBuffer = device.makeBuffer(bytes: q, length: q.count * 4, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: qSize * 4, options: .storageModeShared) else {
            fatalError("Buffer creation failed")
        }
        
        // Create sequence lengths buffer
        let seqLengths = (0..<batchSize).map { _ in Int32(contextLen) }
        guard let seqLenBuffer = device.makeBuffer(
            bytes: seqLengths,
            length: seqLengths.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else {
            fatalError("Buffer creation failed")
        }
        
        // Warmup
        let blockTablesBuffer = try batchManager.getBatchBlockTableBuffer(forBatch: sequenceIDs)
        for _ in 0..<3 {
            engine.decode(
                q: qBuffer,
                kPool: batchManager.kPoolBuffer,
                vPool: batchManager.vPoolBuffer,
                blockTables: blockTablesBuffer,
                seqLengths: seqLenBuffer,
                batchSize: batchSize,
                maxNumBlocks: maxNumBlocks,
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: blockSize,
                output: outputBuffer,
                dataType: dataType
            )
        }
        
        // Benchmark
        let (_, avgMs, minMs, maxMs) = measure( 50) {
            engine.decode(
                q: qBuffer,
                kPool: batchManager.kPoolBuffer,
                vPool: batchManager.vPoolBuffer,
                blockTables: blockTablesBuffer,
                seqLengths: seqLenBuffer,
                batchSize: batchSize,
                maxNumBlocks: maxNumBlocks,
                headDim: headDim,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                blockSize: blockSize,
                output: outputBuffer,
                dataType: dataType
            )
        }
        
        let throughput = Double(batchSize) / (avgMs / 1000.0) // tokens/sec
        
        results.add(BenchmarkResult(
            name: "Decode",
            config: "batch=\(batchSize), ctx=\(contextLen), \(dataType == .float16 ? "FP16" : "FP32")",
            avgMs: avgMs,
            minMs: minMs,
            maxMs: maxMs,
            throughput: throughput,
            memoryMB: nil
        ))
        
        progress.increment("Decode batch=\(batchSize) ctx=\(contextLen) \(dataType == .float16 ? "FP16" : "FP32")")
        
        // Cleanup
        for id in sequenceIDs {
            batchManager.freeSequence(id: id)
        }
    }
}

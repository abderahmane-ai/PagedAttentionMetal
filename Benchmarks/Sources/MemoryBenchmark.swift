import Foundation
import Metal
import PagedAttentionMetal

struct MemoryBenchmark {
    let device: MTLDevice

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = device
    }

    func run(results: inout ResultsTable, config: BenchmarkConfig? = nil) throws {
        if let config {
            let mem = try measurePagedMemory(
                numSequences: config.batchSize,
                avgSeqLen: config.seqLen,
                headDim: config.headDim,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize,
                dataType: config.dataType
            )
            results.add(BenchmarkResult(
                name: "Memory",
                config: "batch=\(config.batchSize), seqLen=\(config.seqLen), \(config.dataType == .float32 ? "FP32" : config.dataType == .float16 ? "FP16" : "FP8")",
                memoryMB: mem
            ))
            return
        }

        try runConcurrencyVariants(results: &results)
        try runDataTypeComparison(results: &results)
        try runPrefixCacheSavings(results: &results)
    }

    private func runConcurrencyVariants(results: inout ResultsTable) throws {
        let concurrencyLevels = [1, 2, 4, 8, 16]
        var progress = ProgressBar(total: concurrencyLevels.count)
        for numSeqs in concurrencyLevels {
            let mem = try measurePagedMemory(
                numSequences: numSeqs,
                avgSeqLen: 2048,
                headDim: 128,
                numKVHeads: 2,
                blockSize: 16,
                dataType: .float16
            )
            results.add(BenchmarkResult(
                name: "MemConcurrency",
                config: "seqs=\(numSeqs)",
                memoryMB: mem
            ))
            progress.increment("seqs=\(numSeqs)")
        }
    }

    private func runDataTypeComparison(results: inout ResultsTable) throws {
        let types: [PagedAttentionDataType] = [.float32, .float16, .float8]
        var progress = ProgressBar(total: types.count)
        for dtype in types {
            let mem = try measurePagedMemory(
                numSequences: 10,
                avgSeqLen: 1024,
                headDim: 128,
                numKVHeads: 2,
                blockSize: 16,
                dataType: dtype
            )
            let label = dtype == .float32 ? "FP32" : dtype == .float16 ? "FP16" : "FP8"
            results.add(BenchmarkResult(
                name: "MemDataType",
                config: label,
                memoryMB: mem
            ))
            progress.increment(label)
        }
    }

    private func runPrefixCacheSavings(results: inout ResultsTable) throws {
        var progress = ProgressBar(total: 3)

        let cacheManager1 = KVCacheManager(
            device: device,
            maxBlocks: 1024,
            blockSize: 16,
            headDim: 128,
            numKVHeads: 2,
            dataType: .float16
        )
        for id in 0..<5 {
            try cacheManager1.allocateSequence(id: id)
            try cacheManager1.appendTokens(toSequence: id, count: 1024)
        }
        let stats1 = cacheManager1.memoryStats()
        results.add(BenchmarkResult(
            name: "MemPrefix",
            config: "no-sharing",
            memoryMB: Double(stats1.usedMemoryBytes) / 1_048_576
        ))
        progress.increment("no prefix cache")

        let prefixBlocks = 32
        let cacheManager2 = KVCacheManager(
            device: device,
            maxBlocks: 1024,
            blockSize: 16,
            headDim: 128,
            numKVHeads: 2,
            dataType: .float16
        )
        try cacheManager2.allocateSequence(id: 0)
        try cacheManager2.appendTokens(toSequence: 0, count: 1024)
        let blocks0 = try cacheManager2.getBlockTable(forSequence: 0)
        for i in 0..<min(prefixBlocks, blocks0.count) {
            cacheManager2.registerBlockHash(UInt64(i), physicalBlock: blocks0[i])
        }
        for id in 1..<5 {
            try cacheManager2.allocateSequence(id: id)
            try cacheManager2.appendTokens(toSequence: id, count: 1024, blockHashes: Array(0..<UInt64(prefixBlocks)))
        }
        let stats2 = cacheManager2.memoryStats()
        results.add(BenchmarkResult(
            name: "MemPrefix",
            config: "prefix-512",
            memoryMB: Double(stats2.usedMemoryBytes) / 1_048_576
        ))
        progress.increment("prefix cache")

        let savings = stats1.usedMemoryBytes > 0 ?
            (Double(stats1.usedMemoryBytes) - Double(stats2.usedMemoryBytes)) / Double(stats1.usedMemoryBytes) * 100 : 0
        results.add(BenchmarkResult(
            name: "MemPrefix",
            config: "savings-pct",
            memoryMB: savings
        ))

        for id in 0..<5 { cacheManager1.freeSequence(id: id) }
        for id in 0..<5 { cacheManager2.freeSequence(id: id) }
    }

    private func measurePagedMemory(
        numSequences: Int,
        avgSeqLen: Int,
        headDim: Int,
        numKVHeads: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType
    ) throws -> Double {
        let blocksPerSeq = (avgSeqLen + blockSize - 1) / blockSize
        let totalBlocks = numSequences * blocksPerSeq
        let maxBlocks = Int(Double(totalBlocks) * 1.2)

        let cacheManager = KVCacheManager(
            device: device,
            maxBlocks: max(maxBlocks, 1),
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )

        for id in 0..<numSequences {
            try cacheManager.allocateSequence(id: id)
            try cacheManager.appendTokens(toSequence: id, count: avgSeqLen)
        }

        let bytesPerToken = numKVHeads * headDim * dataType.byteWidth
        let bytesPerBlock = blockSize * bytesPerToken
        let totalBytes = max(maxBlocks, 1) * bytesPerBlock * 2
        let memoryMB = Double(totalBytes) / 1_048_576

        for id in 0..<numSequences {
            cacheManager.freeSequence(id: id)
        }

        return memoryMB
    }
}

func runFP8MemoryComparison() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal device not available")
    }
    print("\n=== FP8 vs FP16 Memory Comparison ===")
    let configs: [(numKVHeads: Int, seqLen: Int)] = [
        (2, 1024),
        (4, 4096),
        (8, 8192),
    ]
    for (numKVHeads, seqLen) in configs {
        let blockSize = 16
        let headDim = 128
        let numSequences = 4
        let blocksPerSeq = (seqLen + blockSize - 1) / blockSize
        let totalBlocks = numSequences * blocksPerSeq
        let maxBlocks = Int(Double(totalBlocks) * 1.2)

        for dataType in [PagedAttentionDataType.float16, PagedAttentionDataType.float8] {
            let cache = KVCacheManager(
                device: device,
                maxBlocks: max(maxBlocks, 1),
                blockSize: blockSize,
                headDim: headDim,
                numKVHeads: numKVHeads,
                dataType: dataType
            )
            for id in 0..<numSequences {
                try cache.allocateSequence(id: id)
                try cache.appendTokens(toSequence: id, count: seqLen)
            }
            let bytesPerToken = numKVHeads * headDim * dataType.byteWidth
            let bytesPerBlock = blockSize * bytesPerToken
            let totalBytes = max(maxBlocks, 1) * bytesPerBlock * 2
            let memoryMB = Double(totalBytes) / 1_048_576
            for id in 0..<numSequences {
                cache.freeSequence(id: id)
            }
            let label = dataType == .float16 ? "FP16" : "FP8"
            print("  heads=\(numKVHeads), seqLen=\(seqLen): \(label)=\(String(format: "%.1f", memoryMB))MB")
        }
    }
}

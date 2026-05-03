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
    
    func run(results: inout ResultsTable) throws {
        print("\nMemory Usage Benchmarks")
        print("Comparing paged vs contiguous allocation")
        
        let configs: [(numSequences: Int, avgSeqLen: Int, headDim: Int, numKVHeads: Int, blockSize: Int)] = [
            (10, 1024, 128, 2, 16),
            (20, 1024, 128, 2, 16),
            (50, 512, 128, 2, 16),
        ]
        
        var progress = ProgressBar(total: configs.count * 2)
        
        for config in configs {
            // Paged allocation
            let pagedMB = try measurePagedMemory(
                numSequences: config.numSequences,
                avgSeqLen: config.avgSeqLen,
                headDim: config.headDim,
                numKVHeads: config.numKVHeads,
                blockSize: config.blockSize
            )
            
            results.add(BenchmarkResult(
                name: "Memory (Paged)",
                config: "seqs=\(config.numSequences), len=\(config.avgSeqLen)",
                avgMs: 0,
                minMs: 0,
                maxMs: 0,
                throughput: nil,
                memoryMB: pagedMB
            ))
            
            progress.increment("Paged seqs=\(config.numSequences)")
            
            // Contiguous allocation
            let contiguousMB = try measureContiguousMemory(
                numSequences: config.numSequences,
                avgSeqLen: config.avgSeqLen,
                headDim: config.headDim,
                numKVHeads: config.numKVHeads
            )
            
            results.add(BenchmarkResult(
                name: "Memory (Contiguous)",
                config: "seqs=\(config.numSequences), len=\(config.avgSeqLen)",
                avgMs: 0,
                minMs: 0,
                maxMs: 0,
                throughput: nil,
                memoryMB: contiguousMB
            ))
            
            progress.increment("Contiguous seqs=\(config.numSequences)")
        }
    }
    
    private func measurePagedMemory(
        numSequences: Int,
        avgSeqLen: Int,
        headDim: Int,
        numKVHeads: Int,
        blockSize: Int
    ) throws -> Double {
        // Calculate required blocks
        let blocksPerSeq = (avgSeqLen + blockSize - 1) / blockSize
        let totalBlocks = numSequences * blocksPerSeq
        let maxBlocks = Int(Double(totalBlocks) * 1.2) // 20% overhead for fragmentation
        
        let cacheManager = KVCacheManager(
            device: device,
            maxBlocks: maxBlocks,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: .float16
        )
        
        // Allocate sequences
        for id in 0..<numSequences {
            try cacheManager.allocateSequence(id: id)
            try cacheManager.appendTokens(toSequence: id, count: avgSeqLen)
        }
        
        // Calculate actual memory used
        let bytesPerToken = numKVHeads * headDim * 2 // FP16
        let bytesPerBlock = blockSize * bytesPerToken
        let totalBytes = maxBlocks * bytesPerBlock * 2 // K + V
        let memoryMB = Double(totalBytes) / 1_048_576
        
        // Cleanup
        for id in 0..<numSequences {
            cacheManager.freeSequence(id: id)
        }
        
        return memoryMB
    }
    
    private func measureContiguousMemory(
        numSequences: Int,
        avgSeqLen: Int,
        headDim: Int,
        numKVHeads: Int
    ) throws -> Double {
        // Contiguous allocation: each sequence gets max length buffer
        let maxSeqLen = avgSeqLen * 2 // Assume 2x for worst case
        let bytesPerToken = numKVHeads * headDim * 2 // FP16
        let bytesPerSequence = maxSeqLen * bytesPerToken * 2 // K + V
        let totalBytes = numSequences * bytesPerSequence
        let memoryMB = Double(totalBytes) / 1_048_576
        
        return memoryMB
    }
}

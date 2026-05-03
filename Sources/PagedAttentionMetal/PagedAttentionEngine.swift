import Foundation
import Metal

public enum PagedAttentionError: Error {
    case deviceInitializationFailed
    case libraryInitializationFailed
    case pipelineCreationFailed
}

public enum PagedAttentionDataType {
    case float32
    case float16
}

public class PagedAttentionEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Prefill kernels
    private let singlePassPipeline: MTLComputePipelineState
    private let singlePassPipelineF16: MTLComputePipelineState
    private let splitPhase1Pipeline: MTLComputePipelineState
    private let splitPhase1PipelineF16: MTLComputePipelineState
    private let splitPhase2Pipeline: MTLComputePipelineState
    private let splitPhase2PipelineF16: MTLComputePipelineState
    
    // Decode kernels
    private let decodePipeline: MTLComputePipelineState
    private let decodePipelineF16: MTLComputePipelineState
    
    // KV cache append kernels
    private let appendPipeline: MTLComputePipelineState
    private let appendPipelineF16: MTLComputePipelineState
    
    // Backward kernel
    private let backwardPipeline: MTLComputePipelineState
    
    public var splitThreshold: Int = 1024
    
    public static var defaultLibrary: MTLLibrary {
        // Try bundle first
        if let url = Bundle.module.url(forResource: "kernels", withExtension: "metal"),
           let source = try? String(contentsOf: url),
           let device = MTLCreateSystemDefaultDevice(),
           let library = try? device.makeLibrary(source: source, options: nil) {
            return library
        }
        
        // Fallback: try source directory (for Xcode builds)
        let sourcePath = #filePath.replacingOccurrences(of: "PagedAttentionEngine.swift", with: "kernels.metal")
        if let source = try? String(contentsOfFile: sourcePath),
           let device = MTLCreateSystemDefaultDevice(),
           let library = try? device.makeLibrary(source: source, options: nil) {
            return library
        }
        
        fatalError("kernels.metal not found in bundle or source directory")
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw PagedAttentionError.deviceInitializationFailed
        }
        
        self.device = device
        self.commandQueue = queue
        
        let library = PagedAttentionEngine.defaultLibrary
        
        guard let funcSingle = library.makeFunction(name: "paged_attention_single"),
              let funcSingleF16 = library.makeFunction(name: "paged_attention_single_f16"),
              let funcSplit1 = library.makeFunction(name: "paged_attention_split_phase1"),
              let funcSplit1F16 = library.makeFunction(name: "paged_attention_split_phase1_f16"),
              let funcSplit2 = library.makeFunction(name: "paged_attention_split_phase2"),
              let funcSplit2F16 = library.makeFunction(name: "paged_attention_split_phase2_f16"),
              let funcDecode = library.makeFunction(name: "paged_decode_single"),
              let funcDecodeF16 = library.makeFunction(name: "paged_decode_single_f16"),
              let funcAppend = library.makeFunction(name: "kv_cache_append"),
              let funcAppendF16 = library.makeFunction(name: "kv_cache_append_f16"),
              let funcBackward = library.makeFunction(name: "paged_attention_backward") else {
            throw PagedAttentionError.pipelineCreationFailed
        }
        
        self.singlePassPipeline = try device.makeComputePipelineState(function: funcSingle)
        self.singlePassPipelineF16 = try device.makeComputePipelineState(function: funcSingleF16)
        self.splitPhase1Pipeline = try device.makeComputePipelineState(function: funcSplit1)
        self.splitPhase1PipelineF16 = try device.makeComputePipelineState(function: funcSplit1F16)
        self.splitPhase2Pipeline = try device.makeComputePipelineState(function: funcSplit2)
        self.splitPhase2PipelineF16 = try device.makeComputePipelineState(function: funcSplit2F16)
        self.decodePipeline = try device.makeComputePipelineState(function: funcDecode)
        self.decodePipelineF16 = try device.makeComputePipelineState(function: funcDecodeF16)
        self.appendPipeline = try device.makeComputePipelineState(function: funcAppend)
        self.appendPipelineF16 = try device.makeComputePipelineState(function: funcAppendF16)
        self.backwardPipeline = try device.makeComputePipelineState(function: funcBackward)
    }

    
    // MARK: - Prefill (Process full prompt)
    
    public func prefill(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        causal: Bool = true,
        output: MTLBuffer,
        dataType: PagedAttentionDataType = .float16
    ) {
        if seqLen <= splitThreshold {
            prefillSinglePass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable,
                seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                blockSize: blockSize, causal: causal, output: output, dataType: dataType
            )
        } else {
            prefillSplitPass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable,
                seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                blockSize: blockSize, causal: causal, output: output, dataType: dataType
            )
        }
    }
    
    private func prefillSinglePass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, dataType: PagedAttentionDataType
    ) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        let pipeline = (dataType == .float16) ? singlePassPipelineF16 : singlePassPipeline
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)

        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        var causalVar = UInt32(causal ? 1 : 0)
        
        enc.setBytes(&seqLenVar, length: 4, index: 5)
        enc.setBytes(&headDimVar, length: 4, index: 6)
        enc.setBytes(&numHeadsVar, length: 4, index: 7)
        enc.setBytes(&numKVVar, length: 4, index: 8)
        enc.setBytes(&blockSizeVar, length: 4, index: 9)
        enc.setBytes(&causalVar, length: 4, index: 10)

        let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride
        enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 2)

        let numQTiles = (seqLen + blockSize - 1) / blockSize
        let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
        let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)

        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func prefillSplitPass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, dataType: PagedAttentionDataType
    ) {
        let numBlocks = (seqLen + blockSize - 1) / blockSize
        
        let partialOut = device.makeBuffer(
            length: numHeads * seqLen * numBlocks * headDim * MemoryLayout<Float>.stride,
            options: .storageModePrivate)!
        let partialM = device.makeBuffer(
            length: numHeads * seqLen * numBlocks * MemoryLayout<Float>.stride,
            options: .storageModePrivate)!
        let partialL = device.makeBuffer(
            length: numHeads * seqLen * numBlocks * MemoryLayout<Float>.stride,
            options: .storageModePrivate)!

        guard let cb = commandQueue.makeCommandBuffer() else { return }

        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numBlocksVar = UInt32(numBlocks)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        var causalVar = UInt32(causal ? 1 : 0)

        // Phase 1
        let enc1 = cb.makeComputeCommandEncoder()!
        let p1 = (dataType == .float16) ? splitPhase1PipelineF16 : splitPhase1Pipeline
        enc1.setComputePipelineState(p1)
        enc1.setBuffer(q, offset: 0, index: 0)
        enc1.setBuffer(kPool, offset: 0, index: 1)
        enc1.setBuffer(vPool, offset: 0, index: 2)
        enc1.setBuffer(blockTable, offset: 0, index: 3)
        enc1.setBuffer(partialOut, offset: 0, index: 4)
        enc1.setBuffer(partialM, offset: 0, index: 5)
        enc1.setBuffer(partialL, offset: 0, index: 6)
        enc1.setBytes(&seqLenVar, length: 4, index: 7)
        enc1.setBytes(&headDimVar, length: 4, index: 8)
        enc1.setBytes(&numBlocksVar, length: 4, index: 9)
        enc1.setBytes(&numHeadsVar, length: 4, index: 10)
        enc1.setBytes(&numKVVar, length: 4, index: 11)
        enc1.setBytes(&blockSizeVar, length: 4, index: 12)
        enc1.setBytes(&causalVar, length: 4, index: 13)
        
        // Threadgroup memory for acc_o
        enc1.setThreadgroupMemoryLength(headDim * MemoryLayout<Float>.stride, index: 0)

        let grid1 = MTLSize(width: numBlocks, height: seqLen, depth: numHeads)
        let tgroup1 = MTLSize(width: 1, height: 1, depth: 1)
        enc1.dispatchThreads(grid1, threadsPerThreadgroup: tgroup1)
        enc1.endEncoding()

        // Phase 2
        let enc2 = cb.makeComputeCommandEncoder()!
        let p2 = (dataType == .float16) ? splitPhase2PipelineF16 : splitPhase2Pipeline
        enc2.setComputePipelineState(p2)
        enc2.setBuffer(partialOut, offset: 0, index: 0)
        enc2.setBuffer(partialM, offset: 0, index: 1)
        enc2.setBuffer(partialL, offset: 0, index: 2)
        enc2.setBuffer(output, offset: 0, index: 3)
        enc2.setBytes(&seqLenVar, length: 4, index: 4)
        enc2.setBytes(&headDimVar, length: 4, index: 5)
        enc2.setBytes(&numBlocksVar, length: 4, index: 6)
        enc2.setBytes(&numHeadsVar, length: 4, index: 7)

        let grid2 = MTLSize(width: headDim, height: seqLen, depth: numHeads)
        let tgroup2 = MTLSize(width: headDim, height: 1, depth: 1)
        enc2.dispatchThreads(grid2, threadsPerThreadgroup: tgroup2)
        enc2.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()
    }

    
    // MARK: - Decode (Generate one new token per sequence)
    
    public func decode(
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
        dataType: PagedAttentionDataType = .float16
    ) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        let pipeline = (dataType == .float16) ? decodePipelineF16 : decodePipeline
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTables, offset: 0, index: 3)
        enc.setBuffer(seqLengths, offset: 0, index: 4)
        enc.setBuffer(output, offset: 0, index: 5)

        var batchSizeVar = UInt32(batchSize)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        var maxNumBlocksVar = UInt32(maxNumBlocks)
        
        enc.setBytes(&batchSizeVar, length: 4, index: 6)
        enc.setBytes(&headDimVar, length: 4, index: 7)
        enc.setBytes(&numHeadsVar, length: 4, index: 8)
        enc.setBytes(&numKVVar, length: 4, index: 9)
        enc.setBytes(&blockSizeVar, length: 4, index: 10)
        enc.setBytes(&maxNumBlocksVar, length: 4, index: 11)

        let grid = MTLSize(width: headDim, height: numHeads, depth: batchSize)
        let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    
    // MARK: - KV Cache Append
    
    public func appendToCache(
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
        dataType: PagedAttentionDataType = .float16
    ) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        let pipeline = (dataType == .float16) ? appendPipelineF16 : appendPipeline
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(keys, offset: 0, index: 0)
        enc.setBuffer(values, offset: 0, index: 1)
        enc.setBuffer(kPool, offset: 0, index: 2)
        enc.setBuffer(vPool, offset: 0, index: 3)
        enc.setBuffer(blockTable, offset: 0, index: 4)

        var tokenOffsetVar = UInt32(tokenOffset)
        var numNewTokensVar = UInt32(numNewTokens)
        var numKVHeadsVar = UInt32(numKVHeads)
        var headDimVar = UInt32(headDim)
        var blockSizeVar = UInt32(blockSize)
        
        enc.setBytes(&tokenOffsetVar, length: 4, index: 5)
        enc.setBytes(&numNewTokensVar, length: 4, index: 6)
        enc.setBytes(&numKVHeadsVar, length: 4, index: 7)
        enc.setBytes(&headDimVar, length: 4, index: 8)
        enc.setBytes(&blockSizeVar, length: 4, index: 9)

        let grid = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
        let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    // MARK: - Backward Pass
    
    public func backward(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        dO: MTLBuffer,
        m: MTLBuffer,
        l: MTLBuffer,
        dQ: MTLBuffer,
        dKPool: MTLBuffer,
        dVPool: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int
    ) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(backwardPipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(dO, offset: 0, index: 4)
        enc.setBuffer(m, offset: 0, index: 5)
        enc.setBuffer(l, offset: 0, index: 6)
        enc.setBuffer(dQ, offset: 0, index: 7)
        enc.setBuffer(dKPool, offset: 0, index: 8)
        enc.setBuffer(dVPool, offset: 0, index: 9)

        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        
        enc.setBytes(&seqLenVar, length: 4, index: 10)
        enc.setBytes(&headDimVar, length: 4, index: 11)
        enc.setBytes(&numHeadsVar, length: 4, index: 12)
        enc.setBytes(&numKVVar, length: 4, index: 13)
        enc.setBytes(&blockSizeVar, length: 4, index: 14)

        let grid = MTLSize(width: headDim, height: seqLen, depth: numHeads)
        let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

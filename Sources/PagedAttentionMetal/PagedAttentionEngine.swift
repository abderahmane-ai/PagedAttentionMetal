import Foundation
import Metal

/// Errors that can occur during the initialization or execution of the `PagedAttentionEngine`.
public enum PagedAttentionError: Error {
    /// The Metal system default device could not be created (likely running on an unsupported architecture).
    case deviceInitializationFailed
    /// The Metal shader library could not be dynamically JIT-compiled.
    case libraryInitializationFailed
    /// A compute pipeline state could not be created from a compiled Metal function.
    case pipelineCreationFailed
}

/// Defines the underlying precision of the memory buffers passed to the engine.
public enum PagedAttentionDataType {
    /// Standard 32-bit floating point precision.
    case float32
    /// 16-bit half precision. Uses Mixed-Precision under the hood (memory in Float16, compute in Float32).
    case float16
}

/// A highly optimized, hardware-accelerated orchestrator for Paged Attention on Apple Silicon.
///
/// `PagedAttentionEngine` abstracts away the complexity of GPU compute pipelines, threadgroup memory
/// allocation, and kernel dispatching. It acts as a traffic controller, automatically dynamically 
/// routing your tensors to either standard or V2 Split-K Map-Reduce execution paths based on the 
/// context window length, ensuring maximum ALU saturation.
public class PagedAttentionEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let singlePassPipeline: MTLComputePipelineState
    private let splitPhase1Pipeline: MTLComputePipelineState
    private let splitPhase2Pipeline: MTLComputePipelineState

    private let singlePassPipelineF16: MTLComputePipelineState
    private let splitPhase1PipelineF16: MTLComputePipelineState
    private let splitPhase2PipelineF16: MTLComputePipelineState
    
    /// The context window sequence length threshold that triggers the V2 Split-K kernel.
    ///
    /// If the `seqLen` passed to `forward()` is greater than this threshold, the engine 
    /// will automatically switch from a single-pass threadgroup to the massive scale
    /// Map-Reduce architecture to prevent GPU memory exhaustion. 
    /// Default is `1024`.
    public var splitThreshold: Int = 1024
    
    /// The default JIT-compiled Metal library containing all compute kernels.
    public static var defaultLibrary: MTLLibrary {
        let url = Bundle.module.url(forResource: "kernels", withExtension: "metal")!
        let source = try! String(contentsOf: url)
        return try! MTLCreateSystemDefaultDevice()!.makeLibrary(source: source, options: nil)
    }

    /// Initializes a new Paged Attention hardware engine.
    ///
    /// - Throws: `PagedAttentionError` if the Apple Silicon GPU cannot be accessed or if 
    ///   the Metal compute shaders fail to dynamically compile.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw PagedAttentionError.deviceInitializationFailed
        }
        
        self.device = device
        self.commandQueue = queue
        
        let library = PagedAttentionEngine.defaultLibrary
        
        guard let funcSingle = library.makeFunction(name: "paged_attention_single"),
              let funcSplit1 = library.makeFunction(name: "paged_attention_split_phase1"),
              let funcSplit2 = library.makeFunction(name: "paged_attention_split_phase2"),
              let funcSingleF16 = library.makeFunction(name: "paged_attention_single_f16"),
              let funcSplit1F16 = library.makeFunction(name: "paged_attention_split_phase1_f16"),
              let funcSplit2F16 = library.makeFunction(name: "paged_attention_split_phase2_f16") else {
            throw PagedAttentionError.pipelineCreationFailed
        }
        
        self.singlePassPipeline = try device.makeComputePipelineState(function: funcSingle)
        self.splitPhase1Pipeline = try device.makeComputePipelineState(function: funcSplit1)
        self.splitPhase2Pipeline = try device.makeComputePipelineState(function: funcSplit2)
        
        self.singlePassPipelineF16 = try device.makeComputePipelineState(function: funcSingleF16)
        self.splitPhase1PipelineF16 = try device.makeComputePipelineState(function: funcSplit1F16)
        self.splitPhase2PipelineF16 = try device.makeComputePipelineState(function: funcSplit2F16)
    }
    
    /// Executes the forward attention pass directly on the GPU.
    ///
    /// Automatically dispatches to the single-pass kernel for short sequences and
    /// the V2 Split-K Map-Reduce kernel for sequences longer than `splitThreshold`.
    ///
    /// - Parameters:
    ///   - q: Query buffer — layout `[seq_len, num_heads, head_dim]` (`Float`).
    ///   - kPool: Key cache pool — layout `[num_blocks, block_size, num_kv_heads, head_dim]` (`Float`).
    ///   - vPool: Value cache pool — same layout as `kPool` (`Float`).
    ///   - blockTable: Virtual memory mapping from logical blocks to physical blocks (`Int32`).
    ///   - seqLen: Logical sequence length (context window).
    ///   - headDim: Vector dimension of each attention head.
    ///   - numHeads: Number of query heads. All heads are dispatched in parallel on the GPU.
    ///   - numKVHeads: Number of key/value heads. Set equal to `numHeads` for MHA, 1 for MQA, or any divisor for GQA.
    ///   - numBlocks: Total number of physical KV blocks allocated in the pool.
    ///   - blockSize: Tokens stored per physical block. Default is 16.
    ///   - output: Destination buffer — layout `[seq_len, num_heads, head_dim]` (`Float` or `Float16`).
    ///   - dataType: The underlying buffer memory precision. Default is `.float32`.
    public func forward(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        numBlocks: Int,
        blockSize: Int = 16,
        output: MTLBuffer,
        dataType: PagedAttentionDataType = .float32
    ) {
        if seqLen <= splitThreshold {
            forwardSinglePass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable,
                seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                blockSize: blockSize, output: output, dataType: dataType
            )
        } else {
            forwardSplitPass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable,
                seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads,
                numBlocks: numBlocks, blockSize: blockSize, output: output, dataType: dataType
            )
        }
    }

    private func forwardSinglePass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int, output: MTLBuffer,
        dataType: PagedAttentionDataType
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

        var seqLenVar    = UInt32(seqLen)
        var headDimVar   = UInt32(headDim)
        var numHeadsVar  = UInt32(numHeads)
        var numKVVar     = UInt32(numKVHeads)
        enc.setBytes(&seqLenVar,   length: 4, index: 5)
        enc.setBytes(&headDimVar,  length: 4, index: 6)
        enc.setBytes(&numHeadsVar, length: 4, index: 7)
        enc.setBytes(&numKVVar,    length: 4, index: 8)

        let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride
        enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 2)

        let numQTiles   = (seqLen + blockSize - 1) / blockSize
        let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
        // depth = numHeads: all heads execute in parallel
        let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)

        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func forwardSplitPass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int,
        numBlocks: Int, blockSize: Int, output: MTLBuffer,
        dataType: PagedAttentionDataType
    ) {
        // Intermediate buffers sized for all heads: [num_heads, seq_len, num_blocks, ...]
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

        var seqLenVar    = UInt32(seqLen)
        var headDimVar   = UInt32(headDim)
        var numBlocksVar = UInt32(numBlocks)
        var numHeadsVar  = UInt32(numHeads)
        var numKVVar     = UInt32(numKVHeads)

        // --- Phase 1: per-block partial attention (parallelized across heads via depth) ---
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
        enc1.setBytes(&seqLenVar,    length: 4, index: 7)
        enc1.setBytes(&headDimVar,   length: 4, index: 8)
        enc1.setBytes(&numBlocksVar, length: 4, index: 9)
        enc1.setBytes(&numHeadsVar,  length: 4, index: 10)
        enc1.setBytes(&numKVVar,     length: 4, index: 11)

        let grid1   = MTLSize(width: numBlocks, height: seqLen, depth: numHeads)
        let tgroup1 = MTLSize(width: 1, height: 1, depth: 1)
        enc1.dispatchThreads(grid1, threadsPerThreadgroup: tgroup1)
        enc1.endEncoding()

        // --- Phase 2: global Online Safe Softmax reduction across all blocks ---
        let enc2 = cb.makeComputeCommandEncoder()!
        let p2 = (dataType == .float16) ? splitPhase2PipelineF16 : splitPhase2Pipeline
        enc2.setComputePipelineState(p2)
        enc2.setBuffer(partialOut, offset: 0, index: 0)
        enc2.setBuffer(partialM, offset: 0, index: 1)
        enc2.setBuffer(partialL, offset: 0, index: 2)
        enc2.setBuffer(output, offset: 0, index: 3)
        enc2.setBytes(&seqLenVar,    length: 4, index: 4)
        enc2.setBytes(&headDimVar,   length: 4, index: 5)
        enc2.setBytes(&numBlocksVar, length: 4, index: 6)
        enc2.setBytes(&numHeadsVar,  length: 4, index: 7)

        let grid2   = MTLSize(width: headDim, height: seqLen, depth: numHeads)
        let tgroup2 = MTLSize(width: headDim, height: 1, depth: 1)
        enc2.dispatchThreads(grid2, threadsPerThreadgroup: tgroup2)
        enc2.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()
    }
}

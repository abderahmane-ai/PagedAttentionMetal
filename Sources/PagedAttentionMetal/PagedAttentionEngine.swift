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
              let funcSplit2 = library.makeFunction(name: "paged_attention_split_phase2") else {
            throw PagedAttentionError.pipelineCreationFailed
        }
        
        self.singlePassPipeline = try device.makeComputePipelineState(function: funcSingle)
        self.splitPhase1Pipeline = try device.makeComputePipelineState(function: funcSplit1)
        self.splitPhase2Pipeline = try device.makeComputePipelineState(function: funcSplit2)
    }
    
    /// Executes the forward attention pass directly on the GPU.
    ///
    /// - Parameters:
    ///   - q: A contiguous buffer containing the query vectors (`Float`).
    ///   - kPool: The pre-allocated KV pool buffer storing Key vectors (`Float`).
    ///   - vPool: The pre-allocated KV pool buffer storing Value vectors (`Float`).
    ///   - blockTable: The virtual memory mapping matching logical sequence blocks to physical blocks (`Int32`).
    ///   - seqLen: The logical sequence length (context window).
    ///   - headDim: The vector dimension of each attention head.
    ///   - numBlocks: The total number of physical KV blocks allocated in the pool.
    ///   - blockSize: The number of tokens stored inside each physical memory block. Default is 16.
    ///   - output: The destination buffer where the final attention matrix will be written (`Float`).
    public func forward(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numBlocks: Int,
        blockSize: Int = 16,
        output: MTLBuffer
    ) {
        if seqLen <= splitThreshold {
            forwardSinglePass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable, 
                seqLen: seqLen, headDim: headDim, blockSize: blockSize, output: output
            )
        } else {
            forwardSplitPass(
                q: q, kPool: kPool, vPool: vPool, blockTable: blockTable, 
                seqLen: seqLen, headDim: headDim, numBlocks: numBlocks, blockSize: blockSize, output: output
            )
        }
    }
    
    private func forwardSinglePass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, blockSize: Int, output: MTLBuffer
    ) {
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        
        enc.setComputePipelineState(singlePassPipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)
        
        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        enc.setBytes(&seqLenVar, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.setBytes(&headDimVar, length: MemoryLayout<UInt32>.stride, index: 6)
        
        let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride
        enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 2)
        
        let numQTiles = (seqLen + blockSize - 1) / blockSize
        let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
        let threadgroups = MTLSize(width: 1, height: numQTiles, depth: 1)
        
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    
    private func forwardSplitPass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numBlocks: Int, blockSize: Int, output: MTLBuffer
    ) {
        // Allocate intermediate buffers for Phase 1
        let partialOut = device.makeBuffer(length: seqLen * numBlocks * headDim * MemoryLayout<Float>.stride, options: .storageModePrivate)!
        let partialM = device.makeBuffer(length: seqLen * numBlocks * MemoryLayout<Float>.stride, options: .storageModePrivate)!
        let partialL = device.makeBuffer(length: seqLen * numBlocks * MemoryLayout<Float>.stride, options: .storageModePrivate)!
        
        guard let cb = commandQueue.makeCommandBuffer() else { return }
        
        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numBlocksVar = UInt32(numBlocks)
        
        // --- Phase 1 ---
        let enc1 = cb.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(splitPhase1Pipeline)
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
        
        let grid1 = MTLSize(width: numBlocks, height: seqLen, depth: 1)
        let tgroup1 = MTLSize(width: 1, height: 1, depth: 1)
        enc1.dispatchThreads(grid1, threadsPerThreadgroup: tgroup1)
        enc1.endEncoding()
        
        // --- Phase 2 ---
        let enc2 = cb.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(splitPhase2Pipeline)
        enc2.setBuffer(partialOut, offset: 0, index: 0)
        enc2.setBuffer(partialM, offset: 0, index: 1)
        enc2.setBuffer(partialL, offset: 0, index: 2)
        enc2.setBuffer(output, offset: 0, index: 3)
        enc2.setBytes(&seqLenVar, length: 4, index: 4)
        enc2.setBytes(&headDimVar, length: 4, index: 5)
        enc2.setBytes(&numBlocksVar, length: 4, index: 6)
        
        let grid2 = MTLSize(width: headDim, height: seqLen, depth: 1)
        let tgroup2 = MTLSize(width: headDim, height: 1, depth: 1)
        enc2.dispatchThreads(grid2, threadsPerThreadgroup: tgroup2)
        enc2.endEncoding()
        
        cb.commit()
        cb.waitUntilCompleted()
    }
}

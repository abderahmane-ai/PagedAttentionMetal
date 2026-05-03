import Foundation
import Metal

/// Errors thrown by the KV Cache Manager during sequence operations.
public enum KVCacheError: Error {
    case outOfMemory
    case sequenceNotFound
    case sequenceAlreadyExists
}

/// Represents a logical generation sequence in the LLM runtime.
public struct LogicalSequence {
    public let id: Int
    /// The list of physical GPU memory blocks assigned to this sequence.
    public internal(set) var blockTable: [Int32]
    /// The current total number of tokens generated in this sequence.
    public internal(set) var sequenceLength: Int
    
    public init(id: Int) {
        self.id = id
        self.blockTable = []
        self.sequenceLength = 0
    }
}

/// The Operating System for Apple Silicon GPU Memory.
///
/// `KVCacheManager` dynamically allocates, tracks, and frees physical GPU memory blocks
/// for multiple concurrent LLM generation sequences. It completely abstracts away the 
/// complex virtual-to-physical PagedAttention block mapping.
public class KVCacheManager {
    public let device: MTLDevice
    
    /// The total number of tokens each physical block can hold.
    public let blockSize: Int
    public let headDim: Int
    public let numKVHeads: Int
    public let dataType: PagedAttentionDataType
    
    /// The maximum number of physical blocks available in the GPU pool.
    public let maxBlocks: Int
    
    /// The contiguous GPU memory buffer holding the Key Cache.
    public let kPoolBuffer: MTLBuffer
    /// The contiguous GPU memory buffer holding the Value Cache.
    public let vPoolBuffer: MTLBuffer
    
    /// A stack representing available physical blocks. Popping from the end is O(1).
    private var freeBlocks: [Int32]
    
    /// The active sequences currently being processed, mapped by their unique ID.
    private var sequences: [Int: LogicalSequence] = [:]
    
    /// Creates a new KV Cache memory manager and permanently allocates the GPU VRAM pool.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - maxBlocks: The total number of physical blocks to allocate in VRAM.
    ///   - blockSize: The number of tokens per block (default is 16).
    ///   - headDim: The vector dimension of each attention head.
    ///   - numKVHeads: The number of Key/Value heads.
    ///   - dataType: The underlying buffer precision (`.float32` or `.float16`).
    public init(
        device: MTLDevice,
        maxBlocks: Int,
        blockSize: Int = 16,
        headDim: Int,
        numKVHeads: Int,
        dataType: PagedAttentionDataType = .float16
    ) {
        self.device = device
        self.maxBlocks = maxBlocks
        self.blockSize = blockSize
        self.headDim = headDim
        self.numKVHeads = numKVHeads
        self.dataType = dataType
        
        let stride = (dataType == .float16) ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        let poolBytes = maxBlocks * blockSize * numKVHeads * headDim * stride
        
        self.kPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared)!
        self.vPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared)!
        
        // Initialize the free list with all available blocks (from maxBlocks - 1 down to 0)
        // We push them in reverse so popping from the end yields 0, 1, 2...
        self.freeBlocks = (0..<Int32(maxBlocks)).reversed()
    }
    
    /// Registers a new empty LLM sequence.
    ///
    /// - Parameter id: A unique integer identifying this user/generation.
    /// - Throws: `KVCacheError.sequenceAlreadyExists` if the ID is already active.
    public func allocateSequence(id: Int) throws {
        guard sequences[id] == nil else {
            throw KVCacheError.sequenceAlreadyExists
        }
        sequences[id] = LogicalSequence(id: id)
    }
    
    /// Appends new tokens to an active sequence, dynamically reserving GPU memory if necessary.
    ///
    /// If the sequence crosses a block boundary (e.g. going from 16 to 17 tokens), 
    /// the manager automatically pops a free block from VRAM and adds it to the sequence's Block Table.
    ///
    /// - Parameters:
    ///   - id: The active sequence ID.
    ///   - count: The number of new tokens being appended.
    /// - Throws: `KVCacheError.sequenceNotFound` or `KVCacheError.outOfMemory`.
    public func appendTokens(toSequence id: Int, count: Int) throws {
        guard var sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        
        let currentLen = sequence.sequenceLength
        let newLen = currentLen + count
        
        let blocksNeededNow = (currentLen + blockSize - 1) / blockSize
        let blocksNeededAfter = (newLen + blockSize - 1) / blockSize
        
        let additionalBlocksRequired = blocksNeededAfter - blocksNeededNow
        
        if additionalBlocksRequired > 0 {
            guard freeBlocks.count >= additionalBlocksRequired else {
                throw KVCacheError.outOfMemory
            }
            
            for _ in 0..<additionalBlocksRequired {
                let physicalBlockId = freeBlocks.removeLast()
                sequence.blockTable.append(physicalBlockId)
            }
        }
        
        sequence.sequenceLength = newLen
        sequences[id] = sequence
    }
    
    /// Terminates a sequence and immediately recycles all its GPU memory back into the Free Pool.
    ///
    /// - Parameter id: The sequence ID.
    public func freeSequence(id: Int) {
        if let sequence = sequences.removeValue(forKey: id) {
            // Recycle physical blocks back to the free stack
            for blockId in sequence.blockTable {
                freeBlocks.append(blockId)
            }
        }
    }
    
    /// Retrieves the Block Table array for a sequence so it can be passed to the Metal Engine.
    ///
    /// - Parameter id: The sequence ID.
    /// - Returns: An array of `Int32` representing physical block allocations.
    /// - Throws: `KVCacheError.sequenceNotFound`.
    public func getBlockTable(forSequence id: Int) throws -> [Int32] {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence.blockTable
    }
    
    /// Creates a zero-copy MTLBuffer of the Block Table for direct GPU dispatch.
    ///
    /// - Parameter id: The sequence ID.
    /// - Returns: A shared `MTLBuffer` containing the sequence's block mapping.
    public func getBlockTableBuffer(forSequence id: Int) throws -> MTLBuffer {
        let table = try getBlockTable(forSequence: id)
        
        // Even if empty, we create a 4-byte buffer to prevent Metal API crashing
        let bytes = table.isEmpty ? [Int32(0)] : table
        
        var buffer: MTLBuffer!
        bytes.withUnsafeBytes { ptr in
            buffer = device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: bytes.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        }
        return buffer
    }
    
    /// Returns the active logical sequence.
    public func getSequence(id: Int) throws -> LogicalSequence {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence
    }
    
    /// Returns the number of unallocated physical GPU blocks.
    public var availableBlocks: Int {
        return freeBlocks.count
    }
}

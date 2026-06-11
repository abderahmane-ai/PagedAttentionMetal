import Foundation
import Metal
import os

/// Errors thrown by the KV Cache Manager during sequence operations.
public enum KVCacheError: Error {
    /// No free physical blocks are available for allocation.
    case outOfMemory
    /// The specified sequence ID does not exist.
    case sequenceNotFound
    /// The specified sequence ID is already registered.
    case sequenceAlreadyExists
    /// The request parameters are invalid.
    case invalidRequest(String)
}

/// Strategy for selecting which block to evict when the free list is reorganised.
public enum BlockEvictionPolicy: String, Sendable {
    /// First-in, first-out eviction.
    case fifo
    /// Last-in, first-out eviction.
    case lifo
    /// Evict the block whose size best matches the allocation request.
    case bestFit
}

public struct KVCacheMemoryStats: Sendable, Equatable {
    public let totalBlocks: Int
    public let usedBlocks: Int
    public let freeBlocks: Int
    public let activeSequences: Int
    public let totalMemoryBytes: Int
    public let usedMemoryBytes: Int
    public let partiallyFilledBlocks: Int
    public let fragmentationRatio: Float
    public let prefixCacheHits: Int
    public let prefixCacheEntries: Int
    public let sharedBlocks: Int
    public let evictionPolicy: BlockEvictionPolicy
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
public class KVCacheManager: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "kvcache")

    /// The Metal device used for buffer allocation.
    public let device: MTLDevice
    
    /// The total number of tokens each physical block can hold.
    public let blockSize: Int
    /// The dimension of each attention head.
    public let headDim: Int
    /// The number of key/value attention heads.
    public let numKVHeads: Int
    /// The element data type of the KV cache buffers.
    public let dataType: PagedAttentionDataType
    
    /// The maximum number of physical blocks available in the GPU pool.
    public let maxBlocks: Int
    
    /// The contiguous GPU memory buffer holding the Key Cache.
    public let kPoolBuffer: MTLBuffer
    /// The contiguous GPU memory buffer holding the Value Cache.
    public let vPoolBuffer: MTLBuffer
    
    /// A stack representing available physical blocks. Popping from the end is O(1).
    private var freeBlocks: [Int32]
    /// The block eviction policy used when returning blocks to the free pool.
    public var evictionPolicy: BlockEvictionPolicy = .fifo {
        didSet { rebuildFreeList() }
    }
    private var freeBlockSizes: [Int32: Int] = [:]

    /// The active sequences currently being processed, mapped by their unique ID.
    private var sequences: [Int: LogicalSequence] = [:]

    // MARK: - Prefix Cache

    /// Maps content hash → physical block index for prefix sharing
    private var prefixCache: [UInt64: Int32]
    /// Refcount per physical block (1 = exclusive, >1 = shared)
    private var blockRefCounts: [Int32]
    /// Total prefix cache hits since creation
    private(set) var prefixCacheHits: Int = 0

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
        self.freeBlocks = (0..<Int32(maxBlocks)).reversed()
        self.prefixCache = [:]
        self.blockRefCounts = [Int32](repeating: 0, count: maxBlocks)
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
    /// If the sequence crosses a block boundary, a free block is popped from VRAM.
    /// When `blockHashes` is provided, the prefix cache is consulted: if a block with
    /// a matching hash already exists, the existing block is shared (COW happens on write).
    ///
    /// - Parameters:
    ///   - id: The active sequence ID.
    ///   - count: The number of new tokens being appended.
    ///   - blockHashes: Optional per-new-block hashes for prefix cache matching.
    /// - Throws: `KVCacheError.sequenceNotFound` or `KVCacheError.outOfMemory`.
    public func appendTokens(toSequence id: Int, count: Int, blockHashes: [UInt64]? = nil) throws {
        guard count >= 0 else {
            throw KVCacheError.invalidRequest("count must be non-negative")
        }
        if count == 0 { return }
        guard var sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        
        let currentLen = sequence.sequenceLength
        let newLen = currentLen + count
        
        let blocksNeededNow = (currentLen + blockSize - 1) / blockSize
        let blocksNeededAfter = (newLen + blockSize - 1) / blockSize
        
        let additionalBlocksRequired = blocksNeededAfter - blocksNeededNow
        
        // COW: if the last existing block is shared and we'll write to it, duplicate it
        if blocksNeededNow > 0 && currentLen % blockSize != 0 {
            let lastBlockIndex = blocksNeededNow - 1
            let lastBlock = sequence.blockTable[lastBlockIndex]
            if blockRefCounts[Int(lastBlock)] > 1 {
                try copyOnWrite(lastBlock, at: lastBlockIndex, in: &sequence)
            }
        }
        
        if additionalBlocksRequired > 0 {
            guard freeBlocks.count >= additionalBlocksRequired else {
                os_log(.error, log: Self.log, "OOM: sequence %d requested %d blocks, only %d available", id, additionalBlocksRequired, freeBlocks.count)
                throw KVCacheError.outOfMemory
            }
            
            for i in 0..<additionalBlocksRequired {
                
                // Check prefix cache if hashes are provided
                if let hashes = blockHashes, i < hashes.count {
                    let blockHash = hashes[i]
                    if let existingBlock = prefixCache[blockHash] {
                        // Prefix cache hit — share the existing block
                        sequence.blockTable.append(existingBlock)
                        blockRefCounts[Int(existingBlock)] += 1
                        prefixCacheHits += 1
                        os_log(.debug, log: Self.log, "Prefix cache hit: block %d reused for hash %llu", existingBlock, blockHash)
                        continue
                    }
                }
                
                let physicalBlockId = allocateBlock()
                sequence.blockTable.append(physicalBlockId)
                blockRefCounts[Int(physicalBlockId)] = 1
                
                // Insert into prefix cache if hash provided
                if let hashes = blockHashes, i < hashes.count {
                    let blockHash = hashes[i]
                    prefixCache[blockHash] = physicalBlockId
                }
            }
        }
        
        sequence.sequenceLength = newLen
        sequences[id] = sequence
    }
    
    // MARK: - Copy-on-Write
    
    private func copyOnWrite(_ physicalBlock: Int32, at index: Int, in sequence: inout LogicalSequence) throws {
        guard !freeBlocks.isEmpty else {
            throw KVCacheError.outOfMemory
        }
        let newBlock = allocateBlock()
        let bytesPerBlock = blockSize * numKVHeads * headDim * dataType.byteWidth
        
        let srcK = kPoolBuffer.contents().advanced(by: Int(physicalBlock) * bytesPerBlock)
        let dstK = kPoolBuffer.contents().advanced(by: Int(newBlock) * bytesPerBlock)
        dstK.copyMemory(from: srcK, byteCount: bytesPerBlock)
        
        let srcV = vPoolBuffer.contents().advanced(by: Int(physicalBlock) * bytesPerBlock)
        let dstV = vPoolBuffer.contents().advanced(by: Int(newBlock) * bytesPerBlock)
        dstV.copyMemory(from: srcV, byteCount: bytesPerBlock)
        
        blockRefCounts[Int(physicalBlock)] -= 1
        blockRefCounts[Int(newBlock)] = 1
        sequence.blockTable[index] = newBlock
    }
    
    // MARK: - Prefix Cache Registration
    
    /// Registers a block hash for future prefix sharing.
    /// Call this after a new block has been fully written to the pool.
    /// - Parameters:
    ///   - hash: The content hash of the block.
    ///   - physicalBlock: The physical block index to associate with the hash.
    public func registerBlockHash(_ hash: UInt64, physicalBlock: Int32) {
        prefixCache[hash] = physicalBlock
    }
    
    /// Performs copy-on-write for a shared physical block.
    /// Call before writing to a block that may be shared.
    /// - Parameters:
    ///   - sequenceId: The sequence that needs exclusive access.
    ///   - logicalBlockIndex: The logical block index within the sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` or `KVCacheError.outOfMemory`.
    public func ensureExclusiveAccess(sequenceId: Int, logicalBlockIndex: Int) throws {
        guard var sequence = sequences[sequenceId] else {
            throw KVCacheError.sequenceNotFound
        }
        guard logicalBlockIndex < sequence.blockTable.count else {
            throw KVCacheError.invalidRequest("block index out of range")
        }
        let physicalBlock = sequence.blockTable[logicalBlockIndex]
        if blockRefCounts[Int(physicalBlock)] > 1 {
            try copyOnWrite(physicalBlock, at: logicalBlockIndex, in: &sequence)
        }
        sequences[sequenceId] = sequence
    }
    
    /// Terminates a sequence and recycles its GPU memory.
    ///
    /// Blocks with refcount > 1 (shared via prefix cache) are only decremented,
    /// not freed. Blocks whose refcount reaches 0 are returned to the free pool.
    ///
    /// - Parameter id: The sequence ID.
    public func freeSequence(id: Int) {
        if let sequence = sequences.removeValue(forKey: id) {
            let totalBlocks = sequence.blockTable.count
            let remainder = sequence.sequenceLength % blockSize
            let lastTokens = remainder == 0 ? blockSize : remainder
            for (i, blockId) in sequence.blockTable.enumerated() {
                let idx = Int(blockId)
                blockRefCounts[idx] -= 1
                if blockRefCounts[idx] <= 0 {
                    switch evictionPolicy {
                    case .fifo:
                        freeBlocks.insert(blockId, at: 0)
                    case .lifo:
                        freeBlocks.append(blockId)
                    case .bestFit:
                        freeBlockSizes[blockId] = i < totalBlocks - 1 ? blockSize : lastTokens
                        freeBlocks.append(blockId)
                    }
                }
            }
        }
    }
    
    private func allocateBlock() -> Int32 {
        switch evictionPolicy {
        case .fifo, .lifo:
            return freeBlocks.removeLast()
        case .bestFit:
            return allocateBestFit()
        }
    }
    
    private func allocateBestFit() -> Int32 {
        var bestIdx = 0
        var bestDelta = Int.max
        for i in freeBlocks.indices {
            let blockId = freeBlocks[i]
            let held = freeBlockSizes[blockId, default: blockSize]
            let delta = abs(held - blockSize)
            if delta < bestDelta {
                bestDelta = delta
                bestIdx = i
            }
        }
        let blockId = freeBlocks.remove(at: bestIdx)
        freeBlockSizes.removeValue(forKey: blockId)
        return blockId
    }
    
    private func rebuildFreeList() {
        switch evictionPolicy {
        case .fifo:
            freeBlocks.reverse()
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .lifo:
            freeBlocks.reverse()
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .bestFit:
            for blockId in freeBlocks {
                if freeBlockSizes[blockId] == nil {
                    freeBlockSizes[blockId] = blockSize
                }
            }
            freeBlocks.sort { freeBlockSizes[$0, default: 0] > freeBlockSizes[$1, default: 0] }
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
    
    /// Returns the active logical sequence for a given ID.
    /// - Parameter id: The sequence ID.
    /// - Returns: The logical sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getSequence(id: Int) throws -> LogicalSequence {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence
    }

    /// Returns the current token length of a sequence.
    /// - Parameter id: The sequence ID.
    /// - Returns: The number of tokens in the sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getSequenceLength(_ id: Int) throws -> Int {
        try getSequence(id: id).sequenceLength
    }

    /// Returns the number of physical blocks allocated to a sequence.
    /// - Parameter id: The sequence ID.
    /// - Returns: The block count.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getNumBlocks(forSequence id: Int) throws -> Int {
        try getSequence(id: id).blockTable.count
    }
    
    /// Returns the number of unallocated physical GPU blocks.
    public var availableBlocks: Int {
        return freeBlocks.count
    }

    /// The number of active (allocated) sequences currently tracked.
    public var activeSequenceCount: Int {
        sequences.count
    }

    /// Returns detailed memory usage and fragmentation statistics.
    public func memoryStats() -> KVCacheMemoryStats {
        let usedBlocks = maxBlocks - freeBlocks.count
        let bytesPerBlock = blockSize * numKVHeads * headDim * dataType.byteWidth * 2
        let partiallyFilled = sequences.values.reduce(0) { count, sequence in
            let remainder = sequence.sequenceLength % blockSize
            return count + ((remainder > 0 && remainder < blockSize) ? 1 : 0)
        }
        let active = max(sequences.count, 1)
        let sharedBlocks = blockRefCounts.filter { $0 > 1 }.count
        return KVCacheMemoryStats(
            totalBlocks: maxBlocks,
            usedBlocks: usedBlocks,
            freeBlocks: freeBlocks.count,
            activeSequences: sequences.count,
            totalMemoryBytes: maxBlocks * bytesPerBlock,
            usedMemoryBytes: usedBlocks * bytesPerBlock,
            partiallyFilledBlocks: partiallyFilled,
            fragmentationRatio: Float(partiallyFilled) / Float(active),
            prefixCacheHits: prefixCacheHits,
            prefixCacheEntries: prefixCache.count,
            sharedBlocks: sharedBlocks,
            evictionPolicy: evictionPolicy
        )
    }
}

// MARK: - Batch KV Cache Manager

/// Manages KV cache for multiple concurrent sequences with batch processing support.
public class BatchKVCacheManager: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "batchkvcache")

    /// The Metal device used for buffer allocation.
    public let device: MTLDevice
    /// The maximum number of sequences that can be batched together.
    public let maxBatchSize: Int
    /// The maximum number of physical blocks any single sequence can hold.
    public let maxSequenceBlocks: Int
    /// The number of tokens per physical block.
    public let blockSize: Int
    /// The dimension of each attention head.
    public let headDim: Int
    /// The number of key/value attention heads.
    public let numKVHeads: Int
    /// The element data type of the KV cache buffers.
    public let dataType: PagedAttentionDataType
    /// The total number of physical blocks in the pool.
    public let maxBlocks: Int
    
    /// The contiguous GPU memory buffer holding the Key Cache.
    public let kPoolBuffer: MTLBuffer
    /// The contiguous GPU memory buffer holding the Value Cache.
    public let vPoolBuffer: MTLBuffer
    
    /// Pre-allocated slab buffer for batched block tables [maxBatchSize × maxSequenceBlocks × Int32].
    public let blockTablesSlab: MTLBuffer
    /// Pre-allocated slab buffer for per-sequence lengths [maxBatchSize × UInt32].
    public let seqLengthsSlab: MTLBuffer
    
    private var freeBlocks: [Int32]
    /// The block eviction policy used when returning blocks to the free pool.
    public var evictionPolicy: BlockEvictionPolicy = .fifo {
        didSet { rebuildFreeList() }
    }
    private var freeBlockSizes: [Int32: Int] = [:]
    private var sequences: [Int: LogicalSequence] = [:]

    // MARK: - Prefix Cache

    private var prefixCache: [UInt64: Int32]
    private var blockRefCounts: [Int32]
    private(set) var prefixCacheHits: Int = 0

    /// Creates a new batch KV cache manager and allocates GPU memory pools.
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - maxBatchSize: Maximum batch size for batched operations.
    ///   - maxSequenceBlocks: Maximum blocks per sequence.
    ///   - maxBlocks: Total number of physical blocks in the pool.
    ///   - blockSize: Number of tokens per block.
    ///   - headDim: Dimension of each attention head.
    ///   - numKVHeads: Number of key/value heads.
    ///   - dataType: Element data type for the cache buffers.
    public init(
        device: MTLDevice,
        maxBatchSize: Int,
        maxSequenceBlocks: Int,
        maxBlocks: Int,
        blockSize: Int = 16,
        headDim: Int,
        numKVHeads: Int,
        dataType: PagedAttentionDataType = .float16
    ) {
        self.device = device
        self.maxBatchSize = maxBatchSize
        self.maxSequenceBlocks = maxSequenceBlocks
        self.maxBlocks = maxBlocks
        self.blockSize = blockSize
        self.headDim = headDim
        self.numKVHeads = numKVHeads
        self.dataType = dataType
        
        let stride = (dataType == .float16) ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        let poolBytes = maxBlocks * blockSize * numKVHeads * headDim * stride
        
        self.kPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared)!
        self.vPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared)!
        
        self.freeBlocks = (0..<Int32(maxBlocks)).reversed()
        self.prefixCache = [:]
        self.blockRefCounts = [Int32](repeating: 0, count: maxBlocks)
        
        let slabSize = maxBatchSize * maxSequenceBlocks * MemoryLayout<Int32>.stride
        self.blockTablesSlab = device.makeBuffer(length: slabSize, options: .storageModeShared)!
        self.seqLengthsSlab = device.makeBuffer(length: maxBatchSize * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    }
    
    /// Registers a new empty sequence with the batch cache manager.
    /// - Parameter id: A unique integer identifying the sequence.
    /// - Throws: `KVCacheError.sequenceAlreadyExists` if the ID is already active.
    public func allocateSequence(id: Int) throws {
        guard sequences[id] == nil else {
            throw KVCacheError.sequenceAlreadyExists
        }
        sequences[id] = LogicalSequence(id: id)
    }
    
    /// Appends new tokens to a sequence, dynamically allocating physical blocks as needed.
    /// - Parameters:
    ///   - id: The active sequence ID.
    ///   - count: The number of new tokens being appended.
    ///   - blockHashes: Optional per-block hashes for prefix cache matching.
    /// - Throws: `KVCacheError.sequenceNotFound` or `KVCacheError.outOfMemory`.
    public func appendTokens(toSequence id: Int, count: Int, blockHashes: [UInt64]? = nil) throws {
        guard count >= 0 else {
            throw KVCacheError.invalidRequest("count must be non-negative")
        }
        if count == 0 { return }
        guard var sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        
        let currentLen = sequence.sequenceLength
        let newLen = currentLen + count
        
        let blocksNeededNow = (currentLen + blockSize - 1) / blockSize
        let blocksNeededAfter = (newLen + blockSize - 1) / blockSize
        
        let additionalBlocksRequired = blocksNeededAfter - blocksNeededNow
        
        // COW: if the last existing block is shared, duplicate it before write
        if blocksNeededNow > 0 && currentLen % blockSize != 0 {
            let lastBlockIndex = blocksNeededNow - 1
            let lastBlock = sequence.blockTable[lastBlockIndex]
            if blockRefCounts[Int(lastBlock)] > 1 {
                try copyOnWrite(lastBlock, at: lastBlockIndex, in: &sequence)
            }
        }
        
        if additionalBlocksRequired > 0 {
            guard freeBlocks.count >= additionalBlocksRequired else {
                os_log(.error, log: Self.log, "OOM: sequence %d requested %d blocks, only %d available", id, additionalBlocksRequired, freeBlocks.count)
                throw KVCacheError.outOfMemory
            }
            
            for i in 0..<additionalBlocksRequired {
                
                if let hashes = blockHashes, i < hashes.count {
                    let blockHash = hashes[i]
                    if let existingBlock = prefixCache[blockHash] {
                        sequence.blockTable.append(existingBlock)
                        blockRefCounts[Int(existingBlock)] += 1
                        prefixCacheHits += 1
                        os_log(.debug, log: Self.log, "Prefix cache hit (batch): block %d reused for hash %llu", existingBlock, blockHash)
                        continue
                    }
                }
                
                let physicalBlockId = allocateBlock()
                sequence.blockTable.append(physicalBlockId)
                blockRefCounts[Int(physicalBlockId)] = 1
                
                if let hashes = blockHashes, i < hashes.count {
                    let blockHash = hashes[i]
                    prefixCache[blockHash] = physicalBlockId
                }
            }
        }
        
        sequence.sequenceLength = newLen
        sequences[id] = sequence
    }
    
    private func copyOnWrite(_ physicalBlock: Int32, at index: Int, in sequence: inout LogicalSequence) throws {
        guard !freeBlocks.isEmpty else {
            throw KVCacheError.outOfMemory
        }
        let newBlock = allocateBlock()
        let bytesPerBlock = blockSize * numKVHeads * headDim * dataType.byteWidth
        
        let srcK = kPoolBuffer.contents().advanced(by: Int(physicalBlock) * bytesPerBlock)
        let dstK = kPoolBuffer.contents().advanced(by: Int(newBlock) * bytesPerBlock)
        dstK.copyMemory(from: srcK, byteCount: bytesPerBlock)
        
        let srcV = vPoolBuffer.contents().advanced(by: Int(physicalBlock) * bytesPerBlock)
        let dstV = vPoolBuffer.contents().advanced(by: Int(newBlock) * bytesPerBlock)
        dstV.copyMemory(from: srcV, byteCount: bytesPerBlock)
        
        blockRefCounts[Int(physicalBlock)] -= 1
        blockRefCounts[Int(newBlock)] = 1
        sequence.blockTable[index] = newBlock
    }
    
    /// Registers a block hash for future prefix sharing in the batch cache.
    /// - Parameters:
    ///   - hash: The content hash of the block.
    ///   - physicalBlock: The physical block index.
    public func registerBlockHash(_ hash: UInt64, physicalBlock: Int32) {
        prefixCache[hash] = physicalBlock
    }
    
    /// Performs copy-on-write for a shared physical block in the batch cache.
    /// - Parameters:
    ///   - sequenceId: The sequence that needs exclusive access.
    ///   - logicalBlockIndex: The logical block index within the sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` or `KVCacheError.outOfMemory`.
    public func ensureExclusiveAccess(sequenceId: Int, logicalBlockIndex: Int) throws {
        guard var sequence = sequences[sequenceId] else {
            throw KVCacheError.sequenceNotFound
        }
        guard logicalBlockIndex < sequence.blockTable.count else {
            throw KVCacheError.invalidRequest("block index out of range")
        }
        let physicalBlock = sequence.blockTable[logicalBlockIndex]
        if blockRefCounts[Int(physicalBlock)] > 1 {
            try copyOnWrite(physicalBlock, at: logicalBlockIndex, in: &sequence)
        }
        sequences[sequenceId] = sequence
    }
    
    /// Terminates a sequence and recycles its GPU memory blocks.
    /// - Parameter id: The sequence ID.
    public func freeSequence(id: Int) {
        if let sequence = sequences.removeValue(forKey: id) {
            let totalBlocks = sequence.blockTable.count
            let remainder = sequence.sequenceLength % blockSize
            let lastTokens = remainder == 0 ? blockSize : remainder
            for (i, blockId) in sequence.blockTable.enumerated() {
                let idx = Int(blockId)
                blockRefCounts[idx] -= 1
                if blockRefCounts[idx] <= 0 {
                    switch evictionPolicy {
                    case .fifo:
                        freeBlocks.insert(blockId, at: 0)
                    case .lifo:
                        freeBlocks.append(blockId)
                    case .bestFit:
                        freeBlockSizes[blockId] = i < totalBlocks - 1 ? blockSize : lastTokens
                        freeBlocks.append(blockId)
                    }
                }
            }
        }
    }
    
    private func allocateBlock() -> Int32 {
        switch evictionPolicy {
        case .fifo, .lifo:
            return freeBlocks.removeLast()
        case .bestFit:
            return allocateBestFit()
        }
    }
    
    private func allocateBestFit() -> Int32 {
        var bestIdx = 0
        var bestDelta = Int.max
        for i in freeBlocks.indices {
            let blockId = freeBlocks[i]
            let held = freeBlockSizes[blockId, default: blockSize]
            let delta = abs(held - blockSize)
            if delta < bestDelta {
                bestDelta = delta
                bestIdx = i
            }
        }
        let blockId = freeBlocks.remove(at: bestIdx)
        freeBlockSizes.removeValue(forKey: blockId)
        return blockId
    }
    
    private func rebuildFreeList() {
        switch evictionPolicy {
        case .fifo:
            freeBlocks.reverse()
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .lifo:
            freeBlocks.reverse()
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .bestFit:
            for blockId in freeBlocks {
                if freeBlockSizes[blockId] == nil {
                    freeBlockSizes[blockId] = blockSize
                }
            }
            freeBlocks.sort { freeBlockSizes[$0, default: 0] > freeBlockSizes[$1, default: 0] }
        }
    }
    
    /// Returns the active logical sequence for a given ID.
    /// - Parameter id: The sequence ID.
    /// - Returns: The logical sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getSequence(id: Int) throws -> LogicalSequence {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence
    }

    /// Returns the current token length of a sequence.
    /// - Parameter id: The sequence ID.
    /// - Returns: The number of tokens in the sequence.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getSequenceLength(_ id: Int) throws -> Int {
        try getSequence(id: id).sequenceLength
    }

    /// Returns the number of physical blocks allocated to a sequence.
    /// - Parameter id: The sequence ID.
    /// - Returns: The block count.
    /// - Throws: `KVCacheError.sequenceNotFound` if the ID is not registered.
    public func getNumBlocks(forSequence id: Int) throws -> Int {
        try getSequence(id: id).blockTable.count
    }
    
    /// Returns a flat [batchSize × maxSequenceBlocks] MTLBuffer for GPU dispatch.
    /// - Parameter ids: Array of sequence IDs in batch order.
    /// - Returns: A shared MTLBuffer containing the concatenated block tables.
    /// - Throws: `KVCacheError.outOfMemory` or `KVCacheError.sequenceNotFound`.
    public func getBatchBlockTableBuffer(forBatch ids: [Int]) throws -> MTLBuffer {
        guard ids.count <= maxBatchSize else {
            throw KVCacheError.outOfMemory
        }
        
        let ptr = blockTablesSlab.contents().assumingMemoryBound(to: Int32.self)
        let slabWidth = maxSequenceBlocks
        
        for (batchIdx, seqId) in ids.enumerated() {
            guard let sequence = sequences[seqId] else {
                throw KVCacheError.sequenceNotFound
            }
            guard sequence.blockTable.count <= maxSequenceBlocks else {
                throw KVCacheError.invalidRequest("sequence \(seqId) requires \(sequence.blockTable.count) blocks, maxSequenceBlocks is \(maxSequenceBlocks)")
            }
            let offset = batchIdx * slabWidth
            for (blockIdx, physicalBlock) in sequence.blockTable.enumerated() {
                ptr[offset + blockIdx] = physicalBlock
            }
            // Zero out remaining entries
            for blockIdx in sequence.blockTable.count..<maxSequenceBlocks {
                ptr[offset + blockIdx] = 0
            }
        }
        
        return blockTablesSlab
    }
    
    /// Returns an MTLBuffer containing current lengths for the given batch of sequences.
    /// - Parameter ids: Array of sequence IDs in batch order.
    /// - Returns: A shared MTLBuffer of UInt32 sequence lengths.
    /// - Throws: `KVCacheError.sequenceNotFound` if any ID is not registered.
    public func getSeqLengthsBuffer(forBatch ids: [Int]) throws -> MTLBuffer {
        let ptr = seqLengthsSlab.contents().assumingMemoryBound(to: UInt32.self)
        
        for (idx, seqId) in ids.enumerated() {
            guard let sequence = sequences[seqId] else {
                throw KVCacheError.sequenceNotFound
            }
            ptr[idx] = UInt32(sequence.sequenceLength)
        }
        
        return seqLengthsSlab
    }
    
    /// Returns the number of unallocated physical GPU blocks.
    public var availableBlocks: Int {
        return freeBlocks.count
    }

    /// The number of active (allocated) sequences currently tracked.
    public var activeSequenceCount: Int {
        sequences.count
    }

    /// Returns detailed memory usage and fragmentation statistics.
    public func memoryStats() -> KVCacheMemoryStats {
        let usedBlocks = maxBlocks - freeBlocks.count
        let bytesPerBlock = blockSize * numKVHeads * headDim * dataType.byteWidth * 2
        let partiallyFilled = sequences.values.reduce(0) { count, sequence in
            let remainder = sequence.sequenceLength % blockSize
            return count + ((remainder > 0 && remainder < blockSize) ? 1 : 0)
        }
        let active = max(sequences.count, 1)
        let sharedBlocks = blockRefCounts.filter { $0 > 1 }.count
        return KVCacheMemoryStats(
            totalBlocks: maxBlocks,
            usedBlocks: usedBlocks,
            freeBlocks: freeBlocks.count,
            activeSequences: sequences.count,
            totalMemoryBytes: maxBlocks * bytesPerBlock,
            usedMemoryBytes: usedBlocks * bytesPerBlock,
            partiallyFilledBlocks: partiallyFilled,
            fragmentationRatio: Float(partiallyFilled) / Float(active),
            prefixCacheHits: prefixCacheHits,
            prefixCacheEntries: prefixCache.count,
            sharedBlocks: sharedBlocks,
            evictionPolicy: evictionPolicy
        )
    }
}

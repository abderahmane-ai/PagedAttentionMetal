import Foundation
import Metal
import os
import Collections

public enum KVCacheError: Error {
    case outOfMemory
    case sequenceNotFound
    case sequenceAlreadyExists
    case invalidRequest(String)
}

public enum BlockEvictionPolicy: String, Sendable {
    case fifo
    case lifo
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

public struct LogicalSequence {
    public let id: Int
    public internal(set) var blockTable: [Int32]
    public internal(set) var sequenceLength: Int

    public init(id: Int) {
        self.id = id
        self.blockTable = []
        self.sequenceLength = 0
    }
}

public class KVCacheManager: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "kvcache")

    public let device: MTLDevice

    public let blockSize: Int
    public let headDim: Int
    public let numKVHeads: Int
    public let dataType: PagedAttentionDataType

    public let maxBlocks: Int

    public let kPoolBuffer: MTLBuffer
    public let vPoolBuffer: MTLBuffer

    private let lock = OSAllocatedUnfairLock()

    private var freeBlocks: Deque<Int32>
    public var evictionPolicy: BlockEvictionPolicy = .fifo {
        didSet { rebuildFreeList() }
    }
    private var freeBlockSizes: [Int32: Int] = [:]

    private var sequences: [Int: LogicalSequence] = [:]

    private var prefixCache: [UInt64: Int32]
    private var blockRefCounts: [Int32]
    private(set) var prefixCacheHits: Int = 0

    public init(
        device: MTLDevice,
        maxBlocks: Int,
        blockSize: Int = 16,
        headDim: Int,
        numKVHeads: Int,
        dataType: PagedAttentionDataType = .float16
    ) throws {
        self.device = device
        self.maxBlocks = maxBlocks
        self.blockSize = blockSize
        self.headDim = headDim
        self.numKVHeads = numKVHeads
        self.dataType = dataType

        let stride = dataType.byteWidth
        let poolBytes = maxBlocks * blockSize * numKVHeads * headDim * stride

        guard let kPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared),
              let vPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("Failed to allocate KV pool buffers")
        }
        self.kPoolBuffer = kPoolBuffer
        self.vPoolBuffer = vPoolBuffer

        self.freeBlocks = Deque((0..<Int32(maxBlocks)).reversed())
        self.prefixCache = [:]
        self.blockRefCounts = [Int32](repeating: 0, count: maxBlocks)
    }

    public func allocateSequence(id: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard sequences[id] == nil else {
            throw KVCacheError.sequenceAlreadyExists
        }
        sequences[id] = LogicalSequence(id: id)
    }

    public func appendTokens(toSequence id: Int, count: Int, blockHashes: [UInt64]? = nil) throws {
        lock.lock()
        defer { lock.unlock() }
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
                        os_log(.debug, log: Self.log, "Prefix cache hit: block %d reused for hash %llu", existingBlock, blockHash)
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

    public func registerBlockHash(_ hash: UInt64, physicalBlock: Int32) {
        prefixCache[hash] = physicalBlock
    }

    public func ensureExclusiveAccess(sequenceId: Int, logicalBlockIndex: Int) throws {
        lock.lock()
        defer { lock.unlock() }
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

    public func freeSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
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
                        freeBlocks.prepend(blockId)
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
            freeBlocks = Deque(freeBlocks.reversed())
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .lifo:
            freeBlocks = Deque(freeBlocks.reversed())
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .bestFit:
            for blockId in freeBlocks {
                if freeBlockSizes[blockId] == nil {
                    freeBlockSizes[blockId] = blockSize
                }
            }
            freeBlocks = Deque(freeBlocks.sorted { freeBlockSizes[$0, default: 0] > freeBlockSizes[$1, default: 0] })
        }
    }

    private func _getSequence(id: Int) throws -> LogicalSequence {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence
    }

    private func _getBlockTable(forSequence id: Int) throws -> [Int32] {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence.blockTable
    }

    public func getBlockTable(forSequence id: Int) throws -> [Int32] {
        lock.lock()
        defer { lock.unlock() }
        return try _getBlockTable(forSequence: id)
    }

    public func getBlockTableBuffer(forSequence id: Int) throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        let table = try _getBlockTable(forSequence: id)

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

    public func getSequence(id: Int) throws -> LogicalSequence {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id)
    }

    public func getSequenceLength(_ id: Int) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id).sequenceLength
    }

    public func getNumBlocks(forSequence id: Int) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id).blockTable.count
    }

    public var availableBlocks: Int {
        lock.lock()
        defer { lock.unlock() }
        return freeBlocks.count
    }

    public var activeSequenceCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return sequences.count
    }

    public func memoryStats() -> KVCacheMemoryStats {
        lock.lock()
        defer { lock.unlock() }
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

public class BatchKVCacheManager: @unchecked Sendable {
    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "batchkvcache")

    public let device: MTLDevice
    public let maxBatchSize: Int
    public let maxSequenceBlocks: Int
    public let blockSize: Int
    public let headDim: Int
    public let numKVHeads: Int
    public let dataType: PagedAttentionDataType
    public let maxBlocks: Int

    public let kPoolBuffer: MTLBuffer
    public let vPoolBuffer: MTLBuffer

    public let blockTablesSlab: MTLBuffer
    public let seqLengthsSlab: MTLBuffer

    private let lock = OSAllocatedUnfairLock()

    private var freeBlocks: Deque<Int32>
    public var evictionPolicy: BlockEvictionPolicy = .fifo {
        didSet { rebuildFreeList() }
    }
    private var freeBlockSizes: [Int32: Int] = [:]
    private var sequences: [Int: LogicalSequence] = [:]

    private var prefixCache: [UInt64: Int32]
    private var blockRefCounts: [Int32]
    private(set) var prefixCacheHits: Int = 0

    public init(
        device: MTLDevice,
        maxBatchSize: Int,
        maxSequenceBlocks: Int,
        maxBlocks: Int,
        blockSize: Int = 16,
        headDim: Int,
        numKVHeads: Int,
        dataType: PagedAttentionDataType = .float16
    ) throws {
        self.device = device
        self.maxBatchSize = maxBatchSize
        self.maxSequenceBlocks = maxSequenceBlocks
        self.maxBlocks = maxBlocks
        self.blockSize = blockSize
        self.headDim = headDim
        self.numKVHeads = numKVHeads
        self.dataType = dataType

        let stride = dataType.byteWidth
        let poolBytes = maxBlocks * blockSize * numKVHeads * headDim * stride

        guard let kPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared),
              let vPoolBuffer = device.makeBuffer(length: poolBytes, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("Failed to allocate KV pool buffers")
        }
        self.kPoolBuffer = kPoolBuffer
        self.vPoolBuffer = vPoolBuffer

        self.freeBlocks = Deque((0..<Int32(maxBlocks)).reversed())
        self.prefixCache = [:]
        self.blockRefCounts = [Int32](repeating: 0, count: maxBlocks)

        let slabSize = maxBatchSize * maxSequenceBlocks * MemoryLayout<Int32>.stride
        guard let blockTablesSlab = device.makeBuffer(length: slabSize, options: .storageModeShared),
              let seqLengthsSlab = device.makeBuffer(length: maxBatchSize * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("Failed to allocate batch slab buffers")
        }
        self.blockTablesSlab = blockTablesSlab
        self.seqLengthsSlab = seqLengthsSlab
    }

    public func allocateSequence(id: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard sequences[id] == nil else {
            throw KVCacheError.sequenceAlreadyExists
        }
        sequences[id] = LogicalSequence(id: id)
    }

    public func appendTokens(toSequence id: Int, count: Int, blockHashes: [UInt64]? = nil) throws {
        lock.lock()
        defer { lock.unlock() }
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

    public func registerBlockHash(_ hash: UInt64, physicalBlock: Int32) {
        lock.lock()
        defer { lock.unlock() }
        prefixCache[hash] = physicalBlock
    }

    public func ensureExclusiveAccess(sequenceId: Int, logicalBlockIndex: Int) throws {
        lock.lock()
        defer { lock.unlock() }
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

    public func freeSequence(id: Int) {
        lock.lock()
        defer { lock.unlock() }
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
                        freeBlocks.prepend(blockId)
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
            freeBlocks = Deque(freeBlocks.reversed())
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .lifo:
            freeBlocks = Deque(freeBlocks.reversed())
            freeBlockSizes.removeAll(keepingCapacity: true)
        case .bestFit:
            for blockId in freeBlocks {
                if freeBlockSizes[blockId] == nil {
                    freeBlockSizes[blockId] = blockSize
                }
            }
            freeBlocks = Deque(freeBlocks.sorted { freeBlockSizes[$0, default: 0] > freeBlockSizes[$1, default: 0] })
        }
    }

    private func _getSequence(id: Int) throws -> LogicalSequence {
        guard let sequence = sequences[id] else {
            throw KVCacheError.sequenceNotFound
        }
        return sequence
    }

    public func getSequence(id: Int) throws -> LogicalSequence {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id)
    }

    public func getSequenceLength(_ id: Int) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id).sequenceLength
    }

    public func getNumBlocks(forSequence id: Int) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        return try _getSequence(id: id).blockTable.count
    }

    public func getBatchBlockTableBuffer(forBatch ids: [Int]) throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
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
            for blockIdx in sequence.blockTable.count..<maxSequenceBlocks {
                ptr[offset + blockIdx] = 0
            }
        }

        return blockTablesSlab
    }

    public func getSeqLengthsBuffer(forBatch ids: [Int]) throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        let ptr = seqLengthsSlab.contents().assumingMemoryBound(to: UInt32.self)

        for (idx, seqId) in ids.enumerated() {
            guard let sequence = sequences[seqId] else {
                throw KVCacheError.sequenceNotFound
            }
            ptr[idx] = UInt32(sequence.sequenceLength)
        }

        return seqLengthsSlab
    }

    public var availableBlocks: Int {
        return freeBlocks.count
    }

    public var activeSequenceCount: Int {
        sequences.count
    }

    public func memoryStats() -> KVCacheMemoryStats {
        lock.lock()
        defer { lock.unlock() }
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

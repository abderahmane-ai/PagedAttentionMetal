import Foundation
import Metal

private final class PAMContext {
    let engine: PagedAttentionEngine
    var lastError: UnsafeMutablePointer<CChar>?

    init() throws {
        self.engine = try PagedAttentionEngine()
    }

    deinit {
        if let lastError {
            free(lastError)
        }
    }

    func setError(_ error: Error) -> Int32 {
        if let lastError {
            free(lastError)
        }
        let message: String
        if let pagedError = error as? PagedAttentionError {
            message = pagedError.description
        } else {
            message = error.localizedDescription
        }
        lastError = strdup(message)
        return -1
    }

    func clearError() {
        if let lastError {
            free(lastError)
            self.lastError = nil
        }
    }
}

private final class PAMCache {
    let cache: KVCacheManager

    init(cache: KVCacheManager) {
        self.cache = cache
    }
}

private func context(from pointer: UnsafeMutableRawPointer?) -> PAMContext? {
    guard let pointer else { return nil }
    return Unmanaged<PAMContext>.fromOpaque(pointer).takeUnretainedValue()
}

private func cache(from pointer: UnsafeMutableRawPointer?) -> PAMCache? {
    guard let pointer else { return nil }
    return Unmanaged<PAMCache>.fromOpaque(pointer).takeUnretainedValue()
}

private func metalBuffer(from pointer: UnsafeMutableRawPointer?) -> MTLBuffer? {
    guard let pointer else { return nil }
    let object = Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue()
    return object as? MTLBuffer
}

private func dtype(from raw: Int32) throws -> PagedAttentionDataType {
    guard let dtype = PagedAttentionDataType(rawValue: raw) else {
        throw PagedAttentionError.invalidConfiguration("unknown dtype \(raw); use 0 for float32 or 1 for float16")
    }
    return dtype
}

@_cdecl("pam_create_context")
public func pam_create_context() -> UnsafeMutableRawPointer? {
    do {
        let context = try PAMContext()
        return Unmanaged.passRetained(context).toOpaque()
    } catch {
        return nil
    }
}

@_cdecl("pam_destroy_context")
public func pam_destroy_context(_ contextPointer: UnsafeMutableRawPointer?) {
    guard let contextPointer else { return }
    Unmanaged<PAMContext>.fromOpaque(contextPointer).release()
}

@_cdecl("pam_last_error")
public func pam_last_error(_ contextPointer: UnsafeMutableRawPointer?) -> UnsafePointer<CChar>? {
    context(from: contextPointer)?.lastError.map { UnsafePointer($0) }
}

@_cdecl("pam_create_cache")
public func pam_create_cache(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ maxBlocks: Int32,
    _ blockSize: Int32,
    _ headDim: Int32,
    _ numKVHeads: Int32,
    _ dataType: Int32
) -> UnsafeMutableRawPointer? {
    guard let context = context(from: contextPointer) else { return nil }
    do {
        let dtype = try dtype(from: dataType)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PagedAttentionError.deviceInitializationFailed
        }
        let layer = PagedLayerSpec(
            headDim: Int(headDim),
            numHeads: Int(numKVHeads),
            numKVHeads: Int(numKVHeads),
            blockSize: Int(blockSize),
            dataType: dtype
        )
        try layer.validate()
        guard maxBlocks > 0 else {
            throw PagedAttentionError.invalidConfiguration("maxBlocks must be positive")
        }
        let cache = try KVCacheManager(
            device: device,
            maxBlocks: Int(maxBlocks),
            blockSize: Int(blockSize),
            headDim: Int(headDim),
            numKVHeads: Int(numKVHeads),
            dataType: dtype
        )
        context.clearError()
        return Unmanaged.passRetained(PAMCache(cache: cache)).toOpaque()
    } catch {
        _ = context.setError(error)
        return nil
    }
}

@_cdecl("pam_destroy_cache")
public func pam_destroy_cache(_ cachePointer: UnsafeMutableRawPointer?) {
    guard let cachePointer else { return }
    Unmanaged<PAMCache>.fromOpaque(cachePointer).release()
}

@_cdecl("pam_reserve_sequence")
public func pam_reserve_sequence(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64
) -> Int32 {
    guard let context = context(from: contextPointer), let cache = cache(from: cachePointer) else { return -1 }
    do {
        try cache.cache.allocateSequence(id: Int(sequenceID))
        context.clearError()
        return 0
    } catch {
        return context.setError(error)
    }
}

@_cdecl("pam_append_tokens")
public func pam_append_tokens(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64,
    _ count: Int32
) -> Int32 {
    guard let context = context(from: contextPointer), let cache = cache(from: cachePointer) else { return -1 }
    do {
        try cache.cache.appendTokens(toSequence: Int(sequenceID), count: Int(count))
        context.clearError()
        return 0
    } catch {
        return context.setError(error)
    }
}

@_cdecl("pam_free_sequence")
public func pam_free_sequence(
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64
) {
    cache(from: cachePointer)?.cache.freeSequence(id: Int(sequenceID))
}

@_cdecl("pam_available_blocks")
public func pam_available_blocks(_ cachePointer: UnsafeMutableRawPointer?) -> Int32 {
    guard let cache = cache(from: cachePointer) else { return -1 }
    return Int32(cache.cache.availableBlocks)
}

@_cdecl("pam_sequence_length")
public func pam_sequence_length(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64
) -> Int32 {
    guard let context = context(from: contextPointer), let cache = cache(from: cachePointer) else { return -1 }
    do {
        let length = try cache.cache.getSequenceLength(Int(sequenceID))
        context.clearError()
        return Int32(length)
    } catch {
        return context.setError(error)
    }
}

@_cdecl("pam_append_kv")
public func pam_append_kv(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64,
    _ keysPointer: UnsafeMutableRawPointer?,
    _ valuesPointer: UnsafeMutableRawPointer?,
    _ tokenOffset: Int32,
    _ numNewTokens: Int32,
    _ numHeads: Int32
) -> Int32 {
    guard let context = context(from: contextPointer),
          let cache = cache(from: cachePointer),
          let keys = metalBuffer(from: keysPointer),
          let values = metalBuffer(from: valuesPointer) else {
        return -1
    }
    do {
        let blockTable = try cache.cache.getBlockTableBuffer(forSequence: Int(sequenceID))
        let layer = PagedLayerSpec(
            headDim: cache.cache.headDim,
            numHeads: Int(numHeads),
            numKVHeads: cache.cache.numKVHeads,
            blockSize: cache.cache.blockSize,
            dataType: cache.cache.dataType
        )
        try context.engine.appendToCache(PagedKVAppendRequest(
            keys: keys,
            values: values,
            kPool: cache.cache.kPoolBuffer,
            vPool: cache.cache.vPoolBuffer,
            blockTable: blockTable,
            tokenOffset: Int(tokenOffset),
            numNewTokens: Int(numNewTokens),
            layer: layer
        ))
        context.clearError()
        return 0
    } catch {
        return context.setError(error)
    }
}

@_cdecl("pam_prefill")
public func pam_prefill(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ sequenceID: Int64,
    _ queryPointer: UnsafeMutableRawPointer?,
    _ outputPointer: UnsafeMutableRawPointer?,
    _ seqLen: Int32,
    _ numHeads: Int32,
    _ causal: Bool
) -> Int32 {
    guard let context = context(from: contextPointer),
          let cache = cache(from: cachePointer),
          let query = metalBuffer(from: queryPointer),
          let output = metalBuffer(from: outputPointer) else {
        return -1
    }
    do {
        let layer = PagedLayerSpec(
            headDim: cache.cache.headDim,
            numHeads: Int(numHeads),
            numKVHeads: cache.cache.numKVHeads,
            blockSize: cache.cache.blockSize,
            dataType: cache.cache.dataType
        )
        let blockTable = try cache.cache.getBlockTableBuffer(forSequence: Int(sequenceID))
        try context.engine.prefill(PagedAttentionPrefillRequest(
            q: query,
            kPool: cache.cache.kPoolBuffer,
            vPool: cache.cache.vPoolBuffer,
            blockTable: blockTable,
            output: output,
            seqLen: Int(seqLen),
            layer: layer,
            causal: causal
        ))
        context.clearError()
        return 0
    } catch {
        return context.setError(error)
    }
}

@_cdecl("pam_decode")
public func pam_decode(
    _ contextPointer: UnsafeMutableRawPointer?,
    _ cachePointer: UnsafeMutableRawPointer?,
    _ batchIDsPointer: UnsafePointer<Int64>?,
    _ batchSize: Int32,
    _ queryPointer: UnsafeMutableRawPointer?,
    _ outputPointer: UnsafeMutableRawPointer?,
    _ numHeads: Int32
) -> Int32 {
    guard let context = context(from: contextPointer),
          let cache = cache(from: cachePointer),
          let batchIDsPointer,
          let query = metalBuffer(from: queryPointer),
          let output = metalBuffer(from: outputPointer) else {
        return -1
    }
    do {
        let ids = (0..<Int(batchSize)).map { Int(batchIDsPointer[$0]) }
        let maxNumBlocks = max(1, try ids.map { try cache.cache.getNumBlocks(forSequence: $0) }.max() ?? 1)
        let batchManager = try BatchKVCacheView(
            cache: cache.cache,
            ids: ids,
            maxNumBlocks: maxNumBlocks
        )
        let layer = PagedLayerSpec(
            headDim: cache.cache.headDim,
            numHeads: Int(numHeads),
            numKVHeads: cache.cache.numKVHeads,
            blockSize: cache.cache.blockSize,
            dataType: cache.cache.dataType
        )
        try context.engine.decode(PagedAttentionDecodeRequest(
            q: query,
            kPool: cache.cache.kPoolBuffer,
            vPool: cache.cache.vPoolBuffer,
            blockTables: batchManager.blockTablesBuffer,
            seqLengths: batchManager.seqLengthsBuffer,
            output: output,
            batchSize: Int(batchSize),
            maxNumBlocks: maxNumBlocks,
            layer: layer
        ))
        context.clearError()
        return 0
    } catch {
        return context.setError(error)
    }
}

private struct BatchKVCacheView {
    let blockTablesBuffer: MTLBuffer
    let seqLengthsBuffer: MTLBuffer

    init(cache: KVCacheManager, ids: [Int], maxNumBlocks: Int) throws {
        let batchSize = ids.count
        let slabSize = batchSize * maxNumBlocks * MemoryLayout<Int32>.stride
        guard let blockTables = cache.device.makeBuffer(length: slabSize, options: .storageModeShared),
              let seqLengths = cache.device.makeBuffer(length: batchSize * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("failed to create C ABI batch buffers")
        }
        self.blockTablesBuffer = blockTables
        self.seqLengthsBuffer = seqLengths

        let btPtr = blockTables.contents().assumingMemoryBound(to: Int32.self)
        for (batchIndex, id) in ids.enumerated() {
            let sequence = try cache.getSequence(id: id)
            let offset = batchIndex * maxNumBlocks
            for (blockIndex, physicalBlock) in sequence.blockTable.enumerated() {
                btPtr[offset + blockIndex] = physicalBlock
            }
            // Zero-fill padding slots to prevent out-of-bounds GPU reads
            for blockIndex in sequence.blockTable.count..<maxNumBlocks {
                btPtr[offset + blockIndex] = 0
            }
        }

        let slPtr = seqLengths.contents().assumingMemoryBound(to: UInt32.self)
        for (batchIndex, id) in ids.enumerated() {
            let sequence = try cache.getSequence(id: id)
            slPtr[batchIndex] = UInt32(sequence.sequenceLength)
        }
    }
}

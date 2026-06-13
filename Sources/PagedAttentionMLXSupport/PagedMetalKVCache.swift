import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal

public enum PagedAttentionMLXError: Error, CustomStringConvertible {
    case metalDeviceUnavailable
    case unsupportedShape(String)
    case bufferCreationFailed(String)
    case unsupportedMask(String)

    public var description: String {
        switch self {
        case .metalDeviceUnavailable:
            return "Metal device is unavailable."
        case .unsupportedShape(let message):
            return "Unsupported MLX tensor shape: \(message)"
        case .bufferCreationFailed(let name):
            return "Failed to create Metal buffer for \(name)."
        case .unsupportedMask(let message):
            return "Unsupported MLX attention mask: \(message)"
        }
    }
}

public final class PagedMetalKVCache: KVCache, CustomDebugStringConvertible {
    public let device: MTLDevice
    public let engine: PagedAttentionEngine
    public let sequenceID: Int
    public let layer: PagedLayerSpec
    public let cacheManager: KVCacheManager
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    private let fallbackCache = KVCacheSimple()

    public init(
        sequenceID: Int,
        layer: PagedLayerSpec,
        maxBlocks: Int,
        device: MTLDevice? = MTLCreateSystemDefaultDevice(),
        engine: PagedAttentionEngine? = nil
    ) throws {
        guard let device else {
            throw PagedAttentionMLXError.metalDeviceUnavailable
        }
        try layer.validate()
        self.device = device
        self.engine = try engine ?? PagedAttentionEngine()
        self.sequenceID = sequenceID
        self.layer = layer
        self.cacheManager = try KVCacheManager(
            device: device,
            maxBlocks: maxBlocks,
            blockSize: layer.blockSize,
            headDim: layer.headDim,
            numKVHeads: layer.numKVHeads,
            dataType: layer.dataType
        )
        try cacheManager.allocateSequence(id: sequenceID)
    }

    public func innerState() -> [MLXArray] {
        fallbackCache.innerState()
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = fallbackCache.update(keys: keys, values: values)
        offset = fallbackCache.offset
        return result
    }

    public func pagedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) throws -> MLXArray {
        try validateMask(mask)
        let (batch, numHeads, seqLen, headDim) = try decodeQueryShape(queries)
        guard batch == 1 else {
            throw PagedAttentionMLXError.unsupportedShape("PagedMetalKVCache currently supports batch size 1 for MLX prefill/decode; got \(batch).")
        }
        guard numHeads == layer.numHeads, headDim == layer.headDim else {
            throw PagedAttentionMLXError.unsupportedShape("queries are [\(batch), \(numHeads), \(seqLen), \(headDim)], expected heads=\(layer.numHeads), headDim=\(layer.headDim).")
        }
        let (_, kvHeads, kvSeqLen, kvHeadDim) = try decodeKVShape(keys, name: "keys")
        guard kvHeads == layer.numKVHeads, kvSeqLen == seqLen, kvHeadDim == layer.headDim else {
            throw PagedAttentionMLXError.unsupportedShape("keys/values must match seqLen=\(seqLen), kvHeads=\(layer.numKVHeads), headDim=\(layer.headDim).")
        }

        let currentOffset = offset
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        let keyBuffer = try metalBuffer(from: keys.transposed(0, 2, 1, 3), name: "keys")
        let valueBuffer = try metalBuffer(from: values.transposed(0, 2, 1, 3), name: "values")
        let blockTable = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)

        try engine.appendToCache(PagedKVAppendRequest(
            keys: keyBuffer,
            values: valueBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: blockTable,
            tokenOffset: currentOffset,
            numNewTokens: seqLen,
            layer: layer
        ))

        offset += seqLen
        _ = fallbackCache.update(keys: keys, values: values)

        let queryBuffer = try metalBuffer(from: queries.transposed(0, 2, 1, 3), name: "queries")
        guard let outputBuffer = device.makeBuffer(
            length: seqLen * layer.numHeads * layer.headDim * layer.dataType.byteWidth,
            options: .storageModeShared
        ) else {
            throw PagedAttentionMLXError.bufferCreationFailed("output")
        }

        if seqLen == 1 {
            let batchTable = try batchBlockTableBuffer()
            let seqLengths = [UInt32(offset)]
            guard let seqLengthBuffer = seqLengths.withUnsafeBytes({ (ptr: UnsafeRawBufferPointer) in
                device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)
            }) else {
                throw PagedAttentionMLXError.bufferCreationFailed("sequence lengths")
            }
            try engine.decode(PagedAttentionDecodeRequest(
                q: queryBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTables: batchTable,
                seqLengths: seqLengthBuffer,
                output: outputBuffer,
                batchSize: 1,
                maxNumBlocks: max(1, try cacheManager.getNumBlocks(forSequence: sequenceID)),
                layer: layer
            ))
        } else {
            try engine.prefill(PagedAttentionPrefillRequest(
                q: queryBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: blockTable,
                output: outputBuffer,
                seqLen: seqLen,
                layer: layer,
                causal: maskIsCausal(mask)
            ))
        }

        return try mlxArray(from: outputBuffer, seqLen: seqLen).transposed(0, 2, 1, 3)
    }

    public var state: [MLXArray] {
        get { fallbackCache.state }
        set {
            fallbackCache.state = newValue
            offset = fallbackCache.offset
        }
    }

    public var metaState: [String] {
        get { ["paged-metal", String(sequenceID), String(offset), String(layer.blockSize)] }
        set {
            if newValue.count >= 3, let parsedOffset = Int(newValue[2]) {
                offset = parsedOffset
            }
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        0
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(MLXLMCommon.createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public func copy() -> any KVCache {
        let copied = KVCacheSimple()
        copied.state = fallbackCache.state
        return copied
    }

    public var debugDescription: String {
        "PagedMetalKVCache(sequenceID: \(sequenceID), offset: \(offset), layer: \(layer), stats: \(cacheManager.memoryStats()))"
    }

    private func metalBuffer(from array: MLXArray, name: String) throws -> MTLBuffer {
        guard let buffer = array.asMTLBuffer(device: device, noCopy: false) else {
            throw PagedAttentionMLXError.bufferCreationFailed(name)
        }
        return buffer
    }

    private func mlxArray(from buffer: MTLBuffer, seqLen: Int) throws -> MLXArray {
        let shape = [1, seqLen, layer.numHeads, layer.headDim]
        let data = Data(bytes: buffer.contents(), count: shape.reduce(1, *) * layer.dataType.byteWidth)
        switch layer.dataType {
        case .float32:
            return MLXArray(data, shape, dtype: .float32)
        case .float16:
            return MLXArray(data, shape, dtype: .float16)
        case .float8:
            throw PagedAttentionError.unsupported("MLX support for FP8 not yet implemented")
        }
    }

    private func batchBlockTableBuffer() throws -> MTLBuffer {
        let sequence = try cacheManager.getSequence(id: sequenceID)
        let table = sequence.blockTable.isEmpty ? [Int32(0)] : sequence.blockTable
        let maxBlocks = max(1, try cacheManager.getNumBlocks(forSequence: sequenceID))
        
        // Pad table to maxBlocks
        var paddedTable = table
        if paddedTable.count < maxBlocks {
            paddedTable.append(contentsOf: Array(repeating: Int32(0), count: maxBlocks - paddedTable.count))
        }
        
        guard let buffer = paddedTable.withUnsafeBytes({ (ptr: UnsafeRawBufferPointer) in
            device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)
        }) else {
            throw PagedAttentionMLXError.bufferCreationFailed("block table")
        }
        return buffer
    }
}

public func pagedAttentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    if let cache = cache as? PagedMetalKVCache,
       abs(scale - pow(Float(cache.layer.headDim), -0.5)) < 0.0001,
       let output = try? cache.pagedAttention(queries: queries, keys: keys, values: values, mask: mask) {
        return output
    }
    return MLXLMCommon.attentionWithCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: scale,
        mask: mask
    )
}

private func decodeQueryShape(_ array: MLXArray) throws -> (Int, Int, Int, Int) {
    guard array.shape.count == 4 else {
        throw PagedAttentionMLXError.unsupportedShape("queries must be [B, heads, seqLen, headDim]; got \(array.shape)")
    }
    return (array.dim(0), array.dim(1), array.dim(2), array.dim(3))
}

private func decodeKVShape(_ array: MLXArray, name: String) throws -> (Int, Int, Int, Int) {
    guard array.shape.count == 4 else {
        throw PagedAttentionMLXError.unsupportedShape("\(name) must be [B, kvHeads, seqLen, headDim]; got \(array.shape)")
    }
    return (array.dim(0), array.dim(1), array.dim(2), array.dim(3))
}

private func validateMask(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) throws {
    switch mask {
    case .none, .causal:
        return
    case .array, .arrays:
        throw PagedAttentionMLXError.unsupportedMask("materialized masks fall back to native MLX attention")
    }
}

private func maskIsCausal(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
    switch mask {
    case .causal:
        return true
    case .none:
        return false
    case .array, .arrays:
        return true
    }
}

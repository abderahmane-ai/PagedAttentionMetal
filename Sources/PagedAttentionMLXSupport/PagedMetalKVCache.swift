import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal

/// Errors thrown by the PagedAttention MLX integration layer.
public enum PagedAttentionMLXError: Error, CustomStringConvertible {
    /// The Metal device is unavailable (nil).
    case metalDeviceUnavailable
    /// The MLX tensor shape is not supported.
    case unsupportedShape(String)
    /// Failed to create a Metal buffer from an MLX array.
    case bufferCreationFailed(String)
    /// The attention mask mode is not supported.
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

/// A paged attention KV cache implementation that integrates with MLX's `KVCache` protocol.
///
/// Uses the `PagedAttentionEngine` for GPU-accelerated attention and the `KVCacheManager`
/// for efficient GPU memory management. Falls back to `KVCacheSimple` for state tracking.
public final class PagedMetalKVCache: KVCache, CustomDebugStringConvertible {
    /// The Metal device used for GPU operations.
    public let device: MTLDevice
    /// The underlying paged attention engine.
    public let engine: PagedAttentionEngine
    /// The unique identifier for this sequence.
    public let sequenceID: Int
    /// The attention layer specification.
    public let layer: PagedLayerSpec
    /// The KV cache memory manager.
    public let cacheManager: KVCacheManager
    /// The current token offset in the cache.
    public var offset: Int = 0
    /// Unlimited maximum size for the KV cache.
    public var maxSize: Int? { nil }

    private let fallbackCache = KVCacheSimple()

    /// Creates a new paged Metal KV cache.
    /// Creates a new paged Metal KV cache.
    /// - Parameters:
    ///   - sequenceID: The unique sequence identifier.
    ///   - layer: The attention layer specification.
    ///   - maxBlocks: The maximum number of physical cache blocks.
    ///   - device: The Metal device (defaults to the system default device).
    ///   - engine: An optional pre-configured engine (creates a new one if nil).
    /// - Throws: `PagedAttentionMLXError` or `PagedAttentionError` on failure.
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
        self.cacheManager = KVCacheManager(
            device: device,
            maxBlocks: maxBlocks,
            blockSize: layer.blockSize,
            headDim: layer.headDim,
            numKVHeads: layer.numKVHeads,
            dataType: layer.dataType
        )
        try cacheManager.allocateSequence(id: sequenceID)
    }

    /// Returns the inner state of the fallback cache as MLX arrays.
    public func innerState() -> [MLXArray] {
        fallbackCache.innerState()
    }

    /// Updates the fallback cache with new key/value tensors.
    /// - Parameters:
    ///   - keys: The key tensor.
    ///   - values: The value tensor.
    /// - Returns: A tuple of (keys, values) from the cache.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = fallbackCache.update(keys: keys, values: values)
        offset = fallbackCache.offset
        return result
    }

    /// Executes paged attention for the given queries, automatically selecting prefill or decode.
    /// - Parameters:
    ///   - queries: The query tensor [1, heads, seqLen, headDim].
    ///   - keys: The key tensor [1, kvHeads, seqLen, headDim].
    ///   - values: The value tensor [1, kvHeads, seqLen, headDim].
    ///   - mask: The attention mask mode.
    /// - Returns: The output tensor [1, seqLen, heads, headDim].
    /// - Throws: `PagedAttentionMLXError` or `PagedAttentionError` on failure.
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
            guard let seqLengthBuffer = seqLengths.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
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

    /// The state of the fallback cache as MLX arrays.
    public var state: [MLXArray] {
        get { fallbackCache.state }
        set {
            fallbackCache.state = newValue
            offset = fallbackCache.offset
        }
    }

    /// Metadata state used for serialization: ["paged-metal", sequenceID, offset, blockSize].
    public var metaState: [String] {
        get { ["paged-metal", String(sequenceID), String(offset), String(layer.blockSize)] }
        set {
            if newValue.count >= 3, let parsedOffset = Int(newValue[2]) {
                offset = parsedOffset
            }
        }
    }

    /// Whether this cache supports trimming (always false for paged attention).
    public var isTrimmable: Bool { false }

    /// Trims the cache (no-op for paged attention; always returns 0).
    @discardableResult
    public func trim(_ n: Int) -> Int {
        0
    }

    /// Creates an attention mask appropriate for the current cache state.
    /// - Parameters:
    ///   - n: The number of new tokens.
    ///   - windowSize: Optional sliding window constraint.
    ///   - returnArray: Whether to return a materialized array mask (vs. a mode enum).
    /// - Returns: The appropriate `ScaledDotProductAttentionMaskMode`.
    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    /// Creates a copy of this cache (uses the fallback `KVCacheSimple` implementation).
    public func copy() -> any KVCache {
        let copied = KVCacheSimple()
        copied.state = fallbackCache.state
        return copied
    }

    /// A human-readable description of this cache for debugging.
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
        guard let buffer = table.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
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
    return attentionWithCacheUpdate(
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

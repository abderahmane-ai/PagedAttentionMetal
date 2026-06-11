import Foundation
import Metal

/// Errors thrown by the PagedAttention engine.
public enum PagedAttentionError: Error, CustomStringConvertible, Equatable {
    /// Metal device or command queue could not be initialized.
    case deviceInitializationFailed
    /// Metal library could not be loaded or compiled from the provided source.
    case libraryInitializationFailed(String)
    /// A compute pipeline state could not be created for the named kernel.
    case pipelineCreationFailed(String)
    /// The provided configuration parameters are invalid.
    case invalidConfiguration(String)
    /// A Metal buffer is too small for the requested operation.
    case bufferTooSmall(name: String, expected: Int, actual: Int)
    /// Failed to encode commands into a Metal command buffer.
    case commandEncodingFailed(String)
    /// Metal command execution returned an error status.
    case commandExecutionFailed(String)
    /// The requested feature or configuration is not supported.
    case unsupported(String)
    /// A GPU operation exceeded the timeout threshold.
    case gpuTimeout(String)
    /// GPU resources (memory, block count, etc.) are exhausted.
    case resourceExhausted(String)
    /// Data corruption detected during a GPU operation.
    case dataCorruption(String)
    /// The engine is in an invalid state for the requested operation.
    case invalidState(String)
    /// The same operation has failed repeatedly; circuit breaker is engaged.
    case recurringError(String, attemptCount: Int)

    public var description: String {
        switch self {
        case .deviceInitializationFailed:
            return "Metal device or command queue could not be initialized."
        case .libraryInitializationFailed(let message):
            return "Metal library initialization failed: \(message)"
        case .pipelineCreationFailed(let name):
            return "Metal compute pipeline creation failed for \(name)."
        case .invalidConfiguration(let message):
            return "Invalid paged attention configuration: \(message)"
        case .bufferTooSmall(let name, let expected, let actual):
            return "\(name) buffer is too small: expected at least \(expected) bytes, got \(actual)."
        case .commandEncodingFailed(let message):
            return "Metal command encoding failed: \(message)"
        case .commandExecutionFailed(let message):
            return "Metal command execution failed: \(message)"
        case .unsupported(let message):
            return "Unsupported paged attention path: \(message)"
        case .gpuTimeout(let message):
            return "GPU operation timed out: \(message)"
        case .resourceExhausted(let message):
            return "GPU resource exhausted: \(message)"
        case .dataCorruption(let message):
            return "Data corruption detected: \(message)"
        case .invalidState(let message):
            return "Invalid engine state: \(message)"
        case .recurringError(let operation, let attemptCount):
            return "Recurring failure in '\(operation)' after \(attemptCount) attempts — circuit breaker engaged."
        }
    }
}

/// The numeric data type used for KV cache buffers and attention computation.
public enum PagedAttentionDataType: Int32, Sendable {
    /// 32-bit floating point (single precision).
    case float32 = 0
    /// 16-bit floating point (half precision).
    case float16 = 1
    /// 8-bit floating point (FP8 quantized).
    case float8 = 2

    /// The size in bytes of a single element of this data type.
    public var byteWidth: Int {
        switch self {
        case .float32: return MemoryLayout<Float>.stride
        case .float16: return MemoryLayout<Float16>.stride
        case .float8: return 1
        }
    }

    /// Whether this data type uses quantization (true for `.float8`).
    public var isQuantized: Bool {
        switch self {
        case .float8: return true
        default: return false
        }
    }
}

/// Configuration for a single attention layer in the paged attention model.
public struct PagedLayerSpec: Sendable, Equatable {
    /// The dimension of each attention head.
    public var headDim: Int
    /// The number of query attention heads.
    public var numHeads: Int
    /// The number of key/value attention heads (may differ from numHeads for GQA/MQA).
    public var numKVHeads: Int
    /// The number of tokens per physical KV cache block.
    public var blockSize: Int
    /// The numeric data type for KV cache and computation.
    public var dataType: PagedAttentionDataType
    /// Sliding window size (0 disables windowing).
    public var windowSize: Int

    /// Creates a new layer specification.
    public init(
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int = 16,
        dataType: PagedAttentionDataType = .float16,
        windowSize: Int = 0
    ) {
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.blockSize = blockSize
        self.dataType = dataType
        self.windowSize = windowSize
    }

    /// Validates that all layer parameters are positive and internally consistent.
    /// - Throws: `PagedAttentionError.invalidConfiguration` if any constraint is violated.
    public func validate() throws {
        guard headDim > 0 else {
            throw PagedAttentionError.invalidConfiguration("headDim must be positive.")
        }
        guard numHeads > 0 else {
            throw PagedAttentionError.invalidConfiguration("numHeads must be positive.")
        }
        guard numKVHeads > 0 else {
            throw PagedAttentionError.invalidConfiguration("numKVHeads must be positive.")
        }
        guard numHeads % numKVHeads == 0 else {
            throw PagedAttentionError.invalidConfiguration("numHeads must be divisible by numKVHeads for GQA/MQA mapping.")
        }
        guard blockSize > 0 else {
            throw PagedAttentionError.invalidConfiguration("blockSize must be positive.")
        }
    }

    /// The number of query elements per token (numHeads × headDim).
    public var qElementsPerToken: Int { numHeads * headDim }
    /// The number of key/value elements per token (numKVHeads × headDim).
    public var kvElementsPerToken: Int { numKVHeads * headDim }
    /// The size in bytes of query data per token.
    public var qBytesPerToken: Int { qElementsPerToken * dataType.byteWidth }
    /// The size in bytes of key/value data per token.
    public var kvBytesPerToken: Int { kvElementsPerToken * dataType.byteWidth }
}

public struct PagedAttentionConfig: Sendable, Equatable {
    public var layer: PagedLayerSpec
    public var splitThreshold: Int

    public init(layer: PagedLayerSpec, splitThreshold: Int = 1024) {
        self.layer = layer
        self.splitThreshold = splitThreshold
    }
}

/// Statistics collected from the most recent engine operation.
public struct PagedAttentionStats: Sendable, Equatable {
    /// The kind of operation that was performed.
    public enum Operation: String, Sendable {
        /// No operation has been performed.
        case none
        /// A single-pass prefill was executed.
        case prefillSinglePass
        /// A split-pass prefill was executed.
        case prefillSplitPass
        /// A tiled prefill pass was used.
        case prefillTiledPass
        /// A decode step was performed.
        case decode
        /// KV cache append was performed.
        case append
        /// A backward pass was performed.
        case backward
    }

    /// The type of operation recorded.
    public var operation: Operation
    /// The batch size used in the operation.
    public var batchSize: Int
    /// The total sequence length processed.
    public var sequenceLength: Int
    /// The number of KV cache blocks involved.
    public var numBlocks: Int
    /// Temporary scratch buffer bytes allocated.
    public var temporaryBytes: Int
    /// Whether the split/tiled pass was used.
    public var usedSplitPass: Bool
    /// Wall-clock time in milliseconds for the operation.
    public var elapsedMs: Double

    /// All-zero stats representing no operation.
    public static let empty = PagedAttentionStats(
        operation: .none,
        batchSize: 0,
        sequenceLength: 0,
        numBlocks: 0,
        temporaryBytes: 0,
        usedSplitPass: false,
        elapsedMs: 0
    )
}

/// Input parameters for a single-sequence paged attention prefill operation.
public struct PagedAttentionPrefillRequest {
    /// The query tensor buffer [seqLen × numHeads × headDim].
    public var q: MTLBuffer
    /// The key cache pool buffer.
    public var kPool: MTLBuffer
    /// The value cache pool buffer.
    public var vPool: MTLBuffer
    /// The block table mapping logical blocks to physical cache slots.
    public var blockTable: MTLBuffer
    /// The output buffer for the attention result.
    public var output: MTLBuffer
    /// The number of tokens in the input sequence.
    public var seqLen: Int
    /// The layer specification.
    public var layer: PagedLayerSpec
    /// Whether to apply causal masking.
    public var causal: Bool
    /// Optional FP8 key scale factors buffer.
    public var kScaleBuffer: MTLBuffer?
    /// Optional FP8 value scale factors buffer.
    public var vScaleBuffer: MTLBuffer?

    /// Creates a new prefill request.
    public init(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        output: MTLBuffer,
        seqLen: Int,
        layer: PagedLayerSpec,
        causal: Bool = true,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) {
        self.q = q
        self.kPool = kPool
        self.vPool = vPool
        self.blockTable = blockTable
        self.output = output
        self.seqLen = seqLen
        self.layer = layer
        self.causal = causal
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

/// Input parameters for a batched paged attention decode (single-token generation) operation.
public struct PagedAttentionDecodeRequest {
    /// The query tensor buffer [batchSize × numHeads × headDim].
    public var q: MTLBuffer
    /// The key cache pool buffer.
    public var kPool: MTLBuffer
    /// The value cache pool buffer.
    public var vPool: MTLBuffer
    /// The batched block tables buffer [batchSize × maxNumBlocks].
    public var blockTables: MTLBuffer
    /// The per-sequence current length buffer [batchSize × UInt32].
    public var seqLengths: MTLBuffer
    /// The output buffer for decoded token logits.
    public var output: MTLBuffer
    /// The number of sequences in the batch.
    public var batchSize: Int
    /// The maximum number of blocks across all sequences in the batch.
    public var maxNumBlocks: Int
    /// The layer specification.
    public var layer: PagedLayerSpec
    /// Optional FP8 key scale factors buffer.
    public var kScaleBuffer: MTLBuffer?
    /// Optional FP8 value scale factors buffer.
    public var vScaleBuffer: MTLBuffer?

    /// Creates a new decode request.
    public init(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTables: MTLBuffer,
        seqLengths: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        maxNumBlocks: Int,
        layer: PagedLayerSpec,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) {
        self.q = q
        self.kPool = kPool
        self.vPool = vPool
        self.blockTables = blockTables
        self.seqLengths = seqLengths
        self.output = output
        self.batchSize = batchSize
        self.maxNumBlocks = maxNumBlocks
        self.layer = layer
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }

    /// Attaches FP8 scale factor buffers to an existing decode request.
    public mutating func addFP8Scales(kScaleBuffer: MTLBuffer, vScaleBuffer: MTLBuffer) {
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

/// Input parameters for appending keys and values into the paged KV cache.
/// Input parameters for appending keys and values into the paged KV cache.
public struct PagedKVAppendRequest {
    /// The new keys tensor buffer [numNewTokens × numKVHeads × headDim].
    public var keys: MTLBuffer
    /// The new values tensor buffer [numNewTokens × numKVHeads × headDim].
    public var values: MTLBuffer
    /// The key cache pool buffer.
    public var kPool: MTLBuffer
    /// The value cache pool buffer.
    public var vPool: MTLBuffer
    /// The block table mapping logical blocks to physical cache slots.
    public var blockTable: MTLBuffer
    /// The token offset (starting position) in the sequence for this append.
    public var tokenOffset: Int
    /// The number of new tokens being appended.
    public var numNewTokens: Int
    /// The layer specification.
    public var layer: PagedLayerSpec
    /// Optional FP8 key scale factors buffer.
    public var kScaleBuffer: MTLBuffer?
    /// Optional FP8 value scale factors buffer.
    public var vScaleBuffer: MTLBuffer?

    /// Creates a new KV append request.
    public init(
        keys: MTLBuffer,
        values: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        layer: PagedLayerSpec,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) {
        self.keys = keys
        self.values = values
        self.kPool = kPool
        self.vPool = vPool
        self.blockTable = blockTable
        self.tokenOffset = tokenOffset
        self.numNewTokens = numNewTokens
        self.layer = layer
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

/// Input parameters for a fused append-then-prefill operation in a single GPU kernel.
public struct PagedAttentionFusedPrefillRequest {
    /// The query tensor buffer [seqLen × numHeads × headDim].
    public var q: MTLBuffer
    /// The raw (unpaged) keys buffer to append before prefill.
    public var rawK: MTLBuffer
    /// The raw (unpaged) values buffer to append before prefill.
    public var rawV: MTLBuffer
    /// The block table mapping logical blocks to physical cache slots.
    public var blockTable: MTLBuffer
    /// The key cache pool buffer.
    public var kPool: MTLBuffer
    /// The value cache pool buffer.
    public var vPool: MTLBuffer
    /// The output buffer for the attention result.
    public var output: MTLBuffer
    /// The number of tokens in the input sequence.
    public var seqLen: Int
    /// The layer specification.
    public var layer: PagedLayerSpec
    /// Whether to apply causal masking.
    public var causal: Bool
    /// Optional FP8 key scale factors buffer.
    public var kScaleBuffer: MTLBuffer?
    /// Optional FP8 value scale factors buffer.
    public var vScaleBuffer: MTLBuffer?

    /// Creates a new fused prefill request.
    public init(
        q: MTLBuffer,
        rawK: MTLBuffer,
        rawV: MTLBuffer,
        blockTable: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        output: MTLBuffer,
        seqLen: Int,
        layer: PagedLayerSpec,
        causal: Bool = true,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) {
        self.q = q
        self.rawK = rawK
        self.rawV = rawV
        self.blockTable = blockTable
        self.kPool = kPool
        self.vPool = vPool
        self.output = output
        self.seqLen = seqLen
        self.layer = layer
        self.causal = causal
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

public struct PagedAttentionLayerBatch {
    public var layers: [PagedLayerSpec]

    public init(layers: [PagedLayerSpec]) {
        self.layers = layers
    }
}

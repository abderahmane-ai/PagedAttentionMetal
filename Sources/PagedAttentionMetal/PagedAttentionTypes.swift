import Foundation
import Metal

/// Errors that can occur during PagedAttention operations.
public enum PagedAttentionError: Error, CustomStringConvertible, Equatable {
    /// Failed to initialize the Metal device or command queue.
    case deviceInitializationFailed
    /// Failed to initialize the Metal library with the provided source/URL.
    case libraryInitializationFailed(String)
    /// Failed to create a compute pipeline state.
    case pipelineCreationFailed(String)
    /// The configuration parameters are invalid.
    case invalidConfiguration(String)
    /// A buffer provided is too small to fit the expected size of data.
    case bufferTooSmall(name: String, expected: Int, actual: Int)
    /// Failed to create a compute command encoder.
    case commandEncodingFailed(String)
    /// A command buffer execution failed with a Metal/GPU error.
    case commandExecutionFailed(String)
    /// The requested feature or precision path is unsupported.
    case unsupported(String)
    /// A GPU operation timed out.
    case gpuTimeout(String)
    /// The GPU memory or block resource is exhausted.
    case resourceExhausted(String)
    /// Inconsistency or corruption detected in metadata or states.
    case dataCorruption(String)
    /// The engine or cache manager is in an invalid state.
    case invalidState(String)
    /// A recurring failure occurred in a specific operation, engaging a circuit breaker.
    case recurringError(String, attemptCount: Int)

    /// A user-friendly string description of the error.
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

/// The data type and precision mode for PagedAttention keys, values, and queries.
public enum PagedAttentionDataType: Int32, Sendable {
    /// 32-bit single-precision floating point.
    case float32 = 0
    /// 16-bit half-precision floating point.
    case float16 = 1
    /// 8-bit quantized floating point (FP8).
    case float8 = 2

    /// The size of a single element in bytes.
    public var byteWidth: Int {
        switch self {
        case .float32: return MemoryLayout<Float>.stride
        case .float16: return MemoryLayout<Float16>.stride
        case .float8: return 1
        }
    }

    /// Whether this data type is quantized (e.g. FP8).
    public var isQuantized: Bool {
        switch self {
        case .float8: return true
        default: return false
        }
    }
}

/// Specifications for a PagedAttention model layer.
public struct PagedLayerSpec: Sendable, Equatable {
    /// Dimension size of each attention head.
    public var headDim: Int
    /// Number of query attention heads.
    public var numHeads: Int
    /// Number of key/value attention heads (supporting Grouped-Query Attention).
    public var numKVHeads: Int
    /// Number of tokens stored in a single KV block.
    public var blockSize: Int
    /// Precision type for cache allocations.
    public var dataType: PagedAttentionDataType
    /// Sliding window size for localized attention (0 means full context).
    public var windowSize: Int

    /// Initializes a new layer specification.
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

    /// Validates configuration parameters.
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

    /// Number of elements in a query token representation.
    public var qElementsPerToken: Int { numHeads * headDim }
    /// Number of elements in a key/value token representation.
    public var kvElementsPerToken: Int { numKVHeads * headDim }
    /// Number of bytes in a query token representation.
    public var qBytesPerToken: Int { qElementsPerToken * dataType.byteWidth }
    /// Number of bytes in a key/value token representation.
    public var kvBytesPerToken: Int { kvElementsPerToken * dataType.byteWidth }
}

/// Global PagedAttention engine configurations.
public struct PagedAttentionConfig: Sendable, Equatable {
    /// Layer spec definitions.
    public var layer: PagedLayerSpec
    /// Threshold to fall back to split-pass execution in prefill (in tokens).
    public var splitThreshold: Int

    /// Initializes a new configuration.
    public init(layer: PagedLayerSpec, splitThreshold: Int = 1024) {
        self.layer = layer
        self.splitThreshold = splitThreshold
    }
}

/// Performance statistics for PagedAttention GPU dispatches.
public struct PagedAttentionStats: Sendable, Equatable {
    /// Type of execution operation.
    public enum Operation: String, Sendable {
        case none
        case prefillSinglePass
        case prefillMMA
        case prefillFlash
        case prefillTiledPass
        case decode
        case append
        case backward
    }

    /// The operation performed.
    public var operation: Operation
    /// The size of the batch.
    public var batchSize: Int
    /// Total sequence length (tokens).
    public var sequenceLength: Int
    /// Number of KV blocks processed.
    public var numBlocks: Int
    /// Bytes allocated for temporary scratch/reductions.
    public var temporaryBytes: Int
    /// Whether the split-pass pipeline was chosen.
    public var usedSplitPass: Bool
    /// Elapsed GPU execution latency (ms).
    public var elapsedMs: Double

    /// A zeroed, empty stats structure.
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

/// Request parameters for a single prefill operation.
public struct PagedAttentionPrefillRequest {
    /// Buffer holding query tokens.
    public var q: MTLBuffer
    /// Physical block memory pool buffer for keys.
    public var kPool: MTLBuffer
    /// Physical block memory pool buffer for values.
    public var vPool: MTLBuffer
    /// Buffer containing logical-to-physical block index mappings.
    public var blockTable: MTLBuffer
    /// Output buffer to write attention computation results.
    public var output: MTLBuffer
    /// The length of the sequence being prefilled.
    public var seqLen: Int
    /// Specifications of the layer.
    public var layer: PagedLayerSpec
    /// Whether to apply causal mask.
    public var causal: Bool
    /// Optional buffer holding FP8 scale parameters for keys.
    public var kScaleBuffer: MTLBuffer?
    /// Optional buffer holding FP8 scale parameters for values.
    public var vScaleBuffer: MTLBuffer?

    /// Initializes a new prefill request.
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

/// Request parameters for a batched decode operation.
public struct PagedAttentionDecodeRequest {
    /// Buffer holding query tokens for the batch.
    public var q: MTLBuffer
    /// Physical block memory pool buffer for keys.
    public var kPool: MTLBuffer
    /// Physical block memory pool buffer for values.
    public var vPool: MTLBuffer
    /// Flattened buffer containing logical-to-physical block table rows for all batched sequences.
    public var blockTables: MTLBuffer
    /// Buffer containing the current sequence lengths for all batched sequences.
    public var seqLengths: MTLBuffer
    /// Output buffer to write attention computation results.
    public var output: MTLBuffer
    /// Total number of sequences in the batch.
    public var batchSize: Int
    /// Maximum number of blocks allocated for any single sequence in this batch.
    public var maxNumBlocks: Int
    /// Specifications of the layer.
    public var layer: PagedLayerSpec
    /// Optional buffer holding FP8 scale parameters for keys.
    public var kScaleBuffer: MTLBuffer?
    /// Optional buffer holding FP8 scale parameters for values.
    public var vScaleBuffer: MTLBuffer?

    /// Initializes a new decode request.
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

    /// Configures the scale buffers for FP8 quantization.
    public mutating func addFP8Scales(kScaleBuffer: MTLBuffer, vScaleBuffer: MTLBuffer) {
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

/// Request parameters for appending new tokens to the physical cache pools.
public struct PagedKVAppendRequest {
    /// Buffer containing the new key states to append.
    public var keys: MTLBuffer
    /// Buffer containing the new value states to append.
    public var values: MTLBuffer
    /// Cache manager's key pool buffer.
    public var kPool: MTLBuffer
    /// Cache manager's value pool buffer.
    public var vPool: MTLBuffer
    /// Block table buffer mapping the sequence blocks.
    public var blockTable: MTLBuffer
    /// Starting offset of the new tokens in the sequence context.
    public var tokenOffset: Int
    /// Number of new tokens being appended.
    public var numNewTokens: Int
    /// Layer spec configurations.
    public var layer: PagedLayerSpec
    /// Optional scale parameters for keys (FP8 quantization).
    public var kScaleBuffer: MTLBuffer?
    /// Optional scale parameters for values (FP8 quantization).
    public var vScaleBuffer: MTLBuffer?

    /// Initializes a new KV cache append request.
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

/// Request parameters for a fused prefill operation (copying and calculating attention).
public struct PagedAttentionFusedPrefillRequest {
    /// Query states buffer.
    public var q: MTLBuffer
    /// Key states buffer before cache packing.
    public var rawK: MTLBuffer
    /// Value states buffer before cache packing.
    public var rawV: MTLBuffer
    /// Block table buffer mapping logical blocks to physical coordinates.
    public var blockTable: MTLBuffer
    /// Key cache pool buffer.
    public var kPool: MTLBuffer
    /// Value cache pool buffer.
    public var vPool: MTLBuffer
    /// Output buffer.
    public var output: MTLBuffer
    /// Length of the prefill sequence context.
    public var seqLen: Int
    /// Layer specs parameters.
    public var layer: PagedLayerSpec
    /// Whether to apply causal mask.
    public var causal: Bool
    /// Key cache FP8 scale.
    public var kScaleBuffer: MTLBuffer?
    /// Value cache FP8 scale.
    public var vScaleBuffer: MTLBuffer?

    /// Initializes a new fused prefill request.
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

/// Specifies specifications for a batch of layers in PagedAttention operations.
public struct PagedAttentionLayerBatch {
    /// Specifications of the layers in this batch.
    public var layers: [PagedLayerSpec]

    /// Initializes a new layers batch.
    public init(layers: [PagedLayerSpec]) {
        self.layers = layers
    }
}

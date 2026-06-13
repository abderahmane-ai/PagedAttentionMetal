import Foundation
import Metal

public enum PagedAttentionError: Error, CustomStringConvertible, Equatable {
    case deviceInitializationFailed
    case libraryInitializationFailed(String)
    case pipelineCreationFailed(String)
    case invalidConfiguration(String)
    case bufferTooSmall(name: String, expected: Int, actual: Int)
    case commandEncodingFailed(String)
    case commandExecutionFailed(String)
    case unsupported(String)
    case gpuTimeout(String)
    case resourceExhausted(String)
    case dataCorruption(String)
    case invalidState(String)
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

public enum PagedAttentionDataType: Int32, Sendable {
    case float32 = 0
    case float16 = 1
    case float8 = 2

    public var byteWidth: Int {
        switch self {
        case .float32: return MemoryLayout<Float>.stride
        case .float16: return MemoryLayout<Float16>.stride
        case .float8: return 1
        }
    }

    public var isQuantized: Bool {
        switch self {
        case .float8: return true
        default: return false
        }
    }
}

public struct PagedLayerSpec: Sendable, Equatable {
    public var headDim: Int
    public var numHeads: Int
    public var numKVHeads: Int
    public var blockSize: Int
    public var dataType: PagedAttentionDataType
    public var windowSize: Int

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

    public var qElementsPerToken: Int { numHeads * headDim }
    public var kvElementsPerToken: Int { numKVHeads * headDim }
    public var qBytesPerToken: Int { qElementsPerToken * dataType.byteWidth }
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

public struct PagedAttentionStats: Sendable, Equatable {
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

    public var operation: Operation
    public var batchSize: Int
    public var sequenceLength: Int
    public var numBlocks: Int
    public var temporaryBytes: Int
    public var usedSplitPass: Bool
    public var elapsedMs: Double

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

public struct PagedAttentionPrefillRequest {
    public var q: MTLBuffer
    public var kPool: MTLBuffer
    public var vPool: MTLBuffer
    public var blockTable: MTLBuffer
    public var output: MTLBuffer
    public var seqLen: Int
    public var layer: PagedLayerSpec
    public var causal: Bool
    public var kScaleBuffer: MTLBuffer?
    public var vScaleBuffer: MTLBuffer?

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

public struct PagedAttentionDecodeRequest {
    public var q: MTLBuffer
    public var kPool: MTLBuffer
    public var vPool: MTLBuffer
    public var blockTables: MTLBuffer
    public var seqLengths: MTLBuffer
    public var output: MTLBuffer
    public var batchSize: Int
    public var maxNumBlocks: Int
    public var layer: PagedLayerSpec
    public var kScaleBuffer: MTLBuffer?
    public var vScaleBuffer: MTLBuffer?

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

    public mutating func addFP8Scales(kScaleBuffer: MTLBuffer, vScaleBuffer: MTLBuffer) {
        self.kScaleBuffer = kScaleBuffer
        self.vScaleBuffer = vScaleBuffer
    }
}

public struct PagedKVAppendRequest {
    public var keys: MTLBuffer
    public var values: MTLBuffer
    public var kPool: MTLBuffer
    public var vPool: MTLBuffer
    public var blockTable: MTLBuffer
    public var tokenOffset: Int
    public var numNewTokens: Int
    public var layer: PagedLayerSpec
    public var kScaleBuffer: MTLBuffer?
    public var vScaleBuffer: MTLBuffer?

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

public struct PagedAttentionFusedPrefillRequest {
    public var q: MTLBuffer
    public var rawK: MTLBuffer
    public var rawV: MTLBuffer
    public var blockTable: MTLBuffer
    public var kPool: MTLBuffer
    public var vPool: MTLBuffer
    public var output: MTLBuffer
    public var seqLen: Int
    public var layer: PagedLayerSpec
    public var causal: Bool
    public var kScaleBuffer: MTLBuffer?
    public var vScaleBuffer: MTLBuffer?

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

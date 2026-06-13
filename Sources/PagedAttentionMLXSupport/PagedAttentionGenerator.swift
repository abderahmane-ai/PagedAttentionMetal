import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal
import os

private final class CacheHolder: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()
    private var _caches: [KVCacheManager] = []
    
    var caches: [KVCacheManager] {
        lock.lock()
        defer { lock.unlock() }
        return _caches
    }
    
    func setCaches(_ newCaches: [KVCacheManager]) {
        lock.lock()
        defer { lock.unlock() }
        _caches = newCaches
    }
}

/// Protocol defining the interface for adapting external models (e.g. Llama)
/// to the custom PagedAttention projection and generation structures.
public protocol ModelAdapterProtocol: AnyObject, Sendable {
    /// Hidden dimension size of the model.
    var hiddenSize: Int { get }
    /// Vocabulary size of the model.
    var vocabSize: Int { get }
    /// Number of attention layers in the model.
    var numLayers: Int { get }
    /// Specifications of each layer.
    var layerSpecs: [PagedLayerSpec] { get }

    /// Converts token identifiers into hidden state embeddings.
    func embed(tokens: [Int]) throws -> MLXArray

    /// Projects embedded states into Metal query, key, and value buffers.
    func projectQKV(hidden: MLXArray, layer: Int, offset: Int) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer)

    /// Applies projection weight parameters, residual addition, layernorm, and MLP steps to the output of attention computation.
    func applyAttentionOutput(hidden: MLXArray, attentionFloats: [Float], layer: Int) throws -> MLXArray

    /// Evaluates final normalization and lm_head projection to return prediction logits.
    func projectOutput(hidden: MLXArray) throws -> MLXArray
}

/// Facilitates token-by-token generation for adapted MLX models using the optimized PagedAttention execution pipeline.
public class PagedAttentionGenerator: @unchecked Sendable {
    /// The coordinated PagedAttention inference orchestrator.
    public let inference: PagedAttentionInference
    /// The adapter configured for mapping the target LLM architecture weights.
    public let modelAdapter: ModelAdapterProtocol
    /// Configuration variables controlling text generation options.
    public let samplingConfig: SamplingConfig
    /// The Metal device.
    public let device: MTLDevice

    /// Initializes a new generator instance.
    ///
    /// - Parameters:
    ///   - inference: The coordinated inference instance.
    ///   - modelAdapter: The adapter mapping the model's weight configurations.
    ///   - samplingConfig: Text generation settings (e.g. temperature, repetition penalty).
    public init(
        inference: PagedAttentionInference,
        modelAdapter: ModelAdapterProtocol,
        samplingConfig: SamplingConfig = SamplingConfig()
    ) {
        self.inference = inference
        self.modelAdapter = modelAdapter
        self.samplingConfig = samplingConfig
        self.device = inference.device
    }

    /// Asynchronously streams generated token identifiers.
    ///
    /// - Parameters:
    ///   - promptTokens: Array of input token IDs representing the generation prompt.
    ///   - maxNewTokens: Maximum count of new tokens to produce.
    ///   - overrideConfig: Optional override configurations for text generation.
    /// - Returns: An asynchronous throwing stream of generated token IDs.
    public func generate(
        promptTokens: [Int],
        maxNewTokens: Int,
        samplingConfig overrideConfig: SamplingConfig? = nil
    ) throws -> AsyncThrowingStream<Int, Error> {
        guard maxNewTokens >= 0 else {
            throw PagedAttentionError.invalidConfiguration("maxNewTokens must be non-negative")
        }
        let config = overrideConfig ?? samplingConfig
        let seqID = inference.addRequest(promptTokenCount: promptTokens.count, maxNewTokens: maxNewTokens)
        guard seqID >= 0 else {
            throw PagedAttentionError.resourceExhausted("max sequences exceeded")
        }

        return AsyncThrowingStream<Int, Error> { [self] continuation in
            let holder = CacheHolder()
            continuation.onTermination = { @Sendable _ in
                for cache in holder.caches {
                    cache.freeSequence(id: seqID)
                }
            }

            Task {
                do {
                    if maxNewTokens == 0 {
                        inference.completeSequence(id: seqID)
                        continuation.finish()
                        return
                    }

                    let perLayerBlocks = max(1, inference.cache.maxBlocks / max(1, modelAdapter.numLayers))
                    let newCaches = try (0..<modelAdapter.numLayers).map { _ in
                        let firstSpec = modelAdapter.layerSpecs[0]
                        return try KVCacheManager(
                            device: device,
                            maxBlocks: perLayerBlocks,
                            blockSize: firstSpec.blockSize,
                            headDim: firstSpec.headDim,
                            numKVHeads: firstSpec.numKVHeads,
                            dataType: firstSpec.dataType
                        )
                    }
                    holder.setCaches(newCaches)

                    try inference.cache.allocateSequence(id: seqID)
                    try inference.cache.appendTokens(toSequence: seqID, count: promptTokens.count)

                    let embedded = try modelAdapter.embed(tokens: promptTokens)
                    let seqLen = promptTokens.count
                    var hiddenStates = embedded

                    for layer in 0..<modelAdapter.numLayers {
                        let caches = holder.caches
                        let cache = caches[layer]
                        try cache.allocateSequence(id: seqID)
                        try cache.appendTokens(toSequence: seqID, count: seqLen)
                        let (qBuf, kBuf, vBuf) = try modelAdapter.projectQKV(hidden: hiddenStates, layer: layer, offset: 0)
                        let blockTable = try cache.getBlockTableBuffer(forSequence: seqID)
                        let layerSpec = modelAdapter.layerSpecs[layer]

                        try inference.engine.appendToCache(PagedKVAppendRequest(
                            keys: kBuf, values: vBuf,
                            kPool: cache.kPoolBuffer,
                            vPool: cache.vPoolBuffer,
                            blockTable: blockTable,
                            tokenOffset: 0,
                            numNewTokens: seqLen,
                            layer: layerSpec
                        ))

                        let outputBuffer = try makeOutputBuffer(tokenCount: seqLen, layer: layerSpec)
                        try inference.engine.prefill(PagedAttentionPrefillRequest(
                            q: qBuf,
                            kPool: cache.kPoolBuffer,
                            vPool: cache.vPoolBuffer,
                            blockTable: blockTable,
                            output: outputBuffer,
                            seqLen: seqLen,
                            layer: layerSpec,
                            causal: true
                        ))

                        let floats = try readOutputBuffer(outputBuffer, tokenCount: seqLen, layer: layerSpec)
                        hiddenStates = try modelAdapter.applyAttentionOutput(
                            hidden: hiddenStates, attentionFloats: floats, layer: layer
                        )
                    }

                    let logitsArr = try modelAdapter.projectOutput(hidden: hiddenStates)
                    let lastLogitRow: [Float] = logitsArr[seqLen - 1, 0..<modelAdapter.vocabSize].asArray(Float.self)
                    inference.scheduler.markPrefilled(id: seqID)

                    var allTokens = promptTokens
                    var lastToken = Sampler.sample(logits: lastLogitRow, config: config)
                    allTokens.append(lastToken)
                    continuation.yield(lastToken)

                    for _ in 1..<maxNewTokens {
                        if Task.isCancelled {
                            inference.completeSequence(id: seqID)
                            continuation.finish(throwing: CancellationError())
                            return
                        }
                        try inference.cache.appendTokens(toSequence: seqID, count: 1)
                        let currentLen = try inference.cache.getSequenceLength(seqID)
                        var singleHidden = try modelAdapter.embed(tokens: [lastToken])

                        for layer in 0..<modelAdapter.numLayers {
                            let caches = holder.caches
                            let cache = caches[layer]
                            try cache.appendTokens(toSequence: seqID, count: 1)
                            let layerSpec = modelAdapter.layerSpecs[layer]
                            let totalSeqLen = try cache.getSequenceLength(seqID)
                            let kvTokenOffset = totalSeqLen - 1
                            let (qBuf, kBuf, vBuf) = try modelAdapter.projectQKV(hidden: singleHidden, layer: layer, offset: kvTokenOffset)
                            let blockTable = try cache.getBlockTableBuffer(forSequence: seqID)
                            let seqLengthsBuf = try makeSeqLengthsBuffer([UInt32(currentLen)])
                            let numBlocks = try cache.getNumBlocks(forSequence: seqID)

                            try inference.engine.appendToCache(PagedKVAppendRequest(
                                keys: kBuf, values: vBuf,
                                kPool: cache.kPoolBuffer,
                                vPool: cache.vPoolBuffer,
                                blockTable: blockTable,
                                tokenOffset: kvTokenOffset,
                                numNewTokens: 1,
                                layer: layerSpec
                            ))

                            let outputBuffer = try makeOutputBuffer(tokenCount: 1, layer: layerSpec)
                            try inference.engine.decode(PagedAttentionDecodeRequest(
                                q: qBuf,
                                kPool: cache.kPoolBuffer,
                                vPool: cache.vPoolBuffer,
                                blockTables: blockTable,
                                seqLengths: seqLengthsBuf,
                                output: outputBuffer,
                                batchSize: 1,
                                maxNumBlocks: numBlocks,
                                layer: layerSpec
                            ))

                            let floats = try readOutputBuffer(outputBuffer, tokenCount: 1, layer: layerSpec)
                            singleHidden = try modelAdapter.applyAttentionOutput(
                                hidden: singleHidden, attentionFloats: floats, layer: layer
                            )
                        }

                        let logitsArr = try modelAdapter.projectOutput(hidden: singleHidden)
                        let logitsRow: [Float] = logitsArr[0, 0..<modelAdapter.vocabSize].asArray(Float.self)

                        lastToken = Sampler.sample(logits: logitsRow, config: config, previousTokens: allTokens)
                        allTokens.append(lastToken)
                        continuation.yield(lastToken)
                        inference.scheduler.markDecoded(id: seqID)
                    }

                    inference.completeSequence(id: seqID)
                    continuation.finish()
                } catch {
                    inference.completeSequence(id: seqID)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func makeOutputBuffer(tokenCount: Int, layer: PagedLayerSpec) throws -> MTLBuffer {
        let outputSize = tokenCount * layer.numHeads * layer.headDim
        guard let buffer = device.makeBuffer(
            length: outputSize * layer.dataType.byteWidth,
            options: .storageModeShared
        ) else {
            throw PagedAttentionError.commandEncodingFailed("attention output buffer")
        }
        return buffer
    }

    private func readOutputBuffer(_ buffer: MTLBuffer, tokenCount: Int, layer: PagedLayerSpec) throws -> [Float] {
        let floatCount = tokenCount * layer.numHeads * layer.headDim
        switch layer.dataType {
        case .float32:
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            return (0..<floatCount).map { Float(ptr[$0]) }
        case .float16:
            let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<floatCount).map { Float(ptr[$0]) }
        case .float8:
            let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<floatCount).map { Float(ptr[$0]) }
        }
    }

    private func makeSeqLengthsBuffer(_ lengths: [UInt32]) throws -> MTLBuffer {
        guard let buffer = lengths.withUnsafeBytes({ (ptr: UnsafeRawBufferPointer) in
            device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count, options: .storageModeShared)
        }) else {
            throw PagedAttentionError.commandEncodingFailed("seq lengths buffer")
        }
        return buffer
    }
}

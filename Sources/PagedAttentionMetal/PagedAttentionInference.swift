import Foundation
import Metal
import os

/// High-level inference orchestration combining the engine, batch KV cache, and scheduler.
///
/// Manages the full lifecycle of LLM generation: request admission, prefill, continuous
/// batching decode, and sequence completion with automatic memory management.
public final class PagedAttentionInference: @unchecked Sendable {
    /// The underlying paged attention GPU engine.
    public let engine: PagedAttentionEngine
    /// The batch KV cache manager.
    public let cache: BatchKVCacheManager
    /// The continuous batching scheduler.
    public let scheduler: ContinuousBatchingScheduler

    /// The Metal device used for all GPU operations.
    public let device: MTLDevice
    /// The layer specifications for each transformer layer.
    public let layerSpecs: [PagedLayerSpec]

    /// The maximum number of sequences per batch.
    public let maxBatchSize: Int
    /// The number of transformer layers.
    public let numLayers: Int

    /// Callback invoked each time a token is generated during decode.
    public var onTokenGenerated: ((Int, Int) -> Void)?

    private var prefilledSequences: Set<Int> = []
    private let lock: UnsafeMutablePointer<os_unfair_lock>
    private var _totalTokensGenerated: Int = 0

    /// The number of `step()` calls executed so far.
    public private(set) var stepCount: Int = 0

    /// Creates a new inference orchestrator.
    /// Creates a new inference orchestrator.
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - maxBlocks: Total number of physical cache blocks to allocate.
    ///   - blockSize: Number of tokens per physical block.
    ///   - layerSpecs: Specifications for each transformer layer.
    ///   - maxBatchSize: Maximum batch size per step.
    ///   - maxSequences: Maximum number of concurrent sequences.
    ///   - memoryFraction: Fraction of GPU memory reserved for KV cache.
    /// - Throws: `PagedAttentionError` if initialization fails.
    public init(
        device: MTLDevice,
        maxBlocks: Int,
        blockSize: Int = 16,
        layerSpecs: [PagedLayerSpec],
        maxBatchSize: Int = 16,
        maxSequences: Int = 128,
        memoryFraction: Float = 0.75
    ) throws {
        guard !layerSpecs.isEmpty else {
            throw PagedAttentionError.invalidConfiguration("at least one layer spec required")
        }

        let first = layerSpecs[0]
        let headDim = first.headDim
        let numKVHeads = first.numKVHeads
        let dataType = first.dataType

        self.device = device
        self.layerSpecs = layerSpecs
        self.maxBatchSize = maxBatchSize
        self.numLayers = layerSpecs.count

        self.engine = try PagedAttentionEngine()
        self.scheduler = ContinuousBatchingScheduler(maxBatchSize: maxBatchSize, maxSequences: maxSequences)

        let maxSequenceBlocks = maxBlocks

        self.cache = BatchKVCacheManager(
            device: device,
            maxBatchSize: maxBatchSize,
            maxSequenceBlocks: maxSequenceBlocks,
            maxBlocks: maxBlocks,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )

        self.lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        lock.pointee = os_unfair_lock()
    }

    deinit {
        lock.deinitialize(count: 1)
        lock.deallocate()
    }

    /// Adds a new generation request to the inference pipeline.
    /// - Parameters:
    ///   - promptTokenCount: The number of tokens in the input prompt.
    ///   - maxNewTokens: The maximum number of new tokens to generate.
    /// - Returns: The unique sequence ID, or -1 if the scheduler is full.
    @discardableResult
    public func addRequest(promptTokenCount: Int, maxNewTokens: Int) -> Int {
        scheduler.addRequest(promptTokenCount: promptTokenCount, maxNewTokens: maxNewTokens)
    }

    /// Completes a sequence, marking it done in the scheduler and freeing its KV cache blocks.
    /// - Parameter id: The sequence ID.
    public func completeSequence(id: Int) {
        scheduler.completeSequence(id: id)
        cache.freeSequence(id: id)
        os_unfair_lock_lock(lock)
        prefilledSequences.remove(id)
        os_unfair_lock_unlock(lock)
    }

    /// Executes one scheduling step: prefill for new sequences and decode for ongoing ones.
    /// - Returns: The batch of sequences processed in this step.
    /// - Throws: `PagedAttentionError` or `KVCacheError` if GPU execution or memory management fails.
    public func step() throws -> [SchedulerSequence] {
        let batch = scheduler.step()
        guard !batch.isEmpty else { return [] }

        var prefillSeqs: [SchedulerSequence] = []
        var decodeSeqs: [SchedulerSequence] = []

        for seq in batch {
            os_unfair_lock_lock(lock)
            let isPrefilled = prefilledSequences.contains(seq.id)
            os_unfair_lock_unlock(lock)

            if isPrefilled {
                decodeSeqs.append(seq)
            } else {
                prefillSeqs.append(seq)
            }
        }

        if !prefillSeqs.isEmpty {
            try handlePrefill(prefillSeqs)
        }

        if !decodeSeqs.isEmpty {
            try handleDecode(decodeSeqs)
        }

        stepCount += 1
        return batch
    }

    /// Runs all queued sequences to completion.
    /// - Throws: `PagedAttentionError` or `KVCacheError` if any step fails.
    public func runAll() throws {
        while waitingCount > 0 || runningCount > 0 {
            let batch = try step()
            if batch.isEmpty { break }
        }
    }

    /// The number of sequences currently in the waiting state.
    public var waitingCount: Int { scheduler.waitingCount }
    /// The number of sequences currently in the running state.
    public var runningCount: Int { scheduler.runningCount }
    /// The number of sequences that have completed generation.
    public var completedCount: Int { scheduler.completedCount }

    /// The total number of tokens generated (prefill + decode) across all sequences.
    public var totalTokensGenerated: Int {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        return _totalTokensGenerated
    }

    // MARK: - Prefill

    private func handlePrefill(_ sequences: [SchedulerSequence]) throws {
        for seq in sequences {
            let neededBlocks = (seq.promptLength + cache.blockSize - 1) / cache.blockSize
            try ensureBlocksAvailable(neededBlocks)

            do {
                try cache.allocateSequence(id: seq.id)
            } catch KVCacheError.sequenceAlreadyExists {
                cache.freeSequence(id: seq.id)
                try cache.allocateSequence(id: seq.id)
            }

            try cache.appendTokens(toSequence: seq.id, count: seq.promptLength)

            let blockTable = try cache.getBatchBlockTableBuffer(forBatch: [seq.id])
            let seqLen = seq.promptLength

            var qBuffers: [MTLBuffer] = []
            var outputBuffers: [MTLBuffer] = []

            for layer in layerSpecs {
                let bufSize = seqLen * layer.qBytesPerToken
                guard let qBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
                      let outBuf = device.makeBuffer(length: bufSize, options: .storageModeShared) else {
                    throw PagedAttentionError.commandEncodingFailed("unable to allocate prefill buffers")
                }
                qBuffers.append(qBuf)
                outputBuffers.append(outBuf)
            }

            try engine.prefillLayers(
                qBuffers: qBuffers,
                kPool: cache.kPoolBuffer,
                vPool: cache.vPoolBuffer,
                blockTable: blockTable,
                outputBuffers: outputBuffers,
                seqLen: seqLen,
                layers: layerSpecs,
                causal: true
            )

            scheduler.markPrefilled(id: seq.id)
            os_unfair_lock_lock(lock)
            prefilledSequences.insert(seq.id)
            _totalTokensGenerated += seq.promptLength
            os_unfair_lock_unlock(lock)
        }
    }

    // MARK: - Decode

    private func handleDecode(_ sequences: [SchedulerSequence]) throws {
        let ids = sequences.map { $0.id }
        let batchSize = ids.count

        let totalExtras = sequences.reduce(0) { sum, seq in
            let currentLen = (try? cache.getSequenceLength(seq.id)) ?? seq.promptLength
            let blocksNow = (currentLen + cache.blockSize - 1) / cache.blockSize
            let blocksAfter = (currentLen + 1 + cache.blockSize - 1) / cache.blockSize
            return sum + max(0, blocksAfter - blocksNow)
        }
        if totalExtras > 0 {
            try ensureBlocksAvailable(totalExtras)
        }

        let blockTables = try cache.getBatchBlockTableBuffer(forBatch: ids)
        let seqLengths = try cache.getSeqLengthsBuffer(forBatch: ids)
        let maxNumBlocks = cache.maxSequenceBlocks

        var qBuffers: [MTLBuffer] = []
        var outputBuffers: [MTLBuffer] = []

        for layer in layerSpecs {
            let bufSize = batchSize * layer.qBytesPerToken
            guard let qBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: bufSize, options: .storageModeShared) else {
                throw PagedAttentionError.commandEncodingFailed("unable to allocate decode buffers")
            }
            qBuffers.append(qBuf)
            outputBuffers.append(outBuf)
        }

        try engine.decodeLayers(
            qBuffers: qBuffers,
            kPool: cache.kPoolBuffer,
            vPool: cache.vPoolBuffer,
            blockTables: blockTables,
            seqLengths: seqLengths,
            outputBuffers: outputBuffers,
            batchSize: batchSize,
            maxNumBlocks: maxNumBlocks,
            layers: layerSpecs
        )

        for (_, seq) in sequences.enumerated() {
            try cache.appendTokens(toSequence: seq.id, count: 1)

            scheduler.markDecoded(id: seq.id)
            os_unfair_lock_lock(lock)
            _totalTokensGenerated += 1
            os_unfair_lock_unlock(lock)

            onTokenGenerated?(seq.id, 0)

            if seq.isDone {
                completeSequence(id: seq.id)
            }
        }
    }

    // MARK: - Memory Management

    private func ensureBlocksAvailable(_ needed: Int) throws {
        guard cache.availableBlocks < needed else { return }

        let preempted = scheduler.handleMemoryPressure(
            requiredBlocks: needed,
            availableBlocks: cache.availableBlocks
        )

        for pid in preempted {
            cache.freeSequence(id: pid)
            os_unfair_lock_lock(lock)
            prefilledSequences.remove(pid)
            os_unfair_lock_unlock(lock)
        }

        guard cache.availableBlocks >= needed else {
            throw KVCacheError.outOfMemory
        }
    }
}

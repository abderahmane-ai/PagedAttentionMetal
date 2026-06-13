import Foundation
import Metal
import os

/// High-level orchestration manager that coordinates the `PagedAttentionEngine`,
/// the `BatchKVCacheManager`, and the `ContinuousBatchingScheduler` for executing LLM batch inference.
public final class PagedAttentionInference: @unchecked Sendable {
    /// The underlying Metal kernel execution engine.
    public let engine: PagedAttentionEngine
    /// The batched KV cache manager that handles block allocations.
    public let cache: BatchKVCacheManager
    /// The batching scheduler responsible for prioritizing prefill and decode sequences.
    public let scheduler: ContinuousBatchingScheduler

    /// The active Metal device utilized for memory allocation and GPU compute.
    public let device: MTLDevice
    /// Configuration specs for all layers inside the attention model.
    public var layerSpecs: [PagedLayerSpec]

    /// The maximum number of sequences that can be batch processed simultaneously.
    public let maxBatchSize: Int
    /// Number of attention layers in the model configuration.
    public let numLayers: Int

    /// Callback invoked whenever a new token is generated.
    /// Parameters are: `(sequenceId: Int, tokenId: Int)`.
    public var onTokenGenerated: ((Int, Int) -> Void)?

    private var prefilledSequences: Set<Int> = []
    private let lock: UnsafeMutablePointer<os_unfair_lock>
    private var _totalTokensGenerated: Int = 0

    /// Counter tracking the total number of inference steps processed.
    public private(set) var stepCount: Int = 0

    /// Initializes a new instance of the inference coordinator.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - maxBlocks: Maximum number of cache blocks to allocate.
    ///   - blockSize: The size of each block (in tokens). Default is `16`.
    ///   - layerSpecs: A list of specifications for each model layer.
    ///   - maxBatchSize: The maximum capacity of concurrently batched decodes. Default is `16`.
    ///   - maxSequences: The maximum number of sequences tracking at once. Default is `128`.
    ///   - memoryFraction: Fraction of memory dedicated to the block pool. Default is `0.75`.
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

        self.cache = try BatchKVCacheManager(
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

    /// Registers a new token generation request to the batch scheduler.
    ///
    /// - Parameters:
    ///   - promptTokenCount: The count of tokens in the input prompt.
    ///   - maxNewTokens: The maximum count of new tokens to generate.
    /// - Returns: The assigned sequence identifier (ID).
    @discardableResult
    public func addRequest(promptTokenCount: Int, maxNewTokens: Int) -> Int {
        scheduler.addRequest(promptTokenCount: promptTokenCount, maxNewTokens: maxNewTokens)
    }

    /// Closes a sequence context, releasing its assigned cache resources.
    ///
    /// - Parameter id: The sequence ID.
    public func completeSequence(id: Int) {
        scheduler.completeSequence(id: id)
        cache.freeSequence(id: id)
        os_unfair_lock_lock(lock)
        prefilledSequences.remove(id)
        os_unfair_lock_unlock(lock)
    }

    /// Executes a single batch generation step, processing prefill and decode sequences in parallel.
    ///
    /// - Returns: A list of sequences currently being processed in this step.
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

    /// Runs generation continuously until all registered requests complete.
    public func runAll() throws {
        while waitingCount > 0 || runningCount > 0 {
            let batch = try step()
            if batch.isEmpty { break }
        }
    }

    /// Number of sequences waiting in the queue.
    public var waitingCount: Int { scheduler.waitingCount }
    /// Number of sequences currently running.
    public var runningCount: Int { scheduler.runningCount }
    /// Number of sequences completed since launch.
    public var completedCount: Int { scheduler.completedCount }

    /// Cumulative total of all tokens generated across active and completed requests.
    public var totalTokensGenerated: Int {
        os_unfair_lock_lock(lock)
        defer { os_unfair_lock_unlock(lock) }
        return _totalTokensGenerated
    }

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

    private func ensureBlocksAvailable(_ needed: Int) throws {
        guard cache.availableBlocks >= needed else {
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
            return
        }
    }
}

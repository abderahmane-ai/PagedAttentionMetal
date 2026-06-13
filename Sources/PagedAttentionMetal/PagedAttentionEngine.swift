import Foundation
import Metal
import os
import os.lock

private final class CommandBufferManager: @unchecked Sendable {
    private let commandQueue: MTLCommandQueue
    private let semaphore: DispatchSemaphore
    private let prepareQueue: DispatchQueue
    private var _prepared: MTLCommandBuffer?
    private let lock: UnsafeMutablePointer<os_unfair_lock>
    private var _pendingCount = 0
    private let pendingLock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
    private let waitSemaphore = DispatchSemaphore(value: 0)

    init(queue: MTLCommandQueue, count: Int = 3) {
        self.commandQueue = queue
        self.semaphore = DispatchSemaphore(value: count)
        self.prepareQueue = DispatchQueue(label: "com.pagedattentionmetal.cmdprep", qos: .userInitiated)
        self._prepared = queue.makeCommandBuffer()
        self.lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        self.pendingLock.initialize(to: os_unfair_lock())
        lock.pointee = os_unfair_lock()
    }

    deinit {
        lock.deinitialize(count: 1)
        lock.deallocate()
        pendingLock.deinitialize(count: 1)
        pendingLock.deallocate()
    }

    private var prepared: MTLCommandBuffer? {
        get { os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }; return _prepared }
        set { os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }; _prepared = newValue }
    }

    func next() -> MTLCommandBuffer {
        semaphore.wait()
        os_unfair_lock_lock(pendingLock)
        _pendingCount += 1
        os_unfair_lock_unlock(pendingLock)
        let cb: MTLCommandBuffer
        if let p = prepared {
            prepared = nil
            cb = p
        } else {
            cb = commandQueue.makeCommandBuffer()!
        }
        prepareQueue.async { [weak self] in
            guard let self = self else { return }
            self.prepared = self.commandQueue.makeCommandBuffer()
        }
        cb.addCompletedHandler { [weak self, semaphore] _ in
            if let self {
                os_unfair_lock_lock(self.pendingLock)
                self._pendingCount -= 1
                let shouldSignal = self._pendingCount == 0
                os_unfair_lock_unlock(self.pendingLock)
                if shouldSignal {
                    self.waitSemaphore.signal()
                }
            }
            semaphore.signal()
        }
        return cb
    }

    func waitForAll() {
        while true {
            os_unfair_lock_lock(pendingLock)
            let count = _pendingCount
            os_unfair_lock_unlock(pendingLock)
            if count == 0 { break }
            waitSemaphore.wait()
        }
    }
}

public class PagedAttentionEngine: @unchecked Sendable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let cmdBufManager: CommandBufferManager

    private let flashPrefillPipelineMMA: [[MTLComputePipelineState]]
    private let flashPrefillPipelineF16: MTLComputePipelineState
    private let flashPrefillPipelineF32: MTLComputePipelineState
    private let singlePassPipeline: MTLComputePipelineState
    private let singlePassPipelineF16: MTLComputePipelineState
    private let singlePassPipelineFP8: MTLComputePipelineState
    private let tiledPipeline: MTLComputePipelineState
    private let tiledPipelineF16: MTLComputePipelineState
    private let tiledPipelineFP8: MTLComputePipelineState

    private let flashDecodePipelineMMA: [MTLComputePipelineState]
    private let flashDecodePipelineF16: MTLComputePipelineState
    private let flashDecodePipelineF32: MTLComputePipelineState
    private let decodePipeline: MTLComputePipelineState
    private let decodePipelineF16: MTLComputePipelineState
    private let decodePipelineFP8: MTLComputePipelineState

    private let appendPipeline: MTLComputePipelineState
    private let appendPipelineF16: MTLComputePipelineState
    private let appendScaleFP8: MTLComputePipelineState
    private let appendPipelineFP8: MTLComputePipelineState

    private let backwardPipeline: MTLComputePipelineState
    private let backwardPipelineF16: MTLComputePipelineState

    private let fusedPrefillPipelineF32: MTLComputePipelineState
    private let fusedPrefillPipelineF16: MTLComputePipelineState
    private let fusedPrefillPipelineFP8: MTLComputePipelineState

    public var splitThreshold: Int = 1024
    public var defaultWindowSize: Int = 0
    public var defaultChunkSize: Int = 0
    public private(set) var memoryFraction: Float = 0.75
    public private(set) var recommendedBlockCount: Int = 0
    public var maxRetries: Int = 3
    public var enableGracefulDegradation: Bool = true

    private var _lastStats: PagedAttentionStats = .empty
    private var _lastError: PagedAttentionError?
    private var consecutiveFailures: Int = 0
    private let lock = OSAllocatedUnfairLock()

    public var lastStats: PagedAttentionStats {
        lock.lock()
        defer { lock.unlock() }
        return _lastStats
    }

    public var lastError: PagedAttentionError? {
        lock.lock()
        defer { lock.unlock() }
        return _lastError
    }

    public static var defaultLibrary: MTLLibrary {
        get throws {
            if let url = Bundle.module.url(forResource: "kernels", withExtension: "metallib"),
               let device = MTLCreateSystemDefaultDevice(),
               let library = try? device.makeLibrary(URL: url) {
                return library
            }
            if let url = Bundle.module.url(forResource: "kernels", withExtension: "metal"),
               let source = try? String(contentsOf: url),
               let device = MTLCreateSystemDefaultDevice(),
               let library = try? device.makeLibrary(source: source, options: nil) {
                return library
            }
            let sourcePath = #filePath.replacingOccurrences(of: "PagedAttentionEngine.swift", with: "kernels.metal")
            if let source = try? String(contentsOfFile: sourcePath),
               let device = MTLCreateSystemDefaultDevice(),
               let library = try? device.makeLibrary(source: source, options: nil) {
                return library
            }
            throw PagedAttentionError.libraryInitializationFailed("kernels.metal not found in bundle or source directory")
        }
    }

    public static func makeDefaultLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Try precompiled .metallib first (faster startup)
        if let url = Bundle.module.url(forResource: "kernels", withExtension: "metallib"),
           let library = try? device.makeLibrary(URL: url) {
            return library
        }
        // Fallback to source compilation
        if let url = Bundle.module.url(forResource: "kernels", withExtension: "metal") {
            do {
                let source = try String(contentsOf: url)
                return try device.makeLibrary(source: source, options: nil)
            } catch {
                throw PagedAttentionError.libraryInitializationFailed(error.localizedDescription)
            }
        }

        let sourcePath = #filePath.replacingOccurrences(of: "PagedAttentionEngine.swift", with: "kernels.metal")
        do {
            let source = try String(contentsOfFile: sourcePath)
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw PagedAttentionError.libraryInitializationFailed(error.localizedDescription)
        }
    }

    public static func recommendedMaxBlocks(
        device: MTLDevice,
        blockSize: Int,
        numKVHeads: Int,
        headDim: Int,
        dataType: PagedAttentionDataType = .float16,
        memoryFraction: Float = 0.75,
        layerCount: Int = 1
    ) -> Int {
        let available = device.recommendedMaxWorkingSetSize
        let workingSet = available > 0 ? available : UInt64(4 * 1_024 * 1_024 * 1_024)
        let reserved = UInt64(Double(workingSet) * Double(memoryFraction))
        let bytesPerBlock = blockSize * numKVHeads * headDim * dataType.byteWidth * 2
        let totalBytesPerLayer = bytesPerBlock
        let maxBlocks = Int(reserved / UInt64(totalBytesPerLayer * layerCount))
        return max(1, maxBlocks)
    }

    public static func gpuInfo(device: MTLDevice) -> String {
        var families = [String]()
        let appleFamilies: [(MTLGPUFamily, String)] = [
            (.apple1, "Apple 1"),
            (.apple2, "Apple 2"),
            (.apple3, "Apple 3"),
            (.apple4, "Apple 4"),
            (.apple5, "Apple 5"),
            (.apple6, "Apple 6"),
            (.apple7, "Apple 7"),
            (.apple8, "Apple 8"),
            (.apple9, "Apple 9"),
        ]
        for (family, name) in appleFamilies {
            if device.supportsFamily(family) {
                families.append(name)
            }
        }
        return """
        GPU: \(device.name)
        Recommended Max Working Set: \(device.recommendedMaxWorkingSetSize) bytes
        Max Buffer Length: \(device.maxBufferLength) bytes
        Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength) bytes
        Apple GPU Family: \(families.joined(separator: ", "))
        """
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw PagedAttentionError.deviceInitializationFailed
        }

        self.device = device
        self.commandQueue = queue
        self.cmdBufManager = CommandBufferManager(queue: queue)

        let library = try PagedAttentionEngine.makeDefaultLibrary(device: device)

        guard let funcFlashF16 = library.makeFunction(name: "flash_attention_prefill_f16"),
              let funcFlashF32 = library.makeFunction(name: "flash_attention_prefill_f32"),
              let funcFlashDecodeMMA = library.makeFunction(name: "flash_decode_mma_f16"),
              let funcFlashDecodeF16 = library.makeFunction(name: "flash_decode_f16"),
              let funcFlashDecodeF32 = library.makeFunction(name: "flash_decode_f32"),
              let funcSingle = library.makeFunction(name: "paged_attention_single"),
              let funcSingleF16 = library.makeFunction(name: "paged_attention_single_f16"),
              let funcSingleFP8 = library.makeFunction(name: "paged_attention_single_fp8"),
              let funcTiled = library.makeFunction(name: "paged_attention_tiled"),
              let funcTiledF16 = library.makeFunction(name: "paged_attention_tiled_f16"),
              let funcTiledFP8 = library.makeFunction(name: "paged_attention_tiled_fp8"),
              let funcDecode = library.makeFunction(name: "paged_decode_single"),
              let funcDecodeF16 = library.makeFunction(name: "paged_decode_single_f16"),
              let funcDecodeFP8 = library.makeFunction(name: "paged_decode_single_fp8"),
              let funcAppend = library.makeFunction(name: "kv_cache_append"),
              let funcAppendF16 = library.makeFunction(name: "kv_cache_append_f16"),
              let funcAppendScaleFP8 = library.makeFunction(name: "kv_cache_scale_fp8"),
              let funcAppendFP8 = library.makeFunction(name: "kv_cache_append_fp8"),
              let funcBackward = library.makeFunction(name: "paged_attention_backward"),
              let funcBackwardF16 = library.makeFunction(name: "paged_attention_backward_f16"),
              let funcFusedF32 = library.makeFunction(name: "paged_attention_fused_prefill_f32"),
              let funcFusedF16 = library.makeFunction(name: "paged_attention_fused_prefill_f16"),
              let funcFusedFP8 = library.makeFunction(name: "paged_attention_fused_prefill_fp8") else {
            throw PagedAttentionError.pipelineCreationFailed("one or more kernels")
        }

        let causalVals: [UInt32] = [0, 1]
        let headDimVals: [UInt32] = [64, 128]
        self.flashPrefillPipelineMMA = try causalVals.map { causalVal in
            try headDimVals.map { hdVal in
                let cv = MTLFunctionConstantValues()
                var v = causalVal
                cv.setConstantValue(&v, type: .uint, index: 0)
                var h = hdVal
                cv.setConstantValue(&h, type: .uint, index: 1)
                guard let f = try? library.makeFunction(name: "flash_attention_mma_f16", constantValues: cv) else {
                    throw PagedAttentionError.pipelineCreationFailed("MMA prefill function constant variant causal=\(causalVal) headDim=\(hdVal)")
                }
                return try device.makeComputePipelineState(function: f)
            }
        }
        self.flashPrefillPipelineF16 = try device.makeComputePipelineState(function: funcFlashF16)
        self.flashPrefillPipelineF32 = try device.makeComputePipelineState(function: funcFlashF32)
        let decodeHDVals: [UInt32] = [64]
        self.flashDecodePipelineMMA = try decodeHDVals.map { hdVal in
            let cv = MTLFunctionConstantValues()
            var h = hdVal
            cv.setConstantValue(&h, type: .uint, index: 1)
            guard let f = try? library.makeFunction(name: "flash_decode_mma_f16", constantValues: cv) else {
                throw PagedAttentionError.pipelineCreationFailed("decode MMA function constant variant headDim=\(hdVal)")
            }
            return try device.makeComputePipelineState(function: f)
        }
        self.flashDecodePipelineF16 = try device.makeComputePipelineState(function: funcFlashDecodeF16)
        self.flashDecodePipelineF32 = try device.makeComputePipelineState(function: funcFlashDecodeF32)
        self.singlePassPipeline = try device.makeComputePipelineState(function: funcSingle)
        self.singlePassPipelineF16 = try device.makeComputePipelineState(function: funcSingleF16)
        self.singlePassPipelineFP8 = try device.makeComputePipelineState(function: funcSingleFP8)
        self.tiledPipeline = try device.makeComputePipelineState(function: funcTiled)
        self.tiledPipelineF16 = try device.makeComputePipelineState(function: funcTiledF16)
        self.tiledPipelineFP8 = try device.makeComputePipelineState(function: funcTiledFP8)
        self.decodePipeline = try device.makeComputePipelineState(function: funcDecode)
        self.decodePipelineF16 = try device.makeComputePipelineState(function: funcDecodeF16)
        self.decodePipelineFP8 = try device.makeComputePipelineState(function: funcDecodeFP8)
        self.appendPipeline = try device.makeComputePipelineState(function: funcAppend)
        self.appendPipelineF16 = try device.makeComputePipelineState(function: funcAppendF16)
        self.appendScaleFP8 = try device.makeComputePipelineState(function: funcAppendScaleFP8)
        self.appendPipelineFP8 = try device.makeComputePipelineState(function: funcAppendFP8)
        self.backwardPipeline = try device.makeComputePipelineState(function: funcBackward)
        self.backwardPipelineF16 = try device.makeComputePipelineState(function: funcBackwardF16)
        self.fusedPrefillPipelineF32 = try device.makeComputePipelineState(function: funcFusedF32)
        self.fusedPrefillPipelineF16 = try device.makeComputePipelineState(function: funcFusedF16)
        self.fusedPrefillPipelineFP8 = try device.makeComputePipelineState(function: funcFusedFP8)
    }

    public convenience init(
        blockSize: Int = 16,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        dataType: PagedAttentionDataType = .float16,
        memoryFraction: Float = 0.75
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PagedAttentionError.deviceInitializationFailed
        }
        let maxBlocks = Self.recommendedMaxBlocks(
            device: device,
            blockSize: blockSize,
            numKVHeads: numKVHeads,
            headDim: headDim,
            dataType: dataType,
            memoryFraction: memoryFraction
        )
        try self.init()
        self.memoryFraction = memoryFraction
        self.recommendedBlockCount = maxBlocks
    }

    private static let log = OSLog(subsystem: "com.pagedattentionmetal", category: "engine")
    private static let perfLog = OSLog(subsystem: "com.pagedattentionmetal", category: "performance")

    private func logMetric(_ name: String, value: Double, unit: String = "ms") {
        os_log(.info, log: Self.perfLog, "%{public}s: %.2f %{public}s", name, value, unit)
    }

    private func logError(_ message: String) {
        os_log(.error, log: Self.log, "%{public}s", message)
    }

    /// Retries a synchronous block operation with exponential backoff.
    /// NOTE: This function uses `usleep` for backoff which blocks the OS thread.
    /// It must only be called from synchronous (non-async) contexts.
    /// Calling from an async Task context will block the cooperative thread pool.
    private func withRetry<T>(operation: String, block: () throws -> T) throws -> T {
        var lastError: Error?
        for attempt in 0..<maxRetries {
            do {
                let result = try block()
                lock.lock()
                consecutiveFailures = 0
                lock.unlock()
                return result
            } catch {
                lastError = error
                lock.lock()
                consecutiveFailures += 1
                if consecutiveFailures >= maxRetries * 2 {
                    let count = consecutiveFailures
                    lock.unlock()
                    throw PagedAttentionError.recurringError(operation, attemptCount: count)
                }
                lock.unlock()
                if attempt > 0 {
                    usleep(UInt32(0.01 * Double(1 << attempt) * 1_000_000))
                }
            }
        }
        throw lastError!
    }

    public func prefill(_ request: PagedAttentionPrefillRequest) throws {
        try request.layer.validate()
        try validatePrefill(request)
        if enableGracefulDegradation {
            do {
                try prefillGPU(request)
            } catch {
                logError("prefill failed: \(error)")
                request.output.contents().initializeMemory(as: UInt8.self, repeating: 0, count: request.output.length)
            }
        } else {
            try prefillGPU(request)
        }
    }

    private func shouldUseMMAPrefill(headDim: Int, dataType: PagedAttentionDataType) -> Bool {
        guard headDim <= 256 && headDim % 64 == 0 else { return false }
        switch dataType {
        case .float16: return true
        default: return false
        }
    }

    private func shouldUseFlashPrefill(headDim: Int, dataType: PagedAttentionDataType) -> Bool {
        guard headDim <= 64 else { return false }
        switch dataType {
        case .float16, .float32: return true
        case .float8: return false
        }
    }

    private func prefillGPU(_ request: PagedAttentionPrefillRequest) throws {
        let useMMA = shouldUseMMAPrefill(headDim: request.layer.headDim, dataType: request.layer.dataType)
        let useFlash = !useMMA && shouldUseFlashPrefill(headDim: request.layer.headDim, dataType: request.layer.dataType)
        let useTiledPass = !useMMA && !useFlash && shouldUseTiledPass(seqLen: request.seqLen, layer: request.layer)
        let start = CFAbsoluteTimeGetCurrent()

        if useMMA {
            try prefillMMA(
                q: request.q,
                kPool: request.kPool,
                vPool: request.vPool,
                blockTable: request.blockTable,
                seqLen: request.seqLen,
                headDim: request.layer.headDim,
                numHeads: request.layer.numHeads,
                numKVHeads: request.layer.numKVHeads,
                blockSize: request.layer.blockSize,
                causal: request.causal,
                output: request.output,
                windowSize: request.layer.windowSize
            )
        } else if useFlash {
            try prefillFlash(
                q: request.q,
                kPool: request.kPool,
                vPool: request.vPool,
                blockTable: request.blockTable,
                seqLen: request.seqLen,
                headDim: request.layer.headDim,
                numHeads: request.layer.numHeads,
                numKVHeads: request.layer.numKVHeads,
                blockSize: request.layer.blockSize,
                causal: request.causal,
                output: request.output,
                dataType: request.layer.dataType,
                windowSize: request.layer.windowSize
            )
        } else if useTiledPass {
            try prefillTiledPass(
                q: request.q,
                kPool: request.kPool,
                vPool: request.vPool,
                blockTable: request.blockTable,
                seqLen: request.seqLen,
                headDim: request.layer.headDim,
                numHeads: request.layer.numHeads,
                numKVHeads: request.layer.numKVHeads,
                blockSize: request.layer.blockSize,
                causal: request.causal,
                output: request.output,
                dataType: request.layer.dataType,
                windowSize: request.layer.windowSize,
                kScaleBuffer: request.kScaleBuffer,
                vScaleBuffer: request.vScaleBuffer
            )
        } else {
            try prefillSinglePass(
                q: request.q,
                kPool: request.kPool,
                vPool: request.vPool,
                blockTable: request.blockTable,
                seqLen: request.seqLen,
                headDim: request.layer.headDim,
                numHeads: request.layer.numHeads,
                numKVHeads: request.layer.numKVHeads,
                blockSize: request.layer.blockSize,
                causal: request.causal,
                output: request.output,
                dataType: request.layer.dataType,
                windowSize: request.layer.windowSize,
                kScaleBuffer: request.kScaleBuffer,
                vScaleBuffer: request.vScaleBuffer
            )
        }

        let numBlocks = logicalBlocks(tokenCount: request.seqLen, blockSize: request.layer.blockSize)
        lock.lock()
        let opName: PagedAttentionStats.Operation = 
            useMMA ? .prefillMMA :
            useFlash ? .prefillFlash :
            useTiledPass ? .prefillTiledPass : .prefillSinglePass
        _lastStats = PagedAttentionStats(
            operation: opName,
            batchSize: 1,
            sequenceLength: request.seqLen,
            numBlocks: numBlocks,
            temporaryBytes: 0,
            usedSplitPass: useTiledPass,
            elapsedMs: (CFAbsoluteTimeGetCurrent() - start) * 1000
        )
        _lastError = nil
        lock.unlock()
    }

    public func decode(_ request: PagedAttentionDecodeRequest) throws {
        try request.layer.validate()
        try validateDecode(request)
        if enableGracefulDegradation {
            do {
                try decodeGPU(request)
            } catch {
                logError("decode failed: \(error)")
                request.output.contents().initializeMemory(as: UInt8.self, repeating: 0, count: request.output.length)
            }
        } else {
            try decodeGPU(request)
        }
    }

    private func decodeGPU(_ request: PagedAttentionDecodeRequest) throws {
        let start = CFAbsoluteTimeGetCurrent()
        try decodeFlat(
            q: request.q,
            kPool: request.kPool,
            vPool: request.vPool,
            blockTables: request.blockTables,
            seqLengths: request.seqLengths,
            batchSize: request.batchSize,
            maxNumBlocks: request.maxNumBlocks,
            headDim: request.layer.headDim,
            numHeads: request.layer.numHeads,
            numKVHeads: request.layer.numKVHeads,
            blockSize: request.layer.blockSize,
            output: request.output,
            dataType: request.layer.dataType,
            windowSize: request.layer.windowSize,
            kScaleBuffer: request.kScaleBuffer,
            vScaleBuffer: request.vScaleBuffer
        )
        lock.lock()
        _lastStats = PagedAttentionStats(
            operation: .decode,
            batchSize: request.batchSize,
            sequenceLength: 0,
            numBlocks: request.maxNumBlocks,
            temporaryBytes: 0,
            usedSplitPass: false,
            elapsedMs: (CFAbsoluteTimeGetCurrent() - start) * 1000
        )
        _lastError = nil
        lock.unlock()
    }

    public func appendToCache(_ request: PagedKVAppendRequest) throws {
        try request.layer.validate()
        try validateAppend(request)
        if enableGracefulDegradation {
            do {
                try appendToCacheGPU(request)
            } catch {
                logError("appendToCache failed: \(error)")
            }
        } else {
            try appendToCacheGPU(request)
        }
    }

    private func appendToCacheGPU(_ request: PagedKVAppendRequest) throws {
        let start = CFAbsoluteTimeGetCurrent()
        try appendToCacheFlat(
            keys: request.keys,
            values: request.values,
            kPool: request.kPool,
            vPool: request.vPool,
            blockTable: request.blockTable,
            tokenOffset: request.tokenOffset,
            numNewTokens: request.numNewTokens,
            numKVHeads: request.layer.numKVHeads,
            headDim: request.layer.headDim,
            blockSize: request.layer.blockSize,
            dataType: request.layer.dataType,
            kScaleBuffer: request.kScaleBuffer,
            vScaleBuffer: request.vScaleBuffer
        )
        lock.lock()
        _lastStats = PagedAttentionStats(
            operation: .append,
            batchSize: 1,
            sequenceLength: request.tokenOffset + request.numNewTokens,
            numBlocks: logicalBlocks(tokenCount: request.tokenOffset + request.numNewTokens, blockSize: request.layer.blockSize),
            temporaryBytes: 0,
            usedSplitPass: false,
            elapsedMs: (CFAbsoluteTimeGetCurrent() - start) * 1000
        )
        _lastError = nil
        lock.unlock()
    }

    public func fusedPrefill(_ request: PagedAttentionFusedPrefillRequest) throws {
        try request.layer.validate()
        guard request.seqLen > 0 else {
            throw PagedAttentionError.invalidConfiguration("seqLen must be positive.")
        }
        let numBlocks = logicalBlocks(tokenCount: request.seqLen, blockSize: request.layer.blockSize)
        try requireBuffer(request.blockTable, name: "blockTable", atLeast: numBlocks * MemoryLayout<Int32>.stride)
        try requireBuffer(request.output, name: "output", atLeast: request.seqLen * request.layer.qBytesPerToken)

        try withRetry(operation: "fusedPrefill") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create fused prefill encoder")
            }

            let pipeline: MTLComputePipelineState
            switch request.layer.dataType {
            case .float32: pipeline = fusedPrefillPipelineF32
            case .float16: pipeline = fusedPrefillPipelineF16
            case .float8: pipeline = fusedPrefillPipelineFP8
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(request.q, offset: 0, index: 0)
            enc.setBuffer(request.rawK, offset: 0, index: 1)
            enc.setBuffer(request.rawV, offset: 0, index: 2)
            enc.setBuffer(request.blockTable, offset: 0, index: 3)
            enc.setBuffer(request.kPool, offset: 0, index: 4)
            enc.setBuffer(request.vPool, offset: 0, index: 5)
            enc.setBuffer(request.output, offset: 0, index: 6)

            var seqLenVar = UInt32(request.seqLen)
            var headDimVar = UInt32(request.layer.headDim)
            var numHeadsVar = UInt32(request.layer.numHeads)
            var numKVVar = UInt32(request.layer.numKVHeads)
            var blockSizeVar = UInt32(request.layer.blockSize)
            var causalVar = UInt32(request.causal ? 1 : 0)
            var windowStartVar = UInt32(request.layer.windowSize > 0 ? max(0, request.seqLen - request.layer.windowSize) : 0)

            enc.setBytes(&seqLenVar, length: 4, index: 7)
            enc.setBytes(&headDimVar, length: 4, index: 8)
            enc.setBytes(&numHeadsVar, length: 4, index: 9)
            enc.setBytes(&numKVVar, length: 4, index: 10)
            enc.setBytes(&blockSizeVar, length: 4, index: 11)
            enc.setBytes(&causalVar, length: 4, index: 12)
            enc.setBytes(&windowStartVar, length: 4, index: 13)

            if request.layer.dataType == .float8 {
                enc.setBuffer(request.kScaleBuffer, offset: 0, index: 14)
                enc.setBuffer(request.vScaleBuffer, offset: 0, index: 15)
            }

            let tileMemSize = request.layer.blockSize * request.layer.headDim * MemoryLayout<Float>.stride
            enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
            enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
            enc.setThreadgroupMemoryLength(tileMemSize, index: 2)

            let numQTiles = (request.seqLen + request.layer.blockSize - 1) / request.layer.blockSize
            let threadsPerTG = MTLSize(width: request.layer.headDim, height: request.layer.blockSize, depth: 1)
            let threadgroups = MTLSize(width: 1, height: numQTiles, depth: request.layer.numHeads)

            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            enc.endEncoding()
            try commitAndWait(cb)
        }

        lock.lock()
        _lastStats = PagedAttentionStats(
            operation: .prefillSinglePass,
            batchSize: 1,
            sequenceLength: request.seqLen,
            numBlocks: numBlocks,
            temporaryBytes: 0,
            usedSplitPass: false,
            elapsedMs: 0
        )
        _lastError = nil
        lock.unlock()
    }

    public func prefill(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        causal: Bool = true,
        output: MTLBuffer,
        dataType: PagedAttentionDataType = .float16,
        windowSize: Int = 0,
        chunkSize: Int = 0
    ) throws {
        let effectiveWindowSize = windowSize > 0 ? windowSize : defaultWindowSize
        let effectiveChunkSize = chunkSize > 0 ? chunkSize : defaultChunkSize
        let layer = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: dataType,
            windowSize: effectiveWindowSize
        )

        if effectiveChunkSize > 0 && seqLen > effectiveChunkSize {
            for chunkStart in stride(from: 0, to: seqLen, by: effectiveChunkSize) {
                let chunkEnd = min(chunkStart + effectiveChunkSize, seqLen)
                let request = PagedAttentionPrefillRequest(
                    q: q,
                    kPool: kPool,
                    vPool: vPool,
                    blockTable: blockTable,
                    output: output,
                    seqLen: chunkEnd,
                    layer: layer,
                    causal: causal
                )
                try prefill(request)
            }
            return
        }

        try prefill(PagedAttentionPrefillRequest(
            q: q,
            kPool: kPool,
            vPool: vPool,
            blockTable: blockTable,
            output: output,
            seqLen: seqLen,
            layer: layer,
            causal: causal
        ))
    }

    private func prefillSinglePass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, dataType: PagedAttentionDataType,
        windowSize: Int = 0,
        kScaleBuffer: MTLBuffer? = nil, vScaleBuffer: MTLBuffer? = nil
    ) throws {
        try withRetry(operation: "prefillSinglePass") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create prefill command buffer or encoder")
            }

        let pipeline: MTLComputePipelineState
        switch dataType {
        case .float8: pipeline = singlePassPipelineFP8
        case .float16: pipeline = singlePassPipelineF16
        case .float32: pipeline = singlePassPipeline
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)

        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        var causalVar = UInt32(causal ? 1 : 0)
        var windowStartVar = UInt32(windowSize > 0 ? max(0, seqLen - windowSize) : 0)

        enc.setBytes(&seqLenVar, length: 4, index: 5)
        enc.setBytes(&headDimVar, length: 4, index: 6)
        enc.setBytes(&numHeadsVar, length: 4, index: 7)
        enc.setBytes(&numKVVar, length: 4, index: 8)
        enc.setBytes(&blockSizeVar, length: 4, index: 9)
        enc.setBytes(&causalVar, length: 4, index: 10)

        guard let mBuffer = device.makeBuffer(length: seqLen * numHeads * MemoryLayout<Float>.stride, options: .storageModePrivate),
              let lBuffer = device.makeBuffer(length: seqLen * numHeads * MemoryLayout<Float>.stride, options: .storageModePrivate) else {
            throw PagedAttentionError.commandEncodingFailed("Failed to allocate M/L buffers")
        }
        enc.setBuffer(mBuffer, offset: 0, index: 11)
        enc.setBuffer(lBuffer, offset: 0, index: 12)
        enc.setBytes(&windowStartVar, length: 4, index: 13)

        if dataType == .float8 {
            enc.setBuffer(kScaleBuffer, offset: 0, index: 14)
            enc.setBuffer(vScaleBuffer, offset: 0, index: 15)
        }

        let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride
        enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
        enc.setThreadgroupMemoryLength(tileMemSize, index: 2)

        let numQTiles = (seqLen + blockSize - 1) / blockSize
        let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
        let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)

        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        try commitAndWait(cb)
        }
    }

    private func prefillTiledPass(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, dataType: PagedAttentionDataType,
        windowSize: Int = 0,
        kScaleBuffer: MTLBuffer? = nil, vScaleBuffer: MTLBuffer? = nil
    ) throws {
        let maxThreadgroupMemory = device.maxThreadgroupMemoryLength
        let pipeline: MTLComputePipelineState
        switch dataType {
        case .float8: pipeline = tiledPipelineFP8
        case .float16: pipeline = tiledPipelineF16
        case .float32: pipeline = tiledPipeline
        }
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let bytesPerElement = MemoryLayout<Float>.stride
        let maxTileFromMemory = maxThreadgroupMemory / (headDim * bytesPerElement)
        let maxTileFromThreads = maxThreads / headDim
        let qTileSize = min(maxTileFromMemory, maxTileFromThreads, 32)

        try withRetry(operation: "prefillTiledPass") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create tiled prefill command buffer or encoder")
            }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(kPool, offset: 0, index: 1)
        enc.setBuffer(vPool, offset: 0, index: 2)
        enc.setBuffer(blockTable, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)

        var seqLenVar = UInt32(seqLen)
        var headDimVar = UInt32(headDim)
        var numHeadsVar = UInt32(numHeads)
        var numKVVar = UInt32(numKVHeads)
        var blockSizeVar = UInt32(blockSize)
        var causalVar = UInt32(causal ? 1 : 0)
        var qTileSizeVar = UInt32(qTileSize)
        var windowStartVar = UInt32(windowSize > 0 ? max(0, seqLen - windowSize) : 0)

        enc.setBytes(&seqLenVar, length: 4, index: 5)
        enc.setBytes(&headDimVar, length: 4, index: 6)
        enc.setBytes(&numHeadsVar, length: 4, index: 7)
        enc.setBytes(&numKVVar, length: 4, index: 8)
        enc.setBytes(&blockSizeVar, length: 4, index: 9)
        enc.setBytes(&causalVar, length: 4, index: 10)
        enc.setBytes(&qTileSizeVar, length: 4, index: 11)
        enc.setBytes(&windowStartVar, length: 4, index: 12)

        if dataType == .float8 {
            enc.setBuffer(kScaleBuffer, offset: 0, index: 13)
            enc.setBuffer(vScaleBuffer, offset: 0, index: 14)
        }

        let numQTiles = (seqLen + qTileSize - 1) / qTileSize
        let threadsPerTG = MTLSize(width: headDim, height: qTileSize, depth: 1)
        let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)
        let tileMemSize = qTileSize * headDim * MemoryLayout<Float>.stride
        enc.setThreadgroupMemoryLength(tileMemSize, index: 0)

        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        enc.endEncoding()
        try commitAndWait(cb)
        }
    }

    private func prefillMMA(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, windowSize: Int = 0
    ) throws {
        try withRetry(operation: "prefillMMA") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create MMA prefill encoder")
            }

            let causalIdx = causal ? 1 : 0
            let hdIdx = headDim == 128 ? 1 : 0
            enc.setComputePipelineState(flashPrefillPipelineMMA[causalIdx][hdIdx])
            enc.setBuffer(q, offset: 0, index: 0)
            enc.setBuffer(kPool, offset: 0, index: 1)
            enc.setBuffer(vPool, offset: 0, index: 2)
            enc.setBuffer(blockTable, offset: 0, index: 3)
            enc.setBuffer(output, offset: 0, index: 4)

            var seqLenVar = UInt32(seqLen)
            var numHeadsVar = UInt32(numHeads)
            var numKVVar = UInt32(numKVHeads)
            var blockSizeVar = UInt32(blockSize)
            var windowStartVar = UInt32(windowSize > 0 ? max(0, seqLen - windowSize) : 0)

            enc.setBytes(&seqLenVar, length: 4, index: 5)
            enc.setBytes(&numHeadsVar, length: 4, index: 6)
            enc.setBytes(&numKVVar, length: 4, index: 7)
            enc.setBytes(&blockSizeVar, length: 4, index: 8)
            enc.setBytes(&windowStartVar, length: 4, index: 9)

            let hd = headDim
            enc.setThreadgroupMemoryLength(32 * hd * MemoryLayout<Float16>.stride, index: 0)
            enc.setThreadgroupMemoryLength(32 * 16 * MemoryLayout<Float16>.stride, index: 1)
            enc.setThreadgroupMemoryLength(32 * hd * MemoryLayout<Float>.stride, index: 2)
            enc.setThreadgroupMemoryLength(64 * MemoryLayout<Float>.stride, index: 3)

            let gqaRatio = numHeads / numKVHeads
            let rowsPerHead = 32 / gqaRatio
            let numQTiles = (seqLen + rowsPerHead - 1) / rowsPerHead
            let threadsPerTG = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: numKVHeads, height: numQTiles, depth: 1)

            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            enc.endEncoding()
            try commitAndWait(cb)
        }
    }

    private func prefillFlash(
        q: MTLBuffer, kPool: MTLBuffer, vPool: MTLBuffer, blockTable: MTLBuffer,
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        causal: Bool, output: MTLBuffer, dataType: PagedAttentionDataType,
        windowSize: Int = 0
    ) throws {
        try withRetry(operation: "prefillFlash") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create flash prefill encoder")
            }

            let pipeline: MTLComputePipelineState
            switch dataType {
            case .float16: pipeline = flashPrefillPipelineF16
            case .float32: pipeline = flashPrefillPipelineF32
            case .float8: throw PagedAttentionError.invalidConfiguration("FlashAttention does not support FP8")
            }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(q, offset: 0, index: 0)
            enc.setBuffer(kPool, offset: 0, index: 1)
            enc.setBuffer(vPool, offset: 0, index: 2)
            enc.setBuffer(blockTable, offset: 0, index: 3)
            enc.setBuffer(output, offset: 0, index: 4)

            var seqLenVar = UInt32(seqLen)
            var headDimVar = UInt32(headDim)
            var numHeadsVar = UInt32(numHeads)
            var numKVVar = UInt32(numKVHeads)
            var blockSizeVar = UInt32(blockSize)
            var causalVar = UInt32(causal ? 1 : 0)
            var windowStartVar = UInt32(windowSize > 0 ? max(0, seqLen - windowSize) : 0)

            enc.setBytes(&seqLenVar, length: 4, index: 5)
            enc.setBytes(&headDimVar, length: 4, index: 6)
            enc.setBytes(&numHeadsVar, length: 4, index: 7)
            enc.setBytes(&numKVVar, length: 4, index: 8)
            enc.setBytes(&blockSizeVar, length: 4, index: 9)
            enc.setBytes(&causalVar, length: 4, index: 10)
            enc.setBytes(&windowStartVar, length: 4, index: 11)

            let gqaRatio = numHeads / numKVHeads
            let rowsPerHead = 32 / gqaRatio
            let numQTiles = (seqLen + rowsPerHead - 1) / rowsPerHead
            let threadsPerTG = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: numKVHeads, height: numQTiles, depth: 1)

            enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            enc.endEncoding()
            try commitAndWait(cb)
        }
    }

    public func decode(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTables: MTLBuffer,
        seqLengths: MTLBuffer,
        batchSize: Int,
        maxNumBlocks: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        output: MTLBuffer,
        dataType: PagedAttentionDataType = .float16,
        windowSize: Int = 0
    ) throws {
        let effectiveWindowSize = windowSize > 0 ? windowSize : defaultWindowSize
        let layer = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: dataType,
            windowSize: effectiveWindowSize
        )
        try decode(PagedAttentionDecodeRequest(
            q: q,
            kPool: kPool,
            vPool: vPool,
            blockTables: blockTables,
            seqLengths: seqLengths,
            output: output,
            batchSize: batchSize,
            maxNumBlocks: maxNumBlocks,
            layer: layer
        ))
    }

    private func shouldUseMMADecode(headDim: Int, dataType: PagedAttentionDataType) -> Bool {
        return headDim == 64 && dataType == .float16
    }

    private func shouldUseFlashDecode(headDim: Int, dataType: PagedAttentionDataType) -> Bool {
        if shouldUseMMADecode(headDim: headDim, dataType: dataType) { return false }
        guard headDim <= 128 else { return false }
        switch dataType {
        case .float16, .float32: return true
        case .float8: return false
        }
    }

    private func decodeFlat(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTables: MTLBuffer,
        seqLengths: MTLBuffer,
        batchSize: Int,
        maxNumBlocks: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        output: MTLBuffer,
        dataType: PagedAttentionDataType,
        windowSize: Int = 0,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) throws {
        let useMMA = shouldUseMMADecode(headDim: headDim, dataType: dataType)
        let useFlashDecode = useMMA ? false : shouldUseFlashDecode(headDim: headDim, dataType: dataType)

        try withRetry(operation: "decodeFlat") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create decode command buffer or encoder")
            }

            let pipeline: MTLComputePipelineState
            if useMMA {
                pipeline = flashDecodePipelineMMA[0]
            } else if useFlashDecode {
                switch dataType {
                case .float16: pipeline = flashDecodePipelineF16
                case .float32: pipeline = flashDecodePipelineF32
                default: pipeline = decodePipeline
                }
            } else {
                switch dataType {
                case .float8: pipeline = decodePipelineFP8
                case .float16: pipeline = decodePipelineF16
                case .float32: pipeline = decodePipeline
                }
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(q, offset: 0, index: 0)
            enc.setBuffer(kPool, offset: 0, index: 1)
            enc.setBuffer(vPool, offset: 0, index: 2)
            enc.setBuffer(blockTables, offset: 0, index: 3)
            enc.setBuffer(seqLengths, offset: 0, index: 4)
            enc.setBuffer(output, offset: 0, index: 5)

            var batchSizeVar = UInt32(batchSize)
            var headDimVar = UInt32(headDim)
            var numHeadsVar = UInt32(numHeads)
            var numKVVar = UInt32(numKVHeads)
            var blockSizeVar = UInt32(blockSize)
            var maxNumBlocksVar = UInt32(maxNumBlocks)
            var windowSizeVar = UInt32(windowSize)

            enc.setBytes(&batchSizeVar, length: 4, index: 6)
            if useMMA {
                enc.setBytes(&numHeadsVar, length: 4, index: 7)
                enc.setBytes(&numKVVar, length: 4, index: 8)
                enc.setBytes(&blockSizeVar, length: 4, index: 9)
                enc.setBytes(&maxNumBlocksVar, length: 4, index: 10)
                enc.setBytes(&windowSizeVar, length: 4, index: 11)
            } else {
                enc.setBytes(&headDimVar, length: 4, index: 7)
                enc.setBytes(&numHeadsVar, length: 4, index: 8)
                enc.setBytes(&numKVVar, length: 4, index: 9)
                enc.setBytes(&blockSizeVar, length: 4, index: 10)
                enc.setBytes(&maxNumBlocksVar, length: 4, index: 11)
                enc.setBytes(&windowSizeVar, length: 4, index: 12)
            }

            if dataType == .float8 {
                enc.setBuffer(kScaleBuffer, offset: 0, index: 13)
                enc.setBuffer(vScaleBuffer, offset: 0, index: 14)
            }

            if useMMA {
                let maxGQA = numHeads / numKVHeads
                let totalPS = maxGQA * 10 * MemoryLayout<Float>.stride
                let mlSize = maxGQA * 2 * MemoryLayout<Float>.stride
                enc.setThreadgroupMemoryLength(totalPS, index: 0)
                enc.setThreadgroupMemoryLength(mlSize, index: 1)
                let grid = MTLSize(width: 1, height: numKVHeads, depth: batchSize)
                let tgroup = MTLSize(width: 256, height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            } else if useFlashDecode {
                let grid = MTLSize(width: headDim, height: numKVHeads, depth: batchSize)
                let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            } else {
                let grid = MTLSize(width: headDim, height: numHeads, depth: batchSize)
                let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            }
            enc.endEncoding()
            try commitAndWait(cb)
        }
    }

    public func appendToCache(
        keys: MTLBuffer,
        values: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType = .float16,
        windowSize: Int = 0
    ) throws {
        let effectiveWindowSize = windowSize > 0 ? windowSize : defaultWindowSize
        let layer = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: dataType,
            windowSize: effectiveWindowSize
        )
        try appendToCache(PagedKVAppendRequest(
            keys: keys,
            values: values,
            kPool: kPool,
            vPool: vPool,
            blockTable: blockTable,
            tokenOffset: tokenOffset,
            numNewTokens: numNewTokens,
            layer: layer
        ))
    }

    private func appendToCacheFlat(
        keys: MTLBuffer,
        values: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        numKVHeads: Int,
        headDim: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType,
        kScaleBuffer: MTLBuffer? = nil,
        vScaleBuffer: MTLBuffer? = nil
    ) throws {
        if dataType == .float8 {
            try appendToCacheCheckedFP8(
                keys: keys, values: values, kPool: kPool, vPool: vPool,
                blockTable: blockTable, tokenOffset: tokenOffset,
                numNewTokens: numNewTokens, numKVHeads: numKVHeads,
                headDim: headDim, blockSize: blockSize,
                kScaleBuffer: kScaleBuffer, vScaleBuffer: vScaleBuffer
            )
            return
        }

        try withRetry(operation: "appendToCacheFlat") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create append command buffer or encoder")
            }

            let pipeline = (dataType == .float16) ? appendPipelineF16 : appendPipeline
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(keys, offset: 0, index: 0)
            enc.setBuffer(values, offset: 0, index: 1)
            enc.setBuffer(kPool, offset: 0, index: 2)
            enc.setBuffer(vPool, offset: 0, index: 3)
            enc.setBuffer(blockTable, offset: 0, index: 4)

            var tokenOffsetVar = UInt32(tokenOffset)
            var numNewTokensVar = UInt32(numNewTokens)
            var numKVHeadsVar = UInt32(numKVHeads)
            var headDimVar = UInt32(headDim)
            var blockSizeVar = UInt32(blockSize)

            enc.setBytes(&tokenOffsetVar, length: 4, index: 5)
            enc.setBytes(&numNewTokensVar, length: 4, index: 6)
            enc.setBytes(&numKVHeadsVar, length: 4, index: 7)
            enc.setBytes(&headDimVar, length: 4, index: 8)
            enc.setBytes(&blockSizeVar, length: 4, index: 9)

            let grid = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
            let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            enc.endEncoding()
            try commitAndWait(cb)
        }
    }

    private func appendToCacheCheckedFP8(
        keys: MTLBuffer,
        values: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        numKVHeads: Int,
        headDim: Int,
        blockSize: Int,
        kScaleBuffer: MTLBuffer?,
        vScaleBuffer: MTLBuffer?
    ) throws {
        try withRetry(operation: "appendToCacheCheckedFP8") {
            let bytesPerBlock = blockSize * numKVHeads * headDim
            let maxBlocks = kPool.length / bytesPerBlock
            let scratchSize = maxBlocks * 2 * MemoryLayout<Float>.stride
            guard let scratchBuffer = device.makeBuffer(length: scratchSize, options: .storageModeShared) else {
                throw PagedAttentionError.commandEncodingFailed("failed to create FP8 scratch buffer")
            }
            scratchBuffer.contents().bindMemory(to: Float.self, capacity: maxBlocks * 2).update(repeating: 0, count: maxBlocks * 2)

            let cb1 = cmdBufManager.next()
            guard let enc1 = cb1.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create FP8 scale command buffer or encoder")
            }
            enc1.setComputePipelineState(appendScaleFP8)
            enc1.setBuffer(keys, offset: 0, index: 0)
            enc1.setBuffer(values, offset: 0, index: 1)
            enc1.setBuffer(blockTable, offset: 0, index: 2)

            var tokenOffsetVar = UInt32(tokenOffset)
            var numNewTokensVar = UInt32(numNewTokens)
            var numKVHeadsVar = UInt32(numKVHeads)
            var headDimVar = UInt32(headDim)
            var blockSizeVar = UInt32(blockSize)

            enc1.setBytes(&tokenOffsetVar, length: 4, index: 3)
            enc1.setBytes(&numNewTokensVar, length: 4, index: 4)
            enc1.setBytes(&numKVHeadsVar, length: 4, index: 5)
            enc1.setBytes(&headDimVar, length: 4, index: 6)
            enc1.setBytes(&blockSizeVar, length: 4, index: 7)
            enc1.setBuffer(scratchBuffer, offset: 0, index: 8)

            let grid = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
            let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
            enc1.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            enc1.endEncoding()
            try commitAndWait(cb1)

            let cb2 = cmdBufManager.next()
            guard let enc2 = cb2.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create FP8 append command buffer or encoder")
            }
            enc2.setComputePipelineState(appendPipelineFP8)
            enc2.setBuffer(keys, offset: 0, index: 0)
            enc2.setBuffer(values, offset: 0, index: 1)
            enc2.setBuffer(kPool, offset: 0, index: 2)
            enc2.setBuffer(vPool, offset: 0, index: 3)
            enc2.setBuffer(blockTable, offset: 0, index: 4)

            enc2.setBytes(&tokenOffsetVar, length: 4, index: 5)
            enc2.setBytes(&numNewTokensVar, length: 4, index: 6)
            enc2.setBytes(&numKVHeadsVar, length: 4, index: 7)
            enc2.setBytes(&headDimVar, length: 4, index: 8)
            enc2.setBytes(&blockSizeVar, length: 4, index: 9)
            enc2.setBuffer(scratchBuffer, offset: 0, index: 10)
            enc2.setBuffer(kScaleBuffer, offset: 0, index: 11)
            enc2.setBuffer(vScaleBuffer, offset: 0, index: 12)

            enc2.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            enc2.endEncoding()
            try commitAndWait(cb2)
        }
    }

    public func backward(
        q: MTLBuffer,
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        dO: MTLBuffer,
        m: MTLBuffer,
        l: MTLBuffer,
        dQ: MTLBuffer,
        dKPool: MTLBuffer,
        dVPool: MTLBuffer,
        seqLen: Int,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockSize: Int,
        dataType: PagedAttentionDataType = .float32,
        windowSize: Int = 0
    ) throws {
        try withRetry(operation: "backward") {
            let cb = cmdBufManager.next()
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create backward command buffer or encoder")
            }

            let pipeline = (dataType == .float16) ? backwardPipelineF16 : backwardPipeline
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(q, offset: 0, index: 0)
            enc.setBuffer(kPool, offset: 0, index: 1)
            enc.setBuffer(vPool, offset: 0, index: 2)
            enc.setBuffer(blockTable, offset: 0, index: 3)
            enc.setBuffer(dO, offset: 0, index: 4)
            enc.setBuffer(m, offset: 0, index: 5)
            enc.setBuffer(l, offset: 0, index: 6)
            enc.setBuffer(dQ, offset: 0, index: 7)
            enc.setBuffer(dKPool, offset: 0, index: 8)
            enc.setBuffer(dVPool, offset: 0, index: 9)

            var seqLenVar = UInt32(seqLen)
            var headDimVar = UInt32(headDim)
            var numHeadsVar = UInt32(numHeads)
            var numKVVar = UInt32(numKVHeads)
            var blockSizeVar = UInt32(blockSize)
            var windowStartVar = UInt32(windowSize > 0 ? max(0, seqLen - windowSize) : 0)

            enc.setBytes(&seqLenVar, length: 4, index: 10)
            enc.setBytes(&headDimVar, length: 4, index: 11)
            enc.setBytes(&numHeadsVar, length: 4, index: 12)
            enc.setBytes(&numKVVar, length: 4, index: 13)
            enc.setBytes(&blockSizeVar, length: 4, index: 14)
            enc.setBytes(&windowStartVar, length: 4, index: 15)

            let grid = MTLSize(width: headDim, height: seqLen, depth: numHeads)
            let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            enc.endEncoding()
            try commitAndWait(cb)
        }
    }

    private func singlePassPipelineFor(dataType: PagedAttentionDataType) -> MTLComputePipelineState {
        switch dataType {
        case .float32: return singlePassPipeline
        case .float16: return singlePassPipelineF16
        case .float8: return singlePassPipelineFP8
        }
    }

    private func tiledPipelineFor(dataType: PagedAttentionDataType) -> MTLComputePipelineState {
        switch dataType {
        case .float32: return tiledPipeline
        case .float16: return tiledPipelineF16
        case .float8: return tiledPipelineFP8
        }
    }

    private func decodePipelineFor(dataType: PagedAttentionDataType, useFlash: Bool = false) -> MTLComputePipelineState {
        if useFlash {
            switch dataType {
            case .float32: return flashDecodePipelineF32
            case .float16: return flashDecodePipelineF16
            case .float8: return decodePipelineFP8
            }
        }
        switch dataType {
        case .float32: return decodePipeline
        case .float16: return decodePipelineF16
        case .float8: return decodePipelineFP8
        }
    }

    public func prefillLayers(
        qBuffers: [MTLBuffer],
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        outputBuffers: [MTLBuffer],
        seqLen: Int,
        layers: [PagedLayerSpec],
        causal: Bool = true,
        windowSize: Int = 0
    ) throws {
        guard qBuffers.count == layers.count && outputBuffers.count == layers.count else {
            throw PagedAttentionError.invalidConfiguration("buffer/layer count mismatch")
        }

        var commandBuffers: [MTLCommandBuffer] = []

        for i in 0..<layers.count {
            let cb = cmdBufManager.next()
            commandBuffers.append(cb)
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create encoder for layer \(i)")
            }

            let layer = layers[i]
            let dataType = layer.dataType
            let headDim = layer.headDim
            let numHeads = layer.numHeads
            let numKVHeads = layer.numKVHeads
            let blockSize = layer.blockSize
            let effectiveWindowSize = windowSize > 0 ? windowSize : layer.windowSize
            let windowStart = effectiveWindowSize > 0 ? max(0, seqLen - effectiveWindowSize) : 0

            let useMMA = shouldUseMMAPrefill(headDim: headDim, dataType: dataType)
            let useFlash = !useMMA && shouldUseFlashPrefill(headDim: headDim, dataType: dataType)

            if useMMA {
                let causalIdx = causal ? 1 : 0
                let hdIdx = headDim == 128 ? 1 : 0
                let pipeline = flashPrefillPipelineMMA[causalIdx][hdIdx]
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(qBuffers[i], offset: 0, index: 0)
                enc.setBuffer(kPool, offset: 0, index: 1)
                enc.setBuffer(vPool, offset: 0, index: 2)
                enc.setBuffer(blockTable, offset: 0, index: 3)
                enc.setBuffer(outputBuffers[i], offset: 0, index: 4)

                var seqLenVar = UInt32(seqLen)
                var numHeadsVar = UInt32(numHeads)
                var numKVVar = UInt32(numKVHeads)
                var blockSizeVar = UInt32(blockSize)
                var windowStartVar = UInt32(effectiveWindowSize > 0 ? max(0, seqLen - effectiveWindowSize) : 0)

                enc.setBytes(&seqLenVar, length: 4, index: 5)
                enc.setBytes(&numHeadsVar, length: 4, index: 6)
                enc.setBytes(&numKVVar, length: 4, index: 7)
                enc.setBytes(&blockSizeVar, length: 4, index: 8)
                enc.setBytes(&windowStartVar, length: 4, index: 9)

                enc.setThreadgroupMemoryLength(32 * headDim * MemoryLayout<Float16>.stride, index: 0)
                enc.setThreadgroupMemoryLength(32 * 16 * MemoryLayout<Float16>.stride, index: 1)
                enc.setThreadgroupMemoryLength(32 * headDim * MemoryLayout<Float>.stride, index: 2)
                enc.setThreadgroupMemoryLength(64 * MemoryLayout<Float>.stride, index: 3)

                let gqaRatio = numHeads / numKVHeads
                let rowsPerHead = 32 / gqaRatio
                let numQTiles = (seqLen + rowsPerHead - 1) / rowsPerHead
                let threadsPerTG = MTLSize(width: 256, height: 1, depth: 1)
                let threadgroups = MTLSize(width: numKVHeads, height: numQTiles, depth: 1)
                enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            } else if useFlash {
                let pipeline: MTLComputePipelineState
                switch dataType {
                case .float16: pipeline = flashPrefillPipelineF16
                case .float32: pipeline = flashPrefillPipelineF32
                case .float8: throw PagedAttentionError.invalidConfiguration("FlashAttention does not support FP8")
                }

                enc.setComputePipelineState(pipeline)
                enc.setBuffer(qBuffers[i], offset: 0, index: 0)
                enc.setBuffer(kPool, offset: 0, index: 1)
                enc.setBuffer(vPool, offset: 0, index: 2)
                enc.setBuffer(blockTable, offset: 0, index: 3)
                enc.setBuffer(outputBuffers[i], offset: 0, index: 4)

                var seqLenVar = UInt32(seqLen)
                var headDimVar = UInt32(headDim)
                var numHeadsVar = UInt32(numHeads)
                var numKVVar = UInt32(numKVHeads)
                var blockSizeVar = UInt32(blockSize)
                var causalVar = UInt32(causal ? 1 : 0)
                var windowStartVar = UInt32(effectiveWindowSize > 0 ? max(0, seqLen - effectiveWindowSize) : 0)

                enc.setBytes(&seqLenVar, length: 4, index: 5)
                enc.setBytes(&headDimVar, length: 4, index: 6)
                enc.setBytes(&numHeadsVar, length: 4, index: 7)
                enc.setBytes(&numKVVar, length: 4, index: 8)
                enc.setBytes(&blockSizeVar, length: 4, index: 9)
                enc.setBytes(&causalVar, length: 4, index: 10)
                enc.setBytes(&windowStartVar, length: 4, index: 11)

                let gqaRatio = numHeads / numKVHeads
                let rowsPerHead = 32 / gqaRatio
                let numQTiles = (seqLen + rowsPerHead - 1) / rowsPerHead
                let threadsPerTG = MTLSize(width: 256, height: 1, depth: 1)
                let threadgroups = MTLSize(width: numKVHeads, height: numQTiles, depth: 1)
                enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            } else {
                let shouldUseTiled = shouldUseTiledPass(seqLen: seqLen, layer: layer)

                let pipeline: MTLComputePipelineState
                if shouldUseTiled {
                    pipeline = tiledPipelineFor(dataType: dataType)
                } else {
                    pipeline = singlePassPipelineFor(dataType: dataType)
                }

                enc.setComputePipelineState(pipeline)
                enc.setBuffer(qBuffers[i], offset: 0, index: 0)
                enc.setBuffer(kPool, offset: 0, index: 1)
                enc.setBuffer(vPool, offset: 0, index: 2)
                enc.setBuffer(blockTable, offset: 0, index: 3)
                enc.setBuffer(outputBuffers[i], offset: 0, index: 4)

                var seqLenVar = UInt32(seqLen)
                var headDimVar = UInt32(headDim)
                var numHeadsVar = UInt32(numHeads)
                var numKVVar = UInt32(numKVHeads)
                var blockSizeVar = UInt32(blockSize)
                var causalVar = UInt32(causal ? 1 : 0)

                if shouldUseTiled {
                    let maxThreadgroupMemory = device.maxThreadgroupMemoryLength
                    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
                    let maxTileFromMemory = maxThreadgroupMemory / (headDim * MemoryLayout<Float>.stride)
                    let maxTileFromThreads = maxThreads / headDim
                    let qTileSizeVal = min(maxTileFromMemory, maxTileFromThreads, 32)
                    var qTileSizeVar = UInt32(qTileSizeVal)
                    var windowStartVar = UInt32(windowStart)

                    enc.setBytes(&seqLenVar, length: 4, index: 5)
                    enc.setBytes(&headDimVar, length: 4, index: 6)
                    enc.setBytes(&numHeadsVar, length: 4, index: 7)
                    enc.setBytes(&numKVVar, length: 4, index: 8)
                    enc.setBytes(&blockSizeVar, length: 4, index: 9)
                    enc.setBytes(&causalVar, length: 4, index: 10)
                    enc.setBytes(&qTileSizeVar, length: 4, index: 11)
                    enc.setBytes(&windowStartVar, length: 4, index: 12)

                    if dataType == .float8 {
                        enc.setBuffer(nil, offset: 0, index: 13)
                        enc.setBuffer(nil, offset: 0, index: 14)
                    }

                    let numQTiles = (seqLen + qTileSizeVal - 1) / qTileSizeVal
                    let threadsPerTG = MTLSize(width: headDim, height: qTileSizeVal, depth: 1)
                    let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)
                    let tileMemSize = qTileSizeVal * headDim * MemoryLayout<Float>.stride
                    enc.setThreadgroupMemoryLength(tileMemSize, index: 0)

                    enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                } else {
                    var windowStartVar = UInt32(windowStart)

                    enc.setBytes(&seqLenVar, length: 4, index: 5)
                    enc.setBytes(&headDimVar, length: 4, index: 6)
                    enc.setBytes(&numHeadsVar, length: 4, index: 7)
                    enc.setBytes(&numKVVar, length: 4, index: 8)
                    enc.setBytes(&blockSizeVar, length: 4, index: 9)
                    enc.setBytes(&causalVar, length: 4, index: 10)

                    guard let mBuffer = device.makeBuffer(length: seqLen * numHeads * MemoryLayout<Float>.stride, options: .storageModePrivate),
                          let lBuffer = device.makeBuffer(length: seqLen * numHeads * MemoryLayout<Float>.stride, options: .storageModePrivate) else {
                        throw PagedAttentionError.commandEncodingFailed("Failed to allocate M/L buffers")
                    }
                    enc.setBuffer(mBuffer, offset: 0, index: 11)
                    enc.setBuffer(lBuffer, offset: 0, index: 12)
                    enc.setBytes(&windowStartVar, length: 4, index: 13)

                    if dataType == .float8 {
                        enc.setBuffer(nil, offset: 0, index: 14)
                        enc.setBuffer(nil, offset: 0, index: 15)
                    }

                    let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride
                    enc.setThreadgroupMemoryLength(tileMemSize, index: 0)
                    enc.setThreadgroupMemoryLength(tileMemSize, index: 1)
                    enc.setThreadgroupMemoryLength(tileMemSize, index: 2)

                    let numQTiles = (seqLen + blockSize - 1) / blockSize
                    let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
                    let threadgroups = MTLSize(width: 1, height: numQTiles, depth: numHeads)

                    enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                }
            }

            enc.endEncoding()
            cb.commit()
        }
        // Wait for all command buffers to complete before checking errors
        cmdBufManager.waitForAll()
        for cb in commandBuffers where cb.status == .error {
            throw PagedAttentionError.commandExecutionFailed(cb.error?.localizedDescription ?? "Metal error in prefillLayers")
        }
    }

    public func decodeLayers(
        qBuffers: [MTLBuffer],
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTables: MTLBuffer,
        seqLengths: MTLBuffer,
        outputBuffers: [MTLBuffer],
        batchSize: Int,
        maxNumBlocks: Int,
        layers: [PagedLayerSpec],
        windowSize: Int = 0
    ) throws {
        guard qBuffers.count == layers.count && outputBuffers.count == layers.count else {
            throw PagedAttentionError.invalidConfiguration("buffer/layer count mismatch")
        }

        var commandBuffers: [MTLCommandBuffer] = []

        for i in 0..<layers.count {
            let cb = cmdBufManager.next()
            commandBuffers.append(cb)
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw PagedAttentionError.commandEncodingFailed("failed to create decode encoder for layer \(i)")
            }

            let layer = layers[i]
            let dataType = layer.dataType
            let headDim = layer.headDim
            let numHeads = layer.numHeads
            let numKVHeads = layer.numKVHeads
            let blockSize = layer.blockSize
            let effectiveWindowSize = windowSize > 0 ? windowSize : layer.windowSize

            let useMMA = shouldUseMMADecode(headDim: headDim, dataType: dataType)
            let useFlashDecode = useMMA ? false : shouldUseFlashDecode(headDim: headDim, dataType: dataType)
            let pipeline: MTLComputePipelineState
            if useMMA {
                pipeline = flashDecodePipelineMMA[0]
            } else {
                pipeline = decodePipelineFor(dataType: dataType, useFlash: useFlashDecode)
            }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(qBuffers[i], offset: 0, index: 0)
            enc.setBuffer(kPool, offset: 0, index: 1)
            enc.setBuffer(vPool, offset: 0, index: 2)
            enc.setBuffer(blockTables, offset: 0, index: 3)
            enc.setBuffer(seqLengths, offset: 0, index: 4)
            enc.setBuffer(outputBuffers[i], offset: 0, index: 5)

            var batchSizeVar = UInt32(batchSize)
            var headDimVar = UInt32(headDim)
            var numHeadsVar = UInt32(numHeads)
            var numKVVar = UInt32(numKVHeads)
            var blockSizeVar = UInt32(blockSize)
            var maxNumBlocksVar = UInt32(maxNumBlocks)
            var windowSizeVar = UInt32(effectiveWindowSize)

            enc.setBytes(&batchSizeVar, length: 4, index: 6)
            if useMMA {
                enc.setBytes(&numHeadsVar, length: 4, index: 7)
                enc.setBytes(&numKVVar, length: 4, index: 8)
                enc.setBytes(&blockSizeVar, length: 4, index: 9)
                enc.setBytes(&maxNumBlocksVar, length: 4, index: 10)
                enc.setBytes(&windowSizeVar, length: 4, index: 11)
            } else {
                enc.setBytes(&headDimVar, length: 4, index: 7)
                enc.setBytes(&numHeadsVar, length: 4, index: 8)
                enc.setBytes(&numKVVar, length: 4, index: 9)
                enc.setBytes(&blockSizeVar, length: 4, index: 10)
                enc.setBytes(&maxNumBlocksVar, length: 4, index: 11)
                enc.setBytes(&windowSizeVar, length: 4, index: 12)
            }

            if useMMA {
                let maxGQA = numHeads / numKVHeads
                let totalPS = maxGQA * 10 * MemoryLayout<Float>.stride
                let mlSize = maxGQA * 2 * MemoryLayout<Float>.stride
                enc.setThreadgroupMemoryLength(totalPS, index: 0)
                enc.setThreadgroupMemoryLength(mlSize, index: 1)
                let grid = MTLSize(width: 1, height: numKVHeads, depth: batchSize)
                let tgroup = MTLSize(width: 256, height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            } else {
                let grid = MTLSize(
                    width: headDim,
                    height: useFlashDecode ? numKVHeads : numHeads,
                    depth: batchSize
                )
                let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
            }
            enc.endEncoding()
            cb.commit()
        }
        // Wait for all command buffers to complete before checking errors
        cmdBufManager.waitForAll()
        for cb in commandBuffers where cb.status == .error {
            throw PagedAttentionError.commandExecutionFailed(cb.error?.localizedDescription ?? "Metal error in decodeLayers")
        }
    }

    public func appendLayers(
        keys: [MTLBuffer],
        values: [MTLBuffer],
        kPool: MTLBuffer,
        vPool: MTLBuffer,
        blockTable: MTLBuffer,
        tokenOffset: Int,
        numNewTokens: Int,
        layers: [PagedLayerSpec],
        kScaleBuffers: [MTLBuffer]? = nil,
        vScaleBuffers: [MTLBuffer]? = nil
    ) throws {
        guard keys.count == layers.count && values.count == layers.count else {
            throw PagedAttentionError.invalidConfiguration("buffer/layer count mismatch")
        }
        if let kScaleBuffers {
            guard kScaleBuffers.count == layers.count else {
                throw PagedAttentionError.invalidConfiguration("kScaleBuffers count must match layers count")
            }
        }
        if let vScaleBuffers {
            guard vScaleBuffers.count == layers.count else {
                throw PagedAttentionError.invalidConfiguration("vScaleBuffers count must match layers count")
            }
        }

        var commandBuffers: [MTLCommandBuffer] = []

        for i in 0..<layers.count {
            let layer = layers[i]
            let dataType = layer.dataType
            let headDim = layer.headDim
            let numKVHeads = layer.numKVHeads
            let blockSize = layer.blockSize

            if dataType == .float8 {
                let bytesPerBlock = blockSize * numKVHeads * headDim
                let maxBlocks = kPool.length / bytesPerBlock
                let scratchSize = maxBlocks * 2 * MemoryLayout<Float>.stride
                guard let scratchBuffer = device.makeBuffer(length: scratchSize, options: .storageModeShared) else {
                    throw PagedAttentionError.commandEncodingFailed("failed to create FP8 scratch buffer")
                }
                scratchBuffer.contents().bindMemory(to: Float.self, capacity: maxBlocks * 2).update(repeating: 0, count: maxBlocks * 2)

                // First pass: compute scales
                let cb1 = cmdBufManager.next()
                commandBuffers.append(cb1)
                guard let enc1 = cb1.makeComputeCommandEncoder() else {
                    throw PagedAttentionError.commandEncodingFailed("failed to create FP8 scale command buffer or encoder")
                }
                enc1.setComputePipelineState(appendScaleFP8)
                enc1.setBuffer(keys[i], offset: 0, index: 0)
                enc1.setBuffer(values[i], offset: 0, index: 1)
                enc1.setBuffer(blockTable, offset: 0, index: 2)

                var tokenOffsetVar1 = UInt32(tokenOffset)
                var numNewTokensVar1 = UInt32(numNewTokens)
                var numKVHeadsVar1 = UInt32(numKVHeads)
                var headDimVar1 = UInt32(headDim)
                var blockSizeVar1 = UInt32(blockSize)

                enc1.setBytes(&tokenOffsetVar1, length: 4, index: 3)
                enc1.setBytes(&numNewTokensVar1, length: 4, index: 4)
                enc1.setBytes(&numKVHeadsVar1, length: 4, index: 5)
                enc1.setBytes(&headDimVar1, length: 4, index: 6)
                enc1.setBytes(&blockSizeVar1, length: 4, index: 7)
                enc1.setBuffer(scratchBuffer, offset: 0, index: 8)

                let grid1 = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
                let tgroup1 = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc1.dispatchThreads(grid1, threadsPerThreadgroup: tgroup1)
                enc1.endEncoding()
                try commitAndWait(cb1)

                // Second pass: append with scales
                let cb2 = cmdBufManager.next()
                commandBuffers.append(cb2)
                guard let enc2 = cb2.makeComputeCommandEncoder() else {
                    throw PagedAttentionError.commandEncodingFailed("failed to create FP8 append command buffer or encoder")
                }
                enc2.setComputePipelineState(appendPipelineFP8)
                enc2.setBuffer(keys[i], offset: 0, index: 0)
                enc2.setBuffer(values[i], offset: 0, index: 1)
                enc2.setBuffer(kPool, offset: 0, index: 2)
                enc2.setBuffer(vPool, offset: 0, index: 3)
                enc2.setBuffer(blockTable, offset: 0, index: 4)

                var tokenOffsetVar = UInt32(tokenOffset)
                var numNewTokensVar = UInt32(numNewTokens)
                var numKVHeadsVar = UInt32(numKVHeads)
                var headDimVar = UInt32(headDim)
                var blockSizeVar = UInt32(blockSize)

                enc2.setBytes(&tokenOffsetVar, length: 4, index: 5)
                enc2.setBytes(&numNewTokensVar, length: 4, index: 6)
                enc2.setBytes(&numKVHeadsVar, length: 4, index: 7)
                enc2.setBytes(&headDimVar, length: 4, index: 8)
                enc2.setBytes(&blockSizeVar, length: 4, index: 9)
                enc2.setBuffer(scratchBuffer, offset: 0, index: 10)
                if let kScaleBuffers {
                    enc2.setBuffer(kScaleBuffers[i], offset: 0, index: 11)
                }
                if let vScaleBuffers {
                    enc2.setBuffer(vScaleBuffers[i], offset: 0, index: 12)
                }

                let grid2 = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
                let tgroup2 = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc2.dispatchThreads(grid2, threadsPerThreadgroup: tgroup2)
                enc2.endEncoding()
                cb2.commit()
            } else {
                let cb = cmdBufManager.next()
                commandBuffers.append(cb)
                guard let enc = cb.makeComputeCommandEncoder() else {
                    throw PagedAttentionError.commandEncodingFailed("failed to create append encoder for layer \(i)")
                }

                let pipeline = (dataType == .float16) ? appendPipelineF16 : appendPipeline
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(keys[i], offset: 0, index: 0)
                enc.setBuffer(values[i], offset: 0, index: 1)
                enc.setBuffer(kPool, offset: 0, index: 2)
                enc.setBuffer(vPool, offset: 0, index: 3)
                enc.setBuffer(blockTable, offset: 0, index: 4)

                var tokenOffsetVar = UInt32(tokenOffset)
                var numNewTokensVar = UInt32(numNewTokens)
                var numKVHeadsVar = UInt32(numKVHeads)
                var headDimVar = UInt32(headDim)
                var blockSizeVar = UInt32(blockSize)

                enc.setBytes(&tokenOffsetVar, length: 4, index: 5)
                enc.setBytes(&numNewTokensVar, length: 4, index: 6)
                enc.setBytes(&numKVHeadsVar, length: 4, index: 7)
                enc.setBytes(&headDimVar, length: 4, index: 8)
                enc.setBytes(&blockSizeVar, length: 4, index: 9)

                let grid = MTLSize(width: headDim, height: numKVHeads, depth: numNewTokens)
                let tgroup = MTLSize(width: min(headDim, 32), height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
                enc.endEncoding()
                cb.commit()
            }
        }

        // Wait for all command buffers to complete before checking errors
        cmdBufManager.waitForAll()
        for cb in commandBuffers where cb.status == .error {
            throw PagedAttentionError.commandExecutionFailed(cb.error?.localizedDescription ?? "Metal error in appendLayers")
        }
    }

    private func validatePrefill(_ request: PagedAttentionPrefillRequest) throws {
        guard request.seqLen > 0 else {
            throw PagedAttentionError.invalidConfiguration("seqLen must be positive.")
        }
        let layer = request.layer
        let numBlocks = logicalBlocks(tokenCount: request.seqLen, blockSize: layer.blockSize)
        try requireBuffer(request.q, name: "Q", atLeast: request.seqLen * layer.qBytesPerToken)
        try requireBuffer(request.blockTable, name: "blockTable", atLeast: numBlocks * MemoryLayout<Int32>.stride)
        try requireBuffer(request.output, name: "output", atLeast: request.seqLen * layer.qBytesPerToken)
        try requireBuffer(request.kPool, name: "K pool", atLeast: poolBytesFor(blocks: numBlocks, layer: layer))
        try requireBuffer(request.vPool, name: "V pool", atLeast: poolBytesFor(blocks: numBlocks, layer: layer))

        if shouldUseTiledPass(seqLen: request.seqLen, layer: layer) {
            try validateTiledPassDimensions(layer: layer)
        } else {
            try validateSinglePassDimensions(layer: layer)
        }
    }

    private func validateDecode(_ request: PagedAttentionDecodeRequest) throws {
        guard request.batchSize > 0 else {
            throw PagedAttentionError.invalidConfiguration("batchSize must be positive.")
        }
        guard request.maxNumBlocks > 0 else {
            throw PagedAttentionError.invalidConfiguration("maxNumBlocks must be positive.")
        }
        guard request.maxNumBlocks <= request.blockTables.length / (request.batchSize * MemoryLayout<Int32>.stride) else {
            throw PagedAttentionError.invalidConfiguration("maxNumBlocks exceeds blockTables buffer capacity")
        }
        let layer = request.layer
        try requireBuffer(request.q, name: "Q", atLeast: request.batchSize * layer.qBytesPerToken)
        try requireBuffer(request.blockTables, name: "blockTables", atLeast: request.batchSize * request.maxNumBlocks * MemoryLayout<Int32>.stride)
        try requireBuffer(request.seqLengths, name: "seqLengths", atLeast: request.batchSize * MemoryLayout<UInt32>.stride)
        try requireBuffer(request.output, name: "output", atLeast: request.batchSize * layer.qBytesPerToken)
        try requireBuffer(request.kPool, name: "K pool", atLeast: poolBytesFor(blocks: request.maxNumBlocks, layer: layer))
        try requireBuffer(request.vPool, name: "V pool", atLeast: poolBytesFor(blocks: request.maxNumBlocks, layer: layer))
    }

    private func validateAppend(_ request: PagedKVAppendRequest) throws {
        guard request.tokenOffset >= 0 else {
            throw PagedAttentionError.invalidConfiguration("tokenOffset must be non-negative.")
        }
        guard request.numNewTokens > 0 else {
            throw PagedAttentionError.invalidConfiguration("numNewTokens must be positive.")
        }
        let layer = request.layer
        let finalLength = request.tokenOffset + request.numNewTokens
        let numBlocks = logicalBlocks(tokenCount: finalLength, blockSize: layer.blockSize)
        try requireBuffer(request.keys, name: "new keys", atLeast: request.numNewTokens * layer.kvBytesPerToken)
        try requireBuffer(request.values, name: "new values", atLeast: request.numNewTokens * layer.kvBytesPerToken)
        try requireBuffer(request.blockTable, name: "blockTable", atLeast: numBlocks * MemoryLayout<Int32>.stride)
        try requireBuffer(request.kPool, name: "K pool", atLeast: poolBytesFor(blocks: numBlocks, layer: layer))
        try requireBuffer(request.vPool, name: "V pool", atLeast: poolBytesFor(blocks: numBlocks, layer: layer))
    }

    private func shouldUseTiledPass(seqLen: Int, layer: PagedLayerSpec) -> Bool {
        let singlePassThreadgroupBytes = layer.blockSize * layer.headDim * MemoryLayout<Float>.stride * 3
        let singlePassThreads = layer.headDim * layer.blockSize
        return singlePassThreadgroupBytes > device.maxThreadgroupMemoryLength ||
            seqLen > splitThreshold ||
            singlePassThreads > singlePassPipeline.maxTotalThreadsPerThreadgroup
    }

    private func validateSinglePassDimensions(layer: PagedLayerSpec) throws {
        let requestedThreads = layer.headDim * layer.blockSize
        let maxThreads = singlePassPipeline.maxTotalThreadsPerThreadgroup
        guard requestedThreads <= maxThreads else {
            throw PagedAttentionError.invalidConfiguration(
                "prefill single-pass requests \(requestedThreads) threads per threadgroup, but the pipeline supports at most \(maxThreads)."
            )
        }
        let requestedThreadgroupBytes = layer.blockSize * layer.headDim * MemoryLayout<Float>.stride * 3
        guard requestedThreadgroupBytes <= device.maxThreadgroupMemoryLength else {
            throw PagedAttentionError.invalidConfiguration(
                "prefill single-pass requests \(requestedThreadgroupBytes) bytes of threadgroup memory, but the device supports at most \(device.maxThreadgroupMemoryLength)."
            )
        }
    }

    private func validateTiledPassDimensions(layer: PagedLayerSpec) throws {
        let maxThreadgroupMemory = device.maxThreadgroupMemoryLength
        let maxThreads = tiledPipeline.maxTotalThreadsPerThreadgroup
        
        // For tiled pass, we need to verify that tile size is feasible
        // The tiled pass uses qTileSize * headDim * sizeof(Float) threadgroup memory
        // and qTileSize * headDim threads per threadgroup
        let maxTileFromMemory = maxThreadgroupMemory / (layer.headDim * MemoryLayout<Float>.stride)
        let maxTileFromThreads = maxThreads / layer.headDim
        let qTileSize = min(maxTileFromMemory, maxTileFromThreads, 32)
        
        guard qTileSize > 0 else {
            throw PagedAttentionError.invalidConfiguration(
                "tiled pass: headDim \(layer.headDim) exceeds device threadgroup memory/thread limits"
            )
        }
    }

    private func requireBuffer(_ buffer: MTLBuffer, name: String, atLeast expected: Int) throws {
        guard buffer.length >= expected else {
            throw PagedAttentionError.bufferTooSmall(name: name, expected: expected, actual: buffer.length)
        }
    }

    private func logicalBlocks(tokenCount: Int, blockSize: Int) -> Int {
        (tokenCount + blockSize - 1) / blockSize
    }

    private func poolBytesFor(blocks: Int, layer: PagedLayerSpec) -> Int {
        blocks * layer.blockSize * layer.kvBytesPerToken
    }

    private func commitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            let message = commandBuffer.error?.localizedDescription ?? "unknown Metal command buffer failure"
            throw PagedAttentionError.commandExecutionFailed(message)
        }
    }
}

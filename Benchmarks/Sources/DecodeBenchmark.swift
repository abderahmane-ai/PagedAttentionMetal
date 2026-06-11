import Foundation
import Metal
import PagedAttentionMetal

struct DecodeBenchmark {
    let device: MTLDevice
    let engine: PagedAttentionEngine

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = device
        self.engine = try PagedAttentionEngine()
    }

    func run(results: inout ResultsTable, config: BenchmarkConfig? = nil) throws {
        if let config {
            let result = try runSingleDecode(config: config)
            results.add(result)
            return
        }

        try runBatchSizeVariants(results: &results)
        try runSeqLenVariants(results: &results)
        try runSlidingWindowVariants(results: &results)
        try runDataTypeVariants(results: &results)
        try runMultiLayerVariants(results: &results)
    }

    private func runBatchSizeVariants(results: inout ResultsTable) throws {
        let batchSizes = [1, 2, 4, 8, 16, 32]
        var progress = ProgressBar(total: batchSizes.count)
        for batch in batchSizes {
            var cfg = BenchmarkConfig()
            cfg.batchSize = batch
            cfg.seqLen = 1024
            cfg.iterations = 15
            cfg.warmupIterations = 3
            let result = try runSingleDecode(config: cfg, name: "DecodeBatch")
            results.add(result)
            progress.increment("batch=\(batch)")
        }
    }

    private func runSeqLenVariants(results: inout ResultsTable) throws {
        let seqLens = [1024, 2048, 4096, 8192]
        var progress = ProgressBar(total: seqLens.count)
        for seqLen in seqLens {
            var cfg = BenchmarkConfig()
            cfg.batchSize = 4
            cfg.seqLen = seqLen
            cfg.iterations = 15
            cfg.warmupIterations = 3
            let result = try runSingleDecode(config: cfg, name: "DecodeSeqLen")
            results.add(result)
            progress.increment("seqLen=\(seqLen)")
        }
    }

    private func runSlidingWindowVariants(results: inout ResultsTable) throws {
        var progress = ProgressBar(total: 2)
        var cfg = BenchmarkConfig()
        cfg.batchSize = 4
        cfg.seqLen = 4096
        cfg.iterations = 15
        cfg.warmupIterations = 3

        cfg.windowSize = 512
        var result = try runSingleDecode(config: cfg, name: "DecodeWindow")
        results.add(result)
        progress.increment("window=512")

        cfg.windowSize = 0
        result = try runSingleDecode(config: cfg, name: "DecodeWindow")
        results.add(result)
        progress.increment("full")
    }

    private func runDataTypeVariants(results: inout ResultsTable) throws {
        let types: [PagedAttentionDataType] = [.float32, .float16, .float8]
        var progress = ProgressBar(total: types.count)
        for dtype in types {
            var cfg = BenchmarkConfig()
            cfg.batchSize = 4
            cfg.seqLen = 1024
            cfg.dataType = dtype
            cfg.iterations = 15
            cfg.warmupIterations = 3
            let result = try runSingleDecode(config: cfg, name: "DecodeDT")
            results.add(result)
            progress.increment(dtype == .float32 ? "FP32" : dtype == .float16 ? "FP16" : "FP8")
        }
    }

    private func runMultiLayerVariants(results: inout ResultsTable) throws {
        let layerCounts = [1, 4, 8, 16]
        var progress = ProgressBar(total: layerCounts.count)
        for layers in layerCounts {
            var cfg = BenchmarkConfig()
            cfg.batchSize = 4
            cfg.seqLen = 1024
            cfg.numLayers = layers
            cfg.iterations = 15
            cfg.warmupIterations = 3
            let result = try runSingleDecode(config: cfg, name: "DecodeLayers")
            results.add(result)
            progress.increment("layers=\(layers)")
        }
    }

    private func configDescription(_ config: BenchmarkConfig) -> String {
        var parts: [String] = []
        parts.append("batch=\(config.batchSize)")
        parts.append("ctx=\(config.seqLen)")
        parts.append(config.dataType == .float32 ? "FP32" : config.dataType == .float16 ? "FP16" : "FP8")
        if config.windowSize > 0 { parts.append("window=\(config.windowSize)") }
        if config.numLayers > 1 { parts.append("layers=\(config.numLayers)") }
        return parts.joined(separator: ", ")
    }

    func runSingleDecode(config: BenchmarkConfig, name: String = "Decode") throws -> BenchmarkResult {
        let maxNumBlocks = (config.seqLen + config.blockSize - 1) / config.blockSize
        let batchManager = BatchKVCacheManager(
            device: device,
            maxBatchSize: config.batchSize,
            maxSequenceBlocks: maxNumBlocks,
            maxBlocks: 2048,
            blockSize: config.blockSize,
            headDim: config.headDim,
            numKVHeads: config.numKVHeads,
            dataType: config.dataType
        )

        let sequenceIDs = (0..<config.batchSize).map { $0 }
        for id in sequenceIDs {
            try batchManager.allocateSequence(id: id)
            try batchManager.appendTokens(toSequence: id, count: config.seqLen)
        }

        let qSize = config.batchSize * config.numHeads * config.headDim
        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }

        let kScale: MTLBuffer?
        let vScale: MTLBuffer?
        if config.dataType == .float8 {
            let scales = makeScaleBuffers(device: device)
            kScale = scales.kScale
            vScale = scales.vScale
        } else {
            kScale = nil
            vScale = nil
        }

        guard let qBuffer = makeTypedBuffer(device: device, floats: q, dataType: config.dataType),
              let outputBuffer = makeTypedOutputBuffer(device: device, elementCount: qSize, dataType: config.dataType) else {
            fatalError("Buffer creation failed")
        }

        let layer = PagedLayerSpec(
            headDim: config.headDim,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            blockSize: config.blockSize,
            dataType: config.dataType,
            windowSize: config.windowSize
        )

        for id in sequenceIDs {
            let kvSize = config.seqLen * config.numKVHeads * config.headDim
            let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
            let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }
            guard let kBuffer = makeTypedBuffer(device: device, floats: k, dataType: config.dataType),
                  let vBuffer = makeTypedBuffer(device: device, floats: v, dataType: config.dataType) else {
                fatalError("K/V buffer creation failed")
            }
            try engine.appendToCache(PagedKVAppendRequest(
                keys: kBuffer,
                values: vBuffer,
                kPool: batchManager.kPoolBuffer,
                vPool: batchManager.vPoolBuffer,
                blockTable: try batchManager.getBatchBlockTableBuffer(forBatch: [id]),
                tokenOffset: 0,
                numNewTokens: config.seqLen,
                layer: layer,
                kScaleBuffer: kScale,
                vScaleBuffer: vScale
            ))
        }

        let seqLengths = (0..<config.batchSize).map { _ in UInt32(config.seqLen) }
        guard let seqLenBuffer = device.makeBuffer(
            bytes: seqLengths,
            length: seqLengths.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            fatalError("Buffer creation failed")
        }

        let blockTablesBuffer = try batchManager.getBatchBlockTableBuffer(forBatch: sequenceIDs)

        let times = try measureGpuTime(iterations: config.iterations, warmup: config.warmupIterations) {
            for _ in 0..<config.numLayers {
                try engine.decode(PagedAttentionDecodeRequest(
                    q: qBuffer,
                    kPool: batchManager.kPoolBuffer,
                    vPool: batchManager.vPoolBuffer,
                    blockTables: blockTablesBuffer,
                    seqLengths: seqLenBuffer,
                    output: outputBuffer,
                    batchSize: config.batchSize,
                    maxNumBlocks: maxNumBlocks,
                    layer: layer,
                    kScaleBuffer: kScale,
                    vScaleBuffer: vScale
                ))
            }
        }

        let meanMs = times.reduce(0, +) / Double(times.count)
        let totalTokens = Double(config.batchSize) * Double(config.numLayers)
        let throughput = totalTokens / (meanMs / 1000.0)

        let cfgDesc = configDescription(config)

        for id in sequenceIDs {
            batchManager.freeSequence(id: id)
        }

        return computeStats(times: times, name: name, config: cfgDesc, throughputTokensPerSec: throughput)
    }
}

func runBatchScalingBenchmark(config: BenchmarkConfig) throws {
    let batchSizes = [1, 2, 4, 8, 16, 32]
    print("\n=== Decode Batch Scaling ===")
    let bench = try DecodeBenchmark()
    for batch in batchSizes {
        var cfg = config
        cfg.batchSize = batch
        cfg.seqLen = 1024
        let result = try bench.runSingleDecode(config: cfg, name: "DecodeBatchScaling")
        print("  batch=\(batch): mean=\(String(format: "%.2f", result.meanMs))ms")
    }
}

func runSeqLenScalingBenchmark(config: BenchmarkConfig) throws {
    let seqLens = [1024, 2048, 4096, 8192]
    print("\n=== Decode Context Length Scaling ===")
    let bench = try DecodeBenchmark()
    for seqLen in seqLens {
        var cfg = config
        cfg.seqLen = seqLen
        cfg.batchSize = 4
        let result = try bench.runSingleDecode(config: cfg, name: "DecodeCtxScaling")
        print("  ctx=\(seqLen): mean=\(String(format: "%.2f", result.meanMs))ms")
    }
}

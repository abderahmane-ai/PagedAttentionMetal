import Foundation
import Metal
import PagedAttentionMetal

struct PrefillBenchmark {
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
            let result = try runSinglePrefill(config: config)
            results.add(result)
            return
        }

        try runSeqLenVariants(results: &results)
        try runSlidingWindowVariants(results: &results)
        try runDataTypeVariants(results: &results)
        try runChunkedVariants(results: &results)
        try runMultiLayerVariants(results: &results)
    }

    private func runSeqLenVariants(results: inout ResultsTable) throws {
        let seqLens = [128, 256, 512, 1024, 2048, 4096]
        var progress = ProgressBar(total: seqLens.count)
        for seqLen in seqLens {
            var cfg = BenchmarkConfig()
            cfg.seqLen = seqLen
            cfg.iterations = 10
            cfg.warmupIterations = 3
            let result = try runSinglePrefill(config: cfg, name: "PrefillSeqLen")
            results.add(result)
            progress.increment("seqLen=\(seqLen)")
        }
    }

    private func runSlidingWindowVariants(results: inout ResultsTable) throws {
        var progress = ProgressBar(total: 2)
        var cfg = BenchmarkConfig()
        cfg.seqLen = 2048
        cfg.iterations = 10
        cfg.warmupIterations = 3

        cfg.windowSize = 512
        var result = try runSinglePrefill(config: cfg, name: "PrefillWindow")
        results.add(result)
        progress.increment("window=512")

        cfg.windowSize = 0
        result = try runSinglePrefill(config: cfg, name: "PrefillWindow")
        results.add(result)
        progress.increment("full")
    }

    private func runDataTypeVariants(results: inout ResultsTable) throws {
        let types: [PagedAttentionDataType] = [.float32, .float16, .float8]
        var progress = ProgressBar(total: types.count)
        for dtype in types {
            var cfg = BenchmarkConfig()
            cfg.seqLen = 1024
            cfg.dataType = dtype
            cfg.iterations = 10
            cfg.warmupIterations = 3
            let result = try runSinglePrefill(config: cfg, name: "PrefillDT")
            results.add(result)
            progress.increment(dtype == .float32 ? "FP32" : dtype == .float16 ? "FP16" : "FP8")
        }
    }

    private func runChunkedVariants(results: inout ResultsTable) throws {
        var progress = ProgressBar(total: 2)
        var cfg = BenchmarkConfig()
        cfg.seqLen = 2048
        cfg.iterations = 10
        cfg.warmupIterations = 3

        cfg.chunkSize = 512
        var result = try runSinglePrefill(config: cfg, name: "PrefillChunked")
        results.add(result)
        progress.increment("chunk=512")

        cfg.chunkSize = 0
        result = try runSinglePrefill(config: cfg, name: "PrefillChunked")
        results.add(result)
        progress.increment("no-chunk")
    }

    private func runMultiLayerVariants(results: inout ResultsTable) throws {
        let layerCounts = [1, 4, 8, 16]
        var progress = ProgressBar(total: layerCounts.count)
        for layers in layerCounts {
            var cfg = BenchmarkConfig()
            cfg.seqLen = 1024
            cfg.numLayers = layers
            cfg.iterations = 10
            cfg.warmupIterations = 3
            let result = try runSinglePrefill(config: cfg, name: "PrefillLayers")
            results.add(result)
            progress.increment("layers=\(layers)")
        }
    }

    private func configDescription(_ config: BenchmarkConfig) -> String {
        var parts: [String] = []
        parts.append("seqLen=\(config.seqLen)")
        if config.batchSize > 1 { parts.append("batch=\(config.batchSize)") }
        parts.append(config.dataType == .float32 ? "FP32" : config.dataType == .float16 ? "FP16" : "FP8")
        if config.windowSize > 0 { parts.append("window=\(config.windowSize)") }
        if config.chunkSize > 0 { parts.append("chunk=\(config.chunkSize)") }
        if config.numLayers > 1 { parts.append("layers=\(config.numLayers)") }
        return parts.joined(separator: ", ")
    }

    func runSinglePrefill(config: BenchmarkConfig, name: String = "Prefill") throws -> BenchmarkResult {
        let cacheManager = KVCacheManager(
            device: device,
            maxBlocks: 2048,
            blockSize: config.blockSize,
            headDim: config.headDim,
            numKVHeads: config.numKVHeads,
            dataType: config.dataType
        )

        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        try cacheManager.appendTokens(toSequence: sequenceID, count: config.seqLen)

        let qSize = config.seqLen * config.numHeads * config.headDim
        let kvSize = config.seqLen * config.numKVHeads * config.headDim
        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }

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
              let kBuffer = makeTypedBuffer(device: device, floats: k, dataType: config.dataType),
              let vBuffer = makeTypedBuffer(device: device, floats: v, dataType: config.dataType),
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
        let blockTableBuffer = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)

        try engine.appendToCache(PagedKVAppendRequest(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: blockTableBuffer,
            tokenOffset: 0,
            numNewTokens: config.seqLen,
            layer: layer,
            kScaleBuffer: kScale,
            vScaleBuffer: vScale
        ))

        let times = try measureGpuTime(iterations: config.iterations, warmup: config.warmupIterations) {
            if config.chunkSize > 0 {
                try engine.prefill(
                    q: qBuffer,
                    kPool: cacheManager.kPoolBuffer,
                    vPool: cacheManager.vPoolBuffer,
                    blockTable: blockTableBuffer,
                    seqLen: config.seqLen,
                    headDim: config.headDim,
                    numHeads: config.numHeads,
                    numKVHeads: config.numKVHeads,
                    blockSize: config.blockSize,
                    causal: true,
                    output: outputBuffer,
                    dataType: config.dataType,
                    windowSize: config.windowSize,
                    chunkSize: config.chunkSize
                )
            } else {
                for _ in 0..<config.numLayers {
                    try engine.prefill(PagedAttentionPrefillRequest(
                        q: qBuffer,
                        kPool: cacheManager.kPoolBuffer,
                        vPool: cacheManager.vPoolBuffer,
                        blockTable: blockTableBuffer,
                        output: outputBuffer,
                        seqLen: config.seqLen,
                        layer: layer,
                        causal: true,
                        kScaleBuffer: kScale,
                        vScaleBuffer: vScale
                    ))
                }
            }
        }

        let meanMs = times.reduce(0, +) / Double(times.count)
        let totalTokens = config.seqLen * config.numLayers
        let throughput = Double(totalTokens) / (meanMs / 1000.0)

        let cfgDesc = configDescription(config)
        cacheManager.freeSequence(id: sequenceID)

        return computeStats(times: times, name: name, config: cfgDesc, throughputTokensPerSec: throughput)
    }
}

func runScalingBenchmark(config: BenchmarkConfig) throws {
    let seqLengths = [128, 256, 512, 1024, 2048, 4096]
    print("\n=== Prefill Scaling (seqLen) ===")
    let bench = try PrefillBenchmark()
    for seqLen in seqLengths {
        var cfg = config
        cfg.seqLen = seqLen
        let result = try bench.runSinglePrefill(config: cfg, name: "PrefillScaling")
        print("  seqLen=\(seqLen): mean=\(String(format: "%.2f", result.meanMs))ms")
    }
}

func runDataTypeComparison(config: BenchmarkConfig) throws {
    let types: [PagedAttentionDataType] = [.float32, .float16]
    print("\n=== Prefill Data Type Comparison ===")
    let bench = try PrefillBenchmark()
    for dataType in types {
        var cfg = config
        cfg.dataType = dataType
        let result = try bench.runSinglePrefill(config: cfg, name: "PrefillDT")
        let label = dataType == .float32 ? "FP32" : "FP16"
        print("  \(label): mean=\(String(format: "%.2f", result.meanMs))ms")
    }
}

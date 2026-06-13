import Foundation
import Metal
import PagedAttentionMetal

struct PrecisionBenchmark {
    let device: MTLDevice
    let engine: PagedAttentionEngine

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = device
        self.engine = try PagedAttentionEngine()
    }

    func run(results: inout ResultsTable) throws {
        try runDataTypePrecision(results: &results)
        try runFP8ErrorAnalysis(results: &results)
        try runGQAPrecision(results: &results)
    }

    func run(results: inout ResultsTable, config: BenchmarkConfig) throws {
        try runDataTypePrecision(results: &results)
        try runFP8ErrorAnalysis(results: &results)
        try runGQAPrecision(results: &results)
    }

    private func runDataTypePrecision(results: inout ResultsTable) throws {
        let seqLen = 1024
        let headDim = 128
        let numHeads = 8
        let numKVHeads = 2
        let blockSize = 16
        let qSize = seqLen * numHeads * headDim
        let kvSize = seqLen * numKVHeads * headDim

        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }

        var progress = ProgressBar(total: 4)

        let fp32Result = try runPrecisionPrefill(
            seqLen: seqLen, headDim: headDim, numHeads: numHeads,
            numKVHeads: numKVHeads, blockSize: blockSize,
            q: q, k: k, v: v, dataType: .float32
        )
        results.add(fp32Result.result)
        progress.increment("FP32")

        let fp16Result = try runPrecisionPrefill(
            seqLen: seqLen, headDim: headDim, numHeads: numHeads,
            numKVHeads: numKVHeads, blockSize: blockSize,
            q: q, k: k, v: v, dataType: .float16
        )
        results.add(fp16Result.result)
        progress.increment("FP16")

        let fp8Result = try runPrecisionPrefill(
            seqLen: seqLen, headDim: headDim, numHeads: numHeads,
            numKVHeads: numKVHeads, blockSize: blockSize,
            q: q, k: k, v: v, dataType: .float8
        )
        results.add(fp8Result.result)
        progress.increment("FP8")

        let fp32Values = decodeOutput(buffer: fp32Result.output, count: qSize, dataType: .float32)
        let fp16Values = decodeOutput(buffer: fp16Result.output, count: qSize, dataType: .float16)
        let fp8Values = decodeOutput(buffer: fp8Result.output, count: qSize, dataType: .float8)

        let fp16Errors: [Double] = zip(fp32Values, fp16Values).map { Double(abs($0 - $1)) }
        let fp8Errors: [Double] = zip(fp32Values, fp8Values).map { Double(abs($0 - $1)) }

        let fp16MAE = fp16Errors.reduce(0.0, +) / Double(fp16Errors.count)
        let fp8MAE = fp8Errors.reduce(0.0, +) / Double(fp8Errors.count)
        let fp16MSE = fp16Errors.reduce(0.0) { $0 + $1 * $1 } / Double(fp16Errors.count)
        let fp8MSE = fp8Errors.reduce(0.0) { $0 + $1 * $1 } / Double(fp8Errors.count)

        results.add(BenchmarkResult(
            name: "PrecisionError",
            config: "FP16-vs-FP32",
            throughputTokensPerSec: fp16MAE,
            memoryMB: fp16MSE
        ))
        results.add(BenchmarkResult(
            name: "PrecisionError",
            config: "FP8-vs-FP32",
            throughputTokensPerSec: fp8MAE,
            memoryMB: fp8MSE
        ))
        progress.increment("Errors")
    }

    private func runFP8ErrorAnalysis(results: inout ResultsTable) throws {
        let seqLen = 512
        let headDim = 128
        let numHeads = 8
        let numKVHeads = 2
        let blockSize = 16
        let qSize = seqLen * numHeads * headDim
        let kvSize = seqLen * numKVHeads * headDim

        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }

        let fp32Result = try runPrecisionPrefill(
            seqLen: seqLen, headDim: headDim, numHeads: numHeads,
            numKVHeads: numKVHeads, blockSize: blockSize,
            q: q, k: k, v: v, dataType: .float32
        )
        let fp8Result = try runPrecisionPrefill(
            seqLen: seqLen, headDim: headDim, numHeads: numHeads,
            numKVHeads: numKVHeads, blockSize: blockSize,
            q: q, k: k, v: v, dataType: .float8
        )

        let fp32Values = decodeOutput(buffer: fp32Result.output, count: qSize, dataType: .float32)
        let fp8Values = decodeOutput(buffer: fp8Result.output, count: qSize, dataType: .float8)
        let errors: [Double] = zip(fp32Values, fp8Values).map { Double(abs($0 - $1)) }

        let bins: [Double] = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        var histogram = [Int](repeating: 0, count: bins.count + 1)
        for err in errors {
            var placed = false
            for (i, threshold) in bins.enumerated() {
                if err < threshold {
                    histogram[i] += 1
                    placed = true
                    break
                }
            }
            if !placed { histogram[bins.count] += 1 }
        }

        var progress = ProgressBar(total: histogram.count)
        for (i, count) in histogram.enumerated() {
            let label: String
            if i == 0 { label = "<0.001" }
            else if i < bins.count { label = "<\(bins[i])" }
            else { label = ">=\(bins.last!)" }
            results.add(BenchmarkResult(
                name: "FP8Histogram",
                config: label,
                throughputTokensPerSec: Double(count),
                memoryMB: Double(count) / Double(errors.count) * 100
            ))
            progress.increment("bin \(label): \(count)")
        }
    }

    private func runGQAPrecision(results: inout ResultsTable) throws {
        let seqLen = 512
        let headDim = 128
        let blockSize = 16
        let qSize = seqLen * 8 * headDim
        let maxKVHeads = 2
        let kvSize = seqLen * maxKVHeads * headDim

        let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
        let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
        let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }

        let gqaRatios: [(numHeads: Int, numKVHeads: Int, label: String)] = [
            (1, 1, "MHA"),
            (8, 2, "GQA-4x"),
            (8, 1, "MQA"),
        ]

        var progress = ProgressBar(total: gqaRatios.count)
        for (numHeads, numKVHeads, label) in gqaRatios {
            let result = try runPrecisionPrefill(
                seqLen: seqLen, headDim: headDim,
                numHeads: numHeads, numKVHeads: numKVHeads,
                blockSize: blockSize,
                q: q, k: k, v: v, dataType: .float32
            )
            results.add(BenchmarkResult(
                name: "GQAPrecision",
                config: label,
                meanMs: result.result.meanMs,
                medianMs: result.result.medianMs,
                minMs: result.result.minMs,
                maxMs: result.result.maxMs,
                stdMs: result.result.stdMs,
                throughputTokensPerSec: Double(seqLen) / (result.result.meanMs / 1000.0),
                memoryMB: 0
            ))
            progress.increment(label)
        }
    }

    struct PrecisionOutput {
        let result: BenchmarkResult
        let output: MTLBuffer
    }

    func runPrecisionPrefill(
        seqLen: Int, headDim: Int, numHeads: Int, numKVHeads: Int, blockSize: Int,
        q: [Float], k: [Float], v: [Float], dataType: PagedAttentionDataType
    ) throws -> PrecisionOutput {
        let cacheManager = try KVCacheManager(
            device: device,
            maxBlocks: 512,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: dataType
        )

        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)

        let qSize = seqLen * numHeads * headDim

        let kScale: MTLBuffer?
        let vScale: MTLBuffer?
        if dataType == .float8 {
            let scales = makeScaleBuffers(device: device)
            kScale = scales.kScale
            vScale = scales.vScale
        } else {
            kScale = nil
            vScale = nil
        }

        guard let qBuffer = makeTypedBuffer(device: device, floats: q, dataType: dataType),
              let kBuffer = makeTypedBuffer(device: device, floats: k, dataType: dataType),
              let vBuffer = makeTypedBuffer(device: device, floats: v, dataType: dataType),
              let outputBuffer = makeTypedOutputBuffer(device: device, elementCount: qSize, dataType: dataType) else {
            fatalError("Buffer creation failed")
        }

        let layer = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: dataType
        )
        let blockTableBuffer = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)

        try engine.appendToCache(PagedKVAppendRequest(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: blockTableBuffer,
            tokenOffset: 0,
            numNewTokens: seqLen,
            layer: layer,
            kScaleBuffer: kScale,
            vScaleBuffer: vScale
        ))

        let times = try measureGpuTime(iterations: 5, warmup: 2) {
            try engine.prefill(PagedAttentionPrefillRequest(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: blockTableBuffer,
                output: outputBuffer,
                seqLen: seqLen,
                layer: layer,
                causal: true,
                kScaleBuffer: kScale,
                vScaleBuffer: vScale
            ))
        }

        let dtypeLabel = dataType == .float32 ? "FP32" : dataType == .float16 ? "FP16" : "FP8"
        let result = computeStats(
            times: times,
            name: "Precision",
            config: "\(dtypeLabel), seqLen=\(seqLen)",
            throughputTokensPerSec: Double(seqLen) / (times.reduce(0, +) / Double(times.count) / 1000.0)
        )

        cacheManager.freeSequence(id: sequenceID)

        return PrecisionOutput(result: result, output: outputBuffer)
    }

    func decodeOutput(buffer: MTLBuffer, count: Int, dataType: PagedAttentionDataType) -> [Float] {
        switch dataType {
        case .float32:
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            return (0..<count).map { ptr[$0] }
        case .float16:
            let ptr = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<count).map { Float(ptr[$0]) }
        case .float8:
            let ptr = buffer.contents().assumingMemoryBound(to: UInt8.self)
            return (0..<count).map { Float(ptr[$0]) / 127.5 - 1.0 }
        }
    }
}

func runFP8vsFP16Comparison() throws {
    let bench = try PrecisionBenchmark()
    let seqLen = 1024
    let headDim = 128
    let numHeads = 8
    let numKVHeads = 2
    let blockSize = 16
    let qSize = seqLen * numHeads * headDim
    let kvSize = seqLen * numKVHeads * headDim
    let q = (0..<qSize).map { _ in Float.random(in: -1...1) }
    let k = (0..<kvSize).map { _ in Float.random(in: -1...1) }
    let v = (0..<kvSize).map { _ in Float.random(in: -1...1) }

    let fp16Result = try bench.runPrecisionPrefill(
        seqLen: seqLen, headDim: headDim, numHeads: numHeads,
        numKVHeads: numKVHeads, blockSize: blockSize,
        q: q, k: k, v: v, dataType: .float16
    )
    let fp8Result = try bench.runPrecisionPrefill(
        seqLen: seqLen, headDim: headDim, numHeads: numHeads,
        numKVHeads: numKVHeads, blockSize: blockSize,
        q: q, k: k, v: v, dataType: .float8
    )

    let fp16Values = bench.decodeOutput(buffer: fp16Result.output, count: qSize, dataType: .float16)
    let fp8Values = bench.decodeOutput(buffer: fp8Result.output, count: qSize, dataType: .float8)
    let diff: [Double] = zip(fp16Values, fp8Values).map { Double(abs($0 - $1)) }
    let mae = diff.reduce(0.0, +) / Double(diff.count)
    let mse = diff.reduce(0.0) { $0 + $1 * $1 } / Double(diff.count)
    let maxErr = diff.max() ?? 0

    print("\n=== FP8 vs FP16 Precision Comparison ===")
    print("  MAE: \(String(format: "%.6f", mae))")
    print("  MSE: \(String(format: "%.6f", mse))")
    print("  MaxErr: \(String(format: "%.6f", maxErr))")
}

import Foundation
import Metal
import PagedAttentionMetal

public struct BenchmarkConfig: Sendable {
    public static let `default` = BenchmarkConfig()
    public var headDim: Int
    public var numHeads: Int
    public var numKVHeads: Int
    public var blockSize: Int
    public var batchSize: Int
    public var seqLen: Int
    public var dataType: PagedAttentionDataType
    public var windowSize: Int
    public var chunkSize: Int
    public var numLayers: Int
    public var iterations: Int
    public var warmupIterations: Int

    public init(
        headDim: Int = 128,
        numHeads: Int = 8,
        numKVHeads: Int = 2,
        blockSize: Int = 16,
        batchSize: Int = 1,
        seqLen: Int = 1024,
        dataType: PagedAttentionDataType = .float16,
        windowSize: Int = 0,
        chunkSize: Int = 0,
        numLayers: Int = 1,
        iterations: Int = 20,
        warmupIterations: Int = 3
    ) {
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.blockSize = blockSize
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.dataType = dataType
        self.windowSize = windowSize
        self.chunkSize = chunkSize
        self.numLayers = numLayers
        self.iterations = iterations
        self.warmupIterations = warmupIterations
    }
}

public struct BenchmarkResult: Sendable {
    public var name: String
    public var config: String
    public var meanMs: Double
    public var medianMs: Double
    public var minMs: Double
    public var maxMs: Double
    public var stdMs: Double
    public var throughputTokensPerSec: Double
    public var memoryMB: Double

    public init(
        name: String,
        config: String = "",
        meanMs: Double = 0,
        medianMs: Double = 0,
        minMs: Double = 0,
        maxMs: Double = 0,
        stdMs: Double = 0,
        throughputTokensPerSec: Double = 0,
        memoryMB: Double = 0
    ) {
        self.name = name
        self.config = config
        self.meanMs = meanMs
        self.medianMs = medianMs
        self.minMs = minMs
        self.maxMs = maxMs
        self.stdMs = stdMs
        self.throughputTokensPerSec = throughputTokensPerSec
        self.memoryMB = memoryMB
    }

    public func formatted() -> String {
        var parts = [
            "[\(name)]",
            config,
            String(format: "mean: %.2fms", meanMs),
            String(format: "median: %.2fms", medianMs),
            String(format: "min: %.2fms", minMs),
            String(format: "max: %.2fms", maxMs),
            String(format: "std: %.2fms", stdMs)
        ]
        if throughputTokensPerSec > 0 {
            parts.append(String(format: "throughput: %.0f tok/s", throughputTokensPerSec))
        }
        if memoryMB > 0 {
            parts.append(String(format: "memory: %.1f MB", memoryMB))
        }
        return parts.joined(separator: " | ")
    }

    public func csv() -> String {
        let throughputStr = throughputTokensPerSec > 0 ? String(format: "%.0f", throughputTokensPerSec) : ""
        let memoryStr = memoryMB > 0 ? String(format: "%.1f", memoryMB) : ""
        return "\(name),\(config),\(String(format: "%.2f", meanMs)),\(String(format: "%.2f", medianMs)),\(String(format: "%.2f", minMs)),\(String(format: "%.2f", maxMs)),\(String(format: "%.2f", stdMs)),\(throughputStr),\(memoryStr)"
    }
}

public func measureGpuTime(iterations: Int, warmup: Int, block: () throws -> Void) rethrows -> [Double] {
    for _ in 0..<warmup {
        try block()
    }
    var times: [Double] = []
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        try block()
        let end = CFAbsoluteTimeGetCurrent()
        times.append((end - start) * 1000)
    }
    return times
}

public func computeStats(times: [Double], name: String, config: String = "", throughputTokensPerSec: Double = 0, memoryMB: Double = 0) -> BenchmarkResult {
    let sorted = times.sorted()
    let count = times.count
    let mean = times.reduce(0, +) / Double(count)
    let median = sorted[count / 2]
    let minVal = sorted.first ?? 0
    let maxVal = sorted.last ?? 0
    let variance = times.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count)
    let std = sqrt(variance)
    return BenchmarkResult(
        name: name,
        config: config,
        meanMs: mean,
        medianMs: median,
        minMs: minVal,
        maxMs: maxVal,
        stdMs: std,
        throughputTokensPerSec: throughputTokensPerSec,
        memoryMB: memoryMB
    )
}

func makeTypedBuffer(device: MTLDevice, floats: [Float], dataType: PagedAttentionDataType) -> MTLBuffer? {
    switch dataType {
    case .float32:
        return device.makeBuffer(
            bytes: floats,
            length: floats.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
    case .float16:
        let halfs = floats.map { Float16($0) }
        return halfs.withUnsafeBytes { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: ptr.count,
                options: .storageModeShared
            )
        }
    case .float8:
        let bytes = floats.map { UInt8(($0 + 1.0) * 127.5) }
        return bytes.withUnsafeBytes { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: ptr.count,
                options: .storageModeShared
            )
        }
    }
}

func makeTypedOutputBuffer(device: MTLDevice, elementCount: Int, dataType: PagedAttentionDataType) -> MTLBuffer? {
    device.makeBuffer(
        length: elementCount * dataType.byteWidth,
        options: .storageModeShared
    )
}

func makeScaleBuffers(device: MTLDevice) -> (kScale: MTLBuffer, vScale: MTLBuffer) {
    let kScale = device.makeBuffer(length: MemoryLayout<Float>.stride * 1024, options: .storageModeShared)!
    let vScale = device.makeBuffer(length: MemoryLayout<Float>.stride * 1024, options: .storageModeShared)!
    kScale.contents().assumingMemoryBound(to: Float.self).pointee = 1.0
    vScale.contents().assumingMemoryBound(to: Float.self).pointee = 1.0
    return (kScale, vScale)
}

class ResultsTable {
    var results: [BenchmarkResult] = []

    func add(_ result: BenchmarkResult) {
        results.append(result)
    }

    func printTable() {
        Swift.print("\n" + String(repeating: "=", count: 120))
        Swift.print("Name                         Config            Mean(ms)  Median(ms) Min(ms)    Max(ms)    Std(ms)    Tok/s")
        Swift.print(String(repeating: "-", count: 120))
        for result in results {
            let name = result.name.padding(toLength: 26, withPad: " ", startingAt: 0).prefix(26)
            let cfg = result.config.padding(toLength: 16, withPad: " ", startingAt: 0).prefix(16)
            let mean = String(format: "%.2f", result.meanMs).padding(toLength: 8, withPad: " ", startingAt: 0)
            let med = String(format: "%.2f", result.medianMs).padding(toLength: 8, withPad: " ", startingAt: 0)
            let min = String(format: "%.2f", result.minMs).padding(toLength: 8, withPad: " ", startingAt: 0)
            let max = String(format: "%.2f", result.maxMs).padding(toLength: 8, withPad: " ", startingAt: 0)
            let std = String(format: "%.2f", result.stdMs).padding(toLength: 8, withPad: " ", startingAt: 0)
            let tp = result.throughputTokensPerSec > 0 ? String(format: "%.0f", result.throughputTokensPerSec) : ""
            Swift.print("\(name) \(cfg) \(mean) \(med) \(min) \(max) \(std) \(tp)")
        }
        Swift.print(String(repeating: "=", count: 120) + "\n")
    }

    func saveCSV(to path: String) throws {
        var csv = "Name,Config,Mean(ms),Median(ms),Min(ms),Max(ms),Std(ms),Throughput(tok/s),Memory(MB)\n"
        for result in results {
            csv += result.csv() + "\n"
        }
        try csv.write(toFile: path, atomically: true, encoding: .utf8)
    }

    func saveJSON(to path: String) throws {
        let dict = results.map { result -> [String: Any] in
            var d: [String: Any] = [
                "name": result.name,
                "config": result.config,
                "mean_ms": result.meanMs,
                "median_ms": result.medianMs,
                "min_ms": result.minMs,
                "max_ms": result.maxMs,
                "std_ms": result.stdMs,
            ]
            if result.throughputTokensPerSec > 0 {
                d["throughput_tok_per_sec"] = result.throughputTokensPerSec
            }
            if result.memoryMB > 0 {
                d["memory_mb"] = result.memoryMB
            }
            return d
        }
        let data = try JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }
}

public func computeStats(times: [Double]) -> BenchmarkResult {
    let sorted = times.sorted()
    let count = times.count
    let mean = times.reduce(0, +) / Double(count)
    let median = sorted[count / 2]
    let minVal = sorted.first ?? 0
    let maxVal = sorted.last ?? 0
    let variance = times.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(count)
    let std = sqrt(variance)
    return BenchmarkResult(
        name: "",
        meanMs: mean,
        medianMs: median,
        minMs: minVal,
        maxMs: maxVal,
        stdMs: std,
        throughputTokensPerSec: 0,
        memoryMB: 0
    )
}

struct ProgressBar {
    let total: Int
    var current: Int = 0

    mutating func increment(_ label: String = "") {
        current += 1
        let percentage = Double(current) / Double(total) * 100
        let filled = Int(percentage / 2)
        let bar = String(repeating: "█", count: filled) + String(repeating: "░", count: 50 - filled)
        print("\r[\(bar)] \(Int(percentage))% \(label)", terminator: "")
        fflush(stdout)
        if current == total {
            print()
        }
    }
}

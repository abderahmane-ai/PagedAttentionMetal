import Foundation
import Metal
import PagedAttentionMetal

enum BenchmarkCommand: String, CaseIterable {
    case all
    case prefill
    case decode
    case memory
    case precision
    case fpgrowth
    case prefillScaling = "prefill-scaling"
    case prefillDT = "prefill-dt"
    case decodeBatchScaling = "decode-batch-scaling"
    case decodeCtxScaling = "decode-ctx-scaling"
    case fp8Memory = "fp8-memory"
    case fp8VsFp16 = "fp8-vs-fp16"
}

func parseArguments() -> (command: BenchmarkCommand, config: BenchmarkConfig) {
    let args = Array(CommandLine.arguments.dropFirst())
    var command = BenchmarkCommand.all
    var config = BenchmarkConfig()

    var i = 0
    while i < args.count {
        switch args[i] {
        case "all": command = .all
        case "prefill": command = .prefill
        case "decode": command = .decode
        case "memory": command = .memory
        case "precision": command = .precision
        case "fpgrowth": command = .fpgrowth
        case "prefill-scaling": command = .prefillScaling
        case "prefill-dt": command = .prefillDT
        case "decode-batch-scaling": command = .decodeBatchScaling
        case "decode-ctx-scaling": command = .decodeCtxScaling
        case "fp8-memory": command = .fp8Memory
        case "fp8-vs-fp16": command = .fp8VsFp16
        case "--head-dim":
            i += 1
            if i < args.count { config.headDim = Int(args[i]) ?? config.headDim }
        case "--num-heads":
            i += 1
            if i < args.count { config.numHeads = Int(args[i]) ?? config.numHeads }
        case "--num-kv-heads":
            i += 1
            if i < args.count { config.numKVHeads = Int(args[i]) ?? config.numKVHeads }
        case "--block-size":
            i += 1
            if i < args.count { config.blockSize = Int(args[i]) ?? config.blockSize }
        case "--batch-size":
            i += 1
            if i < args.count { config.batchSize = Int(args[i]) ?? config.batchSize }
        case "--seq-len":
            i += 1
            if i < args.count { config.seqLen = Int(args[i]) ?? config.seqLen }
        case "--dtype":
            i += 1
            if i < args.count {
                switch args[i] {
                case "fp32": config.dataType = PagedAttentionDataType.float32
                case "fp16": config.dataType = PagedAttentionDataType.float16
                case "fp8": config.dataType = PagedAttentionDataType.float8
                default: break
                }
            }
        case "--window-size":
            i += 1
            if i < args.count { config.windowSize = Int(args[i]) ?? config.windowSize }
        case "--chunk-size":
            i += 1
            if i < args.count { config.chunkSize = Int(args[i]) ?? config.chunkSize }
        case "--num-layers":
            i += 1
            if i < args.count { config.numLayers = Int(args[i]) ?? config.numLayers }
        case "--iterations":
            i += 1
            if i < args.count { config.iterations = Int(args[i]) ?? config.iterations }
        case "--warmup":
            i += 1
            if i < args.count { config.warmupIterations = Int(args[i]) ?? config.warmupIterations }
        case "--help":
            printUsage()
            exit(0)
        default:
            break
        }
        i += 1
    }

    return (command, config)
}

func printUsage() {
    print("PagedAttentionMetal Benchmark Suite")
    print("Usage: Benchmarks [command] [options]")
    print()
    print("Commands:")
    for cmd in BenchmarkCommand.allCases {
        print("  \(cmd.rawValue)")
    }
    print()
    print("Options:")
    print("  --head-dim <n>      Head dimension (default: 128)")
    print("  --num-heads <n>     Number of query heads (default: 8)")
    print("  --num-kv-heads <n>  Number of KV heads (default: 2)")
    print("  --block-size <n>    Block size (default: 16)")
    print("  --batch-size <n>    Batch size (default: 1)")
    print("  --seq-len <n>       Sequence length (default: 1024)")
    print("  --dtype <type>      Data type: fp32, fp16, fp8 (default: fp16)")
    print("  --window-size <n>   Sliding window size, 0 = full (default: 0)")
    print("  --chunk-size <n>    Chunk size for chunked prefill (default: 0)")
    print("  --num-layers <n>    Number of layers to simulate (default: 1)")
    print("  --iterations <n>    Number of benchmark iterations (default: 20)")
    print("  --warmup <n>        Warmup iterations (default: 3)")
    print("  --help              Show this help")
}

func runFPGrowth(results: inout ResultsTable) throws {
    print("\nFPGrowth Scaling Analysis")
    print("Batch size scaling: 1, 2, 4, 8, 16, 32")
    print("Sequence length scaling: 128, 256, 512, 1024, 2048, 4096")

    let batchSizes = [1, 2, 4, 8, 16, 32]
    let seqLens = [128, 256, 512, 1024, 2048, 4096]

    var progress = ProgressBar(total: batchSizes.count + seqLens.count)

    let prefillBench = try PrefillBenchmark()
    for seqLen in seqLens {
        var cfg = BenchmarkConfig()
        cfg.seqLen = seqLen
        cfg.iterations = 10
        cfg.warmupIterations = 3
        cfg.dataType = PagedAttentionDataType.float16
        let result = try prefillBench.runSinglePrefill(config: cfg, name: "FPGrowth-Prefill")
        results.add(result)
        progress.increment("prefill seqLen=\(seqLen)")
    }

    let decodeBench = try DecodeBenchmark()
    for batch in batchSizes {
        var cfg = BenchmarkConfig()
        cfg.batchSize = batch
        cfg.seqLen = 1024
        cfg.iterations = 10
        cfg.warmupIterations = 3
        cfg.dataType = PagedAttentionDataType.float16
        let result = try decodeBench.runSingleDecode(config: cfg, name: "FPGrowth-Decode")
        results.add(result)
        progress.increment("decode batch=\(batch)")
    }
}

print("PagedAttentionMetal Benchmark Suite")
print("====================================\n")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Error: No Metal device available")
    exit(1)
}

print("Device: \(device.name)")
print("Memory: \(device.recommendedMaxWorkingSetSize / 1_073_741_824) GB\n")

let (command, config) = parseArguments()
var results = ResultsTable()

do {
    switch command {
    case .all:
        try PrefillBenchmark().run(results: &results)
        try DecodeBenchmark().run(results: &results)
        try MemoryBenchmark().run(results: &results)
        try PrecisionBenchmark().run(results: &results)
        try runScalingBenchmark(config: config)
        try runBatchScalingBenchmark(config: config)
        try runFP8MemoryComparison()
        try runFP8vsFP16Comparison()
    case .prefill:
        let bench = try PrefillBenchmark()
        if CommandLine.arguments.contains("--seq-len") || CommandLine.arguments.contains("--dtype") || CommandLine.arguments.contains("--batch-size") {
            try bench.run(results: &results, config: config)
        } else {
            try bench.run(results: &results)
        }
    case .decode:
        let bench = try DecodeBenchmark()
        if CommandLine.arguments.contains("--seq-len") || CommandLine.arguments.contains("--batch-size") || CommandLine.arguments.contains("--dtype") {
            try bench.run(results: &results, config: config)
        } else {
            try bench.run(results: &results)
        }
    case .memory:
        let bench = try MemoryBenchmark()
        if CommandLine.arguments.contains("--batch-size") || CommandLine.arguments.contains("--seq-len") {
            try bench.run(results: &results, config: config)
        } else {
            try bench.run(results: &results)
        }
    case .precision:
        try PrecisionBenchmark().run(results: &results)
    case .fpgrowth:
        try runFPGrowth(results: &results)
    case .prefillScaling:
        try runScalingBenchmark(config: config)
    case .prefillDT:
        try runDataTypeComparison(config: config)
    case .decodeBatchScaling:
        try runBatchScalingBenchmark(config: config)
    case .decodeCtxScaling:
        try runSeqLenScalingBenchmark(config: config)
    case .fp8Memory:
        try runFP8MemoryComparison()
    case .fp8VsFp16:
        try runFP8vsFP16Comparison()
    }

    print("\nResults:")
    results.printTable()

    let timestamp = "\(Int(Date().timeIntervalSince1970))"
    let resultsDir = "Benchmarks/Results"

    try FileManager.default.createDirectory(
        atPath: resultsDir,
        withIntermediateDirectories: true
    )

    let csvPath = "\(resultsDir)/benchmark_\(timestamp).csv"

    var csv = "Name,Config,Mean(ms),Median(ms),Min(ms),Max(ms),Std(ms),Throughput(tok/s),Memory(MB)\n"
    for result in results.results {
        csv += "\(result.name),\(result.meanMs),\(result.medianMs),\(result.minMs),\(result.maxMs),\(result.stdMs),\(result.throughputTokensPerSec),\(result.memoryMB)\n"
    }
    try csv.write(toFile: csvPath, atomically: true, encoding: .utf8)

    print("Saved: \(csvPath)\n")

} catch {
    print("Error: \(error)")
    exit(1)
}

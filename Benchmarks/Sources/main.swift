import Foundation
import Metal
import PagedAttentionMetal

print("PagedAttentionMetal Benchmark Suite")
print("====================================\n")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Error: No Metal device available")
    exit(1)
}

print("Device: \(device.name)")
print("Memory: \(device.recommendedMaxWorkingSetSize / 1_073_741_824) GB\n")

var results = ResultsTable()

do {
    let prefillBench = try PrefillBenchmark()
    try prefillBench.run(results: &results)
    
    let decodeBench = try DecodeBenchmark()
    try decodeBench.run(results: &results)
    
    let memoryBench = try MemoryBenchmark()
    try memoryBench.run(results: &results)
    
    let precisionBench = try PrecisionBenchmark()
    try precisionBench.run(results: &results)
    
    print("\nResults:")
    results.print()
    
    let timestamp = ISO8601DateFormatter().string(from: Date())
    let resultsDir = "Benchmarks/Results"
    
    try FileManager.default.createDirectory(
        atPath: resultsDir,
        withIntermediateDirectories: true
    )
    
    let csvPath = "\(resultsDir)/benchmark_\(timestamp).csv"
    let jsonPath = "\(resultsDir)/benchmark_\(timestamp).json"
    
    try results.saveCSV(to: csvPath)
    try results.saveJSON(to: jsonPath)
    
    print("Saved: \(csvPath)")
    print("Saved: \(jsonPath)\n")
    
} catch {
    print("Error: \(error)")
    exit(1)
}

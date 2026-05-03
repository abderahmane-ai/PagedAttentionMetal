import Foundation
import Metal
import PagedAttentionMetal

@available(macOS 11.0, iOS 14.0, *)
func runBenchmark() {
    print("Initializing PagedAttentionMetal Engine for Benchmarks...")
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
    
    let engine = try! PagedAttentionEngine()
    
    // Test parameters: A long sequence to saturate memory bandwidth
    let seqLen = 4096
    let headDim = 128
    let numHeads = 8
    let numKVHeads = 8
    let blockSize = 16
    let numBlocks = (seqLen + blockSize - 1) / blockSize
    let poolSize = numBlocks * blockSize
    
    print("\n--- Benchmark Configuration ---")
    print("Sequence Length: \(seqLen)")
    print("Head Dim:        \(headDim)")
    print("Num Heads:       \(numHeads)")
    print("Tokens in Cache: \(poolSize)")
    
    // === FP32 Setup ===
    let totalFP32Bytes = poolSize * numKVHeads * headDim * MemoryLayout<Float>.stride
    print("FP32 KV Cache Size: \(totalFP32Bytes / 1024 / 1024) MB")
    
    let bufQ_f32 = device.makeBuffer(length: seqLen * numHeads * headDim * 4, options: .storageModeShared)!
    let bufK_f32 = device.makeBuffer(length: poolSize * numKVHeads * headDim * 4, options: .storageModeShared)!
    let bufV_f32 = device.makeBuffer(length: poolSize * numKVHeads * headDim * 4, options: .storageModeShared)!
    let bufO_f32 = device.makeBuffer(length: seqLen * numHeads * headDim * 4, options: .storageModeShared)!
    
    // === FP16 Setup ===
    let totalFP16Bytes = poolSize * numKVHeads * headDim * MemoryLayout<Float16>.stride
    print("FP16 KV Cache Size: \(totalFP16Bytes / 1024 / 1024) MB")
    
    let bufQ_f16 = device.makeBuffer(length: seqLen * numHeads * headDim * 2, options: .storageModeShared)!
    let bufK_f16 = device.makeBuffer(length: poolSize * numKVHeads * headDim * 2, options: .storageModeShared)!
    let bufV_f16 = device.makeBuffer(length: poolSize * numKVHeads * headDim * 2, options: .storageModeShared)!
    let bufO_f16 = device.makeBuffer(length: seqLen * numHeads * headDim * 2, options: .storageModeShared)!
    
    var blockTable = [Int32](repeating: 0, count: numBlocks)
    for i in 0..<numBlocks { blockTable[i] = Int32(i) }
    let bufBT = device.makeBuffer(bytes: blockTable, length: numBlocks * 4, options: .storageModeShared)!
    
    let iterations = 100
    
    // Warmup
    engine.forward(q: bufQ_f32, kPool: bufK_f32, vPool: bufV_f32, blockTable: bufBT, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, numBlocks: numBlocks, blockSize: blockSize, output: bufO_f32, dataType: .float32)
    engine.forward(q: bufQ_f16, kPool: bufK_f16, vPool: bufV_f16, blockTable: bufBT, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, numBlocks: numBlocks, blockSize: blockSize, output: bufO_f16, dataType: .float16)

    // Benchmark FP32
    print("\nBenchmarking FP32 (\(iterations) iterations)...")
    let startF32 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        engine.forward(q: bufQ_f32, kPool: bufK_f32, vPool: bufV_f32, blockTable: bufBT, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, numBlocks: numBlocks, blockSize: blockSize, output: bufO_f32, dataType: .float32)
    }
    let endF32 = CFAbsoluteTimeGetCurrent()
    let timeF32 = (endF32 - startF32) / Double(iterations) * 1000.0
    
    // Benchmark FP16
    print("Benchmarking FP16 (\(iterations) iterations)...")
    let startF16 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        engine.forward(q: bufQ_f16, kPool: bufK_f16, vPool: bufV_f16, blockTable: bufBT, seqLen: seqLen, headDim: headDim, numHeads: numHeads, numKVHeads: numKVHeads, numBlocks: numBlocks, blockSize: blockSize, output: bufO_f16, dataType: .float16)
    }
    let endF16 = CFAbsoluteTimeGetCurrent()
    let timeF16 = (endF16 - startF16) / Double(iterations) * 1000.0
    
    print("\n=== Results ===")
    print("FP32 Latency: \(String(format: "%.3f", timeF32)) ms")
    print("FP16 Latency: \(String(format: "%.3f", timeF16)) ms")
    print("Speedup:      \(String(format: "%.2fx", timeF32 / timeF16))")
}

@main
struct ExampleApp {
    static func main() {
        if #available(macOS 11.0, iOS 14.0, *) {
            runBenchmark()
        } else {
            print("OS version not supported for FP16.")
        }
    }
}

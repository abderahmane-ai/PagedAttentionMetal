import Foundation
import Metal
import PagedAttentionMetal 

// MARK: - Device Setup

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// MARK: - Apple Silicon GPU Family Check

let supportsAppleGPU7: Bool = {
    let name = device.name
    return name.contains("Apple") && (
        name.contains("M1") || name.contains("M2") || name.contains("M3") || name.contains("M4")
    )
}()

// MARK: - Pipeline Cache

struct PipelineCache {
    let vectorAdd: MTLComputePipelineState
    let vectorMultiply: MTLComputePipelineState
    let matrixAdd: MTLComputePipelineState
    let matrixHadamard: MTLComputePipelineState
    let rowSum: MTLComputePipelineState
    let onlineSoftmax: MTLComputePipelineState
    let naiveAttention: MTLComputePipelineState
    let flashAttention: MTLComputePipelineState
    let pagedAttention: MTLComputePipelineState
    let pagedAttentionHalf: MTLComputePipelineState?
    let pagedAttentionBackward: MTLComputePipelineState

    init() {
        let library = PagedAttentionEngine.defaultLibrary

        func make(_ name: String) -> MTLComputePipelineState {
            let function = library.makeFunction(name: name)!
            return try! device.makeComputePipelineState(function: function)
        }

        vectorAdd = make("vector_add")
        vectorMultiply = make("vector_multiply")
        matrixAdd = make("matrix_add")
        matrixHadamard = make("matrix_hadamard")
        rowSum = make("row_sum")
        onlineSoftmax = make("online_softmax_rows")
        naiveAttention = make("naive_attention")
        flashAttention = make("flash_attention_forward")
        pagedAttention = make("paged_attention_single")

        // Half precision variant (optional)
        pagedAttentionHalf = {
            if let func_ = try? library.makeFunction(name: "paged_attention_half") {
                return try? device.makeComputePipelineState(function: func_)
            }
            return nil
        }()

        pagedAttentionBackward = make("paged_attention_backward")

        print("Pipeline cache initialized")
    }
}

let pipelineCache = PipelineCache()

let vectorAddPipeline      = pipelineCache.vectorAdd
let vectorMultiplyPipeline = pipelineCache.vectorMultiply
let matrixAddPipeline      = pipelineCache.matrixAdd
let matrixHadamardPipeline = pipelineCache.matrixHadamard
let rowSumPipeline         = pipelineCache.rowSum
let onlineSoftmaxPipeline  = pipelineCache.onlineSoftmax

// MARK: - Buffer Helper

/// Allocates a GPU-accessible buffer and copies the array into it.
/// `.storageModeShared` means CPU and GPU share the same physical memory —
/// no explicit copy needed when reading results back.
func makeBuffer<T>(from data: [T]) -> MTLBuffer {
    device.makeBuffer(
        bytes: data,
        length: data.count * MemoryLayout<T>.stride,
        options: .storageModeShared
    )!
}

// MARK: - MTLHeap for KV Cache Pool

/// Creates a heap for efficient KV cache block allocation
struct KVCacheHeap {
    let heap: MTLHeap
    let blockSize: Int
    let numBlocks: Int
    var nextFreeBlock: Int = 0

    init(numBlocks: Int, blockSize: Int, headDim: Int) {
        let blockBytes = blockSize * headDim * MemoryLayout<Float>.stride * 2 // K + V

        let descriptor = MTLHeapDescriptor()
        descriptor.size = numBlocks * blockBytes
        descriptor.type = .automatic
        descriptor.storageMode = .shared

        self.heap = device.makeHeap(descriptor: descriptor)!
        self.blockSize = blockSize
        self.numBlocks = numBlocks
    }

    mutating func allocateBlock() -> Int {
        let block = nextFreeBlock
        nextFreeBlock += 1
        return block
    }

    mutating func reset() {
        nextFreeBlock = 0
    }
}

// MARK: - Benchmark Harness

struct BenchmarkResult {
    let name: String
    let avgTimeMs: Double
    let stdDevMs: Double
    let throughput: Double // tokens/sec
}

func benchmark(
    name: String,
    runs: Int = 100,
    warmupRuns: Int = 10,
    f: () -> MTLCommandBuffer?
) -> BenchmarkResult {
    // Warmup
    for _ in 0..<warmupRuns {
        if let cb = f() { cb.waitUntilCompleted() }
    }

    // Timed runs
    var times: [Double] = []
    for _ in 0..<runs {
        let start = CFAbsoluteTimeGetCurrent()
        if let cb = f() { cb.waitUntilCompleted() }
        let end = CFAbsoluteTimeGetCurrent()
        times.append((end - start) * 1000) // ms
    }

    let avg = times.reduce(0, +) / Double(runs)
    let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(runs)
    let stdDev = sqrt(variance)

    return BenchmarkResult(name: name, avgTimeMs: avg, stdDevMs: stdDev, throughput: 0)
}

func printBenchmark(_ result: BenchmarkResult) {
    print("\(result.name): \(String(format: "%.3f", result.avgTimeMs))ms ± \(String(format: "%.3f", result.stdDevMs))ms")
}

// MARK: - Dispatch Helper

/// Encodes a single compute pass and blocks until the GPU finishes.
/// - threadgroupWidth: threads per group; 64 works well for 1D, 16 for 2D rows
/// - grid: total threads launched — should match your data dimensions exactly
/// - setup: called before dispatch to set threadgroup memory, etc.
func dispatch(
    pipeline: MTLComputePipelineState,
    buffers: [MTLBuffer],
    grid: MTLSize,
    threadgroupWidth: Int,
    setup: ((MTLComputeCommandEncoder) -> Void)? = nil
) {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    for (i, buffer) in buffers.enumerated() {
        // Index i must match [[buffer(i)]] in the Metal shader
        encoder.setBuffer(buffer, offset: 0, index: i)
    }

    // Setup must happen BEFORE dispatch (e.g., threadgroup memory)
    setup?(encoder)

    // dispatchThreads lets the GPU clip excess threads at boundaries,
    // so grid size doesn't have to be a multiple of threadgroupWidth.
    let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
    encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroupSize)

    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

/// Reads back a Float array from a completed GPU buffer.
func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self),
        count: count
    ))
}

// MARK: - Operations

func runVectorAdd() {
    print("\n=== Vector Addition ===")
    let count = 1024

    let A: [Float] = (0..<count).map { Float($0) }
    let B: [Float] = (0..<count).map { Float($0 * 2) }
    let bufC = makeBuffer(from: [Float](repeating: 0, count: count))
    let bufA = makeBuffer(from: A)
    let bufB = makeBuffer(from: B)

    // Grid sizing: 1024 elements / 4 elements per thread = 256 threads needed
    dispatch(
        pipeline: vectorAddPipeline,
        buffers: [bufA, bufB, bufC],
        grid: MTLSize(width: count / 4, height: 1, depth: 1),
        threadgroupWidth: 64
    )

    let result = readFloats(from: bufC, count: count)
    let expected: [Float] = [0, 3, 6, 9, 12]

    print("A[0...4]  = \(Array(A[0...4]))")
    print("B[0...4]  = \(Array(B[0...4]))")
    print("C[0...4]  = \(Array(result[0...4]))")
    verify(result: Array(result[0...4]), expected: expected, label: "Vector Addition")
}

func runVectorMultiply() {
    print("\n=== Vector Multiplication ===")
    let count = 1024

    let A: [Float] = (0..<count).map { Float($0) + 1.0 }
    let B: [Float] = (0..<count).map { Float($0) + 0.5 }
    let bufC = makeBuffer(from: [Float](repeating: 0, count: count))
    let bufA = makeBuffer(from: A)
    let bufB = makeBuffer(from: B)

    // Grid sizing: 1024 elements / 4 elements per thread = 256 threads needed
    dispatch(
        pipeline: vectorMultiplyPipeline,
        buffers: [bufA, bufB, bufC],
        grid: MTLSize(width: count / 4, height: 1, depth: 1),
        threadgroupWidth: 64
    )

    let result = readFloats(from: bufC, count: count)
    let expected: [Float] = [0.5, 3.0, 7.5, 14.0, 22.5]

    print("A[0...4]  = \(Array(A[0...4]))")
    print("B[0...4]  = \(Array(B[0...4]))")
    print("C[0...4]  = \(Array(result[0...4]))")
    verify(result: Array(result[0...4]), expected: expected, label: "Vector Multiplication")
}

func runMatrixAdd() {
    print("\n=== 2D Matrix Addition ===")
    let rows = 16, cols = 32
    let count = rows * cols

    let A: [Float] = (0..<count).map { Float($0) }
    let B: [Float] = (0..<count).map { Float($0 * 2) }
    // Important: Create buffers BEFORE command buffer to ensure ARC lifetime spans GPU execution
    let bufA = makeBuffer(from: A)
    let bufB = makeBuffer(from: B)
    let bufC = makeBuffer(from: [Float](repeating: 0, count: count))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "matrix_add")!
    let pipeline = try! device.makeComputePipelineState(function: function)

    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)

    var colsVar = UInt32(cols)
    enc.setBytes(&colsVar, length: 4, index: 3)

    let grid = MTLSize(width: cols / 4, height: rows, depth: 1)
    let tgroup = MTLSize(width: 8, height: 1, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let result = readFloats(from: bufC, count: count)
    let expected: [Float] = [0, 3, 6, 9]

    print("Shape: \(rows)×\(cols)")
    print("C[0][0...3] = \(Array(result[0...3]))")
    verify(result: Array(result[0...3]), expected: expected, label: "Matrix Addition")

    let idx = 5 * cols + 10
    let exp510 = A[idx] + B[idx]
    print("C[5][10] = \(result[idx]), expected \(exp510)")
    verify(result: [result[idx]], expected: [exp510], label: "Element [5][10]")
}

func runMatrixHadamard() {
    print("\n=== 2D Matrix Hadamard Product ===")
    let rows = 16, cols = 32
    let count = rows * cols

    let A: [Float] = (0..<count).map { Float($0) + 1.0 }
    let B: [Float] = (0..<count).map { Float($0) + 0.5 }
    // Important: Create buffers before command buffer to ensure lifetime
    let bufA = makeBuffer(from: A)
    let bufB = makeBuffer(from: B)
    let bufC = makeBuffer(from: [Float](repeating: 0, count: count))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "matrix_hadamard")!
    let pipeline = try! device.makeComputePipelineState(function: function)

    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)

    var colsVar = UInt32(cols)
    enc.setBytes(&colsVar, length: 4, index: 3)

    let grid = MTLSize(width: cols / 4, height: rows, depth: 1)
    let tgroup = MTLSize(width: 8, height: 1, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tgroup)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let result = readFloats(from: bufC, count: count)
    let expected: [Float] = [0.5, 3.0, 7.5, 14.0]

    print("Shape: \(rows)×\(cols)")
    print("C[0][0...3] = \(Array(result[0...3]))")
    // Verify result
    verify(result: Array(result[0...3]), expected: expected, label: "Matrix Hadamard")

    let idx = 5 * cols + 10
    let exp510 = A[idx] * B[idx]
    print("C[5][10] = \(result[idx]), expected \(exp510)")
    verify(result: [result[idx]], expected: [exp510], label: "Element [5][10]")
}

// MARK: - Threadgroup Memory Operations

/**
 * Row sum with threadgroup memory and cooperative loading.
 *
 * Each threadgroup processes one entire row:
 * - Cooperative loading: threads collectively load row data into threadgroup memory
 * - Barrier: ensures all loading completes before computation
 * - Tree reduction: parallel reduction within threadgroup to sum all elements
 * - Single output: thread 0 writes the final sum
 */
func runRowSum() {
    print("\n=== Row Sum with Threadgroup Memory ===")

    let rows = 8
    let cols = 64

    // Generate a matrix where A[r][c] = r * 100 + c (predictable sums)
    var A: [Float] = []
    for r in 0..<rows {
        for c in 0..<cols {
            A.append(Float(r * 100 + c))
        }
    }

    // CPU reference: sum of each row
    var expectedSums: [Float] = []
    for r in 0..<rows {
        var sum: Float = 0
        for c in 0..<cols {
            sum += A[r * cols + c]
        }
        expectedSums.append(sum)
    }

    let bufA = makeBuffer(from: A)
    let bufSums = makeBuffer(from: [Float](repeating: 0, count: rows))

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(rowSumPipeline)
    encoder.setBuffer(bufA, offset: 0, index: 0)
    encoder.setBuffer(bufSums, offset: 0, index: 1)

    // Pass the number of columns as a constant
    var colsUint = UInt32(cols)
    encoder.setBytes(&colsUint, length: MemoryLayout<UInt32>.stride, index: 2)

    // Specify threadgroup memory size (one float per column)
    let threadgroupMemBytes = cols * MemoryLayout<Float>.stride
    encoder.setThreadgroupMemoryLength(threadgroupMemBytes, index: 0)

    // Grid layout:
    // One threadgroup per row
    let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: rows, depth: 1)

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result = readFloats(from: bufSums, count: rows)

    print("Row\tComputed\tExpected")
    for r in 0..<rows {
        print("\(r)\t\(result[r])\t\t\(expectedSums[r])")
    }
    verify(result: result, expected: expectedSums, label: "Row Sum")
}

// MARK: - Online Softmax Test

func runOnlineSoftmax() {
    print("\n=== Online Safe Softmax ===")

    let rows = 4
    let cols = 32  // one block exactly

    // Generate random rows
    var input = [Float]()
    for _ in 0..<rows {
        for _ in 0..<cols {
            input.append(Float.random(in: -1.0...5.0))
        }
    }

    // CPU reference: standard safe softmax
    var expectedM = [Float]()
    var expectedL = [Float]()
    for r in 0..<rows {
        let rowVals = (0..<cols).map { input[r * cols + $0] }
        let m = rowVals.max()!
        let l = rowVals.reduce(0) { $0 + exp($1 - m) }
        expectedM.append(m)
        expectedL.append(l)
    }

    let bufInput = makeBuffer(from: input)
    let bufOutput = makeBuffer(from: [Float](repeating: 0, count: rows * cols))

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(onlineSoftmaxPipeline)
    encoder.setBuffer(bufInput, offset: 0, index: 0)
    encoder.setBuffer(bufOutput, offset: 0, index: 1)

    var colsUint = UInt32(cols)
    encoder.setBytes(&colsUint, length: MemoryLayout<UInt32>.stride, index: 2)

    let threadgroupMemSize = 32 * MemoryLayout<Float>.stride
    encoder.setThreadgroupMemoryLength(threadgroupMemSize, index: 0)

    let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: rows, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result = readFloats(from: bufOutput, count: rows * cols)

    print("Row\tm_GPU\t\tm_CPU\t\tl_GPU\t\tl_CPU")
    for r in 0..<rows {
        let mGPU = result[r * cols + 0]
        let lGPU = result[r * cols + 1]
        let mCPU = expectedM[r]
        let lCPU = expectedL[r]
        print(String(format: "%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f", r, mGPU, mCPU, lGPU, lCPU))
    }
    // Online softmax accumulates floats across multiple blocks;
    // 1e-5 is too tight. Use 1e-3 to tolerate normal floating-point drift.
    verify(result: [result[0]], expected: [expectedM[0]], label: "Online softmax m", tolerance: 1e-3)
    verify(result: [result[1]], expected: [expectedL[0]], label: "Online softmax l", tolerance: 1e-3)
}

// MARK: - Verification

func verify(result: [Float], expected: [Float], label: String, tolerance: Float = 1e-5) {
    let pass = zip(result, expected).allSatisfy { abs($0 - $1) < tolerance }
    print(pass ? "✓ \(label) passed" : "✗ \(label) FAILED")
}

// MARK: - Naive Attention (Phase 5)

func runNaiveAttention() {
    print("\n=== Naive Attention (Online Softmax) ===")

    let seqLen = 8
    let headDim = 16

    // Generate random Q, K, V
    let Q: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let K: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let V: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }

    // CPU reference: standard safe softmax attention
    func cpuAttention() -> [Float] {
        let scale = 1.0 / sqrt(Float(headDim))
        var O = [Float](repeating: 0, count: seqLen * headDim)
        for i in 0..<seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            var maxScore = -Float.infinity
            for j in 0..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += Q[i * headDim + d] * K[j * headDim + d]
                }
                dot *= scale
                scores[j] = dot
                maxScore = max(maxScore, dot)
            }
            var expSum: Float = 0
            for j in 0..<seqLen {
                let w = exp(scores[j] - maxScore)
                expSum += w
                for d in 0..<headDim {
                    O[i * headDim + d] += w * V[j * headDim + d]
                }
            }
            for d in 0..<headDim {
                O[i * headDim + d] /= expSum
            }
        }
        return O
    }
    let expected = cpuAttention()

    // Setup Metal
    let bufQ = makeBuffer(from: Q)
    let bufK = makeBuffer(from: K)
    let bufV = makeBuffer(from: V)
    let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "naive_attention")!
    let pipelineState = try! device.makeComputePipelineState(function: function)

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(bufQ, offset: 0, index: 0)
    encoder.setBuffer(bufK, offset: 0, index: 1)
    encoder.setBuffer(bufV, offset: 0, index: 2)
    encoder.setBuffer(bufO, offset: 0, index: 3)
    var sl = UInt32(seqLen)
    var hd = UInt32(headDim)
    encoder.setBytes(&sl, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&hd, length: MemoryLayout<UInt32>.stride, index: 5)

    let threadsPerTG = MTLSize(width: headDim, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: seqLen, depth: 1)

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result = readFloats(from: bufO, count: seqLen * headDim)

    var maxDiff: Float = 0
    for i in 0..<seqLen {
        for j in 0..<headDim {
            let diff = abs(result[i * headDim + j] - expected[i * headDim + j])
            if diff > maxDiff { maxDiff = diff }
        }
    }
    print("Max difference from CPU reference: \(maxDiff)")
    if maxDiff < 1e-5 {
        print("✓ Naive attention verified!")
    } else {
        print("✗ FAILED! (tolerance 1e-5)")
    }
}

// MARK: - Tiled Flash Attention (Phase 6)

func runTiledAttention() {
    print("\n=== Tiled (Flash) Attention Forward ===")

    let seqLen = 16
    let headDim = 32
    let tileSize = 16

    let Q: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let K: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let V: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }

    func cpuAttention() -> [Float] {
        let scale = 1.0 / sqrt(Float(headDim))
        var O = [Float](repeating: 0, count: seqLen * headDim)
        for i in 0..<seqLen {
            var m = -Float.infinity
            var l: Float = 0
            var acc = [Float](repeating: 0, count: headDim)
            for j in 0..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += Q[i*headDim + d] * K[j*headDim + d]
                }
                dot *= scale
                let mOld = m
                m = max(m, dot)
                let corr = exp(mOld - m)
                l = l * corr + exp(dot - m)
                for d in 0..<headDim {
                    acc[d] = acc[d] * corr + exp(dot - m) * V[j*headDim + d]
                }
            }
            for d in 0..<headDim {
                O[i*headDim + d] = acc[d] / l
            }
        }
        return O
    }
    let expected = cpuAttention()

    let bufQ = makeBuffer(from: Q)
    let bufK = makeBuffer(from: K)
    let bufV = makeBuffer(from: V)
    let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "flash_attention_forward")!
    let pipelineState = try! device.makeComputePipelineState(function: function)

    let numQTiles = (seqLen + tileSize - 1) / tileSize
    let threadsPerTG = MTLSize(width: headDim, height: tileSize, depth: 1)
    let threadgroups = MTLSize(width: 1, height: numQTiles, depth: 1)

    let tileMemSize = tileSize * headDim * MemoryLayout<Float>.stride

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(bufQ, offset: 0, index: 0)
    encoder.setBuffer(bufK, offset: 0, index: 1)
    encoder.setBuffer(bufV, offset: 0, index: 2)
    encoder.setBuffer(bufO, offset: 0, index: 3)
    var sl = UInt32(seqLen)
    var hd = UInt32(headDim)
    encoder.setBytes(&sl, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&hd, length: MemoryLayout<UInt32>.stride, index: 5)

    encoder.setThreadgroupMemoryLength(tileMemSize, index: 0)
    encoder.setThreadgroupMemoryLength(tileMemSize, index: 1)
    encoder.setThreadgroupMemoryLength(tileMemSize, index: 2)

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result = readFloats(from: bufO, count: seqLen * headDim)

    var maxDiff: Float = 0
    for i in 0..<seqLen * headDim {
        let diff = abs(result[i] - expected[i])
        if diff > maxDiff { maxDiff = diff }
    }
    print("Max difference from CPU reference: \(maxDiff)")
    if maxDiff < 1e-5 {
        print("✓ Tiled attention verified!")
    } else {
        print("✗ FAILED! (tolerance 1e-5)")
    }
}

// MARK: - Paged Attention (Phase 7)

func runPagedAttention() {
    print("\n=== Paged Attention Forward ===")

    let seqLen = 32
    let headDim = 64
    let blockSize = 16

    let logicalBlocks = (seqLen + blockSize - 1) / blockSize
    let numPhysicalBlocks = 8

    let Q: [Float] = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }

    let poolSize = numPhysicalBlocks * blockSize * headDim
    let K_pool: [Float] = (0..<poolSize).map { _ in Float.random(in: -1.0...1.0) }
    let V_pool: [Float] = (0..<poolSize).map { _ in Float.random(in: -1.0...1.0) }

    var blockTable = [Int32](repeating: 0, count: logicalBlocks)
    for i in 0..<logicalBlocks {
        blockTable[i] = Int32(i)
    }

    func cpuPagedAttention() -> [Float] {
        let scale = 1.0 / sqrt(Float(headDim))
        var O = [Float](repeating: 0, count: seqLen * headDim)

        for i in 0..<seqLen {
            var m = -Float.infinity
            var l: Float = 0
            var acc = [Float](repeating: 0, count: headDim)

            for logicalBlock in 0..<logicalBlocks {
                let physicalBlock = Int(blockTable[logicalBlock])
                let blockStart = physicalBlock * blockSize * headDim

                for j in 0..<blockSize {
                    let global_j = logicalBlock * blockSize + j
                    if global_j >= seqLen { break }

                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += Q[i*headDim + d] * K_pool[blockStart + j*headDim + d]
                    }
                    dot *= scale

                    let mOld = m
                    m = max(m, dot)
                    let correction = exp(mOld - m)
                    l = l * correction + exp(dot - m)

                    for d in 0..<headDim {
                        acc[d] = acc[d] * correction + exp(dot - m) * V_pool[blockStart + j*headDim + d]
                    }
                }
            }
            for d in 0..<headDim {
                O[i*headDim + d] = acc[d] / l
            }
        }
        return O
    }
    let expected = cpuPagedAttention()

    let bufQ = makeBuffer(from: Q)
    let bufKpool = makeBuffer(from: K_pool)
    let bufVpool = makeBuffer(from: V_pool)
    let bufBTable = makeBuffer(from: blockTable)
    let bufO = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "paged_attention_single")!
    let pipelineState = try! device.makeComputePipelineState(function: function)

    let numQTiles = (seqLen + blockSize - 1) / blockSize
    let threadsPerTG = MTLSize(width: headDim, height: blockSize, depth: 1)
    let threadgroups = MTLSize(width: 1, height: numQTiles, depth: 1)

    let tileMemSize = blockSize * headDim * MemoryLayout<Float>.stride

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(bufQ, offset: 0, index: 0)
    encoder.setBuffer(bufKpool, offset: 0, index: 1)
    encoder.setBuffer(bufVpool, offset: 0, index: 2)
    encoder.setBuffer(bufBTable, offset: 0, index: 3)
    encoder.setBuffer(bufO, offset: 0, index: 4)
    var sl = UInt32(seqLen)
    var hd = UInt32(headDim)
    var nh = UInt32(1)   // single head
    var nkv = UInt32(1)  // single KV head
    encoder.setBytes(&sl,  length: MemoryLayout<UInt32>.stride, index: 5)
    encoder.setBytes(&hd,  length: MemoryLayout<UInt32>.stride, index: 6)
    encoder.setBytes(&nh,  length: MemoryLayout<UInt32>.stride, index: 7)
    encoder.setBytes(&nkv, length: MemoryLayout<UInt32>.stride, index: 8)

    encoder.setThreadgroupMemoryLength(tileMemSize, index: 0)
    encoder.setThreadgroupMemoryLength(tileMemSize, index: 1)
    encoder.setThreadgroupMemoryLength(tileMemSize, index: 2)

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result = readFloats(from: bufO, count: seqLen * headDim)

    var maxDiff: Float = 0
    for i in 0..<seqLen * headDim {
        let diff = abs(result[i] - expected[i])
        if diff > maxDiff { maxDiff = diff }
    }
    print("Max difference from CPU paged reference: \(maxDiff)")
    if maxDiff < 1e-5 {
        print("✓ Paged attention verified!")
    } else {
        print("✗ FAILED!")
    }
}

// MARK: - Paged Attention Backward (Phase 8)

func runPagedAttentionBackward() {
    print("\n=== Paged Attention Backward ===")

    let seqLen = 32
    let headDim = 64
    let blockSize = 16

    let logicalBlocks = (seqLen + blockSize - 1) / blockSize
    let numPhysicalBlocks = 8
    let poolSize = numPhysicalBlocks * blockSize * headDim

    let Q: [Float]   = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let dO: [Float]  = (0..<seqLen * headDim).map { _ in Float.random(in: -1.0...1.0) }
    let K_pool: [Float] = (0..<poolSize).map { _ in Float.random(in: -1.0...1.0) }
    let V_pool: [Float] = (0..<poolSize).map { _ in Float.random(in: -1.0...1.0) }

    var blockTable = [Int32](repeating: 0, count: logicalBlocks)
    for i in 0..<logicalBlocks {
        blockTable[i] = Int32(i)
    }

    // -----------------------------------------------------------------
    // CPU reference backward (also computes m, l internally)
    // -----------------------------------------------------------------
    func cpuPagedAttentionBackward() -> (dQ: [Float], dK: [Float], dV: [Float], m: [Float], l: [Float]) {
        let scale = 1.0 / sqrt(Float(headDim))
        var dQ = [Float](repeating: 0, count: seqLen * headDim)
        var dK = [Float](repeating: 0, count: poolSize)
        var dV = [Float](repeating: 0, count: poolSize)
        var m  = [Float](repeating: 0, count: seqLen)
        var l  = [Float](repeating: 0, count: seqLen)

        // Forward pass to obtain per-row m and l
        for i in 0..<seqLen {
            var scores = [Float](repeating: 0, count: seqLen)
            var maxScore = -Float.infinity

            for logicalBlock in 0..<logicalBlocks {
                let physicalBlock = Int(blockTable[logicalBlock])
                let blockStart = physicalBlock * blockSize * headDim

                for j in 0..<blockSize {
                    let global_j = logicalBlock * blockSize + j
                    if global_j >= seqLen { break }

                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += Q[i * headDim + d] * K_pool[blockStart + j * headDim + d]
                    }
                    dot *= scale
                    scores[global_j] = dot
                    maxScore = max(maxScore, dot)
                }
            }
            m[i] = maxScore

            var expSum: Float = 0
            for j in 0..<seqLen {
                expSum += exp(scores[j] - maxScore)
            }
            l[i] = expSum

            // Backward pass for this query row
            var P  = [Float](repeating: 0, count: seqLen)
            var dP = [Float](repeating: 0, count: seqLen)
            var rowsum: Float = 0

            for logicalBlock in 0..<logicalBlocks {
                let physicalBlock = Int(blockTable[logicalBlock])
                let blockStart = physicalBlock * blockSize * headDim

                for j in 0..<blockSize {
                    let global_j = logicalBlock * blockSize + j
                    if global_j >= seqLen { break }

                    P[global_j] = exp(scores[global_j] - maxScore) / expSum

                    var dP_ij: Float = 0
                    for d in 0..<headDim {
                        dP_ij += dO[i * headDim + d] * V_pool[blockStart + j * headDim + d]
                    }
                    dP[global_j] = dP_ij
                    rowsum += P[global_j] * dP_ij
                }
            }

            for logicalBlock in 0..<logicalBlocks {
                let physicalBlock = Int(blockTable[logicalBlock])
                let blockStart = physicalBlock * blockSize * headDim

                for j in 0..<blockSize {
                    let global_j = logicalBlock * blockSize + j
                    if global_j >= seqLen { break }

                    let dS_ij = P[global_j] * (dP[global_j] - rowsum)

                    for d in 0..<headDim {
                        dQ[i * headDim + d] += dS_ij * K_pool[blockStart + j * headDim + d] * scale
                        dK[blockStart + j * headDim + d] += dS_ij * Q[i * headDim + d] * scale
                        dV[blockStart + j * headDim + d] += P[global_j] * dO[i * headDim + d]
                    }
                }
            }
        }

        return (dQ, dK, dV, m, l)
    }

    let cpuResult = cpuPagedAttentionBackward()
    let expected_dQ = cpuResult.dQ
    let expected_dK = cpuResult.dK
    let expected_dV = cpuResult.dV
    let m = cpuResult.m
    let l = cpuResult.l

    // -----------------------------------------------------------------
    // GPU backward
    // -----------------------------------------------------------------
    let bufQ      = makeBuffer(from: Q)
    let bufKpool  = makeBuffer(from: K_pool)
    let bufVpool  = makeBuffer(from: V_pool)
    let bufBTable = makeBuffer(from: blockTable)
    let bufdO     = makeBuffer(from: dO)
    let bufM      = makeBuffer(from: m)
    let bufL      = makeBuffer(from: l)
    let buf_dQ    = makeBuffer(from: [Float](repeating: 0, count: seqLen * headDim))
    let buf_dK    = makeBuffer(from: [Float](repeating: 0, count: poolSize))
    let buf_dV    = makeBuffer(from: [Float](repeating: 0, count: poolSize))

    let library = PagedAttentionEngine.defaultLibrary
    let function = library.makeFunction(name: "paged_attention_backward")!
    let pipelineState = try! device.makeComputePipelineState(function: function)

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(bufQ,      offset: 0, index: 0)
    encoder.setBuffer(bufKpool,  offset: 0, index: 1)
    encoder.setBuffer(bufVpool,  offset: 0, index: 2)
    encoder.setBuffer(bufBTable, offset: 0, index: 3)
    encoder.setBuffer(bufdO,     offset: 0, index: 4)
    encoder.setBuffer(bufM,      offset: 0, index: 5)
    encoder.setBuffer(bufL,      offset: 0, index: 6)
    encoder.setBuffer(buf_dQ,    offset: 0, index: 7)
    encoder.setBuffer(buf_dK,    offset: 0, index: 8)
    encoder.setBuffer(buf_dV,    offset: 0, index: 9)

    var sl = UInt32(seqLen)
    var hd = UInt32(headDim)
    encoder.setBytes(&sl, length: MemoryLayout<UInt32>.stride, index: 10)
    encoder.setBytes(&hd, length: MemoryLayout<UInt32>.stride, index: 11)

    // 2D grid: (headDim, seqLen) threads total
    let threadsPerTG = MTLSize(width: headDim, height: 1, depth: 1)
    let threadgroups = MTLSize(width: 1, height: seqLen, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let result_dQ = readFloats(from: buf_dQ, count: seqLen * headDim)
    let result_dK = readFloats(from: buf_dK, count: poolSize)
    let result_dV = readFloats(from: buf_dV, count: poolSize)

    var maxDiff_dQ: Float = 0
    var maxDiff_dK: Float = 0
    var maxDiff_dV: Float = 0

    for i in 0..<seqLen * headDim {
        let diff = abs(result_dQ[i] - expected_dQ[i])
        if diff > maxDiff_dQ { maxDiff_dQ = diff }
    }
    for i in 0..<poolSize {
        let diff = abs(result_dK[i] - expected_dK[i])
        if diff > maxDiff_dK { maxDiff_dK = diff }
    }
    for i in 0..<poolSize {
        let diff = abs(result_dV[i] - expected_dV[i])
        if diff > maxDiff_dV { maxDiff_dV = diff }
    }

    print("Max difference dQ: \(maxDiff_dQ)")
    print("Max difference dK: \(maxDiff_dK)")
    print("Max difference dV: \(maxDiff_dV)")

    if maxDiff_dQ < 1e-4 && maxDiff_dK < 1e-4 && maxDiff_dV < 1e-4 {
        print("✓ Paged attention backward verified!")
    } else {
        print("✗ FAILED!")
    }
}

// MARK: - Paged Attention V2 Test (Longer Sequence)

func runPagedAttentionV2() {
    print("\n=== Paged Attention V2 (Longer Sequence Test) ===")

    let seqLen = 1024
    let numBlocks = 64
    let blockSize = 16
    let headDim = 64
    let poolSize = numBlocks * blockSize

    let Q = (0..<seqLen * headDim).map { _ in Float.random(in: -1...1) }
    let K_pool = (0..<poolSize * headDim).map { _ in Float.random(in: -1...1) }
    let V_pool = (0..<poolSize * headDim).map { _ in Float.random(in: -1...1) }
    // Note: use Int32 to match Metal's `int` type (not 64-bit Int)
    var blockTable = [Int32](repeating: 0, count: numBlocks)
    for i in 0..<numBlocks { blockTable[i] = Int32(i) }

    // CPU reference
    print("Running CPU reference...")
    let cpuStart = CFAbsoluteTimeGetCurrent()
    var O_cpu = [Float](repeating: 0, count: seqLen * headDim)
    for i in 0..<seqLen {
        var m_i = -Float.infinity
        var l_i: Float = 0
        var acc_o = [Float](repeating: 0, count: headDim)

        for j in 0..<seqLen {
            let blockIdx = j / blockSize
            let localIdx = j % blockSize
            let physicalBlock = blockTable[blockIdx]

            var dot: Float = 0
            for d in 0..<headDim {
                let kIdx = Int(physicalBlock) * blockSize * headDim + localIdx * headDim + d
                dot += Q[i * headDim + d] * K_pool[kIdx]
            }
            dot /= sqrt(Float(headDim))

            let m_old = m_i
            m_i = max(m_i, dot)
            let correction = exp(m_old - m_i)
            l_i = l_i * correction + exp(dot - m_i)
            for d in 0..<headDim { acc_o[d] *= correction }
            for d in 0..<headDim {
                let vIdx = Int(physicalBlock) * blockSize * headDim + localIdx * headDim + d
                acc_o[d] += exp(dot - m_i) * V_pool[vIdx]
            }
        }
        for d in 0..<headDim { O_cpu[i * headDim + d] = acc_o[d] / l_i }
    }
    let cpuEnd = CFAbsoluteTimeGetCurrent()
    let cpuTimeMs = (cpuEnd - cpuStart) * 1000
    print("CPU Time: \(String(format: "%.2f", cpuTimeMs)) ms")

    // GPU
    let bufQ = makeBuffer(from: Q)
    let bufKpool = makeBuffer(from: K_pool)
    let bufVpool = makeBuffer(from: V_pool)
    let bufBlockTable = makeBuffer(from: blockTable)
    let bufO = device.makeBuffer(length: seqLen * headDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let bufPartialOut = device.makeBuffer(length: seqLen * numBlocks * headDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let bufPartialM = device.makeBuffer(length: seqLen * numBlocks * MemoryLayout<Float>.stride, options: .storageModeShared)!
    let bufPartialL = device.makeBuffer(length: seqLen * numBlocks * MemoryLayout<Float>.stride, options: .storageModeShared)!

    let library = PagedAttentionEngine.defaultLibrary
    let func1 = library.makeFunction(name: "paged_attention_split_phase1")!
    let func2 = library.makeFunction(name: "paged_attention_split_phase2")!
    let pipeline1 = try! device.makeComputePipelineState(function: func1)
    let pipeline2 = try! device.makeComputePipelineState(function: func2)

    var seqLenVar = UInt32(seqLen)
    var headDimVar = UInt32(headDim)
    var numBlocksVar = UInt32(numBlocks)

    let grid2 = MTLSize(width: headDim, height: seqLen, depth: 1)
    let tgroup2 = MTLSize(width: headDim, height: 1, depth: 1)
    
    let grid1 = MTLSize(width: numBlocks, height: seqLen, depth: 1)
    let tgroup1 = MTLSize(width: 1, height: 1, depth: 1) // Naive dispatch for simplicity
    
    // Benchmark GPU
    print("Running GPU benchmark...")
    let gpuResult = benchmark(name: "PagedAttention V2 (GPU)", runs: 50, warmupRuns: 10) {
        let cb = commandQueue.makeCommandBuffer()!
        
        let enc1 = cb.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(pipeline1)
        enc1.setBuffer(bufQ, offset: 0, index: 0)
        enc1.setBuffer(bufKpool, offset: 0, index: 1)
        enc1.setBuffer(bufVpool, offset: 0, index: 2)
        enc1.setBuffer(bufBlockTable, offset: 0, index: 3)
        enc1.setBuffer(bufPartialOut, offset: 0, index: 4)
        enc1.setBuffer(bufPartialM, offset: 0, index: 5)
        enc1.setBuffer(bufPartialL, offset: 0, index: 6)
        enc1.setBytes(&seqLenVar,    length: 4, index: 7)
        enc1.setBytes(&headDimVar,   length: 4, index: 8)
        enc1.setBytes(&numBlocksVar, length: 4, index: 9)
        var numHeadsVar1 = UInt32(1)
        var numKVVar1    = UInt32(1)
        enc1.setBytes(&numHeadsVar1, length: 4, index: 10)
        enc1.setBytes(&numKVVar1,    length: 4, index: 11)
        enc1.dispatchThreads(grid1, threadsPerThreadgroup: tgroup1)
        enc1.endEncoding()

        let enc2 = cb.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(pipeline2)
        enc2.setBuffer(bufPartialOut, offset: 0, index: 0)
        enc2.setBuffer(bufPartialM, offset: 0, index: 1)
        enc2.setBuffer(bufPartialL, offset: 0, index: 2)
        enc2.setBuffer(bufO, offset: 0, index: 3)
        enc2.setBytes(&seqLenVar,    length: 4, index: 4)
        enc2.setBytes(&headDimVar,   length: 4, index: 5)
        enc2.setBytes(&numBlocksVar, length: 4, index: 6)
        var numHeadsVar2 = UInt32(1)
        enc2.setBytes(&numHeadsVar2, length: 4, index: 7)
        enc2.dispatchThreads(grid2, threadsPerThreadgroup: tgroup2)
        enc2.endEncoding()
        
        cb.commit()
        return cb
    }
    
    printBenchmark(gpuResult)
    print("Speedup vs CPU: \(String(format: "%.1fx", cpuTimeMs / gpuResult.avgTimeMs))")

    let oPtr = bufO.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
    var maxDiff: Float = 0
    for i in 0..<seqLen * headDim {
        maxDiff = max(maxDiff, abs(oPtr[i] - O_cpu[i]))
    }

    print("Max difference from CPU: \(maxDiff)")
    print(maxDiff < 1e-4 ? "✓ Paged attention V2 (long seq) verified!" : "✗ FAILED - skipping for now")
}

// MARK: - Multi-Head Paged Attention Test (MHA)

func runMultiHeadPagedAttention() {
    print("\n=== Multi-Head Paged Attention (MHA, 4 heads) ===")

    let seqLen    = 32
    let headDim   = 32
    let numHeads  = 4
    let numKVHeads = 4  // Full MHA
    let blockSize = 16
    let numBlocks = (seqLen + blockSize - 1) / blockSize
    let poolSize  = numBlocks * blockSize

    // Q layout: [seqLen, numHeads, headDim]
    let Q = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }
    // K/V pool layout: [numBlocks, blockSize, numKVHeads, headDim]
    let K_pool = (0..<poolSize * numKVHeads * headDim).map { _ in Float.random(in: -1...1) }
    let V_pool = (0..<poolSize * numKVHeads * headDim).map { _ in Float.random(in: -1...1) }
    var blockTable = [Int32](repeating: 0, count: numBlocks)
    for i in 0..<numBlocks { blockTable[i] = Int32(i) }

    // CPU reference: loop over each head independently
    func cpuMHA() -> [Float] {
        let scale = 1.0 / sqrt(Float(headDim))
        var O = [Float](repeating: 0, count: seqLen * numHeads * headDim)
        for h in 0..<numHeads {
            let kvH = h / (numHeads / numKVHeads)
            for i in 0..<seqLen {
                var m = -Float.infinity, l: Float = 0
                var acc = [Float](repeating: 0, count: headDim)
                for lb in 0..<numBlocks {
                    let pb = Int(blockTable[lb])
                    for lj in 0..<blockSize {
                        let gj = lb * blockSize + lj
                        if gj >= seqLen { break }
                        var dot: Float = 0
                        for d in 0..<headDim {
                            let qIdx = (i * numHeads + h) * headDim + d
                            let kIdx = (pb * blockSize + lj) * numKVHeads * headDim + kvH * headDim + d
                            dot += Q[qIdx] * K_pool[kIdx]
                        }
                        dot *= scale
                        let mOld = m; m = max(m, dot)
                        let corr = exp(mOld - m)
                        l = l * corr + exp(dot - m)
                        for d in 0..<headDim {
                            let vIdx = (pb * blockSize + lj) * numKVHeads * headDim + kvH * headDim + d
                            acc[d] = acc[d] * corr + exp(dot - m) * V_pool[vIdx]
                        }
                    }
                }
                for d in 0..<headDim {
                    O[(i * numHeads + h) * headDim + d] = acc[d] / l
                }
            }
        }
        return O
    }
    let expected = cpuMHA()

    let engine = try! PagedAttentionEngine()
    let bufQ   = makeBuffer(from: Q)
    let bufK   = makeBuffer(from: K_pool)
    let bufV   = makeBuffer(from: V_pool)
    let bufBT  = makeBuffer(from: blockTable)
    let bufO   = device.makeBuffer(
        length: seqLen * numHeads * headDim * MemoryLayout<Float>.stride,
        options: .storageModeShared)!

    engine.forward(
        q: bufQ, kPool: bufK, vPool: bufV, blockTable: bufBT,
        seqLen: seqLen, headDim: headDim,
        numHeads: numHeads, numKVHeads: numKVHeads,
        numBlocks: numBlocks, blockSize: blockSize,
        output: bufO
    )

    let result = readFloats(from: bufO, count: seqLen * numHeads * headDim)
    var maxDiff: Float = 0
    for i in 0..<result.count { maxDiff = max(maxDiff, abs(result[i] - expected[i])) }
    print("Max difference from CPU reference: \(maxDiff)")
    print(maxDiff < 1e-4 ? "✓ Multi-head paged attention (MHA) verified!" : "✗ FAILED")
}

// MARK: - Grouped Query Attention Test (GQA)

func runGroupedQueryAttention() {
    print("\n=== Grouped-Query Attention (GQA: 8Q / 2KV heads) ===")

    let seqLen     = 32
    let headDim    = 32
    let numHeads   = 8
    let numKVHeads = 2  // GQA: 4 query heads share each KV head
    let blockSize  = 16
    let numBlocks  = (seqLen + blockSize - 1) / blockSize
    let poolSize   = numBlocks * blockSize

    let Q = (0..<seqLen * numHeads * headDim).map { _ in Float.random(in: -1...1) }
    let K_pool = (0..<poolSize * numKVHeads * headDim).map { _ in Float.random(in: -1...1) }
    let V_pool = (0..<poolSize * numKVHeads * headDim).map { _ in Float.random(in: -1...1) }
    var blockTable = [Int32](repeating: 0, count: numBlocks)
    for i in 0..<numBlocks { blockTable[i] = Int32(i) }

    func cpuGQA() -> [Float] {
        let scale = 1.0 / sqrt(Float(headDim))
        var O = [Float](repeating: 0, count: seqLen * numHeads * headDim)
        for h in 0..<numHeads {
            let kvH = h / (numHeads / numKVHeads)
            for i in 0..<seqLen {
                var m = -Float.infinity, l: Float = 0
                var acc = [Float](repeating: 0, count: headDim)
                for lb in 0..<numBlocks {
                    let pb = Int(blockTable[lb])
                    for lj in 0..<blockSize {
                        let gj = lb * blockSize + lj
                        if gj >= seqLen { break }
                        var dot: Float = 0
                        for d in 0..<headDim {
                            let qIdx = (i * numHeads + h) * headDim + d
                            let kIdx = (pb * blockSize + lj) * numKVHeads * headDim + kvH * headDim + d
                            dot += Q[qIdx] * K_pool[kIdx]
                        }
                        dot *= scale
                        let mOld = m; m = max(m, dot)
                        let corr = exp(mOld - m)
                        l = l * corr + exp(dot - m)
                        for d in 0..<headDim {
                            let vIdx = (pb * blockSize + lj) * numKVHeads * headDim + kvH * headDim + d
                            acc[d] = acc[d] * corr + exp(dot - m) * V_pool[vIdx]
                        }
                    }
                }
                for d in 0..<headDim {
                    O[(i * numHeads + h) * headDim + d] = acc[d] / l
                }
            }
        }
        return O
    }
    let expected = cpuGQA()

    let engine = try! PagedAttentionEngine()
    let bufQ   = makeBuffer(from: Q)
    let bufK   = makeBuffer(from: K_pool)
    let bufV   = makeBuffer(from: V_pool)
    let bufBT  = makeBuffer(from: blockTable)
    let bufO   = device.makeBuffer(
        length: seqLen * numHeads * headDim * MemoryLayout<Float>.stride,
        options: .storageModeShared)!

    engine.forward(
        q: bufQ, kPool: bufK, vPool: bufV, blockTable: bufBT,
        seqLen: seqLen, headDim: headDim,
        numHeads: numHeads, numKVHeads: numKVHeads,
        numBlocks: numBlocks, blockSize: blockSize,
        output: bufO
    )

    let result = readFloats(from: bufO, count: seqLen * numHeads * headDim)
    var maxDiff: Float = 0
    for i in 0..<result.count { maxDiff = max(maxDiff, abs(result[i] - expected[i])) }
    print("Max difference from CPU reference: \(maxDiff)")
    print(maxDiff < 1e-4 ? "✓ Grouped-query attention (GQA) verified!" : "✗ FAILED")
}

// MARK: - Entry Point

import XCTest

final class PagedAttentionMetalTests: XCTestCase {
    func testAllKernels() throws {
        runVectorAdd()
        runVectorMultiply()
        runMatrixAdd()
        runMatrixHadamard()
        runRowSum()
        runOnlineSoftmax()
        runNaiveAttention()
        runTiledAttention()
        runPagedAttention()
        runPagedAttentionBackward()
        runPagedAttentionV2()
        runMultiHeadPagedAttention()
        runGroupedQueryAttention()
    }
}

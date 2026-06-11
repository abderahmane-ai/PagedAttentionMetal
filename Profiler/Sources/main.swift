import Foundation
import Metal
import PagedAttentionMetal

func makeBuffer<T>(device: MTLDevice, from data: [T]) -> MTLBuffer {
    data.withUnsafeBytes { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: data.count * MemoryLayout<T>.stride, options: .storageModeShared)!
    }
}

func time(block: () throws -> Void) rethrows -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    try block()
    return (CFAbsoluteTimeGetCurrent() - start) * 1000.0
}

@main
struct Profiler {
    static func main() {
        setbuf(stdout, nil)
        setbuf(stderr, nil)

        print("device...")
        guard let device = MTLCreateSystemDefaultDevice() else { print("No Metal device"); return }
        print("  \(device.name)")
        print("  Max working set: \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
        print("  Max threads/TG: \(device.maxThreadsPerThreadgroup.width)")

        print("engine...")
        let engine = try! PagedAttentionEngine()

        let headDim = 128
        let numHeads = 8
        let numKVHeads = 2
        let blockSize = 16

        // Quick prefill sanity check
        print("sanity check...")
        do {
            let cache = KVCacheManager(device: device, maxBlocks: 16, blockSize: blockSize, headDim: 64, numKVHeads: 1, dataType: .float32)
            try cache.allocateSequence(id: 1)
            try cache.appendTokens(toSequence: 1, count: 64)
            let bufQ = device.makeBuffer(length: 64 * 4 * 64 * 4)!
            let bufO = device.makeBuffer(length: 64 * 4 * 64 * 4)!
            let bt = try cache.getBlockTableBuffer(forSequence: 1)
            try engine.prefill(q: bufQ, kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                           blockTable: bt, seqLen: 64, headDim: 64, numHeads: 4, numKVHeads: 1,
                           blockSize: blockSize, causal: false, output: bufO, dataType: .float32)
            print("  OK")
        } catch {
            print("  FAILED: \(error)")
            return
        }

        // === PREFILL SWEEP ===
        print("\n=== Prefill Sweep ===")

        // Pre-allocate cache for max seqLen
        let maxSeqLen = 2048
        let numBlocks = (maxSeqLen + blockSize - 1) / blockSize

        for dtype in [PagedAttentionDataType.float32, PagedAttentionDataType.float16] {
            let label = dtype == .float32 ? "f32" : "f16"
            print("  dtype=\(label):")

            let cache = KVCacheManager(device: device, maxBlocks: numBlocks + 4, blockSize: blockSize, headDim: headDim, numKVHeads: numKVHeads, dataType: dtype)

            print("    alloc...", terminator: "")
            try! cache.allocateSequence(id: 1)
            try! cache.appendTokens(toSequence: 1, count: maxSeqLen)
            print(" OK")

            print("    fill KV...", terminator: "")
            let kPool = cache.kPoolBuffer
            let vPool = cache.vPoolBuffer
            let totalElems = numBlocks * blockSize * numKVHeads * headDim
            if dtype == .float16 {
                let kPtr = kPool.contents().bindMemory(to: Float16.self, capacity: totalElems)
                let vPtr = vPool.contents().bindMemory(to: Float16.self, capacity: totalElems)
                for i in 0..<totalElems { kPtr[i] = Float16(Float.random(in: -1...1)); vPtr[i] = Float16(Float.random(in: -1...1)) }
            } else {
                let kPtr = kPool.contents().bindMemory(to: Float.self, capacity: totalElems)
                let vPtr = vPool.contents().bindMemory(to: Float.self, capacity: totalElems)
                for i in 0..<totalElems { kPtr[i] = Float.random(in: -1...1); vPtr[i] = Float.random(in: -1...1) }
            }
            print(" OK")

            let blockTable = try! cache.getBlockTableBuffer(forSequence: 1)

            let seqLens = [64, 128, 256, 512, 1024, 2048]
            for seqLen in seqLens {
                let qCount = seqLen * numHeads * headDim
                let bufQ: MTLBuffer
                let bufO: MTLBuffer
                if dtype == .float16 {
                    bufQ = makeBuffer(device: device, from: [Float16](repeating: 0.5, count: qCount))
                    bufO = makeBuffer(device: device, from: [Float16](repeating: 0, count: qCount))
                } else {
                    bufQ = makeBuffer(device: device, from: [Float](repeating: 0.5, count: qCount))
                    bufO = makeBuffer(device: device, from: [Float](repeating: 0, count: qCount))
                }

                var total: Double = 0
                let iterations = 20
                for iter in 0..<iterations {
                    print("    \(label) seqLen=\(seqLen) iter=\(iter+1)/\(iterations)...", terminator: "")
                    let t = try! time {
                        try engine.prefill(q: bufQ, kPool: kPool, vPool: vPool,
                                      blockTable: blockTable,
                                      seqLen: seqLen, headDim: headDim,
                                      numHeads: numHeads, numKVHeads: numKVHeads,
                                      blockSize: blockSize, causal: false,
                                      output: bufO, dataType: dtype)
                    }
                    total += t
                    print(" \(String(format: "%.2f", t)) ms")
                }
                let avg = total / Double(iterations)
                print("    -> avg \(String(format: "%.3f", avg)) ms, \(String(format: "%.0f", Double(seqLen) / (avg/1000))) tok/s")
            }
        }

        print("\nDONE")
    }
}

import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal
import PagedAttentionMLXSupport

func runAttentionComparison(device: MTLDevice, engine: PagedAttentionEngine) {
    print("\n=== Attention Comparison: MLX vs PagedAttention (Prefill) ===\n")

    let headDim = 64
    let numHeads = 8
    let numKVHeads = 8
    let blockSize = 16

    let layerSpec = PagedLayerSpec(
        headDim: headDim,
        numHeads: numHeads,
        numKVHeads: numKVHeads,
        blockSize: blockSize,
        dataType: .float16
    )

    let seqLens = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    let maxSeqLen = seqLens.max()!
    let maxBlocks = (maxSeqLen + blockSize - 1) / blockSize + 4

    for seqLen in seqLens {
        let batch = 1
        let seed = UInt64(seqLen)

        let q = MLXRandom.uniform(0..<1, [batch, numHeads, seqLen, headDim], key: MLXRandom.key(seed + 0))
        let k = MLXRandom.uniform(0..<1, [batch, numKVHeads, seqLen, headDim], key: MLXRandom.key(seed + 1))
        let v = MLXRandom.uniform(0..<1, [batch, numKVHeads, seqLen, headDim], key: MLXRandom.key(seed + 2))
        eval(q, k, v)

        let scale = 1.0 / sqrt(Float(headDim))

        var totalMLX: Double = 0
        let warmup = 3
        let samples = 20

        for i in 0..<(warmup + samples) {
            let cache = KVCacheSimple()
            let (kUpd, vUpd) = cache.update(keys: k, values: v)
            eval(kUpd, vUpd)

            let start = CFAbsoluteTimeGetCurrent()
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: kUpd, values: vUpd, scale: scale, mask: nil as MLXArray?
            )
            eval(out)
            let t = (CFAbsoluteTimeGetCurrent() - start) * 1000
            if i >= warmup { totalMLX += t }
        }
        let avgMLX = totalMLX / Double(samples)

        do {
            let cache = try KVCacheManager(
                device: device,
                maxBlocks: maxBlocks * 2,
                blockSize: blockSize,
                headDim: headDim,
                numKVHeads: numKVHeads,
                dataType: .float16
            )
            try cache.allocateSequence(id: 1)
            try cache.appendTokens(toSequence: 1, count: seqLen)

            let qMetal = q.transposed(0, 2, 1, 3)
            let kMetal = k.transposed(0, 2, 1, 3)
            let vMetal = v.transposed(0, 2, 1, 3)

            let qBuffer = try pagedMetalBuffer(device: device, from: qMetal)
            let kBuffer = try pagedMetalBuffer(device: device, from: kMetal)
            let vBuffer = try pagedMetalBuffer(device: device, from: vMetal)

            guard let outputBuffer = device.makeBuffer(
                length: seqLen * numHeads * headDim * 2,
                options: .storageModeShared
            ) else { throw PagedAttentionError.commandEncodingFailed("output buffer") }

            try engine.appendToCache(PagedKVAppendRequest(
                keys: kBuffer, values: vBuffer,
                kPool: cache.kPoolBuffer, vPool: cache.vPoolBuffer,
                blockTable: try cache.getBlockTableBuffer(forSequence: 1),
                tokenOffset: 0, numNewTokens: seqLen, layer: layerSpec
            ))

            let blockTable = try cache.getBlockTableBuffer(forSequence: 1)

            var totalPA: Double = 0
            for i in 0..<(warmup + samples) {
                let start = CFAbsoluteTimeGetCurrent()
                try engine.prefill(
                    q: qBuffer,
                    kPool: cache.kPoolBuffer,
                    vPool: cache.vPoolBuffer,
                    blockTable: blockTable,
                    seqLen: seqLen,
                    headDim: headDim,
                    numHeads: numHeads,
                    numKVHeads: numKVHeads,
                    blockSize: blockSize,
                    causal: false,
                    output: outputBuffer,
                    dataType: .float16
                )
                let t = (CFAbsoluteTimeGetCurrent() - start) * 1000
                if i >= warmup { totalPA += t }
            }
            let avgPA = totalPA / Double(samples)

            let speedup = avgMLX / avgPA
            let winner = speedup > 1.0 ? "Paged" : "MLX"
            let ratio = speedup > 1.0 ? speedup : (1.0 / speedup)
            let faster = speedup > 1.0
            print("  seqLen=\(String(format: "%4d", seqLen))  MLX: \(String(format: "%7.3f", avgMLX)) ms  Paged: \(String(format: "%7.3f", avgPA)) ms  \(winner) \(String(format: "%.1f", ratio))x\(faster ? "  ✓" : "")")

            cache.freeSequence(id: 1)
        } catch {
            print("  FAILED: \(error)")
        }
    }

    print()
    print("Both sides measure ONLY the attention kernel (no cache setup overhead).")
    print("Crossover point = seqLen where PagedAttention matches or beats MLX.")
    print("At longer contexts, Paged's tiled kernel avoids O(n²) memory.\n")
}

func pagedMetalBuffer(device: MTLDevice, from array: MLXArray) throws -> MTLBuffer {
    guard let buffer = array.asMTLBuffer(device: device, noCopy: false) else {
        throw PagedAttentionMLXError.bufferCreationFailed("array")
    }
    return buffer
}

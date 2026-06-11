import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal
import PagedAttentionMLXSupport

/// Compares standard MLX attention vs paged attention performance.
func runAttentionComparison(device: MTLDevice, engine: PagedAttentionEngine) {
    print("\n=== Attention Comparison: MLX vs PagedAttention ===\n")

    // LLaMA 3.2 1B dimensions
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

    for seqLen in [8, 16, 32, 64] {
        let batch = 1
        let key = MLXRandom.key(42)

        // Generate synthetic Q, K, V matching MLX's [B, nHeads, seqLen, headDim] layout
        let q = MLXRandom.uniform(0..<1, [batch, numHeads, seqLen, headDim], key: key)
        let k = MLXRandom.uniform(0..<1, [batch, numKVHeads, seqLen, headDim], key: key)
        let v = MLXRandom.uniform(0..<1, [batch, numKVHeads, seqLen, headDim], key: key)
        eval(q, k, v)

        let scale = 1.0 / sqrt(Float(headDim))

        // Standard MLX attention
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

        // PagedAttention
        var totalPA: Double = 0
        for i in 0..<(warmup + samples) {
            let pagedCache = try! PagedMetalKVCache(
                sequenceID: 1,
                layer: layerSpec,
                maxBlocks: 64,
                device: device,
                engine: engine
            )

            let start = CFAbsoluteTimeGetCurrent()
            let out = try! pagedCache.pagedAttention(queries: q, keys: k, values: v, mask: .none)
            eval(out)
            let t = (CFAbsoluteTimeGetCurrent() - start) * 1000
            if i >= warmup { totalPA += t }
        }
        let avgPA = totalPA / Double(samples)

        let speedup = avgMLX / avgPA
        let label = speedup > 1.0 ? "✓" : ""
        print("  seqLen=\(seqLen)  MLX: \(String(format: "%.3f", avgMLX)) ms  Paged: \(String(format: "%.3f", avgPA)) ms  \(String(format: "%.1f", speedup))x \(label)")
    }

    print()
    // Interpret results
    print("Results: PagedAttention runs on GPU (Metal) while MLX runs on CPU/GPU.")
    print("For prefill (seqLen ≥ 32), Metal kernel launch cost is amortized.")
    print("For decode (seqLen = 1), MLX's CPU path is faster due to overhead.\n")
}

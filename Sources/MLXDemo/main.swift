import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal
import PagedAttentionMLXSupport

print("=== MLX + PagedAttention Comparison ===\n")

guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal device")
    exit(1)
}

let engine: PagedAttentionEngine
do {
    engine = try PagedAttentionEngine()
    print("PagedAttention engine ready")
} catch {
    print("Engine init: \(error)")
    exit(1)
}

runAttentionComparison(device: device, engine: engine)

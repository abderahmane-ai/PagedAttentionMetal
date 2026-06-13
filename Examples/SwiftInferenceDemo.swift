import Foundation
import Metal
import MLX
import MLXLMCommon
import PagedAttentionMetal
import PagedAttentionMLXSupport

final class MinimalAdapter: @unchecked Sendable, ModelAdapterProtocol {
    let hiddenSize = 512
    let vocabSize = 1000
    let numLayers = 1
    let layerSpecs: [PagedLayerSpec]
    let device: MTLDevice

    init(device: MTLDevice) {
        self.device = device
        self.layerSpecs = [
            PagedLayerSpec(
                headDim: 64,
                numHeads: 8,
                numKVHeads: 2,
                blockSize: 16,
                dataType: .float16
            )
        ]
    }

    func embed(tokens: [Int]) throws -> MLXArray {
        var values = [Float](repeating: 0, count: tokens.count * hiddenSize)
        for (i, token) in tokens.enumerated() {
            for dim in 0..<hiddenSize {
                values[i * hiddenSize + dim] = Float((token + dim) % 127) / 127.0 - 0.5
            }
        }
        let data = Data(bytes: values, count: values.count * MemoryLayout<Float>.stride)
        return MLXArray(data, [tokens.count, hiddenSize], dtype: .float32)
    }

    func projectQKV(hidden: MLXArray, layer: Int, offset: Int) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer) {
        let spec = layerSpecs[layer]
        let seqLen = hidden.shape[0]
        let qSize = seqLen * spec.numHeads * spec.headDim * spec.dataType.byteWidth
        let kvSize = seqLen * spec.numKVHeads * spec.headDim * spec.dataType.byteWidth
        guard let q = device.makeBuffer(length: qSize, options: .storageModeShared),
              let k = device.makeBuffer(length: kvSize, options: .storageModeShared),
              let v = device.makeBuffer(length: kvSize, options: .storageModeShared) else {
            throw PagedAttentionError.commandEncodingFailed("Projection buffer failed")
        }
        return (q, k, v)
    }

    func applyAttentionOutput(hidden: MLXArray, attentionFloats: [Float], layer: Int) throws -> MLXArray {
        hidden
    }

    func projectOutput(hidden: MLXArray) throws -> MLXArray {
        let seqLen = hidden.shape[0]
        var logits = [Float](repeating: 0, count: seqLen * vocabSize)
        for row in 0..<seqLen {
            let offset = row * vocabSize
            for token in 0..<vocabSize {
                logits[offset + token] = Float((hidden.shape[1] + token + row) % 97) / 97.0 - 0.5
            }
        }
        let data = Data(bytes: logits, count: logits.count * MemoryLayout<Float>.stride)
        return MLXArray(data, [seqLen, vocabSize], dtype: .float32)
    }
}

@main
struct SwiftInferenceDemo {
    static func main() async {
        print("Starting direct PagedAttention Swift inference demo...")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: Metal is not supported on this device.")
            return
        }
        print("Running on Device: \(device.name)")
        
        do {
            let adapter = MinimalAdapter(device: device)
            
            // Initialize direct inference pipeline
            let inference = try PagedAttentionInference(
                device: device,
                maxBlocks: 512,
                blockSize: 16,
                layerSpecs: adapter.layerSpecs,
                maxBatchSize: 4,
                maxSequences: 16
            )
            
            let generator = PagedAttentionGenerator(
                inference: inference,
                modelAdapter: adapter,
                samplingConfig: SamplingConfig(temperature: 0.7, topK: 20)
            )
            
            // Dummy input prompt tokens
            let prompt = [10, 20, 30, 40, 50]
            print("Prompt Tokens: \(prompt)")
            
            let start = Date()
            let stream = try generator.generate(promptTokens: prompt, maxNewTokens: 15)
            
            var generated: [Int] = []
            for try await token in stream {
                generated.append(token)
                print("  Generated Token ID: \(token)")
            }
            
            let elapsed = Date().timeIntervalSince(start)
            print("Generation complete!")
            print("Response Tokens: \(generated)")
            print("Total Latency: \(String(format: "%.3f", elapsed)) seconds")
            print("Throughput: \(String(format: "%.1f", Double(generated.count) / elapsed)) tokens/second")
            
        } catch {
            print("Error during execution: \(error)")
        }
    }
}

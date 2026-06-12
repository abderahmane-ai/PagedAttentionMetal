import Foundation
import Metal
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import PagedAttentionMetal

public class LlamaModelAdapter: @unchecked Sendable, ModelAdapterProtocol {
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numLayers: Int
    public let layerSpecs: [PagedLayerSpec]
    public let device: MTLDevice

    private let container: ModelContainer
    private let embedTokens: Embedding
    private let inputNorm: [RMSNorm]
    private let qProj: [Linear]
    private let kProj: [Linear]
    private let vProj: [Linear]
    private let woProj: [Linear]
    private let postAttentionNorm: [RMSNorm]
    private let mlpGate: [Linear]
    private let mlpUp: [Linear]
    private let mlpDown: [Linear]
    private let normModule: RMSNorm
    private let outputProj: (MLXArray) -> MLXArray
    private let llamaModel: LlamaModel
    private let ropeModules: [RoPE]

    public init(container: consuming ModelContainer) async throws {
        self.device = MTLCreateSystemDefaultDevice()!
        self.container = container

        let extract = try await container.perform { context -> LlamaExtract in
            guard let llama = context.model as? LlamaModel else {
                throw PagedAttentionError.unsupported("model is not LlamaModel")
            }
            return LlamaExtract(
                handle: Unmanaged.passUnretained(llama).toOpaque(),
                vocabSize: llama.vocabularySize,
                kvHeads: llama.kvHeads
            )
        }

        let llama = Unmanaged<LlamaModel>.fromOpaque(extract.handle).takeUnretainedValue()
        self.llamaModel = llama
        self.vocabSize = extract.vocabSize
        let kvHeadsArr = extract.kvHeads

        var modelDict = [String: Module]()
        llama.visit { key, module in
            modelDict[key] = module
        }

        guard let emb = modelDict["model.embed_tokens"] as? Embedding else {
            throw PagedAttentionError.unsupported("embed_tokens not found")
        }
        self.embedTokens = emb
        self.hiddenSize = emb.weight.shape[1]

        let _qProj = modelDict.filter { $0.key.contains(".self_attn.q_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _kProj = modelDict.filter { $0.key.contains(".self_attn.k_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _vProj = modelDict.filter { $0.key.contains(".self_attn.v_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _woProj = modelDict.filter { $0.key.contains(".self_attn.o_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _inputNorm = modelDict.filter { $0.key.contains(".input_layernorm") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? RMSNorm }
        let _postNorm = modelDict.filter { $0.key.contains(".post_attention_layernorm") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? RMSNorm }
        let _mlpGate = modelDict.filter { $0.key.contains(".mlp.gate_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _mlpUp = modelDict.filter { $0.key.contains(".mlp.up_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        let _mlpDown = modelDict.filter { $0.key.contains(".mlp.down_proj") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? Linear }
        self.ropeModules = modelDict.filter { $0.key.contains(".self_attn.rotary_emb") }
            .sorted { layerIndex($0.key) < layerIndex($1.key) }
            .compactMap { $0.value as? RoPE }

        guard !_qProj.isEmpty, _qProj.count == _kProj.count,
              _qProj.count == _vProj.count, _qProj.count == _woProj.count,
              _qProj.count == _inputNorm.count, _qProj.count == _postNorm.count,
              _qProj.count == _mlpGate.count, _qProj.count == _mlpUp.count,
              _qProj.count == _mlpDown.count, _qProj.count == ropeModules.count else {
            throw PagedAttentionError.unsupported("mismatched per-layer module counts")
        }
        self.qProj = _qProj
        self.kProj = _kProj
        self.vProj = _vProj
        self.woProj = _woProj
        self.inputNorm = _inputNorm
        self.postAttentionNorm = _postNorm
        self.mlpGate = _mlpGate
        self.mlpUp = _mlpUp
        self.mlpDown = _mlpDown
        self.numLayers = _qProj.count

        guard let normMod = modelDict["model.norm"] as? RMSNorm else {
            throw PagedAttentionError.unsupported("model.norm not found")
        }
        self.normModule = normMod

        let kvHeadsCount = kvHeadsArr.first ?? 1
        let qWeight = _qProj[0].weight
        let kWeight = _kProj[0].weight
        let headDim = kWeight.shape[0] / kvHeadsCount
        let numHeads = qWeight.shape[0] / headDim

        let spec = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: kvHeadsCount,
            blockSize: 16,
            dataType: .float16
        )
        self.layerSpecs = Array(repeating: spec, count: numLayers)

        let capturedNorm = normMod
        let capturedEmb = emb
        if let lmHead = modelDict["lm_head"] as? Linear {
            self.outputProj = { hidden in
                let n = capturedNorm(hidden)
                return lmHead(n.asType(.float16))
            }
        } else {
            self.outputProj = { hidden in
                let n = capturedNorm(hidden)
                return capturedEmb.asLinear(n.asType(.float16))
            }
        }
    }

    public func embed(tokens: [Int]) throws -> MLXArray {
        let tokenIds = MLXArray(tokens)
        let result = embedTokens(tokenIds)
        eval(result)
        return result
    }

    private func layerIndex(_ key: String) -> Int {
        guard let range = key.range(of: "model.layers.") else { return Int.max }
        let suffix = key[range.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        return Int(digits) ?? Int.max
    }

    private func applyRotary(_ arr: MLXArray, layer: Int, offset: Int) -> MLXArray {
        ropeModules[layer](arr.reshaped(arr.dim(0), layerSpecs[layer].numHeads, layerSpecs[layer].headDim), offset: offset)
            .reshaped(arr.dim(0), layerSpecs[layer].numHeads * layerSpecs[layer].headDim)
    }

    public func projectQKV(hidden: MLXArray, layer: Int, offset: Int = 0) throws -> (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer) {
        let normed = inputNorm[layer](hidden)
        let h = normed.asType(.float16)
        let spec = layerSpecs[layer]
        let qArr = qProj[layer](h).asType(.float16)
        let kArr = kProj[layer](h).asType(.float16)
        let vArr = vProj[layer](h).asType(.float16)
        let qRotated = applyRotary(qArr, layer: layer, offset: offset)
        let kRotated = applyRotary(kArr, layer: layer, offset: offset)
        eval(qRotated, kRotated, vArr)
        return (try toBuffer(qRotated), try toBuffer(kRotated), try toBuffer(vArr))
    }

    public func applyAttentionOutput(hidden: MLXArray, attentionFloats: [Float], layer: Int) throws -> MLXArray {
        let spec = layerSpecs[layer]
        let attnLen = spec.numHeads * spec.headDim
        let seqLen = attentionFloats.count / attnLen
        let attn = MLXArray(
            Data(bytes: attentionFloats, count: attentionFloats.count * MemoryLayout<Float>.stride),
            [seqLen, attnLen], dtype: .float32
        )

        let woOut = woProj[layer](attn.asType(.float16))
        let h1 = hidden.asType(.float32) + woOut.asType(.float32)
        let mlpIn = postAttentionNorm[layer](h1)
        let gate = mlpGate[layer](mlpIn.asType(.float16))
        let up = mlpUp[layer](mlpIn.asType(.float16))
        let mlpOut = mlpDown[layer](silu(gate) * up)
        let h2 = h1 + mlpOut.asType(.float32)
        eval(h2)
        return h2
    }

    public func projectOutput(hidden: MLXArray) throws -> MLXArray {
        let result = outputProj(hidden)
        eval(result)
        return result
    }

    private func toBuffer(_ arr: MLXArray) throws -> MTLBuffer {
        let d = arr.asData().data
        guard let buf = device.makeBuffer(
            length: d.count, options: .storageModeShared
        ) else {
            throw PagedAttentionError.commandEncodingFailed("buffer allocation")
        }
        d.withUnsafeBytes { ptr in
            buf.contents().copyMemory(from: ptr.baseAddress!, byteCount: d.count)
        }
        return buf
    }
}

private struct LlamaExtract: @unchecked Sendable {
    let handle: UnsafeMutableRawPointer
    let vocabSize: Int
    let kvHeads: [Int]
}

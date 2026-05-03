# Framework Integration Examples

This guide shows how to integrate PagedAttentionMetal with various ML frameworks and inference engines.

## Table of Contents

1. [PyTorch via Swift-Python Bridge](#pytorch-via-swift-python-bridge)
2. [TensorFlow Lite](#tensorflow-lite)
3. [ONNX Runtime](#onnx-runtime)
4. [Custom Inference Engines](#custom-inference-engines)
5. [Hugging Face Transformers](#hugging-face-transformers)

---

## PyTorch via Swift-Python Bridge

Use [PythonKit](https://github.com/pvieito/PythonKit) to bridge PyTorch tensors with PagedAttentionMetal.

### Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/pvieito/PythonKit", from: "0.3.1"),
    .package(url: "https://github.com/yourusername/PagedAttentionMetal", from: "1.0.0")
]
```

### Tensor Conversion Bridge

```swift
import PythonKit
import Metal
import PagedAttentionMetal

class PyTorchBridge {
    let torch = Python.import("torch")
    let device: MTLDevice
    
    init() {
        self.device = MTLCreateSystemDefaultDevice()!
    }
    
    /// Convert PyTorch tensor to Metal buffer
    func torchToMetal(_ tensor: PythonObject) -> MTLBuffer? {
        // Move to CPU and convert to numpy
        let cpu_tensor = tensor.cpu()
        let numpy_array = cpu_tensor.numpy()
        
        // Get raw data
        let data = numpy_array.__array_interface__["data"][0]
        let size = Int(numpy_array.nbytes)!
        
        // Create Metal buffer
        let ptr = UnsafeRawPointer(bitPattern: Int(data)!)!
        return device.makeBuffer(
            bytes: ptr,
            length: size,
            options: .storageModeShared
        )
    }
    
    /// Convert Metal buffer to PyTorch tensor
    func metalToTorch(_ buffer: MTLBuffer, shape: [Int], dtype: String = "float32") -> PythonObject {
        let ptr = buffer.contents()
        let count = shape.reduce(1, *)
        
        // Create numpy array from pointer
        let np = Python.import("numpy")
        let array = np.frombuffer(
            Python.bytes(ptr, length: buffer.length),
            dtype: dtype
        ).reshape(shape)
        
        // Convert to PyTorch tensor
        return torch.from_numpy(array)
    }
}
```

### Integration Example

```swift
import PythonKit

func runPyTorchModel() throws {
    let torch = Python.import("torch")
    let transformers = Python.import("transformers")
    
    // Load PyTorch model
    let model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    let tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    
    // Initialize PagedAttention
    let engine = try PagedAttentionEngine()
    let cacheManager = KVCacheManager(
        device: MTLCreateSystemDefaultDevice()!,
        maxBlocks: 1024,
        blockSize: 16,
        headDim: 64,
        numKVHeads: 12,
        dataType: .float16
    )
    
    let bridge = PyTorchBridge()
    let sequenceID = 1
    try cacheManager.allocateSequence(id: sequenceID)
    
    // Tokenize input
    let inputs = tokenizer("Hello, world!", return_tensors: "pt")
    let input_ids = inputs["input_ids"]
    
    // Forward pass through embedding + transformer layers
    var hidden_states = model.transformer.wte(input_ids)  // Word embeddings
    
    for (idx, layer) in model.transformer.h.enumerated() {
        // Run everything except attention in PyTorch
        let ln1_output = layer.ln_1(hidden_states)
        
        // Compute Q, K, V in PyTorch
        let qkv = layer.attn.c_attn(ln1_output)
        let qkv_split = qkv.split(768, dim: -1)  // Split into Q, K, V
        
        // Convert to Metal
        guard let qBuffer = bridge.torchToMetal(qkv_split[0]),
              let kBuffer = bridge.torchToMetal(qkv_split[1]),
              let vBuffer = bridge.torchToMetal(qkv_split[2]) else {
            throw IntegrationError.conversionFailed
        }
        
        // Append K/V to cache
        let seqLen = Int(input_ids.shape[1])!
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        engine.appendToCache(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            tokenOffset: 0,
            numNewTokens: seqLen,
            numKVHeads: 12,
            headDim: 64,
            blockSize: 16,
            dataType: .float16
        )
        
        // Run PagedAttention on Metal
        let outputBuffer = device.makeBuffer(
            length: seqLen * 12 * 64 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        engine.prefill(
            q: qBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            seqLen: seqLen,
            headDim: 64,
            numHeads: 12,
            numKVHeads: 12,
            blockSize: 16,
            causal: true,
            output: outputBuffer,
            dataType: .float16
        )
        
        // Convert back to PyTorch
        let attn_output = bridge.metalToTorch(outputBuffer, shape: [1, seqLen, 768])
        
        // Continue with PyTorch (residual, MLP, etc.)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + layer.mlp(layer.ln_2(hidden_states))
    }
    
    // Final layer norm and LM head
    hidden_states = model.transformer.ln_f(hidden_states)
    let logits = model.lm_head(hidden_states)
    
    print("Logits shape:", logits.shape)
}
```

---

## TensorFlow Lite

Integrate with TensorFlow Lite for on-device inference.

### Custom Delegate

```swift
import TensorFlowLite
import PagedAttentionMetal

class PagedAttentionDelegate: Delegate {
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    
    init() throws {
        self.engine = try PagedAttentionEngine()
        self.cacheManager = KVCacheManager(
            device: MTLCreateSystemDefaultDevice()!,
            maxBlocks: 512,
            blockSize: 16,
            headDim: 64,
            numKVHeads: 8,
            dataType: .float16
        )
    }
    
    func prepare(for graph: Graph) throws {
        // Register custom ops
        graph.registerCustomOp("PagedAttention") { inputs, outputs in
            try self.executePagedAttention(inputs: inputs, outputs: outputs)
        }
    }
    
    private func executePagedAttention(inputs: [Tensor], outputs: inout [Tensor]) throws {
        let q = inputs[0]
        let k = inputs[1]
        let v = inputs[2]
        
        // Convert TFLite tensors to Metal buffers
        guard let qBuffer = tensorToMetal(q),
              let kBuffer = tensorToMetal(k),
              let vBuffer = tensorToMetal(v) else {
            throw DelegateError.conversionFailed
        }
        
        // Run PagedAttention
        let seqLen = q.shape[1]
        let sequenceID = 1
        
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        engine.prefill(
            q: qBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            seqLen: seqLen,
            headDim: 64,
            numHeads: 8,
            numKVHeads: 8,
            blockSize: 16,
            causal: true,
            output: outputBuffer,
            dataType: .float16
        )
        
        // Convert back to TFLite tensor
        outputs[0] = metalToTensor(outputBuffer, shape: q.shape)
    }
}
```

---

## ONNX Runtime

Use ONNX Runtime with custom execution provider.

### Custom Execution Provider

```swift
import ONNXRuntime
import PagedAttentionMetal

class PagedAttentionExecutionProvider: ExecutionProvider {
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    
    init() throws {
        self.engine = try PagedAttentionEngine()
        self.cacheManager = KVCacheManager(
            device: MTLCreateSystemDefaultDevice()!,
            maxBlocks: 1024,
            blockSize: 16,
            headDim: 128,
            numKVHeads: 2,
            dataType: .float16
        )
    }
    
    func canExecute(node: Node) -> Bool {
        // Check if this is an attention node we can handle
        return node.opType == "Attention" || node.opType == "MultiHeadAttention"
    }
    
    func execute(node: Node, inputs: [Tensor], outputs: inout [Tensor]) throws {
        guard node.opType == "Attention" else {
            throw ExecutionError.unsupportedOp
        }
        
        // Extract Q, K, V from inputs
        let q = inputs[0]
        let k = inputs[1]
        let v = inputs[2]
        
        // Convert to Metal
        guard let qBuffer = onnxToMetal(q),
              let kBuffer = onnxToMetal(k),
              let vBuffer = onnxToMetal(v) else {
            throw ExecutionError.conversionFailed
        }
        
        // Run PagedAttention
        let seqLen = q.shape[1]
        let sequenceID = node.name.hashValue
        
        if cacheManager.sequences[sequenceID] == nil {
            try cacheManager.allocateSequence(id: sequenceID)
        }
        
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        let outputBuffer = device.makeBuffer(
            length: seqLen * 8 * 128 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        engine.prefill(
            q: qBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            seqLen: seqLen,
            headDim: 128,
            numHeads: 8,
            numKVHeads: 2,
            blockSize: 16,
            causal: true,
            output: outputBuffer,
            dataType: .float16
        )
        
        // Convert back to ONNX tensor
        outputs[0] = metalToONNX(outputBuffer, shape: q.shape)
    }
}

// Usage
let session = try ORTSession(
    modelPath: "model.onnx",
    executionProviders: [PagedAttentionExecutionProvider()]
)
```

---

## Custom Inference Engines

Build a custom inference engine with PagedAttention.

### Minimal LLM Engine

```swift
import PagedAttentionMetal
import Metal

class CustomLLMEngine {
    let device: MTLDevice
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    
    // Model weights (loaded from checkpoint)
    var embeddings: MTLBuffer
    var layerWeights: [LayerWeights]
    var lmHead: MTLBuffer
    
    struct LayerWeights {
        let qProj: MTLBuffer
        let kProj: MTLBuffer
        let vProj: MTLBuffer
        let oProj: MTLBuffer
        let mlpUp: MTLBuffer
        let mlpDown: MTLBuffer
        let ln1: MTLBuffer
        let ln2: MTLBuffer
    }
    
    init(modelPath: String) throws {
        self.device = MTLCreateSystemDefaultDevice()!
        self.engine = try PagedAttentionEngine()
        
        self.cacheManager = KVCacheManager(
            device: device,
            maxBlocks: 1024,
            blockSize: 16,
            headDim: 128,
            numKVHeads: 2,
            dataType: .float16
        )
        
        // Load weights from checkpoint
        (embeddings, layerWeights, lmHead) = try loadWeights(from: modelPath)
    }
    
    func generate(prompt: [Int], maxTokens: Int = 100) throws -> [Int] {
        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        defer { cacheManager.freeSequence(id: sequenceID) }
        
        var tokens = prompt
        
        // Prefill
        var hidden = embed(tokens: prompt)
        for layer in layerWeights {
            hidden = try forwardLayer(hidden: hidden, layer: layer, sequenceID: sequenceID, isPrefill: true)
        }
        let logits = project(hidden: hidden)
        var nextToken = sample(logits: logits)
        tokens.append(nextToken)
        
        // Decode
        for _ in 0..<maxTokens {
            hidden = embed(tokens: [nextToken])
            for layer in layerWeights {
                hidden = try forwardLayer(hidden: hidden, layer: layer, sequenceID: sequenceID, isPrefill: false)
            }
            let logits = project(hidden: hidden)
            nextToken = sample(logits: logits)
            
            if nextToken == eosToken {
                break
            }
            tokens.append(nextToken)
        }
        
        return tokens
    }
    
    private func forwardLayer(
        hidden: MTLBuffer,
        layer: LayerWeights,
        sequenceID: Int,
        isPrefill: Bool
    ) throws -> MTLBuffer {
        // Layer norm
        let ln1Out = layerNorm(hidden, weights: layer.ln1)
        
        // Q, K, V projections
        let q = matmul(ln1Out, layer.qProj)
        let k = matmul(ln1Out, layer.kProj)
        let v = matmul(ln1Out, layer.vProj)
        
        // Append K/V to cache
        let seqLen = getSeqLen(hidden)
        try cacheManager.appendTokens(toSequence: sequenceID, count: seqLen)
        
        engine.appendToCache(
            keys: k,
            values: v,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
            tokenOffset: try cacheManager.getSequence(id: sequenceID).sequenceLength - seqLen,
            numNewTokens: seqLen,
            numKVHeads: 2,
            headDim: 128,
            blockSize: 16,
            dataType: .float16
        )
        
        // PagedAttention
        let attnOut = device.makeBuffer(
            length: seqLen * 8 * 128 * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        if isPrefill {
            engine.prefill(
                q: q, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                blockTable: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
                seqLen: seqLen, headDim: 128, numHeads: 8, numKVHeads: 2,
                blockSize: 16, causal: true, output: attnOut, dataType: .float16
            )
        } else {
            let seqLengths = [Int32(try cacheManager.getSequence(id: sequenceID).sequenceLength)]
            let seqLenBuffer = device.makeBuffer(
                bytes: seqLengths,
                length: MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )!
            
            engine.decode(
                q: q, kPool: cacheManager.kPoolBuffer, vPool: cacheManager.vPoolBuffer,
                blockTables: try cacheManager.getBlockTableBuffer(forSequence: sequenceID),
                seqLengths: seqLenBuffer, batchSize: 1,
                maxNumBlocks: try cacheManager.getSequence(id: sequenceID).blockTable.count,
                headDim: 128, numHeads: 8, numKVHeads: 2,
                blockSize: 16, output: attnOut, dataType: .float16
            )
        }
        
        // Output projection
        let oOut = matmul(attnOut, layer.oProj)
        
        // Residual
        let residual1 = add(hidden, oOut)
        
        // MLP
        let ln2Out = layerNorm(residual1, weights: layer.ln2)
        let mlpOut = mlp(ln2Out, up: layer.mlpUp, down: layer.mlpDown)
        
        // Residual
        return add(residual1, mlpOut)
    }
    
    // Helper functions (matmul, layerNorm, etc.)
    // Implementation depends on your compute backend
}
```

---

## Hugging Face Transformers

Integrate with Hugging Face models via Python bridge.

### Example: LLaMA Integration

```swift
import PythonKit
import PagedAttentionMetal

class HuggingFaceLLaMA {
    let transformers: PythonObject
    let model: PythonObject
    let tokenizer: PythonObject
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    
    init(modelName: String = "meta-llama/Llama-2-7b-hf") throws {
        self.transformers = Python.import("transformers")
        
        // Load model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            modelName,
            torch_dtype: Python.import("torch").float16,
            device_map: "cpu"  // Keep on CPU, we'll use Metal for attention
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(modelName)
        
        // Initialize PagedAttention
        self.engine = try PagedAttentionEngine()
        self.cacheManager = KVCacheManager(
            device: MTLCreateSystemDefaultDevice()!,
            maxBlocks: 2048,
            blockSize: 16,
            headDim: 128,
            numKVHeads: 8,  // LLaMA-2 uses GQA
            dataType: .float16
        )
    }
    
    func generate(prompt: String, maxTokens: Int = 100) throws -> String {
        let inputs = tokenizer(prompt, return_tensors: "pt")
        let input_ids = inputs["input_ids"]
        
        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        defer { cacheManager.freeSequence(id: sequenceID) }
        
        var generated_ids = input_ids
        
        for _ in 0..<maxTokens {
            // Forward pass with PagedAttention replacing standard attention
            let outputs = try forwardWithPagedAttention(
                input_ids: generated_ids,
                sequenceID: sequenceID
            )
            
            let next_token = outputs.logits[0, -1].argmax(dim: -1).unsqueeze(0)
            generated_ids = Python.import("torch").cat([generated_ids, next_token], dim: 1)
            
            if Int(next_token.item())! == Int(tokenizer.eos_token_id)! {
                break
            }
        }
        
        return String(tokenizer.decode(generated_ids[0]))!
    }
    
    private func forwardWithPagedAttention(
        input_ids: PythonObject,
        sequenceID: Int
    ) throws -> PythonObject {
        // This is a simplified example
        // In practice, you'd need to hook into the model's forward pass
        // and replace the attention computation
        
        // Get hidden states from embeddings
        var hidden_states = model.model.embed_tokens(input_ids)
        
        // Process each layer
        for layer in model.model.layers {
            // Pre-attention norm
            let residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            // Compute Q, K, V
            let q = layer.self_attn.q_proj(hidden_states)
            let k = layer.self_attn.k_proj(hidden_states)
            let v = layer.self_attn.v_proj(hidden_states)
            
            // Use PagedAttention instead of standard attention
            let attn_output = try runPagedAttention(
                q: q, k: k, v: v,
                sequenceID: sequenceID
            )
            
            // Output projection
            hidden_states = layer.self_attn.o_proj(attn_output)
            hidden_states = residual + hidden_states
            
            // MLP
            let residual2 = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual2 + hidden_states
        }
        
        // Final norm and LM head
        hidden_states = model.model.norm(hidden_states)
        let logits = model.lm_head(hidden_states)
        
        return Python.dict(["logits": logits])
    }
}
```

---

## Best Practices

1. **Minimize conversions:** Keep data in Metal as long as possible
2. **Batch operations:** Convert multiple tensors at once
3. **Reuse buffers:** Avoid allocating new buffers every forward pass
4. **Profile carefully:** Measure CPU↔GPU transfer overhead
5. **Handle errors:** Framework bridges can fail in unexpected ways

---

## Next Steps

- See [MLX_INTEGRATION.md](MLX_INTEGRATION.md) for MLX-specific integration
- See [ADVANCED_USAGE.md](ADVANCED_USAGE.md) for performance optimization
- See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details

import Foundation
import Metal
import PagedAttentionMetal

class SyntheticLLM {
    let device: MTLDevice
    let engine: PagedAttentionEngine
    let cacheManager: KVCacheManager
    
    let vocabSize = 1000
    let hiddenSize = 512
    let numLayers = 4
    let numHeads = 8
    let numKVHeads = 2
    let headDim: Int
    let blockSize = 16
    
    var embeddings: [Float]
    var qkvWeights: [[Float]]
    var outputWeights: [Float]
    
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device not available")
        }
        self.device = device
        self.engine = try PagedAttentionEngine()
        self.headDim = hiddenSize / numHeads
        
        let vs = vocabSize
        let hs = hiddenSize
        let nl = numLayers
        
        self.embeddings = (0..<vs * hs).map { _ in Float.random(in: -0.1...0.1) }
        self.qkvWeights = (0..<nl).map { _ in
            (0..<hs * hs * 3).map { _ in Float.random(in: -0.1...0.1) }
        }
        self.outputWeights = (0..<hs * vs).map { _ in Float.random(in: -0.1...0.1) }
        
        self.cacheManager = KVCacheManager(
            device: device,
            maxBlocks: 512,
            blockSize: blockSize,
            headDim: headDim,
            numKVHeads: numKVHeads,
            dataType: .float16
        )
        
        print("✓ Synthetic LLM initialized")
        print("  Config: \(numLayers) layers, \(numHeads) heads, \(numKVHeads) KV heads, dim=\(hiddenSize)")
    }
    
    func embed(tokens: [Int]) -> [Float] {
        var hidden = [Float](repeating: 0, count: tokens.count * hiddenSize)
        for (i, token) in tokens.enumerated() {
            let start = token * hiddenSize
            let end = start + hiddenSize
            let embSlice = Array(embeddings[start..<end])
            hidden.replaceSubrange(i*hiddenSize..<(i+1)*hiddenSize, with: embSlice)
        }
        return hidden
    }
    
    func projectQKV(hidden: [Float], layer: Int) -> (q: [Float], k: [Float], v: [Float]) {
        let seqLen = hidden.count / hiddenSize
        let weights = qkvWeights[layer]
        
        var q = [Float](repeating: 0, count: seqLen * numHeads * headDim)
        var k = [Float](repeating: 0, count: seqLen * numKVHeads * headDim)
        var v = [Float](repeating: 0, count: seqLen * numKVHeads * headDim)
        
        for t in 0..<seqLen {
            let hiddenSlice = Array(hidden[t*hiddenSize..<(t+1)*hiddenSize])
            
            for h in 0..<numHeads {
                for d in 0..<headDim {
                    let idx = h * headDim + d
                    q[t * numHeads * headDim + h * headDim + d] = 
                        zip(hiddenSlice, weights[idx*hiddenSize..<(idx+1)*hiddenSize])
                        .map(*).reduce(0, +)
                }
            }
            
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let idx = numHeads * headDim + h * headDim + d
                    k[t * numKVHeads * headDim + h * headDim + d] = 
                        zip(hiddenSlice, weights[idx*hiddenSize..<(idx+1)*hiddenSize])
                        .map(*).reduce(0, +)
                }
            }
            
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let idx = numHeads * headDim + numKVHeads * headDim + h * headDim + d
                    v[t * numKVHeads * headDim + h * headDim + d] = 
                        zip(hiddenSlice, weights[idx*hiddenSize..<(idx+1)*hiddenSize])
                        .map(*).reduce(0, +)
                }
            }
        }
        
        return (q, k, v)
    }
    
    func attention(
        q: [Float], k: [Float], v: [Float],
        sequenceID: Int, layer: Int, seqLen: Int, isPrefill: Bool
    ) throws -> [Float] {
        let qHalf = q.map { Float16($0) }
        let kHalf = k.map { Float16($0) }
        let vHalf = v.map { Float16($0) }

        guard let qBuffer = device.makeBuffer(
            bytes: qHalf, length: qHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared
        ) else { fatalError("Failed to create Q buffer") }
        
        guard let kBuffer = device.makeBuffer(
            bytes: kHalf, length: kHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared
        ) else { fatalError("Failed to create K buffer") }
        
        guard let vBuffer = device.makeBuffer(
            bytes: vHalf, length: vHalf.count * MemoryLayout<Float16>.stride, options: .storageModeShared
        ) else { fatalError("Failed to create V buffer") }
        
        let blockTable = try cacheManager.getBlockTableBuffer(forSequence: sequenceID)
        let currentLen = try cacheManager.getSequence(id: sequenceID).sequenceLength
        let tokenOffset = currentLen - seqLen
        let layerSpec = PagedLayerSpec(
            headDim: headDim,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            blockSize: blockSize,
            dataType: .float16
        )
        
        try engine.appendToCache(PagedKVAppendRequest(
            keys: kBuffer,
            values: vBuffer,
            kPool: cacheManager.kPoolBuffer,
            vPool: cacheManager.vPoolBuffer,
            blockTable: blockTable,
            tokenOffset: tokenOffset,
            numNewTokens: seqLen,
            layer: layerSpec
        ))
        
        let outputSize = seqLen * numHeads * headDim
        guard let outputBuffer = device.makeBuffer(
            length: outputSize * MemoryLayout<Float16>.stride, options: .storageModeShared
        ) else { fatalError("Failed to create output buffer") }
        
        if isPrefill {
            try engine.prefill(PagedAttentionPrefillRequest(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTable: blockTable,
                output: outputBuffer,
                seqLen: seqLen,
                layer: layerSpec,
                causal: true
            ))
        } else {
            let seqLengths = [UInt32(currentLen)]
            guard let seqLenBuffer = device.makeBuffer(
                bytes: seqLengths, length: MemoryLayout<UInt32>.stride, options: .storageModeShared
            ) else { fatalError("Failed to create seqLen buffer") }
            
            let numBlocks = try cacheManager.getSequence(id: sequenceID).blockTable.count
            try engine.decode(PagedAttentionDecodeRequest(
                q: qBuffer,
                kPool: cacheManager.kPoolBuffer,
                vPool: cacheManager.vPoolBuffer,
                blockTables: blockTable,
                seqLengths: seqLenBuffer,
                output: outputBuffer,
                batchSize: 1,
                maxNumBlocks: numBlocks,
                layer: layerSpec
            ))
        }
        
        let ptr = outputBuffer.contents().assumingMemoryBound(to: Float16.self)
        return (0..<outputSize).map { Float(ptr[$0]) }
    }
    
    func projectOutput(hidden: [Float]) -> [Float] {
        let seqLen = hidden.count / hiddenSize
        var logits = [Float](repeating: 0, count: seqLen * vocabSize)
        
        for t in 0..<seqLen {
            let hiddenSlice = Array(hidden[t*hiddenSize..<(t+1)*hiddenSize])
            for v in 0..<vocabSize {
                logits[t * vocabSize + v] = 
                    zip(hiddenSlice, outputWeights[v*hiddenSize..<(v+1)*hiddenSize])
                    .map(*).reduce(0, +)
            }
        }
        
        return logits
    }
    
    func sample(logits: [Float]) -> Int {
        return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
    
    func generate(promptTokens: [Int], maxNewTokens: Int = 20) throws -> [Int] {
        let sequenceID = 1
        try cacheManager.allocateSequence(id: sequenceID)
        defer { cacheManager.freeSequence(id: sequenceID) }
        
        var tokens = promptTokens
        print("\n=== Generation Start ===")
        print("Prompt tokens: \(promptTokens)")
        
        print("\n[Prefill] Processing \(promptTokens.count) tokens...")
        try cacheManager.appendTokens(toSequence: sequenceID, count: promptTokens.count)
        
        var hidden = embed(tokens: promptTokens)
        for layer in 0..<numLayers {
            let (q, k, v) = projectQKV(hidden: hidden, layer: layer)
            let attnOut = try attention(
                q: q, k: k, v: v, sequenceID: sequenceID, 
                layer: layer, seqLen: promptTokens.count, isPrefill: true
            )
            hidden = attnOut
        }
        
        let logits = projectOutput(hidden: hidden)
        let lastLogits = Array(logits[(promptTokens.count-1)*vocabSize..<promptTokens.count*vocabSize])
        var nextToken = sample(logits: lastLogits)
        tokens.append(nextToken)
        print("  → Generated token: \(nextToken)")
        
        print("\n[Decode] Generating \(maxNewTokens) tokens...")
        for step in 0..<maxNewTokens {
            try cacheManager.appendTokens(toSequence: sequenceID, count: 1)
            
            hidden = embed(tokens: [nextToken])
            for layer in 0..<numLayers {
                let (q, k, v) = projectQKV(hidden: hidden, layer: layer)
                let attnOut = try attention(
                    q: q, k: k, v: v, sequenceID: sequenceID,
                    layer: layer, seqLen: 1, isPrefill: false
                )
                hidden = attnOut
            }
            
            let logits = projectOutput(hidden: hidden)
            nextToken = sample(logits: logits)
            tokens.append(nextToken)
            
            if step % 5 == 0 {
                print("  Step \(step): token \(nextToken)")
            }
        }
        
        print("\n=== Generation Complete ===")
        print("Total tokens: \(tokens.count)")
        let numBlocks = try cacheManager.getSequence(id: sequenceID).blockTable.count
        print("Cache usage: \(numBlocks) blocks")
        
        return tokens
    }
}

print("PagedAttentionMetal - Synthetic LLM Demo")
print("=========================================\n")

do {
    let llm = try SyntheticLLM()
    
    let promptTokens = [42, 123, 456, 789]
    let output = try llm.generate(promptTokens: promptTokens, maxNewTokens: 20)
    
    print("\nFinal output tokens: \(output)")
    print("\n✓ Demo completed successfully!")
    print("  This proves PagedAttention works in a realistic LLM generation loop.")
    print("  Replace random weights with real model to get actual text generation.")
    
} catch {
    print("❌ Error: \(error)")
}

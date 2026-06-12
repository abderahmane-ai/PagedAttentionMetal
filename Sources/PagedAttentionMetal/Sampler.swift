import Foundation

public struct SamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float

    public init(temperature: Float = 1.0, topK: Int = 0, topP: Float = 1.0, repetitionPenalty: Float = 1.0) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }
}

public enum Sampler {

    public static func sample(logits: [Float], config: SamplingConfig, previousTokens: [Int] = []) -> Int {
        var scaled = logits

        if config.repetitionPenalty != 1.0 {
            for token in previousTokens where token >= 0 && token < scaled.count {
                scaled[token] *= config.repetitionPenalty
            }
        }

        if config.temperature == 0 {
            return greedy(logits: logits)
        }

        if config.temperature > 0 {
            let invTemp = 1.0 / config.temperature
            for i in scaled.indices {
                scaled[i] *= invTemp
            }
        }

        let maxVal = scaled.max() ?? 0
        for i in scaled.indices {
            scaled[i] = exp(scaled[i] - maxVal)
        }

        var sum: Float = 0
        for v in scaled { sum += v }

        if config.topK > 0 && config.topK < scaled.count {
            let indexed = scaled.enumerated().sorted { $0.element > $1.element }
            let threshold = indexed[min(config.topK, indexed.count) - 1].element
            for i in scaled.indices where scaled[i] < threshold {
                scaled[i] = 0
            }
            sum = scaled.reduce(0, +)
        }

        if config.topP < 1.0 {
            let indexed = scaled.enumerated().sorted { $0.element > $1.element }
            var cumSum: Float = 0
            var threshold: Float = 0
            for (_, v) in indexed {
                cumSum += v / sum
                if cumSum >= config.topP {
                    threshold = v
                    break
                }
            }
            for i in scaled.indices where scaled[i] < threshold {
                scaled[i] = 0
            }
            sum = scaled.reduce(0, +)
        }

        guard sum > 0 else {
            return scaled.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        var r = Float.random(in: 0..<sum)
        for (i, v) in scaled.enumerated() {
            r -= v
            if r <= 0 { return i }
        }
        return scaled.count - 1
    }

    public static func greedy(logits: [Float]) -> Int {
        logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
}

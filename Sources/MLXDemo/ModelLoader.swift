import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Hub
import Tokenizers
import PagedAttentionMetal

class ModelLoader {
    let modelConfig = LLMRegistry.llama3_2_1B_4bit
    var container: ModelContainer?
    
    func load() async throws -> ModelContainer {
        if let container = container {
            return container
        }
        
        print("Loading \(modelConfig.name)...")
        
        let loaded = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: modelConfig
        ) { progress in
            print("Download: \(Int(progress.fractionCompleted * 100))%")
        }
        
        container = loaded
        print("Model loaded: \(modelConfig.name)")
        return loaded
    }
}

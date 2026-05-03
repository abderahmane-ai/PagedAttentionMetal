// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import PagedAttentionMetal

@main
struct ExampleApp {
    static func main() {
        print("Initializing PagedAttentionMetal Engine...")
        do {
            let _ = try PagedAttentionEngine()
            print("✅ Successfully loaded Paged Attention on Apple Silicon!")
            print("Engine is ready to process tokens.")
        } catch {
            print("❌ Failed to initialize engine: \(error)")
        }
    }
}

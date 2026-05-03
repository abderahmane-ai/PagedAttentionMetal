// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PagedAttentionMetal",
    platforms: [
        .macOS(.v14), .iOS(.v17)
    ],
    products: [
        .library(
            name: "PagedAttentionMetal",
            targets: ["PagedAttentionMetal"]),
        .executable(
            name: "MinimalLLM",
            targets: ["MinimalLLM"]),
        .executable(
            name: "Benchmarks",
            targets: ["Benchmarks"]),
        .executable(
            name: "MLXDemo",
            targets: ["MLXDemo"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "PagedAttentionMetal",
            resources: [.process("kernels.metal")]
        ),
        .executableTarget(
            name: "MinimalLLM",
            dependencies: ["PagedAttentionMetal"],
            exclude: ["README.md"]
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["PagedAttentionMetal"],
            path: "Benchmarks/Sources"
        ),
        .executableTarget(
            name: "MLXDemo",
            dependencies: [
                "PagedAttentionMetal",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXDemo"
        ),
        .testTarget(
            name: "PagedAttentionMetalTests",
            dependencies: ["PagedAttentionMetal"]),
    ],
    swiftLanguageModes: [.v6]
)

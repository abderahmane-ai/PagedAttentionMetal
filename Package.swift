// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PagedAttentionMetal",
    platforms: [
        .macOS(.v14), .iOS(.v17)
    ],
    // "Homepage" and "License" are documented at the repository root (README.md, LICENSE).
    products: [
        .library(
            name: "PagedAttentionMetal",
            targets: ["PagedAttentionMetal"]),
        .library(
            name: "PagedAttentionMLXSupport",
            targets: ["PagedAttentionMLXSupport"]),
        .executable(
            name: "MinimalLLM",
            targets: ["MinimalLLM"]),
        .executable(
            name: "Benchmarks",
            targets: ["Benchmarks"]),
        .executable(
            name: "MLXDemo",
            targets: ["MLXDemo"]),
        .executable(
            name: "Profiler",
            targets: ["Profiler"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "PagedAttentionMetal",
            resources: [.process("kernels.metal")]
        ),
        .target(
            name: "PagedAttentionMLXSupport",
            dependencies: [
                "PagedAttentionMetal",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
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
                "PagedAttentionMLXSupport",
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
        .executableTarget(
            name: "Profiler",
            dependencies: ["PagedAttentionMetal"],
            path: "Profiler/Sources"
        ),
        .testTarget(
            name: "PagedAttentionMetalTests",
            dependencies: ["PagedAttentionMetal", "PagedAttentionMLXSupport"]),
    ],
    swiftLanguageModes: [.v6]
)

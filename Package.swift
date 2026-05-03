// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PagedAttentionMetal",
    platforms: [
        .macOS(.v13), .iOS(.v16)
    ],
    products: [
        .library(
            name: "PagedAttentionMetal",
            targets: ["PagedAttentionMetal"]),
    ],
    targets: [
        .target(
            name: "PagedAttentionMetal",
            resources: [.process("kernels.metal")]
        ),
        .testTarget(
            name: "PagedAttentionMetalTests",
            dependencies: ["PagedAttentionMetal"]),
    ],
    swiftLanguageModes: [.v6]
)

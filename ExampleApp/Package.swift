// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "ExampleApp",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../")
    ],
    targets: [
        .executableTarget(
            name: "ExampleApp",
            dependencies: [
                .product(name: "PagedAttentionMetal", package: "PagedAttentionMetal")
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)

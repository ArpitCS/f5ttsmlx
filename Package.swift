// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "F5TTSMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "F5TTSMLX",
            targets: ["F5TTSMLX"]
        ),
        .executable(
            name: "F5TTSMLXSample",
            targets: ["F5TTSMLXSample"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "F5TTSMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "F5TTSMLXSample",
            dependencies: ["F5TTSMLX"]
        ),
        .testTarget(
            name: "F5TTSMLXTests",
            dependencies: ["F5TTSMLX"]
        ),

    ],
    swiftLanguageModes: [.v6]
)

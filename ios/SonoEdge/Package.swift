// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SonoEdge",
    platforms: [.iOS(.v16)],
    dependencies: [
        .package(url: "https://github.com/tensorflow/tensorflow.git",
                 from: "2.16.0"),
    ],
    targets: [
        .executableTarget(
            name: "SonoEdge",
            dependencies: [
                .product(name: "TensorFlowLiteSwift", package: "tensorflow"),
            ],
            path: "Sources",
            resources: [
                .copy("Models/heart_quality_int8full.tflite"),
                .copy("Models/heart_model_int8full.tflite"),
            ]
        ),
    ]
)

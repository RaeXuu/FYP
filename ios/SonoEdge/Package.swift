// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SonoEdge",
    platforms: [.iOS(.v16)],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "SonoEdge",
            dependencies: [],
            path: "Sources",
            resources: [
                .copy("Models/heart_quality_int8full.tflite"),
                .copy("Models/heart_model_int8full.tflite"),
            ]
        ),
    ]
)

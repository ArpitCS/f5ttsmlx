# F5TTSMLX

F5TTSMLX is an on-device text-to-speech library for Apple platforms, built with MLX Swift.
It follows the F5-TTS MLX architecture and is designed for 4-bit quantized model usage derived from:

- `lucasnewman/f5-tts-mlx`
- `alandao/f5-tts-mlx-4bit`

## Platform Requirements

- Apple Silicon Mac for local macOS execution
- macOS 14+
- iOS 17+

## Installation (Swift Package Manager)

In Xcode:

1. Open your project settings.
2. Go to Package Dependencies.
3. Add this package URL.
4. Select the latest version.

Or in `Package.swift`:

```swift
.dependencies: [
    .package(url: "https://github.com/<your-org>/F5TTSMLX.git", from: "0.1.0")
],
.targets: [
    .target(
        name: "YourApp",
        dependencies: ["F5TTSMLX"]
    )
]
```

## Minimal Usage

```swift
import Foundation
import F5TTSMLX

func synthesizeExample() async throws {
    let config = F5TTSConfig(
        modelSource: .huggingFace(repoId: "alandao/f5-tts-mlx-4bit"),
        maxLength: 128,
        temperature: 0.8,
        seed: 42,
        enableVoiceMatching: false
    )

    let tts = try await F5TTS(config: config)
    let result = try await tts.synthesize("Hello from F5TTSMLX")

    // result.sampleRate == 24_000
    // result.samples is mono PCM float data in memory
    print("Sample rate: \(result.sampleRate), samples: \(result.samples.count)")
}
```

Note: Hugging Face model download/cache resolution is currently a planned path in the loader. Today, local model directories are the most direct way to run end-to-end.

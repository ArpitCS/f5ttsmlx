import Foundation
import F5TTSMLX

@main
struct F5TTSMLXSampleMain {
    static func main() async {
        do {
            let args = CommandLine.arguments
            guard args.count >= 3 else {
                print("Usage: swift run F5TTSMLXSample <hf:repo-id OR /local/path> <output.wav> [text]")
                return
            }

            let source: F5TTSConfig.ModelSource
            if args[1].hasPrefix("hf:") {
                source = .huggingFace(repoId: String(args[1].dropFirst(3)))
            } else {
                source = .localDirectory(URL(fileURLWithPath: args[1], isDirectory: true))
            }

            let outputURL = URL(fileURLWithPath: args[2], isDirectory: false)
            let text = args.count > 3 ? args[3] : "Hello from F5TTSMLX"

            let config = F5TTSConfig(
                modelSource: source,
                maxLength: 64,
                temperature: 0.8,
                seed: 42,
                enableVoiceMatching: false
            )

            let tts = try await F5TTS(config: config)
            let audio = try await tts.synthesize(text)

            try writeWAV(samples: audio.samples, sampleRate: audio.sampleRate, to: outputURL)
            print("Wrote WAV to \(outputURL.path)")
            print("Sample count: \(audio.samples.count), sample rate: \(audio.sampleRate)")
        } catch {
            print("Failed: \(error)")
        }
    }

    private static func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        let channelCount: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample / 8)

        var pcmData = Data(capacity: samples.count * bytesPerSample)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16((clamped * Float(Int16.max)).rounded())
            var le = intSample.littleEndian
            withUnsafeBytes(of: &le) { pcmData.append(contentsOf: $0) }
        }

        let byteRate = UInt32(sampleRate) * UInt32(channelCount) * UInt32(bitsPerSample / 8)
        let blockAlign = UInt16(channelCount * (bitsPerSample / 8))
        let dataSize = UInt32(pcmData.count)
        let riffSize = UInt32(36) + dataSize

        var wav = Data()
        wav.append("RIFF".data(using: .ascii)!)
        appendLE(riffSize, to: &wav)
        wav.append("WAVE".data(using: .ascii)!)

        wav.append("fmt ".data(using: .ascii)!)
        appendLE(UInt32(16), to: &wav)
        appendLE(UInt16(1), to: &wav)
        appendLE(channelCount, to: &wav)
        appendLE(UInt32(sampleRate), to: &wav)
        appendLE(byteRate, to: &wav)
        appendLE(blockAlign, to: &wav)
        appendLE(bitsPerSample, to: &wav)

        wav.append("data".data(using: .ascii)!)
        appendLE(dataSize, to: &wav)
        wav.append(pcmData)

        try wav.write(to: url)
    }

    private static func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}

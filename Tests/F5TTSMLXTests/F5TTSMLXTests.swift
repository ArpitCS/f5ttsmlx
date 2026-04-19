import Foundation
import XCTest
@testable import F5TTSMLX

final class F5TTSMLXTests: XCTestCase {
    func testTokenizerIsDeterministic() {
        let vocab: [String: Int] = [
            "h": 0,
            "e": 1,
            "l": 2,
            "o": 3,
            "he": 4,
            "ll": 5,
            " ": 6,
            "w": 7,
            "r": 8,
            "d": 9,
        ]
        let tokenizer = Tokenizer(vocabulary: vocab)
        let inputs = ["hello", "hello world", "hehe"]

        for text in inputs {
            let first = tokenizer.tokenize(text)
            let second = tokenizer.tokenize(text)
            XCTAssertEqual(first, second, "Tokenization should be deterministic for: \(text)")
        }
    }

    func testSafeTensorsLoaderParsesTinyFile() throws {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("F5TTSMLXTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        let safetensorsURL = temp.appendingPathComponent("tiny.safetensors", isDirectory: false)
        try writeTinySafeTensors(to: safetensorsURL)

        let loader = try SafeTensorsLoader(fileURL: safetensorsURL)
        XCTAssertEqual(loader.tensorNames, ["tiny.weight"])

        let tensor = try loader.loadTensor(named: "tiny.weight")
        XCTAssertEqual(tensor.shape, [2])
        XCTAssertEqual(tensor.dtype, .float32)
        XCTAssertEqual(tensor.data.count, 8)
    }

    func testModelLoaderWithTinyBundleAndF5TTSSmoke() async throws {
        guard ProcessInfo.processInfo.environment["ENABLE_MLX_SMOKE_TESTS"] == "1" else {
            throw XCTSkip("Set ENABLE_MLX_SMOKE_TESTS=1 to run MLX-backed synth smoke test.")
        }

        let bundle = try makeTinyModelBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let loader = try ModelLoader(rootURL: bundle)
        let tokenizer = try loader.makeTokenizer()
        XCTAssertFalse(tokenizer.tokenize("hi").isEmpty)

        let config = F5TTSConfig(
            modelSource: .localDirectory(bundle),
            maxLength: 32,
            temperature: 0.8,
            seed: 1,
            enableVoiceMatching: false
        )

        let tts = try await F5TTS(config: config)
        let result = try await tts.synthesize("hi")

        XCTAssertEqual(result.sampleRate, 24_000)
        XCTAssertFalse(result.samples.isEmpty)
    }

    private func makeTinyModelBundle() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("F5TTSMLXModelBundle-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        try writeEmptySafeTensors(to: root.appendingPathComponent("model.safetensors"))
        try writeEmptySafeTensors(to: root.appendingPathComponent("duration_model.safetensors"))
        try writeEmptySafeTensors(to: root.appendingPathComponent("duration_v2.safetensors"))

        let vocab = ["h", "i", " ", "e", "l", "o", "w", "r", "d"].joined(separator: "\n") + "\n"
        try vocab.write(to: root.appendingPathComponent("vocab.txt"), atomically: true, encoding: .utf8)

        return root
    }

    private func writeTinySafeTensors(to url: URL) throws {
        let payload = float32Data([1.0, 2.0])
        let header: [String: Any] = [
            "tiny.weight": [
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, payload.count],
            ]
        ]
        try writeSafeTensorsFile(header: header, payload: payload, to: url)
    }

    private func writeEmptySafeTensors(to url: URL) throws {
        let header: [String: Any] = ["__metadata__": [:]]
        try writeSafeTensorsFile(header: header, payload: Data(), to: url)
    }

    private func writeSafeTensorsFile(header: [String: Any], payload: Data, to url: URL) throws {
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])

        var result = Data()
        var headerLen = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { result.append(contentsOf: $0) }
        result.append(headerData)
        result.append(payload)

        try result.write(to: url)
    }

    private func float32Data(_ values: [Float]) -> Data {
        return values.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
    }
}

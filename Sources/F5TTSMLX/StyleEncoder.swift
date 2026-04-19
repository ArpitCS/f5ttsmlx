import Foundation

// Internal style/reference encoder skeleton mapped to reference audio
// conditioning in the Python f5-tts-mlx repository.
struct StyleEncoder {
    init() {}

    func forward(referenceAudioURL: URL, referenceText: String?) throws -> [Float] {
        let _ = (referenceAudioURL, referenceText)
        throw NSError(
            domain: "F5TTSMLX.StyleEncoder",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "StyleEncoder.forward(referenceAudioURL:referenceText:) is not implemented yet."]
        )
    }
}

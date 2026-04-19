import Foundation

// Internal text encoder skeleton mapped to the text-conditioning module in
// the Python f5-tts-mlx repository.
struct TextEncoder {
    init() {}

    func forward(tokens: [Int]) throws -> [Float] {
        let _ = tokens
        throw NSError(
            domain: "F5TTSMLX.TextEncoder",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "TextEncoder.forward(tokens:) is not implemented yet."]
        )
    }
}

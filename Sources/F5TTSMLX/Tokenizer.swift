import Foundation

// Internal tokenizer skeleton mapped to the tokenizer and vocab handling in
// the Python f5-tts-mlx repository.
struct Tokenizer {
    let vocabulary: [String: Int]

    init(vocabulary: [String: Int] = [:]) {
        self.vocabulary = vocabulary
    }

    func forward(_ text: String) throws -> [Int] {
        let _ = text
        throw NSError(
            domain: "F5TTSMLX.Tokenizer",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Tokenizer.forward(_:) is not implemented yet."]
        )
    }
}

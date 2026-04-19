import Foundation

// Internal duration predictor skeleton mapped to the duration predictor module
// in the Python f5-tts-mlx repository.
struct DurationPredictor {
    init() {}

    func forward(textEmbedding: [Float], styleEmbedding: [Float]?) throws -> Int {
        let _ = (textEmbedding, styleEmbedding)
        throw NSError(
            domain: "F5TTSMLX.DurationPredictor",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "DurationPredictor.forward(textEmbedding:styleEmbedding:) is not implemented yet."]
        )
    }
}

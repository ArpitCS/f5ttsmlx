import Foundation

// Internal vocoder skeleton mapped to the Vocos vocoder stage used by
// the Python f5-tts-mlx repository.
struct Vocoder {
    let sampleRate: Int

    init(sampleRate: Int = 24_000) {
        self.sampleRate = sampleRate
    }

    func forward(melSpectrogram: [Float]) throws -> [Float] {
        let _ = melSpectrogram
        throw NSError(
            domain: "F5TTSMLX.Vocoder",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Vocoder.forward(melSpectrogram:) is not implemented yet."]
        )
    }
}

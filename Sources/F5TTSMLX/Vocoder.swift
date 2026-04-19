import Foundation
import MLX
import MLXNN

// Internal vocoder mapped to the Vocos stage used by the Python
// f5-tts-mlx repository.
struct Vocoder {
    typealias ExternalWeights = [String: MLXArray]

    struct Configuration {
        var melBins: Int
        var hiddenSize: Int
        var depth: Int
        var kernelSize: Int
        var hopLength: Int

        // 24kHz with 10ms frame rate => 240 samples per mel frame.
        static let f5Approximation = Configuration(
            melBins: 100,
            hiddenSize: 512,
            depth: 8,
            kernelSize: 7,
            hopLength: 240
        )
    }

    private final class ConvNeXt1DBlock {
        let norm: LayerNorm
        let depthwise: Conv1d
        let ffIn: Linear
        let ffOut: Linear

        init(hiddenSize: Int, kernelSize: Int) {
            self.norm = LayerNorm(dimensions: hiddenSize)
            self.depthwise = Conv1d(
                inputChannels: hiddenSize,
                outputChannels: hiddenSize,
                kernelSize: kernelSize,
                stride: 1,
                padding: kernelSize / 2,
                dilation: 1,
                groups: hiddenSize,
                bias: true
            )
            self.ffIn = Linear(hiddenSize, hiddenSize * 4)
            self.ffOut = Linear(hiddenSize * 4, hiddenSize)
        }

        func forward(_ x: MLXArray) -> MLXArray {
            let y = depthwise(norm(x))
            let z = ffOut(gelu(ffIn(y)))
            return x + z
        }
    }

    let sampleRate: Int
    private let config: Configuration
    private let parameterDType: DType
    private let externalWeights: ExternalWeights
    private let inputProjection: Conv1d
    private let blocks: [ConvNeXt1DBlock]
    private let outputNorm: LayerNorm
    private let synthesisHead: Linear

    init(
        sampleRate: Int = 24_000,
        configuration: Configuration = .f5Approximation,
        parameterDType: DType = .float32,
        externalWeights: ExternalWeights = [:]
    ) {
        precondition(sampleRate == 24_000, "Vocoder currently supports 24kHz output")

        self.sampleRate = sampleRate
        self.config = configuration
        self.parameterDType = parameterDType
        self.externalWeights = externalWeights
        self.inputProjection = Conv1d(
            inputChannels: configuration.melBins,
            outputChannels: configuration.hiddenSize,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self.blocks = (0..<configuration.depth).map { _ in
            ConvNeXt1DBlock(
                hiddenSize: configuration.hiddenSize,
                kernelSize: configuration.kernelSize
            )
        }
        self.outputNorm = LayerNorm(dimensions: configuration.hiddenSize)
        self.synthesisHead = Linear(configuration.hiddenSize, configuration.hopLength)
    }

    // mels: [batch, frameCount, melBins]
    // returns waveform: [batch, sampleCount] where sampleCount = frameCount * hopLength.
    func generateAudio(from mels: MLXArray) -> MLXArray {
        precondition(mels.ndim == 3, "mels must have shape [batch, frameCount, melBins]")
        precondition(
            mels.dim(-1) == config.melBins,
            "mel bin count must match Vocoder configuration"
        )

        let batchSize = mels.dim(0)
        let frameCount = mels.dim(1)

        var x = inputProjection(mels)
        for block in blocks {
            x = block.forward(x)
        }

        let perFrameAudio = tanh(synthesisHead(outputNorm(x)))
        let waveform = perFrameAudio.reshaped([batchSize, frameCount * config.hopLength])

        precondition(waveform.dim(0) == batchSize, "Waveform batch dimension mismatch")
        return waveform
    }

    func forward(melSpectrogram: [Float]) throws -> [Float] {
        let _ = melSpectrogram
        throw NSError(
            domain: "F5TTSMLX.Vocoder",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Use generateAudio(from:) with an MLXArray mel tensor."]
        )
    }
}

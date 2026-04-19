import Foundation
import MLX
import MLXNN

// Internal style/reference encoder skeleton mapped to reference audio
// conditioning in the Python f5-tts-mlx repository.
struct StyleEncoder {
    typealias ExternalWeights = [String: MLXArray]

    struct Configuration {
        var sampleRate: Int
        var convChannels: [Int]
        var convKernelSizes: [Int]
        var convStrides: [Int]
        var transformerDepth: Int
        var transformerHeads: Int
        var hiddenSize: Int
        var feedForwardMultiplier: Int
        var maxPositionEmbeddings: Int

        static let f5Approximation = Configuration(
            sampleRate: 24_000,
            convChannels: [1, 128, 256, 512],
            convKernelSizes: [7, 5, 5],
            convStrides: [2, 2, 2],
            transformerDepth: 6,
            transformerHeads: 8,
            hiddenSize: 512,
            feedForwardMultiplier: 4,
            maxPositionEmbeddings: 2_048
        )
    }

    private final class ConvBlock {
        let conv: Conv1d
        let norm: LayerNorm

        init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int) {
            self.conv = Conv1d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: kernelSize,
                stride: stride,
                padding: kernelSize / 2
            )
            self.norm = LayerNorm(dimensions: outputChannels)
        }

        func forward(_ x: MLXArray) -> MLXArray {
            gelu(norm(conv(x)))
        }
    }

    private final class TransformerBlock {
        let attention: MultiHeadAttention
        let norm1: LayerNorm
        let norm2: LayerNorm
        let ffnIn: Linear
        let ffnOut: Linear

        init(hiddenSize: Int, headCount: Int, feedForwardSize: Int) {
            self.attention = MultiHeadAttention(dimensions: hiddenSize, numHeads: headCount)
            self.norm1 = LayerNorm(dimensions: hiddenSize)
            self.norm2 = LayerNorm(dimensions: hiddenSize)
            self.ffnIn = Linear(hiddenSize, feedForwardSize)
            self.ffnOut = Linear(feedForwardSize, hiddenSize)
        }

        func forward(_ x: MLXArray) -> MLXArray {
            let normed = norm1(x)
            let attended = attention(normed, keys: normed, values: normed, mask: nil)
            let residual = x + attended

            let ffnHidden = gelu(ffnIn(norm2(residual)))
            let ffnOutput = ffnOut(ffnHidden)
            return residual + ffnOutput
        }
    }

    private let config: Configuration
    private let parameterDType: DType
    private let externalWeights: ExternalWeights
    private let convBlocks: [ConvBlock]
    private let inputProjection: Linear
    private let positionEmbedding: Embedding
    private let transformerBlocks: [TransformerBlock]
    private let outputNorm: LayerNorm

    init(
        config: Configuration = .f5Approximation,
        parameterDType: DType = .float32,
        externalWeights: ExternalWeights = [:]
    ) {
        precondition(config.convChannels.count >= 2, "convChannels must include at least input and one output channel")
        precondition(config.convKernelSizes.count == config.convChannels.count - 1, "convKernelSizes count must match conv stage count")
        precondition(config.convStrides.count == config.convChannels.count - 1, "convStrides count must match conv stage count")
        precondition(config.hiddenSize % config.transformerHeads == 0, "hiddenSize must be divisible by transformerHeads")

        self.config = config
        self.parameterDType = parameterDType
        self.externalWeights = externalWeights

        self.convBlocks = (0..<(config.convChannels.count - 1)).map { index in
            ConvBlock(
                inputChannels: config.convChannels[index],
                outputChannels: config.convChannels[index + 1],
                kernelSize: config.convKernelSizes[index],
                stride: config.convStrides[index]
            )
        }

        let convOutChannels = config.convChannels.last ?? config.hiddenSize
        self.inputProjection = Linear(convOutChannels, config.hiddenSize)
        self.positionEmbedding = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )

        self.transformerBlocks = (0..<config.transformerDepth).map { _ in
            TransformerBlock(
                hiddenSize: config.hiddenSize,
                headCount: config.transformerHeads,
                feedForwardSize: config.hiddenSize * config.feedForwardMultiplier
            )
        }

        self.outputNorm = LayerNorm(dimensions: config.hiddenSize)
    }

    // Expected input shape: [batch, samples] at 24kHz mono.
    // Returns a style embedding of shape [batch, hiddenSize].
    func forward(audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 2, "audio must have shape [batch, samples]")

        var x = audio.expandedDimensions(axis: -1)

        for block in convBlocks {
            x = block.forward(x)
        }

        x = inputProjection(x)

        let sequenceLength = x.dim(1)
        precondition(
            sequenceLength <= config.maxPositionEmbeddings,
            "Sequence length exceeds maxPositionEmbeddings for StyleEncoder"
        )

        let positionIDs = MLXArray(0..<sequenceLength, [1, sequenceLength])
        x = x + positionEmbedding(positionIDs)

        for block in transformerBlocks {
            x = block.forward(x)
        }

        let refined = outputNorm(x)
        return refined.mean(axis: 1)
    }

    func forward(referenceAudioURL: URL, referenceText: String?) throws -> [Float] {
        let _ = (referenceAudioURL, referenceText)
        throw NSError(
            domain: "F5TTSMLX.StyleEncoder",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "StyleEncoder.forward(referenceAudioURL:referenceText:) is not implemented yet. Use forward(audio:) once audio decoding is wired."]
        )
    }
}

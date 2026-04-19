import Foundation
import MLX
import MLXNN

// Internal duration predictor skeleton mapped to the duration predictor module
// in the Python f5-tts-mlx repository.
struct DurationPredictor {
    enum Variant: String, Sendable {
        case original
        case v2

        var checkpointFileName: String {
            switch self {
            case .original:
                return "duration_model.safetensors"
            case .v2:
                return "duration_v2.safetensors"
            }
        }
    }

    struct Configuration {
        var inputFeatureSize: Int
        var modelSize: Int
        var depth: Int
        var headCount: Int
        var feedForwardMultiplier: Int
        var maxPositionEmbeddings: Int

        static let originalDefault = Configuration(
            inputFeatureSize: 1_024,
            modelSize: 512,
            depth: 4,
            headCount: 8,
            feedForwardMultiplier: 2,
            maxPositionEmbeddings: 4_096
        )

        static let v2Default = Configuration(
            inputFeatureSize: 1_024,
            modelSize: 512,
            depth: 8,
            headCount: 8,
            feedForwardMultiplier: 2,
            maxPositionEmbeddings: 4_096
        )

        static func `default`(for variant: Variant) -> Configuration {
            switch variant {
            case .original:
                return .originalDefault
            case .v2:
                return .v2Default
            }
        }
    }

    private final class TransformerBlock {
        let attention: MultiHeadAttention
        let norm1: LayerNorm
        let norm2: LayerNorm
        let ffnIn: Linear
        let ffnOut: Linear

        init(modelSize: Int, headCount: Int, feedForwardSize: Int) {
            self.attention = MultiHeadAttention(dimensions: modelSize, numHeads: headCount)
            self.norm1 = LayerNorm(dimensions: modelSize)
            self.norm2 = LayerNorm(dimensions: modelSize)
            self.ffnIn = Linear(modelSize, feedForwardSize)
            self.ffnOut = Linear(feedForwardSize, modelSize)
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

    let variant: Variant
    private let config: Configuration
    private let inputProjection: Linear
    private let positionEmbedding: Embedding
    private let blocks: [TransformerBlock]
    private let outputNorm: LayerNorm
    private let durationHead: Linear

    init(variant: Variant = .v2, configuration: Configuration? = nil) {
        let config = configuration ?? .default(for: variant)

        precondition(config.modelSize % config.headCount == 0, "modelSize must be divisible by headCount")

        self.variant = variant
        self.config = config
        self.inputProjection = Linear(config.inputFeatureSize, config.modelSize)
        self.positionEmbedding = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.modelSize
        )
        self.blocks = (0..<config.depth).map { _ in
            TransformerBlock(
                modelSize: config.modelSize,
                headCount: config.headCount,
                feedForwardSize: config.modelSize * config.feedForwardMultiplier
            )
        }
        self.outputNorm = LayerNorm(dimensions: config.modelSize)
        self.durationHead = Linear(config.modelSize, 1)
    }

    // textFeatures is expected to be [batch, tokenCount, hiddenDim].
    // Output is [batch, tokenCount] with positive duration-like values.
    func forward(textFeatures: MLXArray) -> MLXArray {
        precondition(textFeatures.ndim == 3, "textFeatures must have shape [batch, tokenCount, hiddenDim]")
        precondition(
            textFeatures.dim(-1) == config.inputFeatureSize,
            "textFeatures hidden dimension must match inputFeatureSize"
        )

        let tokenCount = textFeatures.dim(1)
        precondition(
            tokenCount <= config.maxPositionEmbeddings,
            "tokenCount exceeds maxPositionEmbeddings"
        )

        let positionIDs = MLXArray(0..<tokenCount, [1, tokenCount])

        var x = inputProjection(textFeatures)
        x = x + positionEmbedding(positionIDs)

        for block in blocks {
            x = block.forward(x)
        }

        x = outputNorm(x)
        let logits = durationHead(x)
        let positiveDurations = softplus(logits)
        return squeezed(positiveDurations, axis: -1)
    }

    func forward(textEmbedding: [Float], styleEmbedding: [Float]?) throws -> Int {
        let _ = (textEmbedding, styleEmbedding)
        throw NSError(
            domain: "F5TTSMLX.DurationPredictor",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Use forward(textFeatures:) with MLXArray input."]
        )
    }
}

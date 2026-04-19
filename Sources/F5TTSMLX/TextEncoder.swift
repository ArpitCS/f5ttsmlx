import Foundation
import MLX
import MLXNN

// Internal text encoder skeleton mapped to the text-conditioning module in
// the Python f5-tts-mlx repository.
struct TextEncoder {
    struct Configuration {
        var vocabularySize: Int
        var hiddenSize: Int
        var depth: Int
        var headCount: Int
        var feedForwardMultiplier: Int
        var maxPositionEmbeddings: Int

        static let f5Approximation = Configuration(
            vocabularySize: 2_048,
            hiddenSize: 1_024,
            depth: 22,
            headCount: 16,
            feedForwardMultiplier: 4,
            maxPositionEmbeddings: 4_096
        )
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
            let attended = attention(norm1(x), keys: norm1(x), values: norm1(x), mask: nil)
            let residual = x + attended

            let ffnHidden = gelu(ffnIn(norm2(residual)))
            let ffnOut = ffnOut(ffnHidden)
            return residual + ffnOut
        }
    }

    private let config: Configuration
    private let tokenEmbedding: Embedding
    private let positionEmbedding: Embedding
    private let blocks: [TransformerBlock]
    private let finalNorm: LayerNorm

    init(config: Configuration = .f5Approximation) {
        self.config = config
        self.tokenEmbedding = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.positionEmbedding = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        self.blocks = (0..<config.depth).map { _ in
            TransformerBlock(
                hiddenSize: config.hiddenSize,
                headCount: config.headCount,
                feedForwardSize: config.hiddenSize * config.feedForwardMultiplier
            )
        }
        self.finalNorm = LayerNorm(dimensions: config.hiddenSize)
    }

    func forward(tokens: MLXArray) -> MLXArray {
        let sequenceLength = tokens.dim(-1)
        let positionIDs = MLXArray(0..<sequenceLength, [1, sequenceLength])

        var x = tokenEmbedding(tokens) + positionEmbedding(positionIDs)

        for block in blocks {
            x = block.forward(x)
        }

        return finalNorm(x)
    }

    func forward(tokens: [Int]) throws -> [Float] {
        let tokenArray = MLXArray(tokens, [1, tokens.count])
        let encoded = forward(tokens: tokenArray)
        let flattened = encoded.reshaped([-1])
        return flattened.asArray(Float.self)
    }
}

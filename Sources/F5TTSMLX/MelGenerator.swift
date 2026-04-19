import Foundation
import MLX
import MLXNN

// Internal mel generator mapped to the flow-matching DiT mel generator in
// the Python f5-tts-mlx repository.
struct MelGenerator {
    struct Configuration {
        var textHiddenSize: Int
        var styleHiddenSize: Int
        var modelSize: Int
        var melBins: Int
        var depth: Int
        var headCount: Int
        var feedForwardMultiplier: Int
        var timeEmbeddingSize: Int
        var maxMelFrames: Int

        static let f5Approximation = Configuration(
            textHiddenSize: 1_024,
            styleHiddenSize: 512,
            modelSize: 1_024,
            melBins: 100,
            depth: 10,
            headCount: 16,
            feedForwardMultiplier: 4,
            timeEmbeddingSize: 256,
            maxMelFrames: 2_048
        )
    }

    private final class TransformerBlock {
        let norm1: LayerNorm
        let norm2: LayerNorm
        let attention: MultiHeadAttention
        let ffnIn: Linear
        let ffnOut: Linear

        init(modelSize: Int, headCount: Int, feedForwardSize: Int) {
            self.norm1 = LayerNorm(dimensions: modelSize)
            self.norm2 = LayerNorm(dimensions: modelSize)
            self.attention = MultiHeadAttention(dimensions: modelSize, numHeads: headCount)
            self.ffnIn = Linear(modelSize, feedForwardSize)
            self.ffnOut = Linear(feedForwardSize, modelSize)
        }

        func forward(_ x: MLXArray) -> MLXArray {
            let attended = attention(norm1(x), keys: norm1(x), values: norm1(x), mask: nil)
            let residual = x + attended
            let ffn = ffnOut(gelu(ffnIn(norm2(residual))))
            return residual + ffn
        }
    }

    private let config: Configuration
    private let latentProjection: Linear
    private let textProjection: Linear
    private let styleProjection: Linear
    private let timeIn: Linear
    private let timeOut: Linear
    private let blocks: [TransformerBlock]
    private let outputNorm: LayerNorm
    private let velocityHead: Linear

    init(configuration: Configuration = .f5Approximation) {
        precondition(
            configuration.modelSize % configuration.headCount == 0,
            "modelSize must be divisible by headCount"
        )

        self.config = configuration
        self.latentProjection = Linear(configuration.melBins, configuration.modelSize)
        self.textProjection = Linear(configuration.textHiddenSize, configuration.modelSize)
        self.styleProjection = Linear(configuration.styleHiddenSize, configuration.modelSize)
        self.timeIn = Linear(configuration.timeEmbeddingSize, configuration.modelSize)
        self.timeOut = Linear(configuration.modelSize, configuration.modelSize)
        self.blocks = (0..<configuration.depth).map { _ in
            TransformerBlock(
                modelSize: configuration.modelSize,
                headCount: configuration.headCount,
                feedForwardSize: configuration.modelSize * configuration.feedForwardMultiplier
            )
        }
        self.outputNorm = LayerNorm(dimensions: configuration.modelSize)
        self.velocityHead = Linear(configuration.modelSize, configuration.melBins)
    }

    // textFeatures: [batch, tokenCount, textHiddenSize]
    // styleEmbedding: [batch, styleHiddenSize]
    // durations: [batch, tokenCount]
    // returns mel spectrogram [batch, frameCount, melBins].
    func generateMels(
        textFeatures: MLXArray,
        styleEmbedding: MLXArray,
        durations: MLXArray,
        maxSteps: Int,
        temperature: Float,
        seed: UInt64? = nil
    ) -> MLXArray {
        precondition(textFeatures.ndim == 3, "textFeatures must have shape [batch, tokenCount, hidden]")
        precondition(styleEmbedding.ndim == 2, "styleEmbedding must have shape [batch, hidden]")
        precondition(maxSteps > 0, "maxSteps must be greater than zero")
        precondition(
            textFeatures.dim(-1) == config.textHiddenSize,
            "textFeatures hidden size must match textHiddenSize"
        )
        precondition(
            styleEmbedding.dim(-1) == config.styleHiddenSize,
            "styleEmbedding hidden size must match styleHiddenSize"
        )

        let batchSize = textFeatures.dim(0)
        let tokenCount = textFeatures.dim(1)
        let durations2D = normalizeDurations(
            durations,
            expectedBatch: batchSize,
            expectedTokens: tokenCount
        )

        let frameCount = estimateFrameCount(durations2D)
        let frameCondition = expandTextToFrames(
            textFeatures: textFeatures,
            durations: durations2D,
            frameCount: frameCount
        )

        let boundedTemperature = max(0.0, temperature)
        var latent = deterministicNoise(
            shape: [batchSize, frameCount, config.melBins],
            seed: seed
        ) * boundedTemperature

        let stepSize = MLXArray(1.0 / Float(maxSteps))

        for step in 0..<maxSteps {
            let tValue = 1.0 - (Float(step) / Float(maxSteps))
            let t = MLXArray(Array(repeating: tValue, count: batchSize))

            let velocity = predictVelocity(
                latent: latent,
                frameCondition: frameCondition,
                styleEmbedding: styleEmbedding,
                time: t
            )

            // Euler integration from high-noise state towards denoised mel.
            latent = latent - velocity * stepSize
        }

        return latent
    }

    func forward(
        textEmbedding: [Float],
        styleEmbedding: [Float]?,
        duration: Int,
        temperature: Float
    ) throws -> [Float] {
        let _ = (textEmbedding, styleEmbedding, duration, temperature)
        throw NSError(
            domain: "F5TTSMLX.MelGenerator",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Use generateMels(textFeatures:styleEmbedding:durations:maxSteps:temperature:seed:) with MLXArray inputs."]
        )
    }

    private func predictVelocity(
        latent: MLXArray,
        frameCondition: MLXArray,
        styleEmbedding: MLXArray,
        time: MLXArray
    ) -> MLXArray {
        var x = latentProjection(latent)
        x = x + textProjection(frameCondition)

        let style = styleProjection(styleEmbedding).expandedDimensions(axis: 1)
        x = x + style

        let timeEmbedding = timestepEmbedding(time: time, dimension: config.timeEmbeddingSize)
        let timeCondition = timeOut(gelu(timeIn(timeEmbedding))).expandedDimensions(axis: 1)
        x = x + timeCondition

        for block in blocks {
            x = block.forward(x)
        }

        return velocityHead(outputNorm(x))
    }

    private func normalizeDurations(
        _ durations: MLXArray,
        expectedBatch: Int,
        expectedTokens: Int
    ) -> MLXArray {
        let normalized: MLXArray
        if durations.ndim == 1 {
            precondition(durations.dim(0) == expectedTokens, "durations token count mismatch")
            normalized = durations.expandedDimensions(axis: 0)
        } else {
            precondition(durations.ndim == 2, "durations must have shape [batch, tokenCount] or [tokenCount]")
            normalized = durations
        }

        precondition(normalized.dim(0) == expectedBatch, "durations batch dimension mismatch")
        precondition(normalized.dim(1) == expectedTokens, "durations token count mismatch")

        return clip(normalized.asType(.float32), min: 1e-4)
    }

    private func estimateFrameCount(_ durations: MLXArray) -> Int {
        let perBatchFrames = durations.sum(axis: 1)
        let clipped = clip(perBatchFrames, min: 1.0, max: Float(config.maxMelFrames))
        let maxFrames = clipped.max().item(Float.self)
        return min(config.maxMelFrames, max(1, Int(Foundation.ceil(Double(maxFrames)))))
    }

    private func expandTextToFrames(
        textFeatures: MLXArray,
        durations: MLXArray,
        frameCount: Int
    ) -> MLXArray {
        let batchSize = textFeatures.dim(0)
        let tokenCount = textFeatures.dim(1)

        let sumDurations = durations.sum(axis: 1, keepDims: true)
        let durationFractions = durations / (sumDurations + 1e-6)

        let tokenEnds = cumsum(durationFractions, axis: 1)
        let tokenStarts = tokenEnds - durationFractions

        let framePositions = (MLXArray(0..<frameCount, [1, frameCount]).asType(.float32) + 0.5)
            / Float(frameCount)

        let starts = tokenStarts.expandedDimensions(axis: 1)
        let ends = tokenEnds.expandedDimensions(axis: 1)
        let centers = (starts + ends) * 0.5
        let widths = clip(ends - starts, min: 1e-4)

        let framePos = framePositions.expandedDimensions(axis: -1)
        let distance = (framePos - centers) / widths
        let weights = exp(-(distance * distance) * 8.0)

        let normalizedWeights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-6)
        let expanded = normalizedWeights.matmul(textFeatures)

        precondition(expanded.dim(0) == batchSize, "Expanded frame conditioning batch mismatch")
        precondition(expanded.dim(1) == frameCount, "Expanded frame conditioning frame mismatch")
        precondition(expanded.dim(2) == config.textHiddenSize, "Expanded frame conditioning hidden size mismatch")

        let _ = tokenCount
        return expanded
    }

    private func timestepEmbedding(time: MLXArray, dimension: Int) -> MLXArray {
        precondition(time.ndim == 1, "time must be 1D [batch]")

        let half = max(1, dimension / 2)
        let idx = MLXArray(0..<half).asType(.float32)
        let denom = Float(max(half - 1, 1))
        let freq = exp((idx / denom) * -Float(log(10_000.0)))

        let angles = time.expandedDimensions(axis: -1) * freq.expandedDimensions(axis: 0)
        var embedding = concatenated([sin(angles), cos(angles)], axis: -1)

        if dimension % 2 != 0 {
            let pad = MLXArray.zeros([time.dim(0), 1], type: Float.self)
            embedding = concatenated([embedding, pad], axis: -1)
        }

        return embedding
    }

    private func deterministicNoise(shape: [Int], seed: UInt64?) -> MLXArray {
        let count = max(1, shape.reduce(1, *))
        let base = MLXArray(0..<count).asType(.float32)

        let seedValue = seed ?? 0
        let phase = Float(seedValue % 65_521) / 65_521.0

        let first = sin(base * 0.017 + phase * 6.2831853)
        let second = cos(base * 0.031 + phase * 12.5663706)
        let noise = (first + second) * 0.70710677

        return noise.reshaped(shape)
    }
}

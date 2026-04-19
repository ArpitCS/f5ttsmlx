import Foundation
@preconcurrency import AVFoundation
import MLX

enum F5TTSRuntimeError: Error {
	case emptyInputText
	case audioLoadFailed(URL)
	case audioConversionFailed(URL)
}

/// Configuration for creating an ``F5TTS`` instance.
///
/// This type captures model location and high-level inference controls.
/// All properties are public so callers can build and inspect configuration
/// values before creating the synthesizer.
public struct F5TTSConfig: Sendable {
	/// Describes where model artifacts should be loaded from.
	public enum ModelSource: Sendable {
		/// Load model artifacts from a Hugging Face repository.
		///
		/// - Parameter repoId: Repository identifier, for example
		///   `owner/model-name`.
		case huggingFace(repoId: String)

		/// Load model artifacts from a local directory.
		///
		/// - Parameter url: Directory URL containing model files.
		case localDirectory(URL)
	}

	/// Source used to locate model artifacts.
	public let modelSource: ModelSource

	/// Maximum generation length for synthesis requests.
	public let maxLength: Int

	/// Sampling temperature used by the generation pipeline.
	public let temperature: Float

	/// Optional random seed for deterministic generation.
	public let seed: UInt64?

	/// Enables style transfer from reference audio when available.
	public let enableVoiceMatching: Bool

	/// Creates a configuration for ``F5TTS``.
	///
	/// - Parameters:
	///   - modelSource: Location of model artifacts.
	///   - maxLength: Maximum generation length.
	///   - temperature: Sampling temperature.
	///   - seed: Optional random seed.
	///   - enableVoiceMatching: Whether voice matching features are enabled.
	public init(
		modelSource: ModelSource,
		maxLength: Int,
		temperature: Float,
		seed: UInt64? = nil,
		enableVoiceMatching: Bool
	) {
		self.modelSource = modelSource
		self.maxLength = maxLength
		self.temperature = temperature
		self.seed = seed
		self.enableVoiceMatching = enableVoiceMatching
	}
}

/// In-memory audio output returned by synthesis APIs.
public struct AudioResult: Sendable {
	/// Sample rate in Hz.
	public let sampleRate: Int

	/// Mono PCM samples in normalized floating-point form.
	public let samples: [Float]

	/// Creates an ``AudioResult``.
	///
	/// - Parameters:
	///   - sampleRate: Sample rate in Hz.
	///   - samples: PCM sample buffer.
	public init(sampleRate: Int, samples: [Float]) {
		self.sampleRate = sampleRate
		self.samples = samples
	}
}

/// Top-level text-to-speech entry point for F5-TTS on MLX Swift.
///
/// This class currently exposes API stubs only. Runtime behavior will be
/// implemented in future iterations.
public final class F5TTS {
	/// Configuration captured at initialization time.
	public let config: F5TTSConfig
	private let modelDirectory: URL
	private let tokenizer: Tokenizer
	private let modelLoader: ModelLoader
	private let textEncoder: TextEncoder
	private let styleEncoder: StyleEncoder
	private let durationPredictor: DurationPredictor
	private let melGenerator: MelGenerator
	private let vocoder: Vocoder

	private static let modelSampleRate = 24_000
	private static let minReferenceSeconds: Double = 5.0
	private static let maxReferenceSeconds: Double = 10.0

	/// Creates and asynchronously prepares a synthesizer instance.
	///
	/// Use this initializer to validate configuration and load model state in
	/// future implementations.
	///
	/// - Parameter config: User-provided synthesizer configuration.
	public init(config: F5TTSConfig) async throws {
		self.config = config
		self.modelDirectory = try await resolveModelDirectory(from: config)

		let loader = try ModelLoader(rootURL: modelDirectory)
		self.modelLoader = loader
		self.tokenizer = try loader.makeTokenizer()

		self.textEncoder = try loader.buildTextEncoder()
		self.styleEncoder = try loader.buildStyleEncoder()
		self.durationPredictor = try loader.buildDurationPredictor(variant: .v2)
		self.melGenerator = try loader.buildMelGenerator()
		self.vocoder = try loader.buildVocoder()
	}

	// Initializes a tokenizer from a resolved model directory that contains
	// vocab.txt (for example, alandao/f5-tts-mlx-4bit after download).
	func makeTokenizer(resolvedModelDirectory: URL) throws -> Tokenizer {
		try Tokenizer(modelDirectory: resolvedModelDirectory)
	}

	/// Synthesizes speech audio for the provided text.
	///
	/// This is a stub and currently throws an unimplemented error.
	///
	/// - Parameters:
	///   - text: Input text to synthesize.
	///   - referenceAudioURL: Optional reference audio for voice matching.
	///   - referenceText: Optional transcript corresponding to reference audio.
	/// - Returns: Generated audio as an ``AudioResult``.
	public func synthesize(
		_ text: String,
		referenceAudioURL: URL? = nil,
		referenceText: String? = nil
	) async throws -> AudioResult {
		let _ = referenceText

		let tokenIDs = tokenizer.tokenize(text)
		guard !tokenIDs.isEmpty else {
			throw F5TTSRuntimeError.emptyInputText
		}

		let tokenLimit = max(1, config.maxLength)
		let cappedTokenIDs = Array(tokenIDs.prefix(tokenLimit))
		let tokenTensor = MLXArray(cappedTokenIDs, [1, cappedTokenIDs.count])

		let styleAudio = try preprocessReferenceAudio(url: referenceAudioURL)
		let styleEmbedding = styleEncoder.forward(audio: styleAudio)

		let textFeatures = textEncoder.forward(tokens: tokenTensor)
		let rawDurations = durationPredictor.forward(textFeatures: textFeatures)
		let durations = sanitizeDurations(rawDurations, tokenCount: cappedTokenIDs.count)

		let maxSteps = max(16, min(128, config.maxLength))
		let mels = melGenerator.generateMels(
			textFeatures: textFeatures,
			styleEmbedding: styleEmbedding,
			durations: durations,
			maxSteps: maxSteps,
			temperature: config.temperature,
			seed: config.seed
		)

		let waveform = vocoder.generateAudio(from: mels)
		let samples = waveform.reshaped([-1]).asArray(Float.self)

		return AudioResult(sampleRate: Self.modelSampleRate, samples: samples)
	}

	private func sanitizeDurations(_ raw: MLXArray, tokenCount: Int) -> MLXArray {
		let rawValues = raw.reshaped([-1]).asArray(Float.self)
		let clipped = Array(rawValues.prefix(tokenCount)).map { value -> Float in
			if value.isFinite {
				return min(Float(config.maxLength), max(1.0, value))
			}
			return 1.0
		}

		let total = clipped.reduce(Float(0), +)
		let maxFrames = Float(max(1, config.maxLength))
		let normalized: [Float]
		if total > maxFrames {
			let scale = maxFrames / total
			normalized = clipped.map { max(1.0, $0 * scale) }
		} else {
			normalized = clipped
		}

		return MLXArray(normalized, [1, tokenCount])
	}

	private func preprocessReferenceAudio(url: URL?) throws -> MLXArray {
		let minSamples = Int(Double(Self.modelSampleRate) * Self.minReferenceSeconds)
		let maxSamples = Int(Double(Self.modelSampleRate) * Self.maxReferenceSeconds)

		if let url {
			let mono24k = try loadResampledMonoAudio(url: url, sampleRate: Double(Self.modelSampleRate))
			let clamped = clampSampleCount(mono24k, minSamples: minSamples, maxSamples: maxSamples)
			return MLXArray(clamped, [1, clamped.count])
		}

		let silence = Array(repeating: Float(0), count: minSamples)
		return MLXArray(silence, [1, silence.count])
	}

	private func clampSampleCount(_ samples: [Float], minSamples: Int, maxSamples: Int) -> [Float] {
		if samples.count > maxSamples {
			return Array(samples.prefix(maxSamples))
		}

		if samples.count < minSamples {
			var padded = samples
			padded.append(contentsOf: Array(repeating: Float(0), count: minSamples - samples.count))
			return padded
		}

		return samples
	}

	private func loadResampledMonoAudio(url: URL, sampleRate: Double) throws -> [Float] {
		do {
			let file = try AVAudioFile(forReading: url)
			let sourceFormat = file.processingFormat

			guard let targetFormat = AVAudioFormat(
				commonFormat: .pcmFormatFloat32,
				sampleRate: sampleRate,
				channels: 1,
				interleaved: false
			) else {
				throw F5TTSRuntimeError.audioConversionFailed(url)
			}

			guard let sourceBuffer = AVAudioPCMBuffer(
				pcmFormat: sourceFormat,
				frameCapacity: AVAudioFrameCount(file.length)
			) else {
				throw F5TTSRuntimeError.audioLoadFailed(url)
			}

			try file.read(into: sourceBuffer)

			guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
				throw F5TTSRuntimeError.audioConversionFailed(url)
			}

			let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
			let estimatedFrames = max(1, Int(Double(sourceBuffer.frameLength) * ratio) + 1_024)
			guard let targetBuffer = AVAudioPCMBuffer(
				pcmFormat: targetFormat,
				frameCapacity: AVAudioFrameCount(estimatedFrames)
			) else {
				throw F5TTSRuntimeError.audioConversionFailed(url)
			}

			do {
				try converter.convert(to: targetBuffer, from: sourceBuffer)
			} catch {
				throw F5TTSRuntimeError.audioConversionFailed(url)
			}

			guard let channelData = targetBuffer.floatChannelData else {
				throw F5TTSRuntimeError.audioConversionFailed(url)
			}

			let count = Int(targetBuffer.frameLength)
			if count == 0 {
				return []
			}

			let mono = UnsafeBufferPointer(start: channelData[0], count: count)
			return Array(mono)
		} catch {
			if error is F5TTSRuntimeError {
				throw error
			}
			throw F5TTSRuntimeError.audioLoadFailed(url)
		}
	}
}

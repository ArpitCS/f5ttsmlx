import Foundation

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

	/// Creates and asynchronously prepares a synthesizer instance.
	///
	/// Use this initializer to validate configuration and load model state in
	/// future implementations.
	///
	/// - Parameter config: User-provided synthesizer configuration.
	public init(config: F5TTSConfig) async {
		self.config = config
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
		let _ = (text, referenceAudioURL, referenceText)
		throw NSError(
			domain: "F5TTSMLX",
			code: 1,
			userInfo: [NSLocalizedDescriptionKey: "synthesize(_:referenceAudioURL:referenceText:) is not implemented yet."]
		)
	}
}

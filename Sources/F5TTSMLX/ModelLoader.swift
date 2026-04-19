import Foundation
import MLX

// Internal model-loading scaffold for F5-TTS MLX weights.
// This loader currently validates model layout and parses safetensors payloads,
// but defers MLX parameter materialization to future work.

enum ModelLoaderError: Error {
    case missingModelDirectory(URL)
    case missingRequiredFile(URL)
    case unsupportedModelSource(String)
    case unsupportedTensorDType(String)
}

struct LoadedTensor: Sendable {
    let name: String
    let shape: [Int]
    let dtype: SafeTensorDType
    let data: Data
}

final class ModelLoader {
    static let modelFileName = "model.safetensors"
    static let durationModelFileName = "duration_model.safetensors"
    static let durationV2FileName = "duration_v2.safetensors"
    static let vocabularyFileName = "vocab.txt"

    private let rootURL: URL
    private let modelLoader: SafeTensorsLoader
    private let durationModelLoader: SafeTensorsLoader
    private let durationV2Loader: SafeTensorsLoader
    private let vocabularyURL: URL

    init(rootURL: URL) throws {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ModelLoaderError.missingModelDirectory(rootURL)
        }

        let modelURL = rootURL.appendingPathComponent(Self.modelFileName, isDirectory: false)
        let durationModelURL = rootURL.appendingPathComponent(Self.durationModelFileName, isDirectory: false)
        let durationV2URL = rootURL.appendingPathComponent(Self.durationV2FileName, isDirectory: false)
        let vocabURL = rootURL.appendingPathComponent(Self.vocabularyFileName, isDirectory: false)

        try Self.assertFileExists(at: modelURL)
        try Self.assertFileExists(at: durationModelURL)
        try Self.assertFileExists(at: durationV2URL)
        try Self.assertFileExists(at: vocabURL)

        self.rootURL = rootURL
        self.modelLoader = try SafeTensorsLoader(fileURL: modelURL)
        self.durationModelLoader = try SafeTensorsLoader(fileURL: durationModelURL)
        self.durationV2Loader = try SafeTensorsLoader(fileURL: durationV2URL)
        self.vocabularyURL = vocabURL
    }

    func buildTextEncoder() throws -> TextEncoder {
        let textEncoderTensors = try loadTensors(
            from: modelLoader,
            matchingPrefixes: [
                "transformer.text_embed",
                "text_embed",
                "text_encoder",
            ]
        )

        let preferredFloatDType = preferredFloatingDType(from: textEncoderTensors)
        let externalWeights = try loadMLXTensors(
            from: modelLoader,
            matchingPrefixes: [
                "transformer.text_embed",
                "text_embed",
                "text_encoder",
            ],
            preferredFloatDType: preferredFloatDType
        )

        // TODO: Map text-encoder tensor names from Python checkpoint to Swift
        // MLX parameter keys and instantiate model weights.
        let _ = textEncoderTensors

        return TextEncoder(parameterDType: preferredFloatDType, externalWeights: externalWeights)
    }

    func buildStyleEncoder() throws -> StyleEncoder {
        let styleEncoderTensors = try loadTensors(
            from: modelLoader,
            matchingPrefixes: [
                "reference_encoder",
                "style_encoder",
                "cond",
            ]
        )

        let preferredFloatDType = preferredFloatingDType(from: styleEncoderTensors)
        let externalWeights = try loadMLXTensors(
            from: modelLoader,
            matchingPrefixes: [
                "reference_encoder",
                "style_encoder",
                "cond",
            ],
            preferredFloatDType: preferredFloatDType
        )

        // TODO: Define and map reference/style encoder checkpoint keys to MLX
        // parameters once the Swift StyleEncoder architecture is finalized.
        let _ = styleEncoderTensors

        return StyleEncoder(parameterDType: preferredFloatDType, externalWeights: externalWeights)
    }

    func buildDurationPredictor(variant: DurationPredictor.Variant? = nil) throws -> DurationPredictor {
        // Prefer v2 by default but allow explicit variant override.
        let selectedVariant: DurationPredictor.Variant
        if let variant {
            selectedVariant = variant
        } else {
            selectedVariant = durationV2Loader.tensorNames.isEmpty ? .original : .v2
        }

        let chosenTensors: [LoadedTensor]
        let chosenLoader: SafeTensorsLoader
        switch selectedVariant {
        case .original:
            chosenTensors = try loadTensors(from: durationModelLoader, matchingPrefixes: [])
            chosenLoader = durationModelLoader
        case .v2:
            chosenTensors = try loadTensors(from: durationV2Loader, matchingPrefixes: [])
            chosenLoader = durationV2Loader
        }

        let preferredFloatDType = preferredFloatingDType(from: chosenTensors)
        let externalWeights = try loadMLXTensors(
            from: chosenLoader,
            matchingPrefixes: [],
            preferredFloatDType: preferredFloatDType
        )

        // TODO: Map duration predictor tensors to Swift DurationPredictor MLX
        // modules and assign weights.
        let _ = chosenTensors

        return DurationPredictor(
            variant: selectedVariant,
            parameterDType: preferredFloatDType,
            externalWeights: externalWeights
        )
    }

    func buildMelGenerator() throws -> MelGenerator {
        let melGeneratorTensors = try loadTensors(
            from: modelLoader,
            matchingPrefixes: [
                "transformer",
                "dit",
                "time_mlp",
                "to_pred",
            ]
        )

        let preferredFloatDType = preferredFloatingDType(from: melGeneratorTensors)
        let externalWeights = try loadMLXTensors(
            from: modelLoader,
            matchingPrefixes: [
                "transformer",
                "dit",
                "time_mlp",
                "to_pred",
            ],
            preferredFloatDType: preferredFloatDType
        )

        // TODO: Map DiT / flow-matching mel generator tensor names into Swift
        // MelGenerator MLX layers.
        let _ = melGeneratorTensors

        return MelGenerator(parameterDType: preferredFloatDType, externalWeights: externalWeights)
    }

    func buildVocoder() throws -> Vocoder {
        let vocoderTensors = try loadTensors(
            from: modelLoader,
            matchingPrefixes: [
                "vocoder",
                "vocos",
            ]
        )

        let preferredFloatDType = preferredFloatingDType(from: vocoderTensors)
        let externalWeights = try loadMLXTensors(
            from: modelLoader,
            matchingPrefixes: [
                "vocoder",
                "vocos",
            ],
            preferredFloatDType: preferredFloatDType
        )

        // TODO: If vocoder weights are colocated in this model directory, map
        // those keys to Swift Vocoder parameters. If not, fetch external Vocos
        // weights and populate from that source.
        let _ = vocoderTensors

        return Vocoder(sampleRate: 24_000, parameterDType: preferredFloatDType, externalWeights: externalWeights)
    }

    func makeTokenizer() throws -> Tokenizer {
        try Tokenizer(modelDirectory: rootURL)
    }

    var modelDirectoryURL: URL {
        rootURL
    }

    var vocabFileURL: URL {
        vocabularyURL
    }

    private func loadTensors(
        from loader: SafeTensorsLoader,
        matchingPrefixes prefixes: [String]
    ) throws -> [LoadedTensor] {
        let names = loader.tensorNames
        let filteredNames: [String]

        if prefixes.isEmpty {
            filteredNames = names
        } else {
            filteredNames = names.filter { name in
                prefixes.contains { prefix in name.hasPrefix(prefix) }
            }
        }

        return try filteredNames.map { name in
            let tensor = try loader.loadTensor(named: name)
            return LoadedTensor(name: name, shape: tensor.shape, dtype: tensor.dtype, data: tensor.data)
        }
    }

    private func loadMLXTensors(
        from loader: SafeTensorsLoader,
        matchingPrefixes prefixes: [String],
        preferredFloatDType: DType
    ) throws -> [String: MLXArray] {
        let loaded = try loadTensors(from: loader, matchingPrefixes: prefixes)
        var tensors: [String: MLXArray] = [:]
        tensors.reserveCapacity(loaded.count)

        for tensor in loaded {
            let mlxArray = try mlxArray(from: tensor, preferredFloatDType: preferredFloatDType)
            tensors[tensor.name] = mlxArray
        }

        return tensors
    }

    private func mlxArray(from tensor: LoadedTensor, preferredFloatDType: DType) throws -> MLXArray {
        let mlxDType: DType
        switch tensor.dtype {
        case .float16:
            mlxDType = .float16
        case .bfloat16:
            mlxDType = .bfloat16
        case .float32:
            mlxDType = .float32
        case .int8:
            mlxDType = .int8
        case .uint8:
            mlxDType = .uint8
        case .int16:
            mlxDType = .int16
        case .uint16:
            mlxDType = .uint16
        case .int32:
            mlxDType = .int32
        case .uint32:
            mlxDType = .uint32
        }

        let base = MLXArray(tensor.data, tensor.shape, dtype: mlxDType)
        if tensor.dtype.isFloatingPoint {
            return base.asType(preferredFloatDType)
        }
        return base
    }

    private func preferredFloatingDType(from tensors: [LoadedTensor]) -> DType {
        // Quantized checkpoints typically keep runtime weights in BF16.
        if tensors.contains(where: { $0.dtype == .bfloat16 }) {
            return .bfloat16
        }
        if tensors.contains(where: { $0.dtype == .float16 }) {
            return .float16
        }
        return .float32
    }

    private static func assertFileExists(at url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelLoaderError.missingRequiredFile(url)
        }
    }
}

func resolveModelDirectory(from config: F5TTSConfig) async throws -> URL {
    switch config.modelSource {
    case let .localDirectory(url):
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ModelLoaderError.missingModelDirectory(url)
        }
        return url

    case let .huggingFace(repoId):
        // TODO: Download or reuse cached Hugging Face model artifacts and return
        // the local directory path containing model.safetensors files + vocab.
        throw ModelLoaderError.unsupportedModelSource(
            "Hugging Face source not implemented yet for repo: \(repoId)"
        )
    }
}

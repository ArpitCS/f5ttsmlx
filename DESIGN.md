# F5TTSMLX Design

## Purpose

`F5TTSMLX` is a Swift re-implementation of F5-TTS on top of MLX Swift APIs.
The immediate product goal is runtime compatibility with the Python MLX project:

- `lucasnewman/f5-tts-mlx`
- 4-bit model packaging used by `alandao/f5-tts-mlx-4bit`

The target deployment is a 4-bit quantized inference path on Apple Silicon, while keeping the architecture modular enough to support additional precisions later.

## Compatibility Targets

This library is designed to be compatible with:

1. Python-side behavior and model layout from `lucasnewman/f5-tts-mlx`.
2. 4-bit checkpoint artifacts and vocabulary packaging from `alandao/f5-tts-mlx-4bit`.

Concretely, we align with:

- Quantized model loading semantics equivalent to Python CLI usage with `--q 4`.
- Expected artifact names in the Hugging Face repo: `model.safetensors`, `duration_model.safetensors`, `duration_v2.safetensors`, and `vocab.txt`.

## End-to-End Pipeline

The inference pipeline is organized as the following stages:

1. **Tokenizer**
- Converts input text to token IDs using the provided `vocab.txt` mapping.
- Must preserve token-index compatibility with upstream checkpoints.

2. **Text Encoder**
- Produces contextual text embeddings consumed by downstream duration and mel generation networks.
- Mirrors upstream F5-TTS text-conditioning behavior.

3. **Style Encoder (Reference Audio Encoder)**
- Encodes reference audio characteristics (speaker/prosody/style) into conditioning features.
- Supplies style information for zero-shot voice matching.

4. **Duration Predictor**
- Predicts or refines target temporal length/alignment between text and acoustic frames.
- Compatible with duration checkpoints (`duration_model.safetensors` / `duration_v2.safetensors`).

5. **Mel Generator (Flow-Matching DiT)**
- Core conditional flow-matching model using a DiT-style backbone.
- Generates mel spectrogram trajectories from conditioned noise to final mel output.

6. **Vocoder (Vocos)**
- Converts generated mel spectrograms to waveform audio.
- Output target: **24 kHz, mono** speech audio.

## High-Level Architecture

The library should be split into composable modules:

- `Tokenizer`: vocabulary loading, text normalization, tokenization.
- `Encoders`: text encoder and style/reference-audio encoder.
- `Duration`: duration predictor loading and inference.
- `Generator`: flow-matching DiT model + ODE sampling utilities.
- `Vocoder`: Vocos wrapper and waveform post-processing.
- `Pipeline`: orchestration layer that wires all modules and exposes a single TTS API.
- `ModelIO`: safetensors loading, quantized weight mapping, artifact validation.

This separation keeps model internals testable and allows independent replacement of components (for example, upgrading duration predictor versions without rewriting the full pipeline).

## Quantization Strategy

Primary support target is **4-bit quantized inference**.

Design expectations:

- Accept 4-bit model weights from compatible repos and map them to Swift/MLX parameter structures.
- Keep quantization handling isolated in `ModelIO` and weight-loading paths.
- Preserve a fallback path for non-quantized checkpoints where possible.

## Data and Artifact Contract

A compatible model bundle should include:

- `model.safetensors` (main mel generator / acoustic model weights)
- `duration_model.safetensors` (legacy/alternate duration predictor weights)
- `duration_v2.safetensors` (newer duration predictor weights)
- `vocab.txt` (token vocabulary)

At load time, the runtime should:

1. Validate required files.
2. Select preferred duration model variant (v2 when available).
3. Load vocabulary before model graph initialization to ensure embedding sizing consistency.

## Output Contract

Default synthesis output should be:

- Sample rate: **24,000 Hz**
- Channels: **1 (mono)**
- Format: PCM float internally, with optional export conversions handled by caller utilities.

## References

- Python MLX implementation: <https://github.com/lucasnewman/f5-tts-mlx>
- Quantized usage (`--q` supports 4 and 8; example includes `--q 4`):
  <https://github.com/lucasnewman/f5-tts-mlx#quantized-models>
- CLI docs (`--q` option):
  <https://github.com/lucasnewman/f5-tts-mlx/blob/main/f5_tts_mlx/README.md>
- 4-bit Hugging Face model files (`model.safetensors`, `duration_model.safetensors`, `duration_v2.safetensors`, `vocab.txt`):
  <https://huggingface.co/alandao/f5-tts-mlx-4bit/tree/main>

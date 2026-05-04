---
license: mit
tags:
  - music-source-separation
  - stem-separation
  - onnx
  - audio
  - hot-step
language:
  - en
---

# HOT-Step CPP SuperSep — ONNX Stem Separation Models

Pre-converted ONNX models for multi-stem audio separation in [HOT-Step CPP](https://github.com/scragnog/HOT-Step-CPP). These run natively via ONNX Runtime GPU — no Python required.

## Models

| File | Architecture | Size | Purpose |
|------|-------------|------|---------|
| `bs_roformer_sw.onnx` | BS-Roformer | 672 MB | **Stage 1**: Primary 6-stem split (vocals, drums, bass, guitar, piano, other) |
| `mel_band_roformer_karaoke.onnx` | Mel-Band RoFormer | 875 MB | **Stage 2**: Vocal sub-separation (lead vs backing) |
| `mdx23c_drumsep.onnx` | MDX23C | 418 MB | **Stage 3**: Drum sub-separation (kick, snare, toms, hi-hat, cymbals) |
| `htdemucs_6s.onnx` | HTDemucs | 105 MB | **Stage 4**: "Other" stem refinement |

**Total: ~2.07 GB**

## Usage

These models are designed for use with the HOT-Step CPP Model Manager. In the app:

1. Open the **Model Manager** (click "Get More Models" in the Models dropdown)
2. Go to the **Stem Separation** tab
3. Click **Download** on each model (or use the Stem Separation starter pack)

Models are downloaded to `models/supersep/` and loaded automatically by the SuperSep engine.

### Technical Details

- **Format**: ONNX (opset 18, legacy TorchScript export)
- **Precision**: FP32
- **Input**: Spectrogram representation (STFT performed in C++ engine)
- **Output**: Separation masks (iSTFT performed in C++ engine)
- **Runtime**: ONNX Runtime 1.25.1+ with CUDA Execution Provider

The models export only the neural network portion — STFT/iSTFT operations are handled natively in C++ for optimal performance.

## Conversion

These were converted from PyTorch checkpoints using the [MSS_ONNX_TensorRT](https://github.com/ZFTurbo/MSS_ONNX_TensorRT) toolset with `dynamo=False` (legacy TorchScript exporter) for compatibility with complex attention architectures.

## Attribution & Licenses

### Training & Checkpoints

- **BS-Roformer** checkpoint by [aufr33](https://github.com/jarredou/mss-oracle-list) — trained on the Music Source Separation framework
- **Mel-Band RoFormer Karaoke** checkpoint by [aufr33 & viperx](https://github.com/jarredou/mss-oracle-list) — SDR 10.1956 on karaoke separation
- **MDX23C DrumSep** checkpoint by [aufr33 & jarredou](https://github.com/jarredou/mss-oracle-list) — drum sub-component isolation
- **HTDemucs** by [Meta / Facebook AI Research](https://github.com/facebookresearch/demucs) — Hybrid Transformer architecture

### Frameworks & Tools

- **[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)** by ZFTurbo — training framework for BS-Roformer, Mel-Band RoFormer, and MDX23C architectures
- **[MSS_ONNX_TensorRT](https://github.com/ZFTurbo/MSS_ONNX_TensorRT)** by ZFTurbo — ONNX conversion tooling with STFT extraction and model validation
- **[Demucs](https://github.com/facebookresearch/demucs)** by Meta Research — HTDemucs architecture and pre-trained weights (MIT License)

### Architecture Papers

- **BS-Roformer**: "Music Source Separation with Band-Split RoFormer" ([arXiv:2309.02612](https://arxiv.org/abs/2309.02612))
- **Mel-Band RoFormer**: Mel-frequency variant of Band-Split RoFormer
- **MDX23C**: Based on TFC-TDF-UNet v3 architecture
- **HTDemucs**: "Hybrid Transformers for Music Source Separation" ([arXiv:2211.08553](https://arxiv.org/abs/2211.08553))

## License

The conversion and packaging is released under MIT. Individual model weights are subject to their original training licenses — see the attribution links above for details.

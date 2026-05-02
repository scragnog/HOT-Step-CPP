# SuperSep Model Conversion

One-time conversion of SuperSep `.ckpt` models to ONNX format for native C++ inference.

## Prerequisites

1. Python 3.10+ with PyTorch
2. Clone [ZFTurbo/MSS_ONNX_TensorRT](https://github.com/ZFTurbo/MSS_ONNX_TensorRT) for model architecture definitions
3. Your SuperSep model checkpoints (from HOT-Step 9000's `models/supersep/` directory)

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/ZFTurbo/MSS_ONNX_TensorRT.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/MSS_ONNX_TensorRT
```

## Convert

```bash
python convert_models.py \
    --model-dir D:/Ace-Step-Latest/hot-step-9000/models/supersep \
    --output-dir D:/Ace-Step-Latest/hot-step-cpp/models/supersep
```

## Models

| Stage | Model | Architecture | Output |
|-------|-------|-------------|--------|
| 1 | BS-Roformer-SW | Band-Split RoPE Transformer | `bs_roformer_sw.onnx` |
| 2 | Mel-Band RoFormer Karaoke | Mel-Band RoFormer | `mel_band_roformer_karaoke.onnx` |
| 3 | MDX23C DrumSep | TFC-TDF-net | `mdx23c_drumsep.onnx` |
| 4 | HTDemucs 6s | Hybrid Transformer Demucs v4 | `htdemucs_6s.onnx` |

## Output

The `.onnx` files go into `models/supersep/` and are loaded by the C++ engine's ONNX Runtime integration at runtime. STFT/iSTFT is handled natively in C++ — the ONNX models only contain the neural network core.

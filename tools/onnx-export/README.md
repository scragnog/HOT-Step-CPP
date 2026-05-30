# ONNX Export Tools

Scripts for exporting HOT-Step model components to ONNX format for use with
TensorRT or ONNX Runtime inference.

## Prerequisites

- Python 3.10+ with the hot-step-9000 venv
- PyTorch with CUDA support
- `diffusers`, `onnx`, `onnxruntime` (or `onnxruntime-gpu` for GPU/TRT)

## Scripts

### `export_vae.py` — VAE Decoder Export

Exports the `AutoencoderOobleck` VAE decoder to ONNX format. Only the decoder
half is exported (encoder is not needed for inference — we decode latents to
audio).

**Tensor spec:**
| Name | Shape | Description |
|------|-------|-------------|
| `latents` (input) | `[B, 64, T]` | Latent channels, latent frames @ 25Hz |
| `audio` (output) | `[B, 2, S]` | Stereo audio, S = T × 1920 @ 48kHz |

Dynamic axes: batch (dim 0) and temporal dims (latent_frames, samples).

**Usage:**
```powershell
# Basic export
& .venv\Scripts\python.exe tools\onnx-export\export_vae.py `
    --vae-path checkpoints\vae `
    --output models\onnx\vae_decoder.onnx

# Export with validation (compares ONNX vs PyTorch output)
& .venv\Scripts\python.exe tools\onnx-export\export_vae.py `
    --vae-path checkpoints\vae `
    --output models\onnx\vae_decoder.onnx `
    --validate
```

**Options:**
- `--vae-path` — Path to VAE checkpoint directory (config.json + safetensors)
- `--output` — Output path for the ONNX file
- `--opset` — ONNX opset version (default: 18)
- `--validate` — Compare ONNX output against PyTorch using onnxruntime

### `test_trt_vae.py` — TensorRT Validation & Benchmark

Benchmarks the exported ONNX model using CUDA EP vs TensorRT EP. Reports
latency, speedup ratio, and numerical accuracy.

**Usage:**
```powershell
& .venv\Scripts\python.exe tools\onnx-export\test_trt_vae.py `
    --onnx models\onnx\vae_decoder.onnx
```

**Options:**
- `--onnx` — Path to the exported ONNX file
- `--trt-cache` — Directory for TRT engine cache (default: `models/onnx/trt_cache/`)
- `--iterations` — Number of benchmark iterations (default: 20)

**Requirements for TRT EP:**
- `onnxruntime-gpu` (not `onnxruntime`)
- TensorRT libraries on PATH
- The script gracefully falls back if TRT EP is unavailable

## Output Files

| File | Size | Git-tracked? |
|------|------|-------------|
| `models/onnx/vae_decoder.onnx` | ~330 MB | ❌ No (gitignored) |
| `models/onnx/trt_cache/*.engine` | ~200 MB | ❌ No (gitignored) |

## Notes

- Export is always in fp32. TensorRT handles fp16 conversion during engine build.
- The VAE is a simple 1D convolutional network (no attention layers), so ONNX
  export is straightforward — no trace-safe patches needed.
- ScragVAE (674MB) can also be exported using the same script with a different
  `--vae-path`.

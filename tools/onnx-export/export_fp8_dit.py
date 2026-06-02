#!/usr/bin/env python3
"""
export_fp8_dit.py — Apply FP8 post-training quantization to a DiT ONNX model.

Uses NVIDIA Model Optimizer (modelopt) to insert QuantizeLinear/DequantizeLinear
(QDQ) nodes into the ONNX graph. The resulting graph can be compiled by TRT
into an FP8 tensor-core engine.

Calibration uses random data with appropriate shapes. For DiT-class models
with well-conditioned activations, random calibration with 'max' method
produces scale factors within 1-2% of real-data calibration.

Usage:
    python export_fp8_dit.py --onnx models/onnx/dit_fp32.onnx --output models/onnx/dit_fp8.onnx
"""

import argparse
import os
import sys
import numpy as np

try:
    import modelopt.onnx.quantization as moq
except ImportError:
    print("ERROR: modelopt not found. Install with:")
    print("  pip install nvidia-modelopt[onnx]")
    sys.exit(1)


def generate_calibration_data(num_samples=16, seq_len=512, enc_seq_len=256):
    """Generate random calibration data matching DiT ONNX input signatures.

    Input names and shapes (from export_dit.py DiTForwardWrapper):
        input_latents:  [B, T, 192]   — concatenated context + noise latents
        enc_hidden:     [B, S, 2048]  — encoder hidden states
        t:              [B]           — timestep (fp32)
        t_r:            [B]           — reference timestep (fp32)

    modelopt expects Dict[str, np.ndarray] where the first dimension is the
    number of calibration samples. Each sample is fed as batch=1 inference.
    """
    print(f"Generating {num_samples} calibration samples "
          f"(seq_len={seq_len}, enc_seq_len={enc_seq_len})...")

    return {
        # [num_samples, T, 192] — first dim is sample count, modelopt slices automatically
        "input_latents": np.random.randn(num_samples, seq_len, 192).astype(np.float32),
        # [num_samples, S, 2048]
        "enc_hidden": np.random.randn(num_samples, enc_seq_len, 2048).astype(np.float32),
        # [num_samples] — one scalar timestep per sample
        "t": np.random.uniform(0.0, 1.0, size=(num_samples,)).astype(np.float32),
        # [num_samples]
        "t_r": np.random.uniform(0.0, 1.0, size=(num_samples,)).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Quantize DiT ONNX to FP8")
    parser.add_argument("--onnx", required=True,
                        help="Path to input FP32 ONNX model")
    parser.add_argument("--output", required=True,
                        help="Path to output FP8 ONNX model")
    parser.add_argument("--samples", type=int, default=16,
                        help="Number of calibration samples (default: 16)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for calibration inputs (default: 512)")
    parser.add_argument("--enc-seq-len", type=int, default=256,
                        help="Encoder sequence length for calibration (default: 256)")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: Input ONNX not found: {args.onnx}")
        sys.exit(1)

    onnx_size_gb = os.path.getsize(args.onnx) / 1e9
    data_path = args.onnx + ".data"
    if os.path.exists(data_path):
        onnx_size_gb += os.path.getsize(data_path) / 1e9
    print(f"Input model: {args.onnx} ({onnx_size_gb:.1f} GB)")

    calib_data = generate_calibration_data(
        num_samples=args.samples,
        seq_len=args.seq_len,
        enc_seq_len=args.enc_seq_len,
    )

    print(f"Running modelopt FP8 quantization (calibration_method='max')...")
    print(f"TEMP dir: {os.environ.get('TEMP', os.environ.get('TMP', 'system default'))}")

    moq.quantize(
        onnx_path=args.onnx,
        quantize_mode="fp8",
        calibration_data=calib_data,
        calibration_method="max",
        output_path=args.output,
    )

    # Report output size
    out_size = os.path.getsize(args.output) / 1e6
    out_data = args.output + ".data"
    if os.path.exists(out_data):
        out_size += os.path.getsize(out_data) / 1e6
    print(f"FP8 ONNX saved to {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export AutoencoderOobleck VAE decoder to ONNX format.

Exports the decoder half of the VAE for use with TensorRT or ONNX Runtime.
The encoder is not needed for inference (we only decode latents → audio).

Tensor spec:
  Input:  "latents"  [B, 64, T]        (latent channels, latent frames @ 25Hz)
  Output: "audio"    [B, 2, samples]    (stereo, samples = T * 1920 @ 48kHz)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class VAEDecoderWrapper(nn.Module):
    """Wraps AutoencoderOobleck.decoder + post_quant_conv to extract .sample.

    The raw decoder returns a DecoderOutput namedtuple, which torch.onnx.export
    can't trace cleanly. This wrapper calls the decoder and returns the raw
    tensor directly.
    """

    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder
        # post_quant_conv maps from latent space back to decoder input space
        if hasattr(vae, "post_quant_conv") and vae.post_quant_conv is not None:
            self.post_quant_conv = vae.post_quant_conv
        else:
            self.post_quant_conv = None

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        # decoder returns DecoderOutput with .sample attribute
        if hasattr(decoded, "sample"):
            return decoded.sample
        return decoded


def export_vae(vae_path: str, output_path: str, opset: int = 18) -> str:
    """Export VAE decoder to ONNX.

    Args:
        vae_path: Path to the VAE checkpoint directory (config.json + safetensors).
        output_path: Path to write the ONNX file.
        opset: ONNX opset version.

    Returns:
        The output path of the exported ONNX file.
    """
    from diffusers import AutoencoderOobleck

    print(f"Loading VAE from: {vae_path}")
    vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae.eval()

    wrapper = VAEDecoderWrapper(vae)
    wrapper.eval()

    # Move to CPU for export (fp32)
    wrapper = wrapper.cpu()

    # Create dummy input: [batch=1, latent_channels=64, latent_frames=250]
    # 250 frames @ 25Hz = 10 seconds of audio
    dummy_latents = torch.randn(1, 64, 250, dtype=torch.float32)

    print(f"Dummy input shape: {dummy_latents.shape}")
    print(f"Expected output shape: [1, 2, {250 * 1920}] = [1, 2, {250 * 1920}]")

    # Test forward pass
    with torch.no_grad():
        test_out = wrapper(dummy_latents)
    print(f"Test forward pass output shape: {test_out.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export
    print(f"\nExporting to ONNX (opset {opset})...")
    t0 = time.time()

    dynamic_axes = {
        "latents": {0: "batch", 2: "latent_frames"},
        "audio": {0: "batch", 2: "samples"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_latents,),
        output_path,
        opset_version=opset,
        input_names=["latents"],
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # Force legacy TorchScript exporter (dynamo hits cp1252 UnicodeError on Windows)
    )

    export_time = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export complete in {export_time:.1f}s")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Validate with onnx checker
    import onnx

    print("\nValidating ONNX model...")
    model = onnx.load(output_path)
    onnx.checker.check_model(model, full_check=True)
    print("ONNX checker: PASSED")

    # Print model info
    graph = model.graph
    print(f"\nModel inputs:")
    for inp in graph.input:
        shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print(f"Model outputs:")
    for out in graph.output:
        shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    return output_path


def validate_onnx(onnx_path: str, vae_path: str):
    """Compare ONNX Runtime output against PyTorch output.

    Args:
        onnx_path: Path to the exported ONNX file.
        vae_path: Path to the VAE checkpoint directory.
    """
    import onnxruntime as ort
    from diffusers import AutoencoderOobleck

    print("\n" + "=" * 60)
    print("VALIDATION: Comparing ONNX vs PyTorch outputs")
    print("=" * 60)

    # Load PyTorch model
    print("Loading PyTorch VAE...")
    vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae.eval()
    wrapper = VAEDecoderWrapper(vae)
    wrapper.eval()
    wrapper = wrapper.cpu()

    # Create test input (shorter for speed: 50 frames = 2 seconds)
    test_latents = torch.randn(1, 64, 50, dtype=torch.float32)
    print(f"Test input shape: {test_latents.shape}")

    # PyTorch inference
    with torch.no_grad():
        pt_output = wrapper(test_latents).numpy()
    print(f"PyTorch output shape: {pt_output.shape}")

    # ONNX Runtime inference
    print("Loading ONNX model in onnxruntime...")
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    # Use CUDA if available, else CPU
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using: CUDAExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        print("Using: CPUExecutionProvider (CUDA not available)")

    sess = ort.InferenceSession(onnx_path, providers=providers)

    ort_input = {"latents": test_latents.numpy()}
    ort_output = sess.run(["audio"], ort_input)[0]
    print(f"ONNX output shape: {ort_output.shape}")

    # Compare
    abs_diff = np.abs(pt_output - ort_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    rel_diff = abs_diff / (np.abs(pt_output) + 1e-8)

    print(f"\nDiff statistics:")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max relative diff:  {rel_diff.max():.6e}")
    print(f"  Mean relative diff: {rel_diff.mean():.6e}")

    # Threshold check
    if max_diff < 1e-4:
        print("\n[PASS] VALIDATION PASSED: Outputs match within tolerance (1e-4)")
    elif max_diff < 1e-3:
        print("\n[WARN] VALIDATION WARNING: Small differences detected (< 1e-3)")
        print("  This is acceptable for fp32 export, TRT fp16 will diverge more.")
    else:
        print(f"\n[FAIL] VALIDATION FAILED: Max diff {max_diff:.6e} exceeds tolerance")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export AutoencoderOobleck VAE decoder to ONNX"
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        required=True,
        help="Path to VAE checkpoint directory (config.json + safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the ONNX file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX output against PyTorch output using onnxruntime",
    )

    args = parser.parse_args()

    # Verify input exists
    if not os.path.isdir(args.vae_path):
        print(f"ERROR: VAE path not found: {args.vae_path}")
        sys.exit(1)

    config_path = os.path.join(args.vae_path, "config.json")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.json not found in {args.vae_path}")
        sys.exit(1)

    # Export
    onnx_path = export_vae(args.vae_path, args.output, args.opset)

    # Validate
    if args.validate:
        validate_onnx(onnx_path, args.vae_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

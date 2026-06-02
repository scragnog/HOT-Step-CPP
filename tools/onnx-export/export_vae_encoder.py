#!/usr/bin/env python3
"""Export AutoencoderOobleck VAE encoder to ONNX format.

Exports the encoder half of the VAE for use with TensorRT or ONNX Runtime.
The encoder converts audio → latent space for timbre/cover VAE encoding.

Tensor spec:
  Input:  "audio"    [B, 2, samples]    (stereo, samples @ 48kHz)
  Output: "latents"  [B, 64, T]         (latent channels, latent frames @ 25Hz)

Note: The encoder output is 128ch (64 mean + 64 scale). We only need the mean
for deterministic encoding, so we slice to the first 64 channels.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class VAEEncoderWrapper(nn.Module):
    """Wraps AutoencoderOobleck.encoder + quant_conv to extract mean latents.

    The raw encoder returns 128ch (mean + scale). For deterministic encoding
    we only need the first 64 channels (mean). This wrapper handles the
    quant_conv and slicing.
    """

    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        # quant_conv maps from encoder output space to latent space
        if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
            self.quant_conv = vae.quant_conv
        else:
            self.quant_conv = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, 2, samples] stereo audio at 48kHz

        Returns:
            latents: [B, 64, T] mean latents (deterministic)
        """
        encoded = self.encoder(audio)
        # encoder returns EncoderOutput with .latent_dist or raw tensor
        if hasattr(encoded, "latent_dist"):
            h = encoded.latent_dist.mean
        elif hasattr(encoded, "sample"):
            h = encoded.sample
        else:
            h = encoded

        if self.quant_conv is not None:
            h = self.quant_conv(h)

        # h is [B, 128, T] — first 64 = mean, last 64 = log_var
        # Only return mean for deterministic encoding
        return h[:, :64, :]


def export_vae_encoder(vae_path: str, output_path: str, opset: int = 18) -> str:
    """Export VAE encoder to ONNX.

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

    wrapper = VAEEncoderWrapper(vae)
    wrapper.eval()

    # Move to CPU for export (fp32)
    wrapper = wrapper.cpu()

    # Create dummy input: [batch=1, channels=2, samples=480000]
    # 480000 samples @ 48kHz = 10 seconds of audio
    dummy_audio = torch.randn(1, 2, 480000, dtype=torch.float32)

    print(f"Dummy input shape: {dummy_audio.shape}")
    print(f"Expected output shape: [1, 64, {480000 // 1920}] = [1, 64, {480000 // 1920}]")

    # Test forward pass
    with torch.no_grad():
        test_out = wrapper(dummy_audio)
    print(f"Test forward pass output shape: {test_out.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Export
    print(f"\nExporting to ONNX (opset {opset})...")
    t0 = time.time()

    dynamic_axes = {
        "audio": {0: "batch", 2: "samples"},
        "latents": {0: "batch", 2: "latent_frames"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        output_path,
        opset_version=opset,
        input_names=["audio"],
        output_names=["latents"],
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
    wrapper = VAEEncoderWrapper(vae)
    wrapper.eval()
    wrapper = wrapper.cpu()

    # Create test input (shorter for speed: 96000 samples = 2 seconds)
    test_audio = torch.randn(1, 2, 96000, dtype=torch.float32)
    print(f"Test input shape: {test_audio.shape}")

    # PyTorch inference
    with torch.no_grad():
        pt_output = wrapper(test_audio).numpy()
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

    ort_input = {"audio": test_audio.numpy()}
    ort_output = sess.run(["latents"], ort_input)[0]
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
        description="Export AutoencoderOobleck VAE encoder to ONNX"
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
    onnx_path = export_vae_encoder(args.vae_path, args.output, args.opset)

    # Validate
    if args.validate:
        validate_onnx(onnx_path, args.vae_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

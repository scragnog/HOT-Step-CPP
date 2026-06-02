#!/usr/bin/env python3
"""Export PP-VAE (LeVo autoencoder_music_1320k.ckpt) to ONNX format.

Exports BOTH encoder and decoder as separate ONNX files for use with
TensorRT or ONNX Runtime in the HOT-Step-CPP engine.

PP-VAE is a different model from scragvae (same Oobleck architecture,
different weights). Source: tencent-ailab/SongGeneration autoencoder_music_1320k

Tensor specs:
  Encoder:
    Input:  "audio"    [B, 2, T_audio]   (stereo 48kHz)
    Output: "latents"  [B, 64, T_latent] (mean-only, deterministic)

  Decoder:
    Input:  "latents"  [B, 64, T_latent] (latent channels @ 25Hz)
    Output: "audio"    [B, 2, T_audio]   (stereo, T_audio = T_latent * 1920)

Usage:
  python export_pp_vae.py --ckpt path/to/autoencoder_music_1320k.ckpt --output-dir models/onnx/
  python export_pp_vae.py  # uses default paths
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Model Architecture (from LeVo / stable-audio) ──────────────────────────
# Matches the architecture in levo-vae/reencode.py exactly.

class Snake1d(nn.Module):
    """Snake activation: y = x + sin²(exp(α)·x) / exp(β)"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        a = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        b = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        return x + (torch.sin(a * x) ** 2) / (b + 1e-9)


def WNConv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResUnit(nn.Module):
    """Residual unit: snake → dilated conv(k=7) → snake → conv(k=1) → + skip"""
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            Snake1d(channels),
            WNConv1d(channels, channels, kernel_size=7, dilation=dilation,
                     padding=3 * dilation),
            Snake1d(channels),
            WNConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    """3× ResUnit → Snake → strided Conv1d (downsample)"""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        layers = []
        for dil in [1, 3, 9]:
            layers.append(ResUnit(in_ch, dil))
        layers.append(Snake1d(in_ch))
        layers.append(WNConv1d(in_ch, out_ch, kernel_size=stride * 2,
                               stride=stride, padding=stride // 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    """Snake → ConvTranspose1d (upsample) → 3× ResUnit"""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        layers = []
        layers.append(Snake1d(in_ch))
        layers.append(WNConvTranspose1d(in_ch, out_ch, kernel_size=stride * 2,
                                         stride=stride, padding=stride // 2))
        for dil in [1, 3, 9]:
            layers.append(ResUnit(out_ch, dil))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=128, c_mults=[1,2,4,8,16],
                 strides=[2,4,4,6,10], latent_dim=128, **kw):
        super().__init__()
        c_mults = [1] + c_mults
        layers = [WNConv1d(in_channels, channels * c_mults[0], kernel_size=7, padding=3)]
        for i, stride in enumerate(strides):
            layers.append(EncoderBlock(channels * c_mults[i], channels * c_mults[i + 1], stride))
        layers.append(Snake1d(channels * c_mults[-1]))
        layers.append(WNConv1d(channels * c_mults[-1], latent_dim, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=128, c_mults=[1,2,4,8,16],
                 strides=[2,4,4,6,10], latent_dim=64, **kw):
        super().__init__()
        c_mults = [1] + c_mults
        c_mults_rev = list(reversed(c_mults))
        strides_rev = list(reversed(strides))
        layers = [WNConv1d(latent_dim, channels * c_mults_rev[0], kernel_size=7, padding=3)]
        for i, stride in enumerate(strides_rev):
            layers.append(DecoderBlock(channels * c_mults_rev[i], channels * c_mults_rev[i + 1], stride))
        layers.append(Snake1d(channels * c_mults_rev[-1]))
        layers.append(WNConv1d(channels * c_mults_rev[-1], out_channels, kernel_size=7, padding=3, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ─── Encoder Wrapper (deterministic, mean-only) ─────────────────────────────

class PPVAEEncoderWrapper(nn.Module):
    """Wraps OobleckEncoder to output only the mean (first 64 of 128 channels).

    The encoder outputs 128 channels: [mean(64), logvar(64)].
    For deterministic encoding we only need the mean.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        h = self.encoder(audio)  # [B, 128, T_latent]
        mean = h[:, :64, :]     # [B, 64, T_latent]
        return mean


# ─── Export Functions ────────────────────────────────────────────────────────

def export_encoder(ckpt_path: str, output_path: str, opset: int = 18) -> str:
    """Export PP-VAE encoder to ONNX (deterministic, mean-only)."""
    print(f"\n{'='*60}")
    print(f"  Exporting PP-VAE Encoder")
    print(f"{'='*60}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    del ckpt

    enc_cfg = {"in_channels": 2, "channels": 128, "c_mults": [1,2,4,8,16],
               "strides": [2,4,4,6,10], "latent_dim": 128}

    encoder = OobleckEncoder(**enc_cfg)
    enc_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd)
    del enc_sd, sd

    wrapper = PPVAEEncoderWrapper(encoder)
    wrapper.eval()

    # Dummy: [1, 2, 10s * 48kHz]
    T_audio = 250 * 1920  # 250 latent frames = 10s
    dummy = torch.randn(1, 2, T_audio, dtype=torch.float32)
    print(f"Dummy input: {dummy.shape}")

    with torch.no_grad():
        test_out = wrapper(dummy)
    print(f"Test output: {test_out.shape} (expected [1, 64, 250])")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX (opset {opset})...")
    t0 = time.time()

    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["audio"],
        output_names=["latents"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "latents": {0: "batch", 2: "latent_frames"},
        },
        do_constant_folding=True,
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported in {elapsed:.1f}s: {output_path} ({size_mb:.1f} MB)")

    # Validate
    _validate_encoder(output_path, wrapper, dummy)

    del wrapper, encoder
    gc.collect()
    return output_path


def export_decoder(ckpt_path: str, output_path: str, opset: int = 18) -> str:
    """Export PP-VAE decoder to ONNX."""
    print(f"\n{'='*60}")
    print(f"  Exporting PP-VAE Decoder")
    print(f"{'='*60}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    del ckpt

    dec_cfg = {"out_channels": 2, "channels": 128, "c_mults": [1,2,4,8,16],
               "strides": [2,4,4,6,10], "latent_dim": 64}

    decoder = OobleckDecoder(**dec_cfg)
    dec_sd = {k.replace("decoder.", ""): v for k, v in sd.items() if k.startswith("decoder.")}
    decoder.load_state_dict(dec_sd)
    del dec_sd, sd

    decoder.eval()

    # Dummy: [1, 64, 250] = 10s of latents
    dummy = torch.randn(1, 64, 250, dtype=torch.float32)
    print(f"Dummy input: {dummy.shape}")

    with torch.no_grad():
        test_out = decoder(dummy)
    print(f"Test output: {test_out.shape} (expected [1, 2, {250 * 1920}])")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX (opset {opset})...")
    t0 = time.time()

    torch.onnx.export(
        decoder,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["latents"],
        output_names=["audio"],
        dynamic_axes={
            "latents": {0: "batch", 2: "latent_frames"},
            "audio": {0: "batch", 2: "samples"},
        },
        do_constant_folding=True,
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported in {elapsed:.1f}s: {output_path} ({size_mb:.1f} MB)")

    # Validate
    _validate_decoder(output_path, decoder, dummy)

    del decoder
    gc.collect()
    return output_path


# ─── Validation ──────────────────────────────────────────────────────────────

def _validate_encoder(onnx_path, pytorch_model, dummy_input):
    """Compare ONNX encoder output against PyTorch reference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not available, skipping validation")
        return

    print("\nValidating encoder ONNX vs PyTorch...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        ref = pytorch_model(dummy_input).numpy()

    ort_out = sess.run(None, {"audio": dummy_input.numpy()})[0]

    diff = np.abs(ref - ort_out)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  Shape match: {ref.shape == ort_out.shape}")

    if diff.max() < 0.01:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — large deviation!")


def _validate_decoder(onnx_path, pytorch_model, dummy_input):
    """Compare ONNX decoder output against PyTorch reference."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not available, skipping validation")
        return

    print("\nValidating decoder ONNX vs PyTorch...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    with torch.no_grad():
        ref = pytorch_model(dummy_input).numpy()

    ort_out = sess.run(None, {"latents": dummy_input.numpy()})[0]

    diff = np.abs(ref - ort_out)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  Shape match: {ref.shape == ort_out.shape}")

    if diff.max() < 0.01:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — large deviation!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    default_ckpt = r"D:\Ace-Step-Latest\levo-vae\autoencoder_music_1320k.ckpt"
    default_output_dir = r"D:\Ace-Step-Latest\hot-step-cpp\models\onnx"

    parser = argparse.ArgumentParser(description="Export PP-VAE to ONNX (encoder + decoder)")
    parser.add_argument("--ckpt", default=default_ckpt,
                        help=f"Path to autoencoder_music_1320k.ckpt (default: {default_ckpt})")
    parser.add_argument("--output-dir", default=default_output_dir,
                        help=f"Output directory for ONNX files (default: {default_output_dir})")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Export only the encoder")
    parser.add_argument("--decoder-only", action="store_true",
                        help="Export only the decoder")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    enc_path = os.path.join(args.output_dir, "pp-vae_encoder.onnx")
    dec_path = os.path.join(args.output_dir, "pp-vae_decoder.onnx")

    if not args.decoder_only:
        export_encoder(args.ckpt, enc_path, args.opset)

    if not args.encoder_only:
        export_decoder(args.ckpt, dec_path, args.opset)

    print(f"\n{'='*60}")
    print(f"  Done!")
    if not args.decoder_only:
        print(f"  Encoder: {enc_path}")
    if not args.encoder_only:
        print(f"  Decoder: {dec_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

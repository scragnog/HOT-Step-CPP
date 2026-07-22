#!/usr/bin/env python3
"""Export Stable Audio 3 SAME-L autoencoder (encoder + decoder) to ONNX.

Part of the SA3 post-processing refiner port. Exports fixed-size chunk graphs;
the C++ engine tiles with overlap-trim exactly like the Oobleck tiled decode.

Tensor specs (fp32, static shapes — TRT-friendly):
  Encoder: "audio"   [1, 2, 524288]  (stereo 44.1kHz, 128-latent chunk)
        -> "latents" [1, 256, 128]
  Decoder: "latents" [1, 256, 128]
        -> "audio"   [1, 2, 524288]

Stochastic decode paths (bottleneck noise_regularize, resampler mask_noise) are
zeroed for determinism — quality impact to be validated by listening test.

Runs in the StableAudio3 uv venv (NOT hot-step-9000):
  cd d:/Ace-Step-Latest/StableAudio3
  uv run --with onnx --with onnxruntime python \
      d:/Ace-Step-Latest/hot-step-cpp/tools/onnx-export/export_sa3_same.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")

# Force the plain/chunked-halo SDPA attention tiers — flash and flex_attention
# do not trace to ONNX.
import stable_audio_3.models.transformer as sat
sat.flash_attn_func = None
sat.flash_attn_kvpacked_func = None
sat.flex_attention_available = False
sat.flex_attention_compiled = None

from stable_audio_3.model_configs import ae_models
from stable_audio_3.loading_utils import load_autoencoder

CHUNK_LATENTS = 128
DOWNSAMPLING = 4096
CHUNK_SAMPLES = CHUNK_LATENTS * DOWNSAMPLING  # 524288


def zero_stochastic_paths(ae):
    ae.bottleneck.noise_regularize = False
    for m in ae.modules():
        if hasattr(m, "mask_noise"):
            m.mask_noise = 0


class EncoderWrapper(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, audio):
        return self.ae.encode(audio)


class DecoderWrapper(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, latents):
        return self.ae.decode(latents)


def parity(name, onnx_path, feed_name, feed, ref):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out = sess.run(None, {feed_name: feed.cpu().numpy()})[0]
    ref_np = ref.cpu().numpy()
    max_abs = np.abs(out - ref_np).max()
    denom = np.linalg.norm(out.ravel()) * np.linalg.norm(ref_np.ravel())
    cos = float(np.dot(out.ravel(), ref_np.ravel()) / denom) if denom > 0 else 0.0
    print(f"  [{name}] ORT-CPU {time.time()-t0:.1f}s  max_abs_diff={max_abs:.3e}  cosine={cos:.6f}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=r"d:/Ace-Step-Latest/hot-step-cpp/models/onnx/sa3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SAME-L...")
    cfg_path, ckpt_path = ae_models["same-l"].resolve()
    ae = load_autoencoder(cfg_path, ckpt_path, device=args.device).eval().requires_grad_(False)
    zero_stochastic_paths(ae)

    torch.manual_seed(0)
    audio = (torch.randn(1, 2, CHUNK_SAMPLES, device=args.device) * 0.1).clamp(-1, 1)

    enc = EncoderWrapper(ae)
    with torch.no_grad():
        ref_latents = enc(audio)
    print(f"Encoder reference: {tuple(ref_latents.shape)}")

    enc_path = os.path.join(args.output_dir, "sa3-same_encoder.onnx")
    torch.onnx.export(
        enc, (audio,), enc_path,
        input_names=["audio"], output_names=["latents"],
        opset_version=18, dynamo=False,
    )
    print(f"Exported {enc_path} ({os.path.getsize(enc_path)/1e9:.2f} GB)")

    dec = DecoderWrapper(ae)
    with torch.no_grad():
        ref_audio = dec(ref_latents)
    print(f"Decoder reference: {tuple(ref_audio.shape)}")

    dec_path = os.path.join(args.output_dir, "sa3-same_decoder.onnx")
    torch.onnx.export(
        dec, (ref_latents,), dec_path,
        input_names=["latents"], output_names=["audio"],
        opset_version=18, dynamo=False,
    )
    print(f"Exported {dec_path} ({os.path.getsize(dec_path)/1e9:.2f} GB)")

    print("Parity vs PyTorch (fp32):")
    c1 = parity("encoder", enc_path, "audio", audio, ref_latents)
    c2 = parity("decoder", dec_path, "latents", ref_latents, ref_audio)
    ok = c1 > 0.999 and c2 > 0.999
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

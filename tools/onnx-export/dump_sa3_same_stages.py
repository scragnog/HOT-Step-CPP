#!/usr/bin/env python3
"""Dump per-stage SAME-L autoencoder activations for GGML parity debugging.

Loads SAME-L exactly like export_sa3_same.py (noise paths zeroed), feeds the
SAME golden inputs used by sa3-ggml-test, and dumps the token sequence
(b, S, 1536) f32 before/after each TransformerBlock:

  <out>/enc_stage0.bin  = input to encoder transformers[0] (folded seq + new tokens)
  <out>/enc_stage<i>.bin = output of encoder transformers[i-1], i = 1..12
  <out>/enc_final.bin   = final latents (1, 256, 128)
  (same for dec_*)

Matches the C++ side: SA3_SAME_STAGE=<n> SA3_SAME_DUMP=<path> sa3-ggml-test
dumps the corresponding [dim, S] token-major tensor.

Run:
  cd d:/Ace-Step-Latest/StableAudio3
  uv run python d:/Ace-Step-Latest/hot-step-cpp/tools/onnx-export/dump_sa3_same_stages.py \
      --goldens <dir-with-same_enc.audio.bin> --out <dir>
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")

# Same attention-tier forcing as the ONNX export (goldens came from this path).
import stable_audio_3.models.transformer as sat
sat.flash_attn_func = None
sat.flash_attn_kvpacked_func = None
sat.flex_attention_available = False
sat.flex_attention_compiled = None

from stable_audio_3.model_configs import ae_models
from stable_audio_3.loading_utils import load_autoencoder


def zero_stochastic_paths(ae):
    ae.bottleneck.noise_regularize = False
    for m in ae.modules():
        if hasattr(m, "mask_noise"):
            m.mask_noise = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default="both", choices=["enc", "dec", "both"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg_path, ckpt_path = ae_models["same-l"].resolve()
    ae = load_autoencoder(cfg_path, ckpt_path, device=args.device).eval().requires_grad_(False)
    zero_stochastic_paths(ae)

    def dump(path, t):
        arr = t.detach().float().cpu().numpy()
        arr.tofile(path)
        print(f"  {os.path.basename(path)}  shape={tuple(arr.shape)}")

    def hook_block(block_module, prefix):
        stages = {}

        def pre_hook(mod, args_, kwargs):
            x = args_[0]
            if 0 not in stages:
                stages[0] = x.detach().clone()

        handles = [block_module.transformers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]

        for i, layer in enumerate(block_module.transformers):
            def post_hook(mod, args_, output, idx=i):
                if idx + 1 not in stages:
                    stages[idx + 1] = output.detach().clone()
            handles.append(layer.register_forward_hook(post_hook))
        return stages, handles

    if args.side in ("enc", "both"):
        audio = np.fromfile(os.path.join(args.goldens, "same_enc.audio.bin"), dtype=np.float32)
        audio = torch.from_numpy(audio.reshape(1, 2, -1)).to(args.device)
        block = ae.encoder.layers[0]
        stages, handles = hook_block(block, "enc")
        with torch.no_grad():
            latents = ae.encode(audio)
        for h in handles:
            h.remove()
        print("[enc]")
        for n, t in sorted(stages.items()):
            dump(os.path.join(args.out, f"enc_stage{n}.bin"), t)
        dump(os.path.join(args.out, "enc_final.bin"), latents)

    if args.side in ("dec", "both"):
        lat = np.fromfile(os.path.join(args.goldens, "same_dec.latents.bin"), dtype=np.float32)
        lat = torch.from_numpy(lat.reshape(1, 256, -1)).to(args.device)
        block = ae.decoder.layers[3]
        stages, handles = hook_block(block, "dec")
        with torch.no_grad():
            audio_out = ae.decode(lat)
        for h in handles:
            h.remove()
        print("[dec]")
        for n, t in sorted(stages.items()):
            dump(os.path.join(args.out, f"dec_stage{n}.bin"), t)
        dump(os.path.join(args.out, "dec_final.bin"), audio_out)


if __name__ == "__main__":
    main()

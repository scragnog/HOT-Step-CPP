#!/usr/bin/env python3
"""Export the Stable Audio 3 medium DiT (1.45B) to ONNX.

Part of the SA3 post-processing refiner port. Exports the single-forward core
(_forward): the model is 8-step distilled at cfg_scale=1.0, so there is no CFG
dual-pass — the sampler loop lives outside the graph (C++/numpy).

Tensor specs (fp32):
  "x"               [1, 256, T]   noised latents (T = latent frames, dynamic)
  "t"               [1]           current timestep in [0,1] (rf convention)
  "cross_attn_cond" [1, S, 768]   prompt tokens + seconds_total embed (S dynamic)
  "cross_attn_mask" [1, S]        bool
  "global_embed"    [1, 768]      seconds_total embed
  "local_add_cond"  [1, 257, T]   inpaint_mask (1ch) + inpaint_masked_input (256ch)
  "padding_mask"    [1, T]        bool, True = valid
  -> "v"            [1, 256, T]   rf_denoiser output

Also verifies whether the traced graph generalizes over T (dynamic_axes) by
running ORT at a different length; if that fails, the C++ side uses bucketed
static graphs + padding_mask instead.

Runs in the StableAudio3 uv venv:
  cd d:/Ace-Step-Latest/StableAudio3
  uv run --with onnx --with onnxruntime python \
      d:/Ace-Step-Latest/hot-step-cpp/tools/onnx-export/export_sa3_dit.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")

import stable_audio_3.models.transformer as sat
sat.flash_attn_func = None
sat.flash_attn_kvpacked_func = None
sat.flex_attention_available = False
sat.flex_attention_compiled = None

# aten::rms_norm has no ONNX symbolic in the TS exporter — decompose to
# primitive ops (identical math; pow/mean/rsqrt export cleanly and TRT likes them).
import torch.nn.functional as F_patch

def _rms_norm_decomposed(input, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    dims = tuple(range(-len(normalized_shape), 0))
    out = input * torch.rsqrt(input.pow(2).mean(dim=dims, keepdim=True) + eps)
    if weight is not None:
        out = out * weight
    return out

F_patch.rms_norm = _rms_norm_decomposed

from stable_audio_3.model import StableAudioModel

T_TRACE = 1024   # latent frames used for tracing (~97s of audio)
T_ALT = 640      # different length to probe dynamic-shape generalization
S_TRACE = 257    # 256 prompt tokens + 1 seconds_total token


class DiTCore(nn.Module):
    """Flattens the conditioning dict interface to plain tensors around _forward."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit  # DiffusionTransformer

    def forward(self, x, t, cross_attn_cond, cross_attn_mask, global_embed,
                local_add_cond, padding_mask):
        return self.dit._forward(
            x, t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            global_embed=global_embed,
            local_add_cond=local_add_cond,
            padding_mask=padding_mask,
        )


def make_inputs(T, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 256, T, generator=g).to(device)
    t = torch.tensor([0.3], dtype=torch.float32, device=device)
    cross = torch.randn(1, S_TRACE, 768, generator=g).to(device)
    cross_mask = torch.ones(1, S_TRACE, dtype=torch.bool, device=device)
    glob = torch.randn(1, 768, generator=g).to(device)
    local = torch.zeros(1, 257, T, device=device)
    pad = torch.ones(1, T, dtype=torch.bool, device=device)
    return (x, t, cross, cross_mask, glob, local, pad)


_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(bool)": np.bool_, "tensor(int64)": np.int64}
INPUT_NAMES = ["x", "t", "cross_attn_cond", "cross_attn_mask",
               "global_embed", "local_add_cond", "padding_mask"]


def run_ort(sess, inputs):
    # Feed by NAME: the exporter prunes graph-unused inputs (e.g. cross_attn_mask —
    # the model never forwards it; learned padding replaces masking), so positional
    # zipping misaligns.
    named = dict(zip(INPUT_NAMES, inputs))
    feed = {}
    for meta in sess.get_inputs():
        arr = named[meta.name].cpu().numpy()
        want = _ORT_DTYPES.get(meta.type)
        if want is not None and arr.dtype != want:
            arr = arr.astype(want)
        feed[meta.name] = arr
    return sess.run(None, feed)[0]


def compare(tag, out, ref):
    ref_np = ref.float().cpu().numpy()
    max_abs = np.abs(out - ref_np).max()
    denom = np.linalg.norm(out.ravel()) * np.linalg.norm(ref_np.ravel())
    cos = float(np.dot(out.ravel(), ref_np.ravel()) / denom) if denom > 0 else 0.0
    print(f"  [{tag}] max_abs_diff={max_abs:.3e}  cosine={cos:.6f}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=r"d:/Ace-Step-Latest/hot-step-cpp/models/onnx/sa3")
    ap.add_argument("--parity-only", action="store_true",
                    help="Skip export; run parity against an existing sa3-dit.onnx")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading stable-audio-3-medium (fp32)...")
    model = StableAudioModel.from_pretrained("medium", model_half=False)
    dit = model.model.model.model  # StableAudioModel -> CondWrapper -> DiTWrapper -> DiffusionTransformer
    dit.eval().requires_grad_(False)
    device = next(dit.parameters()).device
    core = DiTCore(dit)

    inputs = make_inputs(T_TRACE, device)
    with torch.no_grad():
        ref = core(*inputs)
    print(f"Reference out: {tuple(ref.shape)}")

    dit_path = os.path.join(args.output_dir, "sa3-dit.onnx")
    if args.parity_only:
        assert os.path.exists(dit_path), f"{dit_path} not found"
        print("(--parity-only: skipping export)")
        import onnxruntime as ort
        sess = ort.InferenceSession(dit_path, providers=["CPUExecutionProvider"])
        print("Parity at traced length:")
        out = run_ort(sess, inputs)
        c1 = compare(f"T={T_TRACE}", out, ref)
        print("Dynamic-shape probe at different length:")
        alt_inputs = make_inputs(T_ALT, device, seed=1)
        with torch.no_grad():
            alt_ref = core(*alt_inputs)
        try:
            alt_out = run_ort(sess, alt_inputs)
            c2 = compare(f"T={T_ALT}", alt_out, alt_ref)
            dynamic_ok = c2 > 0.999
        except Exception as e:
            print(f"  [T={T_ALT}] FAILED to run: {type(e).__name__}: {str(e)[:300]}")
            dynamic_ok = False
        print(f"PARITY {'OK' if c1 > 0.999 else 'FAILED'}; DYNAMIC-T {'OK' if dynamic_ok else 'NOT SUPPORTED -> use bucketed static graphs'}")
        return 0 if c1 > 0.999 else 1

    t0 = time.time()
    torch.onnx.export(
        core, inputs, dit_path,
        input_names=["x", "t", "cross_attn_cond", "cross_attn_mask",
                     "global_embed", "local_add_cond", "padding_mask"],
        output_names=["v"],
        dynamic_axes={
            "x": {2: "T"}, "local_add_cond": {2: "T"}, "padding_mask": {1: "T"},
            "cross_attn_cond": {1: "S"}, "cross_attn_mask": {1: "S"},
            "v": {2: "T"},
        },
        opset_version=18, dynamo=False,
    )
    total = sum(os.path.getsize(os.path.join(args.output_dir, f))
                for f in os.listdir(args.output_dir)
                if f.startswith("sa3-dit"))
    print(f"Exported {dit_path} ({total/1e9:.2f} GB incl. external data, {time.time()-t0:.0f}s)")

    import onnxruntime as ort
    sess = ort.InferenceSession(dit_path, providers=["CPUExecutionProvider"])

    print("Parity at traced length:")
    out = run_ort(sess, inputs)
    c1 = compare(f"T={T_TRACE}", out, ref)

    print("Dynamic-shape probe at different length:")
    alt_inputs = make_inputs(T_ALT, device, seed=1)
    with torch.no_grad():
        alt_ref = core(*alt_inputs)
    try:
        alt_out = run_ort(sess, alt_inputs)
        c2 = compare(f"T={T_ALT}", alt_out, alt_ref)
        dynamic_ok = c2 > 0.999
    except Exception as e:
        print(f"  [T={T_ALT}] FAILED to run: {type(e).__name__}: {str(e)[:300]}")
        dynamic_ok = False

    print(f"PARITY {'OK' if c1 > 0.999 else 'FAILED'}; DYNAMIC-T {'OK' if dynamic_ok else 'NOT SUPPORTED -> use bucketed static graphs'}")
    return 0 if c1 > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())

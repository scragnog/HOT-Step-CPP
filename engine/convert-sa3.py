#!/usr/bin/env python3
# convert-sa3.py: Stable Audio 3 checkpoints -> GGUF for the StableStep GGML backend.
#
# Produces four GGUFs into models/:
#   sa3-dit-BF16.gguf       arch "sa3-dit"       from stable-audio-3-medium (model.* keys)
#   sa3-same-enc-BF16.gguf  arch "sa3-same-enc"  from pretransform.model.* (encoder side)
#   sa3-same-dec-BF16.gguf  arch "sa3-same-dec"  from pretransform.model.* (decoder side)
#   sa3-text-enc-BF16.gguf  arch "sa3-t5gemma"   from the t5gemma-b-b-ul2 subfolder
#
# Tensor policy: >=2D weights -> BF16; 1D tensors (norms, biases, scales) -> F32
# (precision finding from the ONNX leg: this model's norm/timestep paths are
# fp32-sensitive — measured cosine 0.966 with blanket fp16 vs 0.9995 scoped).
# Tensor names are the source names minus the strip prefix; the C++ graph
# builders consume them as-is. The full model_config.json is embedded verbatim
# under metadata key "sa3.config_json" (the C++ side parses what it needs).
#
# Runs in the StableAudio3 uv venv:
#   cd d:/Ace-Step-Latest/StableAudio3
#   uv run --with gguf python d:/Ace-Step-Latest/hot-step-cpp/engine/convert-sa3.py

import json
import os
import sys

import numpy as np
import gguf

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")
from huggingface_hub import hf_hub_download
from safetensors import safe_open

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
REPO = "stabilityai/stable-audio-3-medium"


def log(msg):
    print(f"[convert-sa3] {msg}", file=sys.stderr, flush=True)


def to_np(t):
    import torch
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def write_sa3_gguf(out_path, arch, tensors, config_json, extra_meta=None):
    """tensors: list of (name, np.float32 array). >=2D stored BF16, 1D stored F32."""
    w = gguf.GGUFWriter(out_path, arch)
    w.add_string("sa3.config_json", config_json)
    for k, v in (extra_meta or {}).items():
        w.add_string(k, v)
    import torch
    n_bf16 = n_f32 = 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim >= 2:
            # raw_dtype does NOT convert — it labels. Convert to bf16 bytes
            # explicitly (uint16 view keeps the logical shape).
            bf16 = torch.from_numpy(arr).to(torch.bfloat16).view(torch.uint16).numpy()
            w.add_tensor(name, bf16, raw_dtype=gguf.GGMLQuantizationType.BF16)
            n_bf16 += 1
        else:
            w.add_tensor(name, arr)  # F32
            n_f32 += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size = os.path.getsize(out_path) / 1e9
    log(f"{os.path.basename(out_path)}: {n_bf16} BF16 + {n_f32} F32 tensors, {size:.2f} GB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg_path = hf_hub_download(REPO, "model_config.json")
    ckpt_path = hf_hub_download(REPO, "model.safetensors")
    with open(cfg_path) as f:
        config_json = f.read()

    # ── Split the combined checkpoint by prefix ─────────────────────────
    dit_tensors, enc_tensors, dec_tensors = [], [], []
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model."):
                dit_tensors.append((key[len("model."):], to_np(f.get_tensor(key))))
            elif key.startswith("pretransform.model."):
                sub = key[len("pretransform.model."):]
                # AudioAutoencoder members: encoder.*, decoder.*, bottleneck.*,
                # pretransform.* (patched — no weights). Bottleneck params go to BOTH
                # (encoder applies scale/bias+running_std, decoder inverts).
                if sub.startswith("encoder."):
                    enc_tensors.append((sub, to_np(f.get_tensor(key))))
                elif sub.startswith("decoder."):
                    dec_tensors.append((sub, to_np(f.get_tensor(key))))
                elif sub.startswith("bottleneck."):
                    t = to_np(f.get_tensor(key))
                    enc_tensors.append((sub, t))
                    dec_tensors.append((sub, t))
            # conditioner.* (learned padding, seconds embedder) rides with the DiT
            # gguf — small and needed by the same backend module.
            elif key.startswith("conditioner."):
                dit_tensors.append((key, to_np(f.get_tensor(key))))

    write_sa3_gguf(os.path.join(OUTPUT_DIR, "sa3-dit-BF16.gguf"),
                  "sa3-dit", dit_tensors, config_json)
    write_sa3_gguf(os.path.join(OUTPUT_DIR, "sa3-same-enc-BF16.gguf"),
                  "sa3-same-enc", enc_tensors, config_json)
    write_sa3_gguf(os.path.join(OUTPUT_DIR, "sa3-same-dec-BF16.gguf"),
                  "sa3-same-dec", dec_tensors, config_json)

    # ── T5Gemma encoder (separate HF model in the repo subfolder) ───────
    t5_cfg = hf_hub_download(REPO, "config.json", subfolder="t5gemma-b-b-ul2")
    t5_ckpt = hf_hub_download(REPO, "model.safetensors", subfolder="t5gemma-b-b-ul2")
    with open(t5_cfg) as f:
        t5_config_json = f.read()
    t5_tensors = []
    with safe_open(t5_ckpt, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Encoder-only: drop the decoder half (never used by SA3)
            if key.startswith("decoder."):
                continue
            t5_tensors.append((key, to_np(f.get_tensor(key))))
    # The SA3 conditioner's learned padding embedding is applied to the text
    # encoder's output (padded positions replaced) — it belongs to this module,
    # so duplicate it here (it also rides in the DiT gguf with the rest of
    # conditioner.*).
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        key = "conditioner.conditioners.prompt.padding_embedding"
        t5_tensors.append((key, to_np(f.get_tensor(key))))
    write_sa3_gguf(os.path.join(OUTPUT_DIR, "sa3-text-enc-BF16.gguf"),
                  "sa3-t5gemma", t5_tensors, t5_config_json,
                  extra_meta={"sa3.parent_config_json": config_json})

    log("Done.")


if __name__ == "__main__":
    main()

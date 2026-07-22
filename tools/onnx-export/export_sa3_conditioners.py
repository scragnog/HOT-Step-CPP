#!/usr/bin/env python3
"""Export the Stable Audio 3 conditioners to ONNX: T5Gemma text encoder + seconds_total embedder.

Part of the SA3 post-processing refiner port.

Tensor specs (fp32):
  Text encoder (tokenization stays outside — HF tokenizer.json, 256 max length,
  pad to max with learned-padding substitution baked into the graph):
    "input_ids"      [1, 256] int64
    "attention_mask" [1, 256] bool
    -> "embeddings"  [1, 256, 768]

  Seconds embedder (replaces hand-porting Fourier-feature math to C++):
    "seconds"     [1] float32   (clamped/normalized inside the graph, max 384s)
    -> "embed"    [1, 768]      (used both as global_embed and, unsqueezed, as
                                 the extra cross-attention token after the prompt)

Runs in the StableAudio3 uv venv:
  cd d:/Ace-Step-Latest/StableAudio3
  uv run --with onnx --with onnxruntime python \
      d:/Ace-Step-Latest/hot-step-cpp/tools/onnx-export/export_sa3_conditioners.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"d:/Ace-Step-Latest/StableAudio3")

from safetensors import safe_open
from stable_audio_3.model_configs import models
from stable_audio_3.factory import create_multi_conditioner_from_conditioning_config

# transformers v5 mask construction (vmap-based) doesn't trace to ONNX. The
# encoder is bidirectional with plain padding, so a broadcast bool keep-mask
# [B,1,Q,K] is equivalent — create_bidirectional_mask's contract accepts a
# prepared 4D mask. Patch at the t5gemma module level (direct name import).
import transformers.models.t5gemma.modeling_t5gemma as t5g_mod


def _trace_friendly_bidirectional_mask(config=None, inputs_embeds=None,
                                       attention_mask=None, **kwargs):
    if attention_mask is None:
        return None
    q = inputs_embeds.shape[1]
    return attention_mask.to(torch.bool)[:, None, None, :].expand(
        attention_mask.shape[0], 1, q, attention_mask.shape[-1]
    )


def _trace_friendly_sliding_window_mask(config=None, inputs_embeds=None,
                                        attention_mask=None, **kwargs):
    window = getattr(config, "sliding_window", None) or 4096
    q = inputs_embeds.shape[1]
    idx = torch.arange(q, device=inputs_embeds.device)
    band = (idx[None, :] - idx[:, None]).abs() < window
    mask = band[None, None, :, :]
    if attention_mask is not None:
        mask = mask & attention_mask.to(torch.bool)[:, None, None, :]
    return mask.expand(inputs_embeds.shape[0], 1, q, q)


t5g_mod.create_bidirectional_mask = _trace_friendly_bidirectional_mask
t5g_mod.create_bidirectional_sliding_window_mask = _trace_friendly_sliding_window_mask


class TextEncWrapper(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond

    def forward(self, input_ids, attention_mask):
        emb = self.cond.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        emb = self.cond.proj_out(emb)
        emb = self.cond.apply_padding(emb, attention_mask)
        return emb


class SecondsWrapper(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond

    def forward(self, seconds):
        x = seconds.clamp(self.cond.min_val, self.cond.max_val)
        x = (x - self.cond.min_val) / (self.cond.max_val - self.cond.min_val)
        return self.cond.embedder(x)


def parity(tag, path, feeds, ref):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, feeds)[0]
    ref_np = ref.float().cpu().numpy()
    max_abs = np.abs(out - ref_np).max()
    denom = np.linalg.norm(out.ravel()) * np.linalg.norm(ref_np.ravel())
    cos = float(np.dot(out.ravel(), ref_np.ravel()) / denom) if denom > 0 else 0.0
    print(f"  [{tag}] max_abs_diff={max_abs:.3e}  cosine={cos:.6f}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=r"d:/Ace-Step-Latest/hot-step-cpp/models/onnx/sa3")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg_path, ckpt_path = models["medium"].resolve()
    with open(cfg_path) as f:
        config = json.load(f)

    print("Building conditioners (T5Gemma from HF subfolder)...")
    conditioner = create_multi_conditioner_from_conditioning_config(
        config["model"]["conditioning"]
    )

    # Learned-padding embeddings etc. live in the main checkpoint under conditioner.*
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        cond_sd = {
            k[len("conditioner."):]: f.get_tensor(k)
            for k in f.keys() if k.startswith("conditioner.")
        }
    missing, unexpected = conditioner.load_state_dict(cond_sd, strict=False)
    print(f"  conditioner tensors loaded: {len(cond_sd)} (missing={len(missing)}, unexpected={len(unexpected)})")

    prompt_cond = conditioner.conditioners["prompt"]
    prompt_cond.model.float().eval().requires_grad_(False)
    prompt_cond.proj_out.float()
    seconds_cond = conditioner.conditioners["seconds_total"].float().eval().requires_grad_(False)

    # --- Text encoder ------------------------------------------------------
    text = "Instrumental punk rock with distorted electric guitars. BPM: 160. Length: 200 seconds."
    enc = prompt_cond.tokenizer(
        [text], truncation=True, max_length=prompt_cond.max_length,
        padding="max_length", return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].to(torch.bool)

    wrapper = TextEncWrapper(prompt_cond).eval()
    # Conditioner stores the HF model outside nn.Module registration (enable_grad
    # False path uses __dict__) — reattach for export.
    wrapper.cond.model.eval()
    with torch.no_grad():
        ref_emb = wrapper(input_ids, attention_mask)
    print(f"Text encoder reference: {tuple(ref_emb.shape)}")

    text_path = os.path.join(args.output_dir, "sa3-text_encoder.onnx")
    torch.onnx.export(
        wrapper, (input_ids, attention_mask), text_path,
        input_names=["input_ids", "attention_mask"], output_names=["embeddings"],
        opset_version=18, dynamo=False,
    )
    print(f"Exported {text_path}")
    c1 = parity("text-enc", text_path,
                {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
                ref_emb)

    # --- Seconds embedder --------------------------------------------------
    sw = SecondsWrapper(seconds_cond).eval()
    seconds = torch.tensor([203.8], dtype=torch.float32)
    with torch.no_grad():
        ref_sec = sw(seconds)
    sec_path = os.path.join(args.output_dir, "sa3-seconds_embedder.onnx")
    torch.onnx.export(
        sw, (seconds,), sec_path,
        input_names=["seconds"], output_names=["embed"],
        opset_version=18, dynamo=False,
    )
    print(f"Exported {sec_path} ({os.path.getsize(sec_path)/1e6:.1f} MB)")
    c2 = parity("seconds", sec_path, {"seconds": seconds.numpy()}, ref_sec)

    ok = c1 > 0.999 and c2 > 0.999
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

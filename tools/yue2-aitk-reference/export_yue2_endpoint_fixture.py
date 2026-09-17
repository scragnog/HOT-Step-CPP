"""Export deterministic endpoint references from the pinned YuE2 ConvRot checkpoint.

Only the three requested embedding rows are sliced from the large embedding
records. Other endpoint weights are loaded by safetensors and evaluated through
the Toolkit ConvRot operator and the source RMSNorm recipe.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

IDS = [0, 1, 152011]
FRAMES = 2

def write(path: Path, x: torch.Tensor):
    a = x.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    path.write_bytes(a.tobytes(order="C"))

def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_key(st, key: str) -> torch.Tensor:
    if key not in st.keys(): raise KeyError(key)
    return st.get_tensor(key)

def convrot(st, cq, x):
    q = load_key(st, cq + ".weight")
    s = load_key(st, cq + ".weight_scale").reshape(-1)
    bias = load_key(st, cq + ".bias").to(x.device).bfloat16()
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    y = convrot_quant._int8_linear_ste_op(convrot_quant.rotate(flat, 256), q, s.view(torch.uint8), bias, 127, "bfloat16")
    return y.reshape(*shape[:-1], q.shape[0])

def rms(x, weight):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * weight

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--toolkit", type=Path, default=Path(r"D:\Ace-Step-Latest\ai-toolkit"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = ap.parse_args()
    if a.output.exists(): raise SystemExit("output directory already exists")
    a.output.mkdir(parents=True)
    if a.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    global convrot_quant
    sys.path.insert(0, str(a.toolkit.resolve()))
    from toolkit.util import convrot_quant as _cq
    convrot_quant = _cq
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(str(a.checkpoint), framework="pt", device=str(dev)) as st:
        def rows(key, ids):
            sl = st.get_slice(key)
            return torch.cat([sl[i:i+1] for i in ids], dim=0)
        # ConvRot embedding is dequantized and rotated exactly as Toolkit CPU.
        eq = rows("text_encoders.model.embed_tokens.weight", IDS).to(dev)
        es = rows("text_encoders.model.embed_tokens.weight_scale", IDS).to(dev).float().view(-1, 1)
        embedding = convrot_quant.rotate(eq.float() * es, 256).to(torch.bfloat16)
        write(a.output / "embedding.f32", embedding)
        hidden = embedding.detach().clone().requires_grad_(True)
        dy = torch.tensor([((i % 17) - 8) * 0.007 for i in range(3 * 2048)], device=dev, dtype=torch.float32).bfloat16().view(3, 2048)
        ar_norm = load_key(st, "text_encoders.model.norm.weight").to(dev).bfloat16()
        ar_out = rms(hidden, ar_norm)
        (ar_out.float() * dy.float()).sum().backward()
        write(a.output / "ar_final.f32", ar_out); write(a.output / "ar_dx.f32", hidden.grad)

        latent = torch.tensor([((i % 13) - 6) * 0.03125 for i in range(FRAMES * 64)], device=dev, dtype=torch.float32).view(1, FRAMES, 64).bfloat16()
        padded = F.pad(latent, (0, 0, 1, 1))
        vae_w = load_key(st, "model.diffusion_model.vae2llm.weight").to(dev).bfloat16()
        vae_b = load_key(st, "model.diffusion_model.vae2llm.bias").to(dev).bfloat16()
        x = F.linear(padded, vae_w, vae_b).bfloat16()
        t = torch.tensor([0.375], device=dev, dtype=torch.float32).bfloat16()
        half = 128; freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=dev, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        time_input = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).bfloat16()
        time0 = convrot(st, "model.diffusion_model.time_embedder.mlp.0", time_input)
        time0 = F.silu(time0).bfloat16()
        time2 = convrot(st, "model.diffusion_model.time_embedder.mlp.2", time0)
        pos = load_key(st, "model.diffusion_model.latent_pos_embed.pe").to(dev).bfloat16()[:FRAMES + 2]
        frontend = (x + time2[:, None, :] + pos[None, :, :]).bfloat16()
        write(a.output / "nar_frontend.f32", frontend)
        nh = frontend.detach().clone().requires_grad_(True)
        nar_norm = load_key(st, "model.diffusion_model.model.norm.weight").to(dev).bfloat16()
        n = rms(nh, nar_norm)
        final = convrot(st, "model.diffusion_model.llm2vae", n)
        final = final.bfloat16()
        ndy = torch.tensor([((i % 11) - 5) * 0.005 for i in range((FRAMES + 2) * 64)], device=dev, dtype=torch.float32).bfloat16().view(1, FRAMES + 2, 64)
        (final.float() * ndy.float()).sum().backward()
        write(a.output / "nar_final.f32", final); write(a.output / "nar_dx.f32", nh.grad)
    schema = {"schema": 1, "ids": IDS, "frames": FRAMES, "device": str(dev), "source_sha256": {"model.py": source_hash(a.toolkit / "extensions_built_in/audio_models/yue2/src/model.py"), "convrot_quant.py": source_hash(a.toolkit / "toolkit/util/convrot_quant.py")}}
    (a.output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(a.output), "device": str(dev)}, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())


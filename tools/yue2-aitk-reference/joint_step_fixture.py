"""Development-only actual Toolkit YuE2 joint AR/NAR LoRA reference.

This script loads the pinned YuE2 model and LoRAModule, restores native-v1
F32 LoRA factors from the resume container, performs one or two synthetic
joint updates with bitsandbytes AdamW8bit, and exports metrics/factors. It is
an executable reference fixture, not the product trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

MAGIC = b"Y2RSUME1"
VERSION = 1
MAX_FILE = 512 * 1024 * 1024
MAX_NAME = 4096
MAX_METADATA = 16 * 1024 * 1024
ABC_END, MUSIC_START, MUSIC_END, CODEC_OFFSET = 151848, 151851, 151852, 151853


def _u32(raw: bytes, off: int) -> tuple[int, int]:
    if off + 4 > len(raw): raise ValueError("truncated resume u32")
    return struct.unpack_from("<I", raw, off)[0], off + 4


def _u64(raw: bytes, off: int) -> tuple[int, int]:
    if off + 8 > len(raw): raise ValueError("truncated resume u64")
    return struct.unpack_from("<Q", raw, off)[0], off + 8


def _fnv(raw: bytes, h: int = 1469598103934665603) -> int:
    for byte in raw:
        h ^= byte
        h = (h * 1099511628211) & 0xffffffffffffffff
    return h


def read_resume(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = path.read_bytes()
    if len(raw) > MAX_FILE or len(raw) < 52 or raw[:8] != MAGIC:
        raise ValueError("invalid native-v1 resume header")
    version, pos = _u32(raw, 8)
    payload_size, pos = _u64(raw, pos)
    metadata_size, pos = _u64(raw, pos)
    checksum, pos = _u64(raw, pos)
    metadata_prefix, pos = _u64(raw, pos)
    if version != VERSION or metadata_prefix != metadata_size or metadata_size > MAX_METADATA:
        raise ValueError("unsupported resume version or metadata bound")
    if metadata_size > len(raw) - pos:
        raise ValueError("resume metadata exceeds file")
    payload_offset = pos + metadata_size
    if payload_size != len(raw) - payload_offset:
        raise ValueError("resume payload length mismatch")
    metadata_bytes = raw[pos:payload_offset]
    payload = raw[payload_offset:]
    if _fnv(payload, _fnv(metadata_bytes)) != checksum:
        raise ValueError("resume checksum mismatch")
    p = 0
    step, p = _u32(payload, p)
    count, p = _u32(payload, p)
    if count == 0 or count > 65536:
        raise ValueError("invalid resume parameter count")
    arrays: dict[str, np.ndarray] = {}
    for _ in range(count):
        name_len, p = _u32(payload, p)
        if not name_len or name_len > MAX_NAME or p + name_len > len(payload):
            raise ValueError("invalid resume parameter name")
        name = payload[p:p + name_len].decode("utf-8")
        p += name_len
        n, p = _u64(payload, p)
        if not n or n > 0xffffffff or p >= len(payload):
            raise ValueError("invalid resume parameter shape")
        kind = payload[p]
        p += 1
        if name in arrays or kind > 1:
            raise ValueError("duplicate or invalid resume parameter")
        n = int(n)
        need = n * 4
        blocks = (n + 255) // 256
        if kind == 0:
            if n >= 4096: raise ValueError("FP32 resume state at or above 4096")
            need += n * 8
        else:
            if n < 4096: raise ValueError("uint8 resume state below 4096")
            need += n * 2 + blocks * 8
        if need > len(payload) - p: raise ValueError("resume slot exceeds payload")
        parameter = np.frombuffer(payload, dtype="<f4", count=n, offset=p).copy(); p += n * 4
        if not np.isfinite(parameter).all(): raise ValueError("nonfinite resume parameter")
        arrays[name] = parameter
        if kind == 0:
            m = np.frombuffer(payload, dtype="<f4", count=n, offset=p).copy(); p += n * 4
            v = np.frombuffer(payload, dtype="<f4", count=n, offset=p).copy(); p += n * 4
            if not np.isfinite(m).all() or not np.isfinite(v).all() or (v < 0).any(): raise ValueError("invalid FP32 optimizer state")
        else:
            p += n * 2 + blocks * 8
    if p != len(payload): raise ValueError("trailing resume payload")
    return arrays, {"step": step, "metadata": metadata_bytes.decode("utf-8")}


class NetworkStub:
    network_type = "lora"
    is_active = True
    is_merged_in = False
    is_lorm = False
    _multiplier = 1.0

    def __init__(self, device: torch.device):
        self.torch_multiplier = torch.ones(1, device=device)


def attach_lora(model, resume: dict[str, np.ndarray], toolkit: Path, device: torch.device):
    sys.path.insert(0, str(toolkit.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_yue2_block_fixture import load_toolkit_lora
    LoRAModule = load_toolkit_lora(toolkit / "toolkit/network_mixins.py", toolkit / "toolkit/lora_special.py")
    network = NetworkStub(device)
    wrappers = {}
    for expert_name, expert, prefix in (("nar", model.nar, "diffusion_model"), ("ar", model.ar, "text_encoders")):
        for layer_index, layer in enumerate(expert.model.layers):
            sites = (("self_attn.qkv_proj", layer.self_attn.qkv_proj), ("self_attn.o_proj", layer.self_attn.o_proj),
                     ("mlp.gate_up_proj", layer.mlp.gate_up_proj), ("mlp.down_proj", layer.mlp.down_proj))
            for site, module in sites:
                base = f"{prefix}.model.layers.{layer_index}.{site}.lora_"
                for suffix, attr in (("A.weight", "lora_down"), ("B.weight", "lora_up")):
                    name = base + suffix
                    if name not in resume: raise RuntimeError(f"resume missing expected LoRA site: {name}")
                wrapper = LoRAModule(base[:-1], module, lora_dim=32, alpha=32, network=network)
                wrapper.to(device=device, dtype=torch.float32)
                wrapper.apply_to()
                with torch.no_grad():
                    if resume[base + "A.weight"].size != wrapper.lora_down.weight.numel() or resume[base + "B.weight"].size != wrapper.lora_up.weight.numel():
                        raise RuntimeError(f"LoRA element-count mismatch at {base}: expected {wrapper.lora_down.weight.numel()}/{wrapper.lora_up.weight.numel()}")
                    a = torch.from_numpy(resume[base + "A.weight"]).reshape(wrapper.lora_down.weight.shape).to(device=device)
                    b = torch.from_numpy(resume[base + "B.weight"]).reshape(wrapper.lora_up.weight.shape).to(device=device)
                    wrapper.lora_down.weight.copy_(a); wrapper.lora_up.weight.copy_(b)
                wrapper.lora_down.weight.requires_grad_(True); wrapper.lora_up.weight.requires_grad_(True)
                wrappers[base] = wrapper
    expected = {f"{prefix}.model.layers.{i}.{site}.lora_{factor}.weight" for prefix in ("diffusion_model", "text_encoders") for i in range(len(model.nar.model.layers)) for site in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj") for factor in ("A", "B")}
    if set(resume) != expected: raise RuntimeError(f"unexpected resume sites: expected {len(expected)}, got {len(resume)}")
    params = [p for wrapper in wrappers.values() for p in (wrapper.lora_down.weight, wrapper.lora_up.weight)]
    if len(wrappers) != 224 or len(params) != 448 or any(p.dtype != torch.float32 or not p.requires_grad for p in params): raise RuntimeError("LoRA parameters are not trainable FP32")
    return network, wrappers, params


def joint_step(model, network, params, device, seed: int):
    torch.manual_seed(seed)
    prefix_ids = torch.tensor([[1, 2]], device=device, dtype=torch.long)
    prefix = model.ar.embed(prefix_ids)[0]
    tokens = torch.tensor([12, 42], device=device, dtype=torch.long)
    ids = torch.cat((torch.tensor([ABC_END, MUSIC_START], device=device), tokens + CODEC_OFFSET, torch.tensor([MUSIC_END], device=device)))
    embeds = torch.cat((prefix, model.ar.embed(ids)), dim=0)[None]
    _, adapted_hidden = model.ar.prefill(embeds, return_hidden=True)
    n = ids.shape[0]; adapted_hidden = adapted_hidden[0, -n - 1:-1]
    logits = model.ar.model.lm_head(adapted_hidden).float()
    ce = F.cross_entropy(logits, ids, reduction="mean")
    old_active = network.is_active; network.is_active = False
    try:
        with torch.no_grad():
            _, base_hidden = model.ar.prefill(embeds, return_hidden=True)
            base_logits = model.ar.model.lm_head(base_hidden[0, -n - 1:-1]).float()
    finally:
        network.is_active = old_active
    kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(base_logits, -1), log_target=True, reduction="batchmean")
    with torch.no_grad():
        cache, _ = model.ar.prefill(embeds)
        cache = [(k.detach(), v.detach()) for k, v in cache]
    clean = torch.tensor([float(int(i % 13) - 6) * .03125 for i in range(128)], device=device, dtype=torch.float32).reshape(1, 2, 64)
    noise = torch.tensor([float(int(i % 17) - 8) * .0625 for i in range(128)], device=device, dtype=torch.float32).reshape(1, 2, 64)
    noisy = .625 * clean + .375 * noise
    target = noise - clean
    pred = model.nar(noisy.to(model.nar.dtype), torch.tensor([.375], device=device, dtype=model.nar.dtype), cache, embeds.shape[1])
    mse = F.mse_loss(pred.float(), target)
    total = ce + .2 * kl + mse
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    return total, ce.detach(), kl.detach(), mse.detach(), float(norm.detach().cpu())


def export_step(wrappers, output: Path, step: int) -> None:
    exported = {}
    for base, wrapper in wrappers.items():
        exported[base + "A.weight"] = wrapper.lora_down.weight.detach().float().cpu()
        exported[base + "B.weight"] = wrapper.lora_up.weight.detach().float().cpu()
        exported[base + "A_grad.weight"] = wrapper.lora_down.weight.grad.detach().float().cpu()
        exported[base + "B_grad.weight"] = wrapper.lora_up.weight.grad.detach().float().cpu()
    save_file(exported, str(output / f"updated_step{step}.safetensors"), metadata={"format": "yue2-joint-reference-v1", "step": str(step)})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path(r"K:\yue2-bakeoff\ai-toolkit\models\checkpoints\yue2_3b_int8_convrot.safetensors"))
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--toolkit", type=Path, default=Path(r"D:\Ace-Step-Latest\ai-toolkit"))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--steps", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    if args.output.exists(): raise SystemExit("output directory must be new")
    args.output.mkdir(parents=True)
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    print("Validate native initial F32 state", flush=True)
    arrays, resume_meta = read_resume(args.resume)
    if resume_meta["step"] != 0: raise SystemExit("reference fixture requires a step-0 resume")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.toolkit.resolve()))
    from safetensors.torch import load_file
    from extensions_built_in.audio_models.yue2.src.model import YuE2Model
    print("Load actual ConvRot checkpoint", flush=True)
    model = YuE2Model.load_from_state_dict(load_file(str(args.checkpoint), device="cpu"), dtype=torch.bfloat16).to(device)
    model.enable_gradient_checkpointing()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    network, wrappers, params = attach_lora(model, arrays, args.toolkit, device)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(params, lr=1e-4, betas=(.9, .999), eps=1e-6, weight_decay=1e-4)
    metrics = []
    for step in range(args.steps):
        print(f"Joint reference update {step + 1}", flush=True)
        optimizer.zero_grad(set_to_none=False)
        total, ce, kl, mse, norm = joint_step(model, network, params, device, 20260917 + step)
        optimizer.step()
        export_step(wrappers, args.output, step + 1)
        metrics.append({"step": step + 1, "loss": float(total.detach().cpu()), "ar_ce": float(ce.cpu()), "ar_kl": float(kl.cpu()), "nar_mse": float(mse.cpu()), "gradient_norm": norm})
        print(json.dumps(metrics[-1]), flush=True)
        (args.output / "steps.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps({"resume": resume_meta, "steps": metrics, "checkpoint_sha256": sha256_file(args.checkpoint)}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

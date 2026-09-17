"""Export a bounded YuE2 ConvRot lm-head loss fixture.

The reference path is the pinned AI Toolkit ``convrot_quant`` custom op and
autograd.  It deliberately keeps logits chunk-local: the 184704-wide head is
evaluated for 128 rows and then one row, so the fixture never stores a full
song's logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROWS = 184704
WIDTH = 256
POSITIONS = 129
CHUNK = 128
ROTATION = 256
KL_WEIGHT = 0.2


def _write(path: Path, value: torch.Tensor, dtype=None) -> None:
    value = value.detach().contiguous().cpu()
    if dtype is not None:
        value = value.to(dtype)
    # NumPy has no bfloat16 dtype. Preserve the exact BF16 payload for the
    # portable gradient fixture instead of silently converting it to F32.
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    path.write_bytes(value.numpy().tobytes())


def _require_cuda_tensor(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cuda" or value.data_ptr() == 0 or not value.is_contiguous():
        raise RuntimeError(f"{name} was not allocated as a contiguous CUDA tensor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260917)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.toolkit / "toolkit/util/convrot_quant.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(args.toolkit.resolve()))
    from toolkit.util import convrot_quant  # pinned source custom op

    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the actual ConvRot custom-op fixture")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # These are deliberately raw ConvRot records: [out, in] I8 codes and one
    # FP32 scale per output row.  No dequantized 47-million-element F32 weight
    # is exported or retained here.
    weight = torch.randint(-127, 128, (ROWS, WIDTH), device=device,
                            dtype=torch.int8, generator=generator)
    scales = torch.rand(ROWS, device=device, dtype=torch.float32,
                        generator=generator).mul_(0.02).add_(1e-5)
    adapted = torch.randn(POSITIONS, WIDTH, device=device, dtype=torch.float32,
                          generator=generator).bfloat16().float()
    base = torch.randn(POSITIONS, WIDTH, device=device, dtype=torch.float32,
                       generator=generator).bfloat16().float()
    targets = torch.randint(0, ROWS, (POSITIONS,), device=device,
                            dtype=torch.int64, generator=generator)
    for name, value in (("weight", weight), ("scales", scales),
                        ("adapted_hidden", adapted), ("base_hidden", base),
                        ("targets", targets)):
        _require_cuda_tensor(name, value)
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise RuntimeError("invalid ConvRot scales")

    grad_f32 = torch.empty_like(adapted)
    ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    kl_sum = torch.zeros((), device=device, dtype=torch.float64)
    for offset in range(0, POSITIONS, CHUNK):
        end = min(offset + CHUNK, POSITIONS)
        # The model passes BF16 hidden states to ConvRot.  Keep the exported
        # inputs as F32 for the native seam, but run the pinned custom op with
        # the same BF16 storage and BF16 output arithmetic.
        ah = adapted[offset:end].detach().clone().bfloat16().requires_grad_(True)
        bh = base[offset:end].detach().bfloat16()
        q = weight
        s = scales.view(torch.uint8)
        # This is the installed training custom op, including its registered
        # STE backward.  Base logits are teacher targets and stay detached.
        al = convrot_quant._int8_linear_ste_op(
            convrot_quant.rotate(ah, ROTATION), q, s, None, 127, "bfloat16")
        with torch.no_grad():
            bl = convrot_quant._int8_linear_ste_op(
                convrot_quant.rotate(bh, ROTATION), q, s, None, 127, "bfloat16")
        log_al = F.log_softmax(al.float(), dim=-1)
        log_bl = F.log_softmax(bl.float(), dim=-1)
        ce = F.cross_entropy(al.float(), targets[offset:end], reduction="sum")
        # log_target=True makes this explicitly KL(base || adapted).
        kl = F.kl_div(log_al, log_bl, reduction="sum", log_target=True)
        ((ce + KL_WEIGHT * kl) / POSITIONS).backward()
        grad_f32[offset:end].copy_(ah.grad.float())
        ce_sum += ce.detach().double()
        kl_sum += kl.detach().double()
        del al, bl, ah, bh, log_al, log_bl, ce, kl
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    _write(args.output / "weight_i8.bin", weight)
    _write(args.output / "scales_f32.bin", scales)
    _write(args.output / "adapted_hidden_f32.bin", adapted)
    _write(args.output / "base_hidden_f32.bin", base)
    _write(args.output / "targets_u32.bin", targets, torch.int32)
    _write(args.output / "expected_hidden_grad_f32.bin", grad_f32)
    _write(args.output / "expected_hidden_grad_bf16.bin", grad_f32.bfloat16())
    values = [float(ce_sum.cpu()), float(kl_sum.cpu()),
              float(ce_sum.cpu() / POSITIONS), float(kl_sum.cpu() / POSITIONS)]
    (args.output / "reference_metrics.bin").write_bytes(struct.pack("<4d", *values))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps({
        "schema": 1, "source_sha256": digest, "reference": "pinned AI Toolkit convrot_quant.py custom op/autograd",
        "rows": ROWS, "width": WIDTH, "positions": POSITIONS, "chunk": CHUNK,
        "rotation": ROTATION, "kl_weight": KL_WEIGHT, "seed": args.seed,
        "target_storage": "uint32 little endian; values validated < rows",
        "hidden_storage": "float32 values rounded through BF16; reference op receives BF16",
        "cuda_arrays_allocated_before_export": True,
        "metrics": ["ce_sum", "kl_base_to_adapted_sum", "ce_mean", "kl_mean"],
        "scope": "bounded head-loss reference; no full-model parity claim",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "positions": POSITIONS,
                      "chunks": [CHUNK, 1], "ce_mean": values[2], "kl_mean": values[3]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

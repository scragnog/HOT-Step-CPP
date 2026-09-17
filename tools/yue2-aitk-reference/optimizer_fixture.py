#!/usr/bin/env python3
"""Export a small, reviewable AdamW8bit reference fixture.

``--inspect`` is deliberately dependency-free: it reports the contract read
from the pinned AI Toolkit and bitsandbytes sources.  ``--export`` needs the
AI Toolkit virtualenv (and NumPy) and constructs the real bitsandbytes
optimizer, but does not call ``step`` on CPU.  ``--execute`` is an explicit
CUDA-only operation for a later, guarded run.

This file does not implement an AdamW approximation.  In particular, a CPU
AdamW step must never be presented as evidence for bitsandbytes parity.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any


TOOLKIT_COMMIT = "e65c4d0fb69251e692390574c49873297dc4bae5"
DEFAULT_NUMEL = 4096
BLOCK_SIZE = 256


def contract_metadata() -> dict[str, Any]:
    """Return the resolved construction and state contract.

    The epsilon is an AI Toolkit override: ``toolkit.optimizer.get_optimizer``
    passes ``eps=1e-6`` even though bitsandbytes' constructor default is 1e-8.
    AdamW8bit hard-codes 8-bit state; its compatibility ``optim_bits`` argument
    remains 32 and must not be changed.  Tensors below ``min_8bit_size`` use
    float32 state, so the native implementation must make this per tensor.
    """
    return {
        "schema": "yue2-aitk-adamw8bit-fixture-v1",
        "reference": {
            "toolkit_commit": TOOLKIT_COMMIT,
            "optimizer_factory": "toolkit.optimizer.get_optimizer",
            "optimizer_type": "adamw8bit",
            "backend": "bitsandbytes.optim.AdamW8bit",
        },
        "construction": {
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-6,
            "weight_decay": 1.0e-4,
            "amsgrad": False,
            "optim_bits_argument": 32,
            "effective_state_bits": 8,
            "min_8bit_size": 4096,
            "percentile_clipping": 100,
            "block_wise": True,
            "block_size": BLOCK_SIZE,
            "max_unorm": 0.0,
            "skip_zeros": False,
            "is_paged": False,
            "trainable_parameter_dtype": "float32",
            "bf16_diagnostic_opt_in": "--bf16-diagnostic",
        },
        "state": {
            "first_update": ["step=0", "state1=zero", "state2=zero"],
            "large_tensor": [
                "state1:uint8", "state2:uint8", "qmap1:dynamic",
                "qmap2:udynamic", "absmax1:float32", "absmax2:float32",
            ],
            "small_tensor": ["state1:float32", "state2:float32"],
            "blockwise": "ceil(numel / 256) independent absmax values per state",
            "step": "incremented before each update; starts at zero",
        },
        "limitations": [
            "CPU inspection and fixture export do not execute optimizer.step.",
            "A CPU AdamW update is not a bitsandbytes equivalence test.",
            "8-bit codebooks and CUDA update kernels are owned by bitsandbytes.",
        ],
    }


def deterministic_fixture(numel: int, seed: int, dtype: Any) -> tuple[Any, Any]:
    """Create deterministic parameter/gradient tensors (no optimizer step)."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameter = torch.randn(numel, generator=generator, dtype=torch.float32).to(dtype)
    gradient = torch.randn(numel, generator=generator, dtype=torch.float32).to(dtype)
    return parameter, gradient


def _sha256_array(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_to_numpy(value: Any) -> Any:
    if not hasattr(value, "detach"):
        return value
    # NumPy has no portable BF16 dtype. Preserve the source dtype in metadata
    # and store the exact values widened to FP32 for interchange.
    value = value.detach().cpu()
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return value.numpy().copy()


def _state_arrays(prefix: str, state: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    """Copy every tensor state value and scalar step into portable arrays."""
    manifest: dict[str, Any] = {}
    for key, value in state.items():
        name = f"{prefix}_{key}"
        if hasattr(value, "detach"):
            arrays[name] = _tensor_to_numpy(value)
            manifest[key] = {"array": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        else:
            manifest[key] = {"scalar": value}
    return manifest


def export_fixture(output_dir: Path, numel: int, seed: int, steps: int, execute: bool, force: bool,
                   bf16_diagnostic: bool, toolkit_source: Path | None) -> None:
    """Export params/grads plus optimizer construction and optional CUDA state."""
    try:
        import numpy as np
        import torch
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "export requires the AI Toolkit environment with torch, numpy and bitsandbytes"
        ) from exc

    if numel < 4096:
        raise ValueError("--numel must be at least 4096; the fixture always includes a 4095-element tensor")
    if steps < 0:
        raise ValueError("--steps must be non-negative")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty fixture directory: {output_dir}")
    dtype = torch.bfloat16 if bf16_diagnostic else torch.float32
    parameter_small, gradient_small = deterministic_fixture(4095, seed, dtype)
    parameter_large, gradient_large = deterministic_fixture(numel, seed + 1, dtype)
    if execute and not torch.cuda.is_available():
        raise RuntimeError("--execute requires a CUDA device; no GPU work was attempted")
    if execute:
        parameter_small, gradient_small = parameter_small.cuda(), gradient_small.cuda()
        parameter_large, gradient_large = parameter_large.cuda(), gradient_large.cuda()

    parameter_small = torch.nn.Parameter(parameter_small)
    parameter_large = torch.nn.Parameter(parameter_large)
    optimizer = bnb.optim.AdamW8bit(
        [parameter_small, parameter_large], lr=1.0e-4, betas=(0.9, 0.999), eps=1.0e-6,
        weight_decay=1.0e-4, min_8bit_size=4096,
        percentile_clipping=100, block_wise=True, is_paged=False,
    )
    arrays = {
        "parameter_small_initial": _tensor_to_numpy(parameter_small),
        "parameter_large_initial": _tensor_to_numpy(parameter_large),
        "gradient_small_step_0": _tensor_to_numpy(gradient_small),
        "gradient_large_step_0": _tensor_to_numpy(gradient_large),
    }
    metadata = contract_metadata()
    import inspect
    source_files = {
        "toolkit_optimizer": toolkit_source,
        "bnb_adamw": Path(inspect.getfile(bnb.optim.AdamW8bit)),
        "bnb_optimizer": Path(inspect.getfile(bnb.optim.AdamW8bit.__mro__[1])),
    }
    metadata["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "bitsandbytes": getattr(bnb, "__version__", "unknown"),
        "source_files": {name: {"path": str(path) if path else None, "sha256": _sha256_file(path) if path else None} for name, path in source_files.items()},
    }
    metadata["fixture"] = {
        "seed": seed, "numel_large": numel, "numel_small": 4095,
        "bf16_diagnostic": bf16_diagnostic,
        "parameter_dtype": str(parameter_large.dtype),
        "device": "cuda" if execute else "cpu", "steps_requested": steps,
        "parameter_sha256": _sha256_array(arrays["parameter_large_initial"]),
        "gradient_sha256": _sha256_array(arrays["gradient_large_step_0"]),
        "optimizer_instantiated": True,
        "optimizer_executed": execute,
    }
    if execute:
        state_snapshots = []
        generator = torch.Generator(device=parameter_large.device).manual_seed(seed + 2)
        for step in range(steps):
            grad_s = torch.randn(parameter_small.shape, generator=generator, device=parameter_small.device, dtype=torch.float32).to(dtype)
            grad_l = torch.randn(parameter_large.shape, generator=generator, device=parameter_large.device, dtype=torch.float32).to(dtype)
            arrays[f"gradient_small_step_{step}"] = _tensor_to_numpy(grad_s)
            arrays[f"gradient_large_step_{step}"] = _tensor_to_numpy(grad_l)
            parameter_small.grad, parameter_large.grad = grad_s, grad_l
            optimizer.step()
            state_snapshots.append({
                "step": step + 1,
                "small": _state_arrays(f"step_{step + 1}_small", optimizer.state[parameter_small], arrays),
                "large": _state_arrays(f"step_{step + 1}_large", optimizer.state[parameter_large], arrays),
            })
            arrays[f"parameter_small_after_step_{step + 1}"] = _tensor_to_numpy(parameter_small)
            arrays[f"parameter_large_after_step_{step + 1}"] = _tensor_to_numpy(parameter_large)
        metadata["state_snapshots"] = state_snapshots
        metadata["fixture"]["gradient_sha256"] = _sha256_array(arrays["gradient_large_step_0"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "optimizer_tensors.npz", **arrays)
    (output_dir / "optimizer_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="print the dependency-free contract JSON")
    parser.add_argument("--export", dest="output_dir", type=Path, help="write portable NPZ tensors and JSON metadata")
    parser.add_argument("--numel", type=int, default=DEFAULT_NUMEL, help="large tensor size; small boundary tensor is always 4095")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--steps", type=int, default=1, help="CUDA update steps to execute")
    parser.add_argument("--bf16-diagnostic", action="store_true", help="use BF16 parameters diagnostically; reference training defaults to FP32")
    parser.add_argument("--toolkit-source", type=Path, help="path to pinned toolkit/optimizer.py for identity hashing")
    parser.add_argument("--force", action="store_true", help="allow writing into a non-empty output directory")
    parser.add_argument("--execute", action="store_true", help="run one real bnb update (CUDA only)")
    args = parser.parse_args()
    if args.numel <= 0:
        parser.error("--numel must be positive")
    if args.execute and not args.output_dir:
        parser.error("--execute requires --export DIR")
    if args.inspect or not args.output_dir:
        print(json.dumps(contract_metadata(), indent=2))
    if args.output_dir:
        export_fixture(args.output_dir, args.numel, args.seed, args.steps, args.execute, args.force,
                       args.bf16_diagnostic, args.toolkit_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

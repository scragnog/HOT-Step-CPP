#!/usr/bin/env python3
"""Bounded AI Toolkit YuE2 adapter layout inspection and interchange.

This module deliberately handles only the two fused linears instantiated by the
AI Toolkit YuE2 model: ``qkv_proj`` and ``gate_up_proj``.  It does not guess at
foreign LoRA namespaces or expand rank when importing a split adapter.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional for header-only inspection
    torch = None


EXPERTS = ("diffusion_model", "text_encoders")
FUSED_SITES = {"qkv_proj": (2048, 1024, 1024), "gate_up_proj": (6144, 6144)}
PAIR_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class Pair:
    expert: str
    module: str
    a: TensorInfo
    b: TensorInfo

    @property
    def site(self) -> str:
        return self.module.rsplit(".", 1)[-1]


@dataclass
class Inventory:
    path: Path
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    pairs: list[Pair]
    errors: list[str]

    @property
    def experts(self) -> dict[str, int]:
        return {e: sum(1 for p in self.pairs if p.expert == e) for e in EXPERTS}


class LayoutError(ValueError):
    pass


def _read_header(path: Path) -> tuple[dict[str, Any], bytes, int]:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise LayoutError(f"{path}: truncated safetensors header length")
        n = struct.unpack("<Q", raw)[0]
        if n > file_size - 8:
            raise LayoutError(f"{path}: header length {n} exceeds file size")
        header = f.read(n)
        if len(header) != n:
            raise LayoutError(f"{path}: truncated safetensors header")
        try:
            obj = json.loads(header)
        except json.JSONDecodeError as exc:
            raise LayoutError(f"{path}: invalid safetensors JSON header: {exc}") from exc
        if not isinstance(obj, dict):
            raise LayoutError(f"{path}: safetensors header is not an object")
        return obj, header, 8 + n


def read_inventory(path: str | Path) -> Inventory:
    path = Path(path)
    header, _, data_offset = _read_header(path)
    data_size = path.stat().st_size - data_offset
    tensors: dict[str, TensorInfo] = {}
    errors: list[str] = []
    dtype_sizes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4,
                   "I32": 4, "U64": 8, "I64": 8, "F8_E4M3": 1, "F16": 2,
                   "BF16": 2, "F32": 4, "F64": 8}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict) or not isinstance(item.get("shape"), list):
            errors.append(f"{name}: malformed tensor header")
            continue
        try:
            shape = tuple(int(x) for x in item["shape"])
            if any(x < 0 for x in shape):
                raise ValueError("negative tensor dimension")
            offsets = tuple(int(x) for x in item["data_offsets"])
            if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
                raise ValueError("invalid data_offsets")
            if offsets[1] > data_size:
                raise ValueError("data_offsets exceed file")
            if str(item["dtype"]) not in dtype_sizes:
                raise ValueError(f"unsupported dtype {item['dtype']}")
            elements = 1
            for dimension in shape:
                elements *= dimension
            if offsets[1] - offsets[0] != elements * dtype_sizes[str(item["dtype"])]:
                raise ValueError("data_offsets byte size does not match shape/dtype")
            tensors[name] = TensorInfo(name, shape, str(item["dtype"]), offsets)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{name}: malformed tensor header ({exc})")

    ranges = sorted((t.data_offsets[0], t.data_offsets[1], name) for name, t in tensors.items())
    for (_, previous_end, previous_name), (current_start, _, current_name) in zip(ranges, ranges[1:]):
        if current_start < previous_end:
            errors.append(f"overlapping data_offsets: {previous_name} and {current_name}")

    pairs: list[Pair] = []
    for name, a in tensors.items():
        if not name.endswith(PAIR_SUFFIXES[0]):
            continue
        bname = name[: -len(PAIR_SUFFIXES[0])] + PAIR_SUFFIXES[1]
        b = tensors.get(bname)
        if b is None:
            errors.append(f"{name}: missing paired B tensor ({bname})")
            continue
        module = name[: -len(PAIR_SUFFIXES[0])]
        expert = module.split(".", 1)[0]
        if expert in EXPERTS:
            pairs.append(Pair(expert, module, a, b))
        else:
            errors.append(f"foreign adapter namespace: {module}")
    for name, b in tensors.items():
        if not name.endswith(PAIR_SUFFIXES[1]):
            continue
        aname = name[: -len(PAIR_SUFFIXES[1])] + PAIR_SUFFIXES[0]
        if aname not in tensors:
            errors.append(f"{name}: missing paired A tensor ({aname})")
    for pair in pairs:
        if len(pair.a.shape) != 2 or len(pair.b.shape) != 2:
            errors.append(f"{pair.module}: LoRA factors must be rank-2")
        elif pair.a.shape[0] != pair.b.shape[1] or pair.a.shape[0] <= 0 or pair.a.shape[1] <= 0:
            errors.append(f"{pair.module}: A/B rank or feature dimensions disagree")
        if pair.a.dtype != pair.b.dtype:
            errors.append(f"{pair.module}: A/B dtype mismatch ({pair.a.dtype} vs {pair.b.dtype})")
    return Inventory(path, dict(header.get("__metadata__", {})), tensors, pairs, errors)


def validate_inventory(inv: Inventory, *, require_both_experts: bool = True) -> list[str]:
    errors = list(inv.errors)
    for expert in EXPERTS:
        count = inv.experts[expert]
        if require_both_experts and count == 0:
            errors.append(f"missing expert: {expert}")
        for site in FUSED_SITES:
            if require_both_experts and not any(p.expert == expert and p.site == site for p in inv.pairs):
                errors.append(f"missing fused {site} pairs for expert: {expert}")
    for pair in inv.pairs:
        site = pair.site
        if site not in FUSED_SITES:
            continue
        expected = FUSED_SITES[site]
        if len(pair.a.shape) != 2 or len(pair.b.shape) != 2:
            continue
        rank, in_features = pair.a.shape
        out_features, b_rank = pair.b.shape
        if b_rank != rank or in_features != 2048 or out_features != sum(expected):
            errors.append(
                f"{pair.module}: fused factors have A={pair.a.shape}, B={pair.b.shape}; "
                f"expected A=[rank,{in_features}], B=[{sum(expected)},rank]"
            )
        if pair.a.dtype != pair.b.dtype:
            errors.append(f"{pair.module}: A/B dtype mismatch ({pair.a.dtype} vs {pair.b.dtype})")
    return errors


def validate(path: str | Path, *, require_both_experts: bool = True) -> Inventory:
    inv = read_inventory(path)
    inv.errors = validate_inventory(inv, require_both_experts=require_both_experts)
    return inv


def _cat(parts: Iterable[Any], dim: int) -> Any:
    parts = tuple(parts)
    if torch is not None and all(hasattr(x, "shape") for x in parts):
        return torch.cat(parts, dim=dim)
    raise LayoutError("tensor conversion requires torch tensors (CPU is sufficient)")


def fused_to_split(tensors: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split fused qkv/gate_up B factors, retaining one shared A per module.

    The returned metadata is copied so alpha/config fields survive interchange.
    No rank expansion or independent A factors are created.
    """
    out = dict(tensors)
    for key, b in list(tensors.items()):
        if not key.endswith(".lora_B.weight"):
            continue
        module = key[: -len(".lora_B.weight")]
        site = module.rsplit(".", 1)[-1]
        if site not in FUSED_SITES:
            continue
        akey = module + ".lora_A.weight"
        if akey not in tensors:
            raise LayoutError(f"{module}: missing shared A factor")
        cuts = FUSED_SITES[site]
        if len(getattr(b, "shape", ())) != 2 or int(b.shape[0]) != sum(cuts):
            raise LayoutError(f"{module}: B shape {getattr(b, 'shape', None)} is not fused shape")
        del out[key]
        start = 0
        labels = ("q", "k", "v") if site == "qkv_proj" else ("gate", "up")
        for label, size in zip(labels, cuts):
            out[f"{module}.{label}.lora_B.weight"] = b[start : start + size, :]
            out[f"{module}.{label}.lora_A.weight"] = tensors[akey]
            start += size
        del out[akey]
    result_metadata = dict(metadata or {})
    result_metadata["yue2_adapter_layout"] = "split_shared_a_v1"
    return out, result_metadata


def _same_tensor(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if torch is not None and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    return a == b


def split_to_fused(tensors: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fuse split q/k/v and gate/up factors, refusing unequal A factors."""
    out = dict(tensors)
    for prefix, labels in (("qkv_proj", ("q", "k", "v")), ("gate_up_proj", ("gate", "up"))):
        marker = f".{prefix}."
        modules = sorted({k.split(marker, 1)[0] for k in tensors if marker in k})
        for module in modules:
            a_keys = [f"{module}.{prefix}.{x}.lora_A.weight" for x in labels]
            b_keys = [f"{module}.{prefix}.{x}.lora_B.weight" for x in labels]
            if not all(k in tensors for k in a_keys + b_keys):
                raise LayoutError(f"{module}.{prefix}: incomplete split factor set")
            a0 = tensors[a_keys[0]]
            if len(getattr(a0, "shape", ())) != 2:
                raise LayoutError(f"{module}.{prefix}: A factors must be rank-2")
            rank, in_features = map(int, a0.shape)
            if rank <= 0 or in_features != 2048:
                raise LayoutError(f"{module}.{prefix}: invalid A shape {tuple(a0.shape)}")
            for ak, bk in zip(a_keys, b_keys):
                a, b = tensors[ak], tensors[bk]
                if len(getattr(b, "shape", ())) != 2 or tuple(map(int, b.shape)) != (FUSED_SITES[prefix][labels.index(ak.split(".")[-3])], rank):
                    raise LayoutError(f"{module}.{prefix}: invalid B shape for {ak}")
                if getattr(a, "dtype", None) != getattr(a0, "dtype", None) or getattr(b, "dtype", None) != getattr(a0, "dtype", None):
                    raise LayoutError(f"{module}.{prefix}: split factor dtypes differ")
            if not all(_same_tensor(tensors[a_keys[0]], tensors[k]) for k in a_keys[1:]):
                raise LayoutError(f"{module}.{prefix}: split A factors differ; refusing rank expansion")
            fused_a = f"{module}.{prefix}.lora_A.weight"
            fused_b = f"{module}.{prefix}.lora_B.weight"
            if (fused_a in out and fused_a not in a_keys) or (fused_b in out and fused_b not in b_keys):
                raise LayoutError(f"{module}.{prefix}: fused output key already exists")
            out[fused_a] = tensors[a_keys[0]]
            out[fused_b] = _cat((tensors[k] for k in b_keys), 0)
            for k in a_keys + b_keys:
                del out[k]
    return out, dict(metadata or {})


def _report(inv: Inventory) -> dict[str, Any]:
    return {"path": str(inv.path), "tensors": len(inv.tensors), "pairs": len(inv.pairs), "experts": inv.experts,
            "fused": {site: sum(1 for p in inv.pairs if p.site == site) for site in FUSED_SITES}, "errors": inv.errors,
            "metadata": inv.metadata}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--export-split", type=Path, help="reserved for torch-enabled safe export")
    args = ap.parse_args(argv)
    inv = validate(args.checkpoint)
    print(json.dumps(_report(inv), indent=2, sort_keys=True))
    if inv.errors:
        return 2
    if args.export_split:
        if args.export_split.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.export_split}")
        if torch is None:
            raise SystemExit("--export-split requires torch and safetensors; no file was written")
        try:
            from safetensors.torch import load_file, save_file  # type: ignore
        except ImportError as exc:
            raise SystemExit("--export-split requires torch and safetensors; no file was written") from exc
        values = {k: v.cpu() for k, v in load_file(str(args.checkpoint), device="cpu").items()}
        split, metadata = fused_to_split(values, metadata=inv.metadata)
        # safetensors refuses duplicate storage.  Keep the mathematical shared-A
        # contract in metadata, while writing independent contiguous file ranges.
        split = {k: (v.detach().cpu().clone().contiguous() if isinstance(v, torch.Tensor) else v)
                 for k, v in split.items()}
        args.export_split.parent.mkdir(parents=True, exist_ok=True)
        save_file(split, str(args.export_split), metadata={k: str(v) for k, v in metadata.items()})
        print(f"wrote {args.export_split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

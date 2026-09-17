#!/usr/bin/env python3
"""Validate and compare complete AI Toolkit AdamW8bit trajectories.

The comparator checks every exported parameter, changing gradient, uint8 state,
dynamic codebook, block scale, and scalar-step snapshot. It does not implement
an optimizer update; a CPU AdamW approximation would invalidate the parity gate.
"""
from __future__ import annotations
import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"AITKOPT1"
DTYPES = {np.dtype("<f4"): 1, np.dtype("u1"): 2, np.dtype("<i4"): 3, np.dtype("<f8"): 4}


def dynamic_map(signed: bool) -> np.ndarray:
    """Reproduce bitsandbytes functional.create_dynamic_map (v0.49.2)."""
    values = []
    max_exponent_bits, total_bits = 7, 8
    non_sign_bits = total_bits - 1
    additional = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        # The upstream expression differs for the signed and unsigned maps.
        count = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1) if signed else int(2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = np.linspace(0.1, 1.0, count, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / np.float32(2.0)
        scale = np.float32(10 ** (-(max_exponent_bits - 1) + i))
        values.extend((scale * means).tolist())
        if signed: values.extend((-scale * means).tolist())
    if additional > 0:
        boundaries = np.linspace(0.1, 1.0, additional + 1, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / np.float32(2.0)
        scale = np.float32(10 ** (-(max_exponent_bits - 1) + i))
        values.extend((scale * means).tolist())
        if signed: values.extend((-scale * means).tolist())
    values.extend([0.0, 1.0])
    values.extend([0.0] * (256 - len(values)))
    return np.sort(np.asarray(values, dtype=np.float32))


def quantize_2d(x: np.ndarray, code: np.ndarray, signed: bool) -> np.ndarray:
    """CPU translation of kernels.cu quantize_2D<1>/<0> midpoint search."""
    x = np.asarray(x, dtype=np.float32)
    idx = np.searchsorted(code, x, side="left").clip(1, 255)
    lo, hi = code[idx - 1], code[idx]
    choose_hi = x > ((lo + hi) * np.float32(0.5))
    out = (idx - 1 + choose_hi.astype(np.uint8)).astype(np.int16)
    return out.astype(np.uint8)


def replay_cpu(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay the source formulas on CPU for diagnostics, never as parity proof."""
    metadata, arrays = load_fixture(root)
    c = metadata["construction"]
    beta1, beta2, eps, lr, wd = map(float, (c["betas"][0], c["betas"][1], c["eps"], c["lr"], c["weight_decay"]))
    signed, unsigned = dynamic_map(True), dynamic_map(False)
    params = {"small": arrays["parameter_small_initial"].astype(np.float32).copy(), "large": arrays["parameter_large_initial"].astype(np.float32).copy()}
    state = {}
    for side in params:
        n = params[side].size
        state[side] = (np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.uint8), np.zeros(n, np.uint8), np.zeros((n + 255) // 256, np.float32), np.zeros((n + 255) // 256, np.float32))
    out = {}
    for step in range(1, int(metadata["fixture"]["steps_requested"]) + 1):
        for side in ("small", "large"):
            grad = arrays[f"gradient_{side}_step_{step - 1}"].astype(np.float32)
            p = params[side]; m, v, q1, q2, a1, a2 = state[side]
            for block, start in enumerate(range(0, p.size, 256)):
                end = min(start + 256, p.size)
                gv = grad[start:end]
                finite = np.isfinite(gv)
                if p.size < 4096:
                    m0, v0 = m[start:end].copy(), v[start:end].copy()
                else:
                    m0 = np.where(finite, signed[q1[start:end].astype(np.int16)] * a1[block], 0.0).astype(np.float32)
                    v0 = np.where(finite, unsigned[q2[start:end].astype(np.int16)] * a2[block], 0.0).astype(np.float32)
                m1 = np.where(finite, m0 * beta1 + gv * (1 - beta1), 0.0).astype(np.float32)
                v1 = np.where(finite, v0 * beta2 + (gv * gv) * (1 - beta2), 0.0).astype(np.float32)
                a1[block], a2[block] = np.max(np.abs(m1)), np.max(np.abs(v1))
                norm1 = np.divide(m1, a1[block], out=np.zeros_like(m1), where=a1[block] != 0)
                norm2 = np.divide(v1, a2[block], out=np.zeros_like(v1), where=a2[block] != 0)
                q1b, q2b = quantize_2d(norm1, signed, True), quantize_2d(norm2, unsigned, False)
                # kernels.cu preserves the sign of the signed moment after quantization.
                qvals = signed[q1b.astype(np.int16)]
                mismatch = np.signbit(qvals) != np.signbit(m1)
                q1b = np.clip(q1b.astype(np.int16) + np.where(mismatch, np.where(m1 > 0, 1, -1), 0), 0, 255).astype(np.uint8)
                m[start:end], v[start:end], q1[start:end], q2[start:end] = m1, v1, q1b, q2b
                correction1 = np.float32(1 - beta1 ** step); correction2 = np.float32(np.sqrt(1 - beta2 ** step))
                step_size = np.float32(-lr * correction2 / correction1)
                p[start:end] += step_size * (m1 / (np.sqrt(v1) + correction2 * eps))
                p[start:end] *= np.float32(1 - lr * wd)
            out[f"parameter_{side}_after_step_{step}"] = p.copy()
            out[f"step_{step}_{side}_state1"] = m.copy() if p.size < 4096 else q1.copy()
            out[f"step_{step}_{side}_state2"] = v.copy() if p.size < 4096 else q2.copy()
            if p.size >= 4096:
                out[f"step_{step}_{side}_absmax1"], out[f"step_{step}_{side}_absmax2"] = a1.copy(), a2.copy()
    return out, {"warning": "CPU replay is a diagnostic translation; CUDA parity remains required."}


def load_fixture(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((root / "optimizer_metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "optimizer_tensors.npz", allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    return metadata, arrays


def validate(root: Path) -> dict[str, Any]:
    metadata, arrays = load_fixture(root)
    if metadata.get("fixture", {}).get("steps_requested") != 5:
        raise ValueError("expected the five-step FP32 CUDA fixture")
    snapshots = metadata.get("state_snapshots", [])
    if len(snapshots) != 5:
        raise ValueError(f"expected 5 state snapshots, found {len(snapshots)}")
    for snap in snapshots:
        for side in ("small", "large"):
            for item in snap[side].values():
                if "array" in item and item["array"] not in arrays:
                    raise ValueError(f"missing state array {item['array']}")
    return {"arrays": len(arrays), "snapshots": len(snapshots), "device": metadata["fixture"]["device"]}


def compare(expected: Path, actual: Path, atol: float, rtol: float) -> dict[str, Any]:
    em, ea = load_fixture(expected)
    am, aa = load_fixture(actual)
    missing = sorted(set(ea) - set(aa))
    extra = sorted(set(aa) - set(ea))
    if missing or extra:
        raise ValueError(f"array key mismatch: missing={missing}, extra={extra}")
    worst = {"name": "", "max_abs": 0.0, "max_rel": 0.0}
    for name in sorted(ea):
        x, y = ea[name], aa[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"array metadata mismatch for {name}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
        max_abs = float(delta.max()) if delta.size else 0.0
        scale = np.maximum(np.abs(x.astype(np.float64)), np.abs(y.astype(np.float64)))
        max_rel = float((delta / np.maximum(scale, 1e-30)).max()) if delta.size else 0.0
        if max_abs > worst["max_abs"]: worst = {"name": name, "max_abs": max_abs, "max_rel": max_rel}
        if not np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
            raise ValueError(f"mismatch in {name}: max_abs={max_abs:g}, max_rel={max_rel:g}")
    return {"arrays": len(ea), "worst": worst, "expected_device": em["fixture"]["device"], "actual_device": am["fixture"]["device"]}


def write_binary(root: Path, output: Path) -> None:
    _, arrays = load_fixture(root)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    with output.open("wb") as out:
        out.write(struct.pack("<8sII", MAGIC, 1, len(arrays)))
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            dtype = DTYPES.get(array.dtype.newbyteorder("<"))
            if dtype is None: raise ValueError(f"unsupported dtype for {name}: {array.dtype}")
            encoded = name.encode("utf-8")
            out.write(struct.pack("<I", len(encoded))); out.write(encoded)
            out.write(struct.pack("<II", dtype, array.ndim))
            out.write(struct.pack("<" + "Q" * array.ndim, *array.shape))
            raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
            out.write(struct.pack("<Q", len(raw))); out.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--cpu-replay", action="store_true", help="run the source-formula CPU diagnostic; not an equivalence claim")
    args = parser.parse_args()
    print(json.dumps(validate(args.fixture), indent=2))
    if args.binary: write_binary(args.fixture, args.binary)
    if args.actual: print(json.dumps(compare(args.fixture, args.actual, args.atol, args.rtol), indent=2))
    if args.cpu_replay:
        replayed, note = replay_cpu(args.fixture)
        _, expected = load_fixture(args.fixture)
        diffs = {}
        for name, value in replayed.items():
            target = expected.get(name)
            diffs[name] = None if target is None else float(np.max(np.abs(value.astype(np.float64) - target.astype(np.float64))))
        print(json.dumps({"cpu_replay": note, "max_abs_by_array": diffs}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())

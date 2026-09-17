"""Compare the native scalar oracle with functions extracted from pinned AI Toolkit.

CPU mode checks extracted reference arithmetic. CUDA mode loads the reference's
actual Triton quantization, custom linear operation and autograd. Neither mode
establishes end-to-end training parity.
"""
import argparse
import ast
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def reference_functions(path):
    source = path.read_text(encoding="utf-8")
    names = {"_cached", "regular_hadamard", "rotate", "quantize_int8_rows",
             "_int8_linear_ste_backward"}
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Reference functions are missing")
    for node in functions:
        node.decorator_list = []
    namespace = {"torch": torch, "_hadamard_cache": {}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def compare(name, actual, expected, atol, rtol):
    diff = (actual.float() - expected.float()).abs()
    # Tensor-scale bound avoids unstable per-element relative errors at cancellation zeros.
    bound = atol + rtol * expected.float().abs().max().item()
    ok = diff.max().item() <= bound
    result = {"name": name, "pass": ok, "max_abs": diff.max().item(),
              "relative_l2": (torch.linalg.vector_norm(diff) /
                              torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item(),
              "atol": atol, "rtol": rtol, "max_abs_bound": bound}
    return result


def run_case(ref, probe, directory, name, rows, width, outputs, rotation, dtype, seed, device):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=generator).to(dtype)
    # Include a zero row to exercise its special scale = 1 contract.
    x[0].zero_()
    if rotation == 1 and rows > 1 and width >= 8:
        x[1].zero_()
        x[1, :8] = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -127, 127], dtype=dtype)
    w = torch.randint(-127, 128, (outputs, width), generator=generator, dtype=torch.int8)
    scales = torch.rand(outputs, generator=generator) * 0.02 + 0.00001
    grad = torch.randn(rows, outputs, generator=generator).to(dtype)
    bias = torch.randn(outputs, generator=generator).to(dtype)
    x, w, scales, grad, bias = (value.to(device) for value in (x, w, scales, grad, bias))
    if device == "cuda":
        x.requires_grad_()
    rotated = ref["rotate"](x, rotation)
    if device == "cuda":
        codes, act_scales = ref["quantize_int8_rows_fused"](rotated.detach())
        expected_y = ref["_int8_linear_ste_op"](rotated, w, scales.view(torch.uint8), bias, 127,
                                               str(dtype).split(".")[-1])
        expected_y.backward(grad)
        expected_dx = x.grad
    else:
        codes, act_scales = ref["quantize_int8_rows"](rotated)
        accum = codes.int() @ w.int().T
        expected_y = (accum.float() * (act_scales[:, None] * scales[None, :]) + bias.float()).to(dtype)
        ctx = SimpleNamespace(saved_tensors=(w, scales.view(torch.uint8)))
        rotated_grad = ref["_int8_linear_ste_backward"](ctx, grad)[0]
        expected_dx = ref["rotate"](rotated_grad, rotation)
    fixture = directory / f"{name}.input.bin"
    output = directory / f"{name}.output.bin"
    with fixture.open("xb") as file:
        file.write(struct.pack("<6I", 0x314B5441, rows, width, outputs, rotation, int(dtype == torch.bfloat16)))
        for tensor in (x.float(), w, scales, grad.float(), bias.float()):
            file.write(tensor.detach().contiguous().cpu().numpy().tobytes())
    subprocess.run([str(probe), str(fixture), str(output)], check=True, capture_output=True, text=True)
    data = output.read_bytes()
    offset = 0

    def take(shape, np_dtype):
        nonlocal offset
        count = int(np.prod(shape))
        array = np.frombuffer(data, dtype=np_dtype, count=count, offset=offset).copy().reshape(shape)
        offset += array.nbytes
        return torch.from_numpy(array).to(device)

    native_rotated = take((rows, width), "<f4")
    native_codes = take((rows, width), "i1")
    native_scales = take((rows,), "<f4")
    native_y = take((rows, outputs), "<f4")
    native_dx = take((rows, width), "<f4")
    if offset != len(data):
        raise ValueError("Unexpected native output length")
    # BF16 inputs are dyadic; these test sizes keep transform sums exact in FP32.
    # FP32 transform reduction ordering can vary, so allow small accumulation error.
    tolerance = 0.0 if dtype == torch.bfloat16 else 2e-6
    checks = [compare("rotation", native_rotated, rotated, tolerance, tolerance),
              compare("activation_codes", native_codes, codes, 0, 0),
              compare("activation_scales", native_scales, act_scales, tolerance, tolerance),
              compare("forward", native_y, expected_y, tolerance, tolerance),
              compare("input_gradient", native_dx, expected_dx, tolerance, tolerance)]
    return {"case": name, "dtype": str(dtype), "shape": [rows, width, outputs],
            "rotation": rotation, "checks": checks, "pass": all(c["pass"] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--production-shapes", action="store_true", help="Also test YuE2 projection dimensions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ref, digest = reference_functions(args.toolkit / "toolkit/util/convrot_quant.py")
    if args.device == "cuda":
        sys.path.insert(0, str(args.toolkit.resolve()))
        from toolkit.util import convrot_quant
        ref = vars(convrot_quant)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for dtype, label in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        for index, (width, rotation) in enumerate(((16, 1), (64, 16), (256, 256), (512, 256))):
            cases.append(run_case(ref, args.probe.resolve(), args.output, f"{label}-{width}-{rotation}",
                                  3, width, 32, rotation, dtype, 71 + index, args.device))
    if args.production_shapes:
        for index, (rows, width, outputs) in enumerate(((33, 2048, 4096), (3, 2048, 12288), (3, 6144, 2048))):
            cases.append(run_case(ref, args.probe.resolve(), args.output, f"bf16-production-{rows}-{width}-{outputs}",
                                  rows, width, outputs, 256, torch.bfloat16, 81 + index, args.device))
    report = {"schema": 1, "source_sha256": digest, "torch": torch.__version__,
              "scope": "Standalone native probe versus reference arithmetic; no full-model parity claim",
              "reference_device": args.device,
              "reference_execution": "actual custom-op/autograd" if args.device == "cuda" else "extracted arithmetic",
              "pass": all(case["pass"] for case in cases), "cases": cases}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(args.output / "report.json"),
                      "cases": [{"case": c["case"], "pass": c["pass"]} for c in cases]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

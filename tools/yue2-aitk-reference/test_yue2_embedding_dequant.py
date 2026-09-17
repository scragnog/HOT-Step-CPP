"""Compare staged native YuE2 ConvRot embedding dequantization with AI Toolkit."""

import argparse
import ast
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch


def reference_functions(path: Path):
    source = path.read_text(encoding="utf-8")
    names = {"_cached", "regular_hadamard", "rotate"}
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("reference rotation functions are missing")
    namespace = {"torch": torch, "_hadamard_cache": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ref, digest = reference_functions(args.toolkit / "toolkit/util/convrot_quant.py")

    rows, cols, rotation, first_row, count = 5, 512, 256, 1, 3
    rng = np.random.default_rng(9017)
    qdata = rng.integers(-127, 128, size=(rows, cols), dtype=np.int8)
    qdata[0, :] = 0
    scales = np.array([1.0, 0.003125, -0.0078125, 0.125, 0.0009765625], dtype=np.float32)
    expected = ref["rotate"](
        torch.from_numpy(qdata.astype(np.float32)) * torch.from_numpy(scales).view(-1, 1), rotation
    )[first_row:first_row + count].to(torch.bfloat16)
    expected_u16 = expected.view(torch.uint16).contiguous().numpy().astype("<u2", copy=False)

    fixture = args.output / "embedding.input.bin"
    output = args.output / "embedding.output.bin"
    with fixture.open("xb") as stream:
        stream.write(struct.pack("<6I", 0x324D4245, rows, cols, rotation, first_row, count))
        stream.write(qdata.tobytes(order="C"))
        stream.write(scales.astype("<f4", copy=False).tobytes())
    command = [str(args.probe.resolve()), str(fixture.resolve()), str(output.resolve())]
    if os.name == "nt":
        # All native Windows probes inherit DLL lookup and noninteractive errors.
        launcher = Path(__file__).with_name("run-native-test.ps1")
        def ps_literal(value):
            return "'" + str(value).replace("'", "''") + "'"
        script = (
            f"& {ps_literal(launcher.resolve())} -Executable {ps_literal(command[0])} "
            f"-TestArguments @({','.join(ps_literal(arg) for arg in command[1:])})"
        )
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output / "probe.stdout.log").write_text(result.stdout, encoding="utf-8")
    (args.output / "probe.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise SystemExit(result.returncode)
    actual_u16 = np.fromfile(output, dtype="<u2").reshape(count, cols)
    mismatch = actual_u16 != expected_u16
    mismatch_positions = np.argwhere(mismatch)
    differences = []
    for row, col in mismatch_positions[:16]:
        differences.append({
            "selected_row": int(first_row + row),
            "col": int(col),
            "native_bf16": f"0x{int(actual_u16[row, col]):04x}",
            "reference_bf16": f"0x{int(expected_u16[row, col]):04x}",
        })
    report = {
        "schema": 1,
        "source_sha256": digest,
        "scope": "CPU selected-row ConvRot dequantization; no full-model claim",
        "shape": [rows, cols],
        "rotation": rotation,
        "selected_rows": [first_row, count],
        "bf16_exact": not bool(mismatch.any()),
        "mismatch_count": int(mismatch.sum()),
        "max_uint16_delta": int(np.abs(actual_u16.astype(np.int32) - expected_u16.astype(np.int32)).max()),
        "differences": differences,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["bf16_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

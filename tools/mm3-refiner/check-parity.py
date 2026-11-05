"""Compare one engine NPY or little-endian F32 dump with a frozen reference NPY."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(actual, reference, limits):
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: engine {actual.shape}, reference {reference.shape}")
    if not actual.size or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("empty or non-finite tensor")
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    delta = a - b
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else float(np.array_equal(a, b))
    metrics = dict(max_abs=float(np.max(np.abs(delta))),
                   relative_rmse=float(np.linalg.norm(delta) / max(norm_b, 1e-30)), cosine=cosine)
    passed = (metrics["max_abs"] <= limits["max_abs"] and
              metrics["relative_rmse"] <= limits["relative_rmse"] and cosine >= limits["cosine"])
    return dict(passed=bool(passed), **metrics, limits=limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("fixtures.json"))
    parser.add_argument("--shape", type=int, nargs="+", help="required for raw F32, in reference C dimension order")
    args = parser.parse_args()
    limits = json.loads(args.contract.read_text())["tolerances"][args.stage]
    reference = np.load(args.reference, allow_pickle=False)
    if args.engine.suffix == ".npy":
        actual = np.load(args.engine, allow_pickle=False)
    else:
        if not args.shape:
            parser.error("raw F32 input requires --shape; dimensions are never inferred")
        actual = np.fromfile(args.engine, dtype="<f4").reshape(args.shape)
    result = compare(actual, reference, limits)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a read-only, dependency-free inventory of the YuE2 AI Toolkit reference.

The exporter deliberately uses only the Python standard library.  In particular,
it never imports torch, safetensors, or any other package which could initialise
CUDA or download a model.  Checkpoint inspection reads only safetensors headers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


PINNED_COMMIT = "e65c4d0fb69251e692390574c49873297dc4bae5"
SCHEMA_VERSION = 1
REFERENCE_SOURCES = (
    "extensions_built_in/audio_models/yue2/yue2_model.py",
    "extensions_built_in/audio_models/yue2/src/model.py",
    "extensions_built_in/audio_models/yue2/src/tokenizer.py",
    "extensions_built_in/audio_models/yue2/src/pipeline.py",
    "toolkit/util/convrot_quant.py", "toolkit/optimizer.py", "toolkit/config_modules.py",
    "jobs/process/BaseSDTrainProcess.py", "extensions_built_in/sd_trainer/SDTrainer.py",
    "ui/src/app/jobs/new/jobConfig.ts", "ui/src/app/jobs/new/options.tsx",
)


class ManifestError(ValueError):
    """Raised when an input cannot be inventoried safely."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    relative = str(resolved)
    if root:
        try:
            relative = str(resolved.relative_to(root.resolve()))
        except ValueError:
            pass
    record: dict[str, Any] = {
        "path": relative,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    return record


def _git(toolkit_root: Path, *args: str, check: bool = True) -> str:
    command = ["git", "-c", f"safe.directory={toolkit_root.resolve()}", "-C", str(toolkit_root), *args]
    result = subprocess.run(command, check=check, capture_output=True, text=True)
    if check:
        return result.stdout.strip()
    return result.stdout.strip()


def git_inventory(toolkit_root: Path, pinned_commit: str) -> dict[str, Any]:
    try:
        head = _git(toolkit_root, "rev-parse", "HEAD")
        probe = subprocess.run(
            ["git", "-c", f"safe.directory={toolkit_root.resolve()}", "-C", str(toolkit_root), "cat-file", "-e", f"{pinned_commit}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        pinned_exists = probe.returncode == 0
        if not pinned_exists:
            raise ManifestError(f"pinned commit is not present: {pinned_commit}")
        tree = _git(toolkit_root, "rev-parse", f"{pinned_commit}^{{tree}}")
        head_tree = _git(toolkit_root, "rev-parse", "HEAD^{tree}")
        status = _git(toolkit_root, "status", "--porcelain=v1", "--untracked-files=all")
        upstream_names = _git(toolkit_root, "diff", "--name-status", pinned_commit, "HEAD")
        worktree_names = _git(toolkit_root, "diff", "HEAD", "--name-status")
        untracked = _git(toolkit_root, "ls-files", "--others", "--exclude-standard")
        patch = _git(toolkit_root, "diff", pinned_commit, "HEAD", "--binary")
        effective_patch = _git(toolkit_root, "diff", pinned_commit, "--binary")
        worktree_patch = _git(toolkit_root, "diff", "HEAD", "--binary")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError(f"unable to inspect git repository: {exc}") from exc

    def parse_names(value: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line in value.splitlines():
            bits = line.split("\t")
            if bits:
                rows.append({"status": bits[0], "path": bits[-1]})
        return rows

    source_paths = sorted(set(REFERENCE_SOURCES) | {row["path"] for row in parse_names(upstream_names) + parse_names(worktree_names)})
    source_hashes = []
    for relative in source_paths:
        current = toolkit_root / relative
        current_hash = sha256_file(current) if current.is_file() else None
        baseline = subprocess.run(
            ["git", "-c", f"safe.directory={toolkit_root.resolve()}", "-C", str(toolkit_root), "show", f"{pinned_commit}:{relative}"],
            capture_output=True,
        )
        source_hashes.append({
            "path": relative,
            "current_sha256": current_hash,
            "pinned_sha256": sha256_bytes(baseline.stdout) if baseline.returncode == 0 else None,
            "classification": ("working_tree_patch" if relative in {r["path"] for r in parse_names(worktree_names)}
                               else "upstream_delta" if relative in {r["path"] for r in parse_names(upstream_names)}
                               else "pinned_source"),
        })

    return {
        "pinned_commit": pinned_commit,
        "head": head,
        "pinned_tree": tree,
        "head_tree": head_tree,
        "head_matches_pinned": head == pinned_commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "upstream_delta": parse_names(upstream_names),
        "working_tree_patch": parse_names(worktree_names),
        "untracked": [line for line in untracked.splitlines() if line],
        "pinned_to_head_patch_sha256": sha256_bytes(patch.encode("utf-8")),
        "pinned_to_worktree_patch_sha256": sha256_bytes(effective_patch.encode("utf-8")),
        "working_tree_patch_sha256": sha256_bytes(worktree_patch.encode("utf-8")),
        "source_hashes": source_hashes,
    }


def parse_requirements(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:==|@)\s*(.+?)\s*$", line)
        if match:
            packages[match.group(1)] = match.group(2)
    return packages


def package_inventory(experiment_root: Path, explicit: Iterable[Path]) -> list[dict[str, Any]]:
    paths = {p.resolve() for p in explicit}
    for name in ("environment.txt", "environment-optimized.txt"):
        candidate = experiment_root / name
        if candidate.is_file():
            paths.add(candidate.resolve())
    result = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        record = file_record(path, experiment_root)
        record["packages"] = parse_requirements(path)
        result.append(record)
    return result


def config_inventory(experiment_root: Path, explicit: Iterable[Path]) -> list[dict[str, Any]]:
    paths = {p.resolve() for p in explicit}
    pattern = re.compile(r"^(?:train[^/]*|config)\.(?:ya?ml)$|^(?:dataset-manifest|toolkit-commit)\.json$|^environment[^/]*\.txt$", re.I)
    for path in experiment_root.rglob("*"):
        if path.is_file() and pattern.match(path.name):
            paths.add(path.resolve())
    return [file_record(path, experiment_root) for path in sorted(paths, key=lambda item: str(item).lower())]


def safetensors_header(path: Path) -> dict[str, Any]:
    """Read and validate the JSON header without reading tensor payloads."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ManifestError(f"{path}: missing safetensors header length")
        header_size = int.from_bytes(size_bytes, "little")
        if header_size <= 0 or header_size > 100 * 1024 * 1024:
            raise ManifestError(f"{path}: unreasonable safetensors header length {header_size}")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ManifestError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{path}: invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise ManifestError(f"{path}: safetensors header is not an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ManifestError(f"{path}: __metadata__ is not an object")
    tensors = []
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or not isinstance(info.get("dtype"), str):
            raise ManifestError(f"{path}: malformed tensor entry {name!r}")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise ManifestError(f"{path}: malformed shape for tensor {name!r}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) and x >= 0 for x in offsets) or offsets[1] < offsets[0]:
            raise ManifestError(f"{path}: malformed data_offsets for tensor {name!r}")
        if offsets[1] > file_size - 8 - header_size:
            raise ManifestError(f"{path}: tensor {name!r} extends beyond the file payload")
        tensors.append({"name": name, "dtype": info["dtype"], "shape": shape, "data_offsets": offsets, "payload_bytes": offsets[1] - offsets[0]})
    record = file_record(path)
    record.update({"header_bytes": header_size, "metadata": metadata, "tensor_count": len(tensors), "tensors": tensors})
    return record


def build_manifest(toolkit_root: Path, experiment_root: Path, checkpoints: Iterable[Path], package_files: Iterable[Path] = (), config_files: Iterable[Path] = (), pinned_commit: str = PINNED_COMMIT) -> dict[str, Any]:
    toolkit_root = toolkit_root.resolve()
    experiment_root = experiment_root.resolve()
    if not toolkit_root.is_dir() or not experiment_root.is_dir():
        raise ManifestError("toolkit and experiment paths must be existing directories")
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": {"toolkit_root": str(toolkit_root), "experiment_root": str(experiment_root), "git": git_inventory(toolkit_root, pinned_commit)},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "gpu_imports": False},
        "packages": package_inventory(experiment_root, package_files),
        "configs": config_inventory(experiment_root, config_files),
        "checkpoints": [safetensors_header(path.resolve()) for path in sorted(checkpoints, key=lambda item: str(item).lower())],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--package-file", type=Path, action="append", default=[])
    parser.add_argument("--config-file", type=Path, action="append", default=[])
    parser.add_argument("--pinned-commit", default=PINNED_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.toolkit_root, args.experiment_root, args.checkpoint, args.package_file, args.config_file, args.pinned_commit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (ManifestError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

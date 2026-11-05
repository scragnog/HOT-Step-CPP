"""Lossless F32 reference GGUFs with source hashes and complete tensor accounting."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contract():
    return json.loads((HERE / "fixtures.json").read_text())


def verify_sources(spec, root=ROOT):
    for relative, expected in spec["source_sha256"].items():
        path = root / relative
        if sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    for name, version in spec["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"package version mismatch: {name}, expected {version}")


def load_tensors(path):
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def convert(component, spec, output, root=ROOT):
    cfg = spec["models"][component]
    source = load_tensors(root / cfg["weights"])
    tensors = {}
    omitted = []
    for name, value in sorted(source.items()):
        if component == "clap" and not name.startswith(("audio_model.", "audio_projection.")):
            omitted.append(dict(name=name, shape=list(value.shape), elements=value.numel(),
                                reason="text tower or training logit scale, not used by audio inference"))
        else:
            tensors[name] = value
    if component == "refiner":
        for name, value in load_tensors(root / cfg["normalization"]).items():
            tensors["normalization." + name] = value
        # A frozen constant is embedded only after its use is approved in the contract.
        if cfg.get("style_mode") == "frozen_mean":
            with np.load(root / cfg["constants"]) as constants:
                tensors["constants.clap_mean"] = torch.from_numpy(constants["clap_mean"].copy())
    names = set(tensors)
    if len(names) != len(tensors) or not names:
        raise ValueError("empty or duplicate tensor inventory")
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f"mm3-refiner-{component}-f32.gguf"
    writer = gguf.GGUFWriter(str(dest), f"mm3-refiner-{component}")
    writer.add_name(f"MM3 refiner {component}")
    writer.add_string("general.license", cfg["license"])
    writer.add_string("mm3.refiner.contract_sha256", sha256(HERE / "fixtures.json"))
    writer.add_string("mm3.refiner.source_revision", cfg["revision"])
    writer.add_string("mm3.refiner.config", (root / cfg["config"]).read_text())
    writer.add_string("mm3.refiner.tensor_layout", "torch C order; GGML dimensions reversed; no transpose or squeeze")
    if "processor" in cfg:
        writer.add_string("mm3.refiner.processor", (root / cfg["processor"]).read_text())
    writer.add_uint32("general.file_type", 0)
    inventory = []
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().float().contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite weights: {name}")
        writer.add_tensor(name, array)
        inventory.append(dict(name=name, shape=list(array.shape), dtype="float32", bytes=array.nbytes,
                              sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # Read back every tensor, including weight-norm parameters and scalar buffers.
    reader = gguf.GGUFReader(str(dest))
    actual = {t.name: t for t in reader.tensors}
    if set(actual) != names:
        raise ValueError("GGUF tensor inventory mismatch")
    for entry in inventory:
        tensor = actual[entry["name"]]
        if hashlib.sha256(tensor.data.tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"GGUF round-trip mismatch: {entry['name']}")
    report = dict(component=component, path=dest.name, bytes=dest.stat().st_size, sha256=sha256(dest),
                  source_tensor_count=len(source), written_tensor_count=len(inventory),
                  tensor_bytes=sum(t["bytes"] for t in inventory), omitted=omitted, tensors=inventory)
    (output / f"{component}-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tensors", "omitted")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+", choices=["refiner", "mert", "clap"], default=["refiner", "mert"])
    parser.add_argument("--out", type=Path, default=HERE / "artifacts/gguf")
    args = parser.parse_args()
    spec = contract()
    if "clap" in args.components and spec["models"]["clap"]["decision"] != "retained":
        parser.error("CLAP conversion is awaiting the parent decision")
    verify_sources(spec)
    for component in args.components:
        convert(component, spec, args.out)


if __name__ == "__main__":
    main()

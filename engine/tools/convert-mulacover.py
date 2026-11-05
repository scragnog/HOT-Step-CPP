#!/usr/bin/env python3
"""Offline MuLaCover safetensors to HeartMuLa-contract GGUF (global + local).

Requires numpy, torch, safetensors and gguf. No model code is imported.
Example:
  python engine/tools/convert-mulacover.py --src <MuLaCover> --codec <HeartCodec-oss> \
      --tokenizer <models>/heartmula/heartmula-tokenizer.gguf \
      --rvq <models>/heartmula/heartcodec-rvq-F32.gguf --out <gguf-dir>

MuLaCover is the HeartMuLa 3B/300M stack plus cover conditioning, so it is written
on the convert-heartmula.py contract as variant "MuLaCover":
heartmula-mulacover-global-BF16.gguf and heartmula-mulacover-local-BF16.gguf.
The cover-only modules ride in the global file: peft_layers.20..27 (GACA cross
attention, gate attention and gates, symbolic adaptor), tag_mlp_layers.20..27 and
qwen_linear. They are full modules, not LoRA factors, so nothing is merged.

Every tensor in model.safetensors.index.json is written; none is dropped. Names are
kept except audio_head -> audio_head.0..6.weight, each transposed to [8197,3072]
as convert-heartmula.py does. Dtype policy (the checkpoint is all BF16): tensors
with two or more dimensions stay BF16, bit for bit; 1-D tensors (norm scales,
biases) widen to F32, which is exact. So every output equals its source exactly.

The tokenizer and RVQ dependencies point at the installed HeartMuLa pack: the
converter refuses unless MuLaCover's tokenizer.json is the pack's, its gen_config
token ids agree with the pack's, and the RVQ file was converted from the same
HeartCodec checkpoint MuLaCover pins (--codec). MuLaCover's own gen_config (which
adds the Qwen and tag ids) is stored as heartmula.cover.gen_config_json.
Existing outputs are refused. A failed run leaves .partial files for diagnosis.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

_spec = importlib.util.spec_from_file_location("convert_heartmula", Path(__file__).with_name("convert-heartmula.py"))
heartmula = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heartmula)
sha256, write_json, source_identity = heartmula.sha256, heartmula.write_json, heartmula.source_identity

VERSION = 1
VARIANT = "MuLaCover"
TOTAL = 4374364776
COVER_LAYERS = range(20, 28)
LICENSE = "CC-BY-NC-4.0 plus MuLaCover MODEL_LICENSE; outputs are non-commercial"
ATTRIBUTION = "MuLaCover by MuLa Labs, https://github.com/HeartMuLa/MuLaCover"


def read_headers(directory):
    """Every tensor's dtype, shape and shard, with extents and the index checked."""
    tensors = {}
    for file in sorted(directory.glob("*.safetensors")):
        with file.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(size))
        end = 0
        for name, info in sorted(((n, i) for n, i in header.items() if n != "__metadata__"),
                                 key=lambda item: item[1]["data_offsets"][0]):
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            if info["dtype"] != "BF16":
                raise ValueError(f"Expected BF16, got {info['dtype']}: {name}")
            start, stop = info["data_offsets"]
            if start != end or stop - start != math.prod(info["shape"]) * 2:
                raise ValueError(f"Invalid tensor extent: {name}")
            tensors[name] = dict(info, file=file.name)
            end = stop
        if 8 + size + end != file.stat().st_size:
            raise ValueError(f"Invalid source byte count: {file}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    if index != {n: i["file"] for n, i in tensors.items()}:
        raise ValueError("Shard index differs from actual headers")
    return tensors


def component(name):
    if name == "audio_head" or name.startswith(("decoder.", "projection.")):
        return "local"
    if name.startswith(("backbone.", "text_embeddings.", "audio_embeddings.", "codebook0_head.", "muq_linear.",
                        "unconditional_text_embedding.", "qwen_linear.")):
        return "global"
    for prefix in ("peft_layers.", "tag_mlp_layers."):
        if name.startswith(prefix) and int(name.split(".")[1]) in COVER_LAYERS:
            return "global"
    raise ValueError(f"Unaccounted source tensor: {name}")


def outputs_for(name, value):
    """(output name, encoded array, raw GGML type, transform) for one source tensor."""
    if name == "audio_head":
        if tuple(value.shape) != (7, 3072, 8197):
            raise ValueError("Unexpected audio_head shape")
        return [(f"audio_head.{i}.weight", value[i].T.contiguous(), f"slice[{i}],transpose(1,0)") for i in range(7)]
    return [(name, value, "identity")]


def encode(tensor):
    if tensor.ndim < 2:
        out = tensor.float()
        if not torch.equal(out.to(torch.bfloat16), tensor):
            raise ValueError("BF16 to F32 widening is not exact")
        return out.numpy(), gguf.GGMLQuantizationType.F32
    return tensor.view(torch.int16).numpy().view(np.uint16), gguf.GGMLQuantizationType.BF16


def verify(path, rows, metadata):
    reader = gguf.GGUFReader(path)
    for key, value in metadata.items():
        if reader.fields[key].contents() != value:
            raise ValueError(f"Metadata round-trip mismatch: {key}")
    expected = {o["name"]: o for row in rows for o in row["outputs"]}
    if len(reader.tensors) != len(expected):
        raise ValueError("GGUF tensor count differs")
    for tensor in reader.tensors:
        row = expected.pop(tensor.name)
        if list(reversed(tensor.shape.tolist())) != row["shape"] or tensor.n_bytes != row["bytes"]:
            raise ValueError(f"GGUF shape/bytes differ: {tensor.name}")
        if hashlib.sha256(tensor.data.tobytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"GGUF payload differs: {tensor.name}")
    if expected:
        raise ValueError(f"GGUF is missing tensors: {sorted(expected)[:5]}")


def convert(src, tensors, part, identity, dependencies, cover_gen_config, out):
    path = out / f"heartmula-{VARIANT.lower()}-{part}-BF16.gguf"
    partial = path.with_suffix(".gguf.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    names = sorted(n for n in tensors if component(n) == part)
    fingerprint = hashlib.sha256(json.dumps(
        {name: info["sha256"] for name, info in identity.items() if name.endswith(".safetensors")},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    metadata = {"general.architecture": "heartmula-" + part, "general.name": path.stem,
                "general.file_type": int(gguf.LlamaFileType.MOSTLY_BF16),
                "general.license": LICENSE, "general.source.url": ATTRIBUTION,
                "heartmula.format_version": VERSION, "heartmula.base_fingerprint": fingerprint,
                "heartmula.component": part, "heartmula.variant": VARIANT,
                "heartmula.source": json.dumps(identity, sort_keys=True),
                "heartmula.config_json": (src / "config.json").read_text(encoding="utf-8"),
                "heartmula.config_sha256": identity["config.json"]["sha256"],
                "heartmula.dependencies": json.dumps(dependencies, sort_keys=True),
                "heartmula.cover.gen_config_json": cover_gen_config,
                "heartmula.tensor_layout": "numpy-C; GGUF dimensions reversed; transforms in tensor_manifest"}
    writer = gguf.GGUFWriter(partial, metadata["general.architecture"], use_temp_file=True)
    rows = []
    shards = {}
    for name in names:
        file = tensors[name]["file"]
        if file not in shards:
            shards[file] = safe_open(src / file, framework="pt", device="cpu")
        row = dict(sources={name: tensors[name]}, outputs=[])
        for out_name, value, transform in outputs_for(name, shards[file].get_tensor(name)):
            array, raw = encode(value)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite tensor: {name}")
            row["outputs"].append(dict(name=out_name, shape=list(value.shape), dtype=raw.name, bytes=array.nbytes,
                                       sha256=hashlib.sha256(array.tobytes()).hexdigest(), transform=transform,
                                       max_abs_error=0.0))
            writer.add_tensor(out_name, array, raw_dtype=raw)
        rows.append(row)
    metadata["heartmula.tensor_manifest"] = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    for key, value in metadata.items():
        if key == "general.architecture":
            continue
        if isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    verify(partial, rows, metadata)
    partial.rename(path)
    outputs = [o for r in rows for o in r["outputs"]]
    print(f"PASS {path.name}: {len(names)} source tensors, {len(outputs)} outputs, "
          f"{sum(o['dtype'] == 'BF16' for o in outputs)} BF16 + {sum(o['dtype'] == 'F32' for o in outputs)} F32",
          flush=True)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path), tensors=rows,
                source_tensors=len(names), source_elements=sum(math.prod(tensors[n]["shape"]) for n in names))


def gguf_str(reader, key):
    field = reader.fields.get(key)
    return field.contents() if field else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="MuLaCover checkpoint directory")
    parser.add_argument("--codec", type=Path, required=True, help="the HeartCodec-oss checkpoint MuLaCover pins")
    parser.add_argument("--tokenizer", type=Path, required=True, help="installed heartmula-tokenizer.gguf")
    parser.add_argument("--rvq", type=Path, required=True, help="installed heartcodec-rvq-F32.gguf")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.out)
    manifest_path = args.out / "mulacover-conversion-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    tensors = read_headers(args.src)
    count = sum(math.prod(t["shape"]) for t in tensors.values())
    if count != TOTAL:
        raise ValueError(f"Unexpected checkpoint element count: {count}")

    # Tokenizer: same tokenizer.json, and every id the pack pins agrees with MuLaCover's gen_config.
    tok = gguf.GGUFReader(args.tokenizer)
    pack = json.loads(gguf_str(tok, "heartmula.tokenizer.source_sha256"))
    if sha256(args.src / "tokenizer.json") != pack["tokenizer.json"]:
        raise ValueError("MuLaCover tokenizer.json differs from the installed HeartMuLa tokenizer")
    cover_gen_config = (args.src / "gen_config.json").read_text(encoding="utf-8")
    gen = json.loads(cover_gen_config)
    for key, field in (("text_bos_id", "text_bos"), ("text_eos_id", "text_eos"), ("audio_eos_id", "audio_eos"),
                       ("empty_id", "empty")):
        if gen[key] != gguf_str(tok, "heartmula.token." + field):
            raise ValueError(f"gen_config {key} differs from the installed HeartMuLa tokenizer")

    # RVQ: converted from the HeartCodec payload MuLaCover pins.
    rvq = gguf.GGUFReader(args.rvq)
    codec = {n: i["sha256"] for n, i in json.loads(gguf_str(rvq, "heartmula.source")).items()
             if n.endswith(".safetensors")}
    pinned = {n: i["sha256"] for n, i in source_identity(args.codec).items() if n.endswith(".safetensors")}
    if codec != pinned:
        raise ValueError("The installed RVQ was converted from a different HeartCodec checkpoint")

    dependencies = dict(tokenizer=pack, rvq_sha256=sha256(args.rvq))
    identity = source_identity(args.src)
    manifest = dict(version=VERSION, converter_sha256=sha256(__file__), variant=VARIANT, dependencies=dependencies,
                    source_tensors=len(tensors), source_elements=count, components=[])
    for part in ("global", "local"):
        manifest["components"].append(convert(args.src, tensors, part, identity, dependencies, cover_gen_config,
                                              args.out))
    written = sum(c["source_tensors"] for c in manifest["components"])
    if written != len(tensors):
        raise ValueError(f"Source accounting differs: {written} of {len(tensors)}")
    write_json(manifest_path, manifest)
    print(f"PASS manifest {manifest_path}: {len(tensors)} source tensors, {count} elements, none dropped", flush=True)


if __name__ == "__main__":
    main()

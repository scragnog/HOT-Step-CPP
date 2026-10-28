#!/usr/bin/env python3
# convert-sheetsage2-refined.py: SheetSage2-Refined (anime-song) -> one overlay GGUF.
#
#   sheetsage2-refined-<type>.gguf     arch "sheetsage2r"
#       dec.*      the fine-tuned decoder, same tensor names and layout as
#                  convert-sheetsage2.py's dec.*, vocabulary 31678 -> 32576
#       bar.proj.* BarFeatures projection (43 bar facts -> 512, added to memory)
#       hd.<k>.*   the frame heads the decoder needs at inference:
#                    bar0, bar1  bar-v5-f0/f1  (bar features + downbeat/meter rescoring)
#                    chord       chord-v1      (root, bass rescoring)
#                    detail      chord-detail-c3 (quality rescoring)
#                    multi       multi-v2      (key, key-boundary rescoring)
#                  source tensor names kept verbatim after the prefix.
#
# NO ENCODER. Refined keeps SheetSage2's encoder, layer mix and projection
# frozen. The runtime takes enc.* from the base sheetsage2-<type>.gguf
# (convert-sheetsage2.py) and dec.* from this file. Measured against the
# refined repo's fp32 sheetsage2/model.safetensors, the shipped base GGUF's
# encoder sits at rel-L2 ~1.7e-3 on every tensor checked (the base was built
# from a bf16 repack; mel buffers are exact), i.e. bf16 rounding of the same
# weights.
#
# Precision: decoder 2-D matmul weights follow --outtype (default f16), as the
# base converter. Everything else, and every head tensor, is F32: the heads
# are small (GRUs over 25 Hz frames) and feed rescoring margins directly.
#
# Constants below are transcribed from the reference package (MIT,
# github.com/anime-song/SheetSage2-Refined): vocab.py block table, heads.py,
# transcribe.py frame-table spans, engine.py rescoring terms, and the model
# repo's config.json (weights, mode bias). Never imported or executed.
#
# DEPENDENCIES: numpy, gguf, safetensors (numpy loader only, no torch).
#
# USAGE
#   python engine/tools/convert-sheetsage2-refined.py \
#       --src <snapshot of anime-song/SheetSage2-Refined> --out models/yue2

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import gguf
from safetensors.numpy import load_file

CONVERTER_VERSION = 1
ARCH = "sheetsage2r"

REPO = "anime-song/SheetSage2-Refined"
REVISION = "e36ce4fc3c355e7a4e740ef1797275107b7ea257"
CODE_REPO = "https://github.com/anime-song/SheetSage2-Refined"
CODE_REVISION = "c9e5d7c62863f21cdeed4d3e8fded7aac962b909"
MODEL_NAME = "SheetSage2-Refined 0.5.0"

# Pinned sha256 of every source file this converter reads.
SOURCES = {
    "config.json": "2bc2768846904f438412421cd8668dc260ed04c9c45125c56409a8f129e07342",
    "decoder.safetensors": "ac7c563bc84e59c5a4fdeae45767464085e1fd46ed5b1ba78fc333c32953594a",
    "bar_features.safetensors": "4a6fc26842fc72a69c70aee7d382ef05a820213df4081b757052ba13529d0fa0",
    "heads/bar-v5-f0.safetensors": "6da8544dc568c11af985ff60e2f5f323de69a858706fa2ba2592464d316376d8",
    "heads/bar-v5-f1.safetensors": "680b50a92897e61a1e0e18d683e47977c0041120147d73dfa10865f4fe2f8bd0",
    "heads/chord-v1.safetensors": "a7a7003789b7f8e56bde74018f511fd1701565190536ef9a40bcac59458ed5a1",
    "heads/chord-detail-c3.safetensors": "301048eaedcafa65d871cc822dedbaa29efddd9d7fae59c35349b0b91934c60a",
    "heads/multi-v2.safetensors": "c454334bcb6d9d338e0f19c007224e01c645e298ed6dae995f86ac0390de1d8a",
}
HEADS = (
    ("bar0", "heads/bar-v5-f0.safetensors"),
    ("bar1", "heads/bar-v5-f1.safetensors"),
    ("chord", "heads/chord-v1.safetensors"),
    ("detail", "heads/chord-detail-c3.safetensors"),
    ("multi", "heads/multi-v2.safetensors"),
)

# -- decoder: identical architecture to SheetSage2 (convert-sheetsage2.py) ---
DEC_DIM, DEC_FF, DEC_BLOCKS, DEC_MAX_POSITIONS = 512, 2048, 6, 5122
BASE_VOCAB_SIZE = 31678
BASE_VOCAB_FINGERPRINT = "5ba3325af0344c7f"   # the base GGUF this overlays

# -- vocabulary: vocab.py block table; blocks after `duration` are appended --
MODES = ("major", "dorian", "phrygian", "lydian", "mixolydian", "minor",
         "locrian", "harmonic_minor", "phrygian_dominant")
QUALITIES = (
    "5", "", "-5", "m", "dim", "aug", "sus2", "sus4", "6", "7", "M7", "m6", "m7",
    "mM7", "7-5", "m7-5", "aug7", "augM7", "7sus4", "dim7", "add9", "madd9", "69",
    "7(9)", "7(13)", "7(b9)", "7(#9)", "7(#11)", "7(b13)", "7-5(b13)", "M7(9)",
    "M7(b9)", "M7(13)", "M7(#11)", "m69", "m7(9)", "m7(11)", "m7(13)", "m7(b9)",
    "mM7(9)", "mM7(13)", "aug7(9)", "augM7(#9)", "add9(#11)", "m7-5(b9)",
    "7(9,13)", "7(9,b13)", "7(9,#11)", "7(b9,#9)", "7(b9,13)", "7(b9,b13)",
    "7(b9,#11)", "7(#9,13)", "7(#9,b13)", "7(#11,13)", "m7(9,11)", "m7(9,13)",
    "M7(9,13)", "M7(9,#11)", "7(9,#11,13)", "7(9,#11,b13)", "M7(9,#11,13)",
)
assert len(QUALITIES) == 62
VOCAB_BLOCKS = (
    ("special", 4), ("prompt", 256), ("shift", 257), ("time", 30000),
    ("meter", 192), ("eighth_position", 256), ("structure", 23), ("key", 24),
    ("chord_majmin", 25), ("chord_full", 361), ("pitch", 256), ("duration", 24),
    ("chord_bass", 12),
    ("chord_detail", 12 * len(QUALITIES) + 1),   # 12*quality + root, N last
    ("key_mode", 12 * len(MODES) + 1),           # 12*mode + tonic, N last
    ("swing", 32),                               # reserved, never generated
)
VOCAB_SIZE = sum(n for _, n in VOCAB_BLOCKS)
assert VOCAB_SIZE == 32576
assert sum(n for _, n in VOCAB_BLOCKS[:12]) == BASE_VOCAB_SIZE
# vocab.py PROMPTS: base 8 plus two appended (ids 4..13)
PROMPT_NAMES = ("timestamp", "downbeat_meter", "structure", "key", "chord_majmin",
                "chord_full", "melody_vocal", "melody_full", "chord_detail", "key_mode")
DEFAULT_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key",
                   "chord_detail", "melody_full")   # config.json "prompts"

# -- inference constants (heads.py / transcribe.py) --------------------------
HEAD_CHUNK = (1500, 375)          # GRU core frames, context either side
BAR_FEATURES = 43                 # bar_beats 16, beats_left 16, bar_groups 8, beat/downbeat/group 3
# frame_table(): frames averaged ahead of "now", and max-pool half-widths
FRAME_LOOKAHEAD = {"root": 12, "bass": 12, "quality": 12, "key": 50, "meter": 8}
FRAME_REACH = {"key_boundary": 25, "downbeat": 2}
RESCORE_HEAD = {"downbeat": "bar", "meter": "bar", "root": "chord", "bass": "chord",
                "quality": "detail", "key": "multi", "key_boundary": "multi"}

LICENSE_ATTRIBUTION = (
    "Weights are CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 "
    "International): non-commercial use only. Fine-tuned decoder and frame heads: "
    "anime-song/SheetSage2-Refined (https://huggingface.co/anime-song/SheetSage2-Refined, "
    "inference code MIT at https://github.com/anime-song/SheetSage2-Refined). "
    "Built on m-a-p/SheetSage2 (https://huggingface.co/m-a-p/SheetSage2) and "
    "m-a-p/MERT-v2-FullSong (https://huggingface.co/m-a-p/MERT-v2-FullSong)."
)

F16_MAX = 65504.0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def die(msg):
    log(f"error: {msg}")
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Source:
    """One safetensors file; every tensor must be consumed exactly once."""

    def __init__(self, path):
        self.path = path
        self.t = load_file(path)
        self.used = set()

    def get(self, name, expect=None):
        if name not in self.t:
            die(f"{os.path.basename(self.path)}: missing {name}")
        a = self.t[name]
        if a.dtype != np.float32:
            die(f"{name}: dtype {a.dtype}, expected float32")
        if expect is not None and tuple(a.shape) != tuple(expect):
            die(f"{name}: shape {tuple(a.shape)}, expected {tuple(expect)}")
        self.used.add(name)
        return a

    def check_consumed(self):
        left = sorted(set(self.t) - self.used)
        if left:
            die(f"{os.path.basename(self.path)}: unconsumed tensors {left[:8]}")


def build_decoder(src, out):
    def put(name, key, shape, narrow=False):
        out.append((name, src.get(key, expect=shape), narrow))

    put("dec.tok_embd", "token_embedding.weight", (VOCAB_SIZE, DEC_DIM), True)
    put("dec.pos_embd", "position_embedding.weight", (DEC_MAX_POSITIONS, DEC_DIM), True)
    put("dec.norm_embd.weight", "embedding_norm.weight", (DEC_DIM,))
    put("dec.norm_embd.bias", "embedding_norm.bias", (DEC_DIM,))
    for m in range(DEC_BLOCKS):
        s, d = f"layers.{m}", f"dec.blk.{m}"
        for part in ("self_attn", "cross_attn"):
            put(f"{d}.{part}_norm.weight", f"{s}.{part}_norm.weight", (DEC_DIM,))
            put(f"{d}.{part}_norm.bias", f"{s}.{part}_norm.bias", (DEC_DIM,))
            for dst, proj in (("q", "q"), ("k", "k"), ("v", "v"), ("output", "out")):
                put(f"{d}.{part}_{dst}.weight", f"{s}.{part}.{proj}.weight", (DEC_DIM, DEC_DIM), True)
                put(f"{d}.{part}_{dst}.bias", f"{s}.{part}.{proj}.bias", (DEC_DIM,))
        put(f"{d}.ffn_up.weight", f"{s}.fc1.weight", (DEC_FF, DEC_DIM), True)
        put(f"{d}.ffn_up.bias", f"{s}.fc1.bias", (DEC_FF,))
        put(f"{d}.ffn_down.weight", f"{s}.fc2.weight", (DEC_DIM, DEC_FF), True)
        put(f"{d}.ffn_down.bias", f"{s}.fc2.bias", (DEC_DIM,))
        put(f"{d}.final_norm.weight", f"{s}.final_norm.weight", (DEC_DIM,))
        put(f"{d}.final_norm.bias", f"{s}.final_norm.bias", (DEC_DIM,))
    src.check_consumed()


def head_info(t):
    """The FrameHead shape facts heads.load_head() infers from a state dict."""

    hidden = t["inp.weight"].shape[0]
    mixes = t["mix"].shape[0] if t["mix"].ndim == 2 else 1
    layers = sum(1 for k in t if k.startswith("rnn.weight_ih_l") and not k.endswith("_reverse"))
    binary = t["binary.weight"].shape[0] if "binary.weight" in t else 0
    for k in ("distill.weight", "offset.weight", "unit.weight", "melody.weight",
              "line.weight", "extra.weight"):
        if k in t:
            die(f"head carries {k}, which this port does not run")
    if t["inp.weight"].shape[1] != 1024 * mixes:
        die(f"inp.weight {t['inp.weight'].shape} does not match {mixes} mixes of 1024")
    if t["convs.0.weight"].shape != (hidden, 32, 9):
        die(f"convs.0.weight {t['convs.0.weight'].shape}, expected ({hidden}, 32, 9)")
    return hidden, mixes, layers, binary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="snapshot dir of " + REPO)
    ap.add_argument("--out", default="models/yue2")
    ap.add_argument("--outtype", choices=("f16", "f32"), default="f16")
    args = ap.parse_args()

    for rel, want in SOURCES.items():
        got = sha256_file(os.path.join(args.src, rel))
        if got != want:
            die(f"{rel}: sha256 {got} does not match the pinned {want} (revision {REVISION})")
    log(f"verified sha256 of {len(SOURCES)} source files")

    config = json.load(open(os.path.join(args.src, "config.json"), encoding="utf-8"))
    if tuple(config["prompts"]) != DEFAULT_PROMPTS:
        die(f"config.json prompts {config['prompts']} != {DEFAULT_PROMPTS}")
    rescoring = config["rescoring"]
    if set(rescoring) != set(RESCORE_HEAD):
        die(f"config.json rescoring terms {sorted(rescoring)} != {sorted(RESCORE_HEAD)}")

    tensors = []   # (name, array, narrow)
    build_decoder(Source(os.path.join(args.src, "decoder.safetensors")), tensors)

    bar = Source(os.path.join(args.src, "bar_features.safetensors"))
    tensors.append(("bar.proj.weight", bar.get("projection.weight", (DEC_DIM, BAR_FEATURES)), False))
    tensors.append(("bar.proj.bias", bar.get("projection.bias", (DEC_DIM,)), False))
    bar.check_consumed()

    infos = {}
    for key, rel in HEADS:
        t = load_file(os.path.join(args.src, rel))
        infos[key] = head_info(t)
        for name in sorted(t):
            tensors.append((f"hd.{key}.{name}", t[name].astype(np.float32), False))
        log(f"  head {key:6s} {rel}: hidden={infos[key][0]} mixes={infos[key][1]} "
            f"gru_layers={infos[key][2]} binary={infos[key][3]} tensors={len(t)}")
    for key in ("bar0", "bar1"):
        if infos[key][3] != 5 or not any(n == f"hd.{key}.bar.weight" for n, _, _ in tensors):
            die(f"{key}: expected a bar head with 5 binary outputs")
    if not any(n == "hd.detail.detail.weight" for n, _, _ in tensors):
        die("detail head has no detail output")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"sheetsage2-refined-{args.outtype}.gguf")
    w = gguf.GGUFWriter(path, ARCH)
    w.add_name(MODEL_NAME)
    w.add_string(f"{ARCH}.license", "CC BY-NC 4.0")
    w.add_string(f"{ARCH}.license_attribution", LICENSE_ATTRIBUTION)
    w.add_uint32(f"{ARCH}.converter_version", CONVERTER_VERSION)
    w.add_string(f"{ARCH}.pinned.repo", REPO)
    w.add_string(f"{ARCH}.pinned.revision", REVISION)
    w.add_string(f"{ARCH}.pinned.code_repo", CODE_REPO)
    w.add_string(f"{ARCH}.pinned.code_revision", CODE_REVISION)
    w.add_string(f"{ARCH}.pinned.config_name", config["name"])
    for rel, sha in SOURCES.items():
        w.add_string(f"{ARCH}.pinned.sha256.{rel}", sha)
    w.add_string(f"{ARCH}.base.vocab_fingerprint", BASE_VOCAB_FINGERPRINT)
    w.add_uint32(f"{ARCH}.base.vocab_size", BASE_VOCAB_SIZE)
    w.add_string(f"{ARCH}.base.note", "encoder, layer mix and projection come from the base sheetsage2 GGUF")

    w.add_uint32(f"{ARCH}.vocab_size", VOCAB_SIZE)
    start = 0
    for name, n in VOCAB_BLOCKS:
        w.add_uint32(f"{ARCH}.vocab.block.{name}.start", start)
        w.add_uint32(f"{ARCH}.vocab.block.{name}.count", n)
        start += n
    w.add_array(f"{ARCH}.vocab.prompt_names", list(PROMPT_NAMES))
    w.add_array(f"{ARCH}.vocab.chord_qualities", list(QUALITIES))
    w.add_array(f"{ARCH}.vocab.modes", list(MODES))
    w.add_array(f"{ARCH}.generation.default_prompts", list(DEFAULT_PROMPTS))
    w.add_bool(f"{ARCH}.window.pad_encoder", False)   # model.py: fine-tuned decoders see unpadded windows

    w.add_uint32(f"{ARCH}.heads.chunk_core", HEAD_CHUNK[0])
    w.add_uint32(f"{ARCH}.heads.chunk_context", HEAD_CHUNK[1])
    w.add_uint32(f"{ARCH}.bar.feature_count", BAR_FEATURES)
    for key, (hidden, mixes, layers, binary) in infos.items():
        w.add_uint32(f"{ARCH}.head.{key}.hidden", hidden)
        w.add_uint32(f"{ARCH}.head.{key}.mixes", mixes)
        w.add_uint32(f"{ARCH}.head.{key}.gru_layers", layers)
        w.add_uint32(f"{ARCH}.head.{key}.binary", binary)
    for term, head in RESCORE_HEAD.items():
        w.add_float32(f"{ARCH}.rescore.{term}.weight", float(rescoring[term]["weight"]))
        w.add_string(f"{ARCH}.rescore.{term}.head", head)
    for term, n in FRAME_LOOKAHEAD.items():
        w.add_uint32(f"{ARCH}.rescore.{term}.lookahead", n)
    for term, n in FRAME_REACH.items():
        w.add_uint32(f"{ARCH}.rescore.{term}.reach", n)
    w.add_float32(f"{ARCH}.key_mode.tau", float(config["mode_tau"]))
    counts = [float(c) for c in config["mode_counts"]]
    if len(counts) != len(MODES):
        die(f"mode_counts has {len(counts)} entries, expected {len(MODES)}")
    w.add_array(f"{ARCH}.key_mode.counts", counts)

    nf16 = 0
    for name, a, narrow in tensors:
        a = np.ascontiguousarray(a, dtype=np.float32)
        if narrow and args.outtype == "f16" and float(np.abs(a).max()) <= F16_MAX:
            w.add_tensor(name, a.astype(np.float16).view(np.uint16),
                         raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.F16)
            nf16 += 1
        else:
            w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size = os.path.getsize(path)
    log(f"wrote {path}: {len(tensors)} tensors ({nf16} F16), {size:,} bytes")

    # Round trip: every tensor reads back with the written shape and values.
    r = gguf.GGUFReader(path)
    back = {t.name: t for t in r.tensors}
    for name, a, narrow in tensors:
        t = back[name]
        got = np.array(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            got = got.view(np.float16)
        got = got.astype(np.float32).reshape(a.shape)
        tol = 1e-3 if t.tensor_type == gguf.GGMLQuantizationType.F16 else 0.0
        rel = float(np.linalg.norm(got - a) / max(np.linalg.norm(a), 1e-30))
        if rel > tol:
            die(f"round trip {name}: rel-L2 {rel:.3e} > {tol}")
    log(f"round trip ok: {len(tensors)} tensors")


if __name__ == "__main__":
    main()

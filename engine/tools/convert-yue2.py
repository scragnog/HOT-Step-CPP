#!/usr/bin/env python3
# convert-yue2.py: YuE2 safetensors -> GGUF for the HOT-Step engine.
#
# Produces up to three files into --out:
#
#   yue2-lm-<type>.gguf                arch "yue2"       AR+NAR backbone, embed/head,
#                                                          flow-matching heads (vae2llm,
#                                                          llm2vae, time_embd, latent_pos_embed),
#                                                          tokenizer carried as GGUF KV.
#   yue2-vae-standard-<type>.gguf      arch "yue2-vae"   decoder (dec.*) + encoder (enc.*)
#   yue2-vae-legacy-<type>.gguf        arch "yue2-vae"   same tensor layout, different weights
#
# This is the working implementation of the contract in
# docs/plans/yue2/05-gguf-layout.md -- read that first; every naming/policy
# decision below has its reasoning written out there, not repeated here.
#
# Companion docs consulted while writing this: 00-oracle-pin.md (runtime pin),
# 03-reference-numerics.md (exact math/dtype boundaries), 04-weights-inventory.md/.json
# (tensor names/shapes read from the real checkpoint headers).
#
# SOURCES (single-checkpoint-per-component; no multi-file alias resolution needed,
# unlike convert-mm3.py, because YuE2 ships exactly one model.safetensors per model dir):
#   --src-lm   K:/yue2/models/YuE2-3B          (config.json, generation_config.json,
#                                                yue2_generation_config.json, model.safetensors)
#   --src-vae         K:/yue2/models/YuE2-Vae         (release_variant: "standard")
#   --src-vae-legacy  K:/yue2/models/YuE2-Vae-legacy  (release_variant: "legacy")
#   --src-vae-rs-x3   YuE2-VAE-RS-x3_v1.safetensors   (Ruach Studio's decoder, goldhub/YuE2-VAE-RS-x3;
#                                                       needs --src-vae for Standard's encoder and config)
#
# DEPENDENCIES: numpy + gguf (llama.cpp gguf-py). No torch needed -- BF16/F32 are
# read directly off the safetensors header+mmap, same small reader convert-mm3.py uses.
#
# USAGE
#   python engine/tools/convert-yue2.py \
#       --src-lm K:/yue2/models/YuE2-3B \
#       --src-vae K:/yue2/models/YuE2-Vae \
#       --src-vae-legacy K:/yue2/models/YuE2-Vae-legacy \
#       --tokenizer-dir models/yue2/tokenizer \
#       --out models/yue2 --components all

import argparse
import json
import mmap
import os
import re
import struct
import subprocess
import sys

import numpy as np
import gguf

CONVERTER_VERSION = 1

# ---------------------------------------------------------------------------
# Model constants. Hardcoded on purpose (docs/plans/yue2/05-gguf-layout.md §8):
# a converter that only reads shapes from the checkpoint cannot reject a wrong
# one. Every one of these was read from K:/yue2/models/YuE2-3B/config.json and
# cross-checked against the safetensors header in 04-weights-inventory.md.
# ---------------------------------------------------------------------------

LM_LAYERS = 28
LM_DIM = 2048
LM_FF = 6144
LM_HEADS = 16
LM_KV_HEADS = 8
LM_HEAD_DIM = 128
LM_VOCAB = 184704
LM_CTX = 24576
LM_RMS_EPS = 1e-6
LM_ROPE_THETA = 1000000.0
LM_LATENT_DIM = 64
LM_MAX_LATENT_FRAMES = 24576
LM_TIMESTEP_SHIFT = 1.0
LM_FREQ_EMBED_SIZE = 256          # TimestepEmbedder's frequency_embedding_size

# -- special/structural token ids (protocol.py, 03-reference-numerics.md §1.1) --
TOK = {
    "eod": 151643,
    "abc_start": 151847, "abc_end": 151848,
    "music_start": 151851, "music_end": 151852,
    "codec_offset": 151853, "codec_size": 32768,
    "latent_start": 184621, "latent_end": 184622, "latent_pad": 184623,
}

# -- generation defaults (protocol.py dataclass defaults -- the ACTUAL runtime
# source, not yue2_generation_config.json; see 03-reference-numerics.md §2.3's
# provenance correction. Values happen to agree with that JSON for this
# checkpoint, so --src-lm's copy is used as the read source below, but this is
# what a loader is really relying on.) ---------------------------------------
SAMPLING_DEFAULTS = {
    "abc": {"temperature": 0.7, "top_p": 0.9, "top_k": 30,
            "repetition_penalty": 1.005, "penalty_window": 100,
            "min_tokens": 32, "max_tokens": 4096},
    "semantic": {"temperature": 1.0, "top_p": 0.95, "top_k": 100,
                 "repetition_penalty": 1.2, "penalty_window": 50,
                 "min_tokens": 200, "max_tokens": 9000},
}
ODE_STEPS = 32
ODE_METHOD = "midpoint"

# -- VAE (both variants share every one of these; only weight VALUES differ) --
VAE_SAMPLE_RATE = 48000
VAE_DOWNSAMPLE_RATIO = 1920
VAE_AUDIO_CHANNELS = 2
VAE_LATENT_DIM = 64                 # decoder input width
VAE_ENCODER_LATENT_DIM = 128        # encoder output width (mean||scale, pre-chunk)
VAE_BASE_CHANNELS = 64
VAE_STRIDES = (2, 2, 4, 4, 5, 6)     # encoder forward order; decoder runs the reverse
VAE_RES_DILATIONS = (1, 3, 9)
VAE_SNAKE_EPS = 1e-9
VAE_FINAL_TANH = False
VAE_DECODE_CORE_FRAMES = 1024
VAE_DECODE_HALO_FRAMES = 16
VAE_REQUIRED_HALO = 12               # computed dependency-interval minimum, 03-ref §4.1
VAE_OUTPUT_CLAMP = (-1.0, 1.0)       # pipeline-level post-process, not a decoder property
# rs-x3 (goldhub/YuE2-VAE-RS-x3, docs/VAE-RS-x3-integration.md): 3 latent-rate units,
# 6 units per block, and a 60-frame receptive field each side (its exact halo).
RS_X3_PRE_DILATIONS = (1, 3, 9)
RS_X3_RES_UNITS = 6
RS_X3_HALO = 60
RS_X3_ATTRIBUTION = (
    "Decoder: YuE2-VAE-RS x3 by Viktor Zhuromskyy / Ruach Studio "
    "(https://huggingface.co/goldhub/YuE2-VAE-RS-x3, DOI 10.5281/zenodo.24167962), an adaptation "
    "of m-a-p's YuE2-Vae, CC BY-NC 4.0. Encoder: m-a-p's YuE2-Vae (standard), unchanged. "
)

# -- licence (K:/yue2/models/*/LICENSE, all three checkpoints) ---------------
LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_ATTRIBUTION = (
    "Weights are CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 "
    "International). Attribution: identify YuE2, the model name, and its source "
    "repository -- https://huggingface.co/m-a-p/YuE2-3B, "
    "https://huggingface.co/m-a-p/YuE2-Vae, https://huggingface.co/m-a-p/YuE2-Vae-legacy"
)

# -- pinned provenance (00-oracle-pin.md §4). Verified against the local
# weights_manifest.json sha256 before being trusted -- see verify_pin(). -----
PINS = {
    "lm": {
        "repo": "m-a-p/YuE2-3B",
        "revision": "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
        "sha256": "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
    },
    "standard": {
        "repo": "m-a-p/YuE2-Vae",
        "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
        "sha256": "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346",
    },
    "legacy": {
        "repo": "m-a-p/YuE2-Vae-legacy",
        "revision": "b54118f0fc462f08999d1ec07e88817f4ee3f770",
        "sha256": "b6d283628913bb41145ba99e2314eef613905ee95f690eb70e8212d5f4965044",
    },
}
ORACLE_RUNTIME = "yue2-infer 0.1.5 + local cuda_graph.py FlashAttention-capability patch"


def log(msg):
    print(f"[convert-yue2] {msg}", file=sys.stderr, flush=True)


def die(msg):
    raise SystemExit(f"[convert-yue2] ERROR: {msg}")


def expect_config(cfg, key, want, label):
    got = cfg.get(key)
    if got != want:
        die(f"{label}: config.json {key}={got!r}, expected {want!r} -- this does not "
            f"look like the YuE2 checkpoint this converter understands")


# ---------------------------------------------------------------------------
# safetensors reader -- identical approach to convert-mm3.py's SafeTensorsFile:
# a small built-in mmap reader so BF16 works without a torch dependency.
# ---------------------------------------------------------------------------

_ST_DTYPES = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}


class SafeTensorsFile:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n).decode("utf-8"))
        self.metadata = self.header.pop("__metadata__", {})
        self._base = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        for name, spec in self.header.items():
            if spec["dtype"] not in _ST_DTYPES:
                die(f"{path}: tensor {name} has unsupported dtype {spec['dtype']}")

    def keys(self):
        return self.header.keys()

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def raw(self, name):
        spec = self.header[name]
        start, end = spec["data_offsets"]
        return self._mm[self._base + start:self._base + end]

    def raw_bf16_u16(self, name):
        """Tensor's raw bytes as a uint16 array, no widening. Requires BF16 storage."""
        if self.dtype(name) != "BF16":
            die(f"{self.path}: {name} is {self.dtype(name)}, expected BF16 for a verbatim copy")
        return np.frombuffer(self.raw(name), dtype=np.uint16).reshape(self.shape(name))

    def get(self, name):
        """Returns a float32 numpy array (widened losslessly from the native dtype)."""
        spec = self.header[name]
        buf = self.raw(name)
        dt = spec["dtype"]
        shape = tuple(spec["shape"])
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        np_dt = {"F64": np.float64, "F32": np.float32, "F16": np.float16}[dt]
        arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
        return arr.astype(np.float32)

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


class Source:
    """Thin wrapper: one safetensors file, a consumed-set, and expect-checked get()."""

    def __init__(self, path):
        self.path = path
        self.file = SafeTensorsFile(path)
        self.consumed = set()

    def has(self, name):
        return name in self.file.header

    def get(self, name, expect=None, label=None):
        if name not in self.file.header:
            die(f"missing tensor {label or name} in {self.path}")
        arr = self.file.get(name)
        self.consumed.add(name)
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has {tuple(arr.shape)} "
                f"in {self.path} -- not the YuE2 checkpoint this converter understands")
        return arr

    def get_bf16_verbatim(self, name, expect=None):
        if name not in self.file.header:
            die(f"missing tensor {name} in {self.path}")
        u16 = self.file.raw_bf16_u16(name)
        self.consumed.add(name)
        if expect is not None and tuple(u16.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has {tuple(u16.shape)}")
        return u16

    def unconsumed(self, predicate):
        return sorted(n for n in self.file.header if predicate(n) and n not in self.consumed)

    def close(self):
        self.file.close()


class RsX3Source(Source):
    """Ruach Studio's YuE2-VAE-RS x3 decoder file (goldhub/YuE2-VAE-RS-x3), read
    under m-a-p's decoder.* names so build_vae binds it like any other decoder.

    The file keeps its own module names (conv1, pre.<i>, blocks.<b>.units.<j>,
    snake1, conv2) with weight norm stored as parametrizations original0 (g) /
    original1 (v). Same unit form as Standard: 3 latent-rate units after conv_in
    (decoder.pre.<i>, ours) and 3 more per block (layers 5..7 of each block)."""

    UNIT = ("snake1", "conv1", "snake2", "conv2")

    def __init__(self, path):
        super().__init__(path)
        meta = self.file.metadata
        cfg = json.loads(meta.get("config", "{}"))
        if (meta.get("variant") != "x3" or cfg.get("pre") != 3 or cfg.get("pre_dil") != list(RS_X3_PRE_DILATIONS)
                or cfg.get("extra") != [3] * 6 or cfg.get("pre_mult") != 1.0 or cfg.get("mult") != [1.0] * 6):
            die(f"{path}: not a YuE2-VAE-RS x3 decoder (metadata variant={meta.get('variant')!r} config={cfg})")
        self.step = meta.get("step", "unknown")

    def raw_name(self, name):
        m = re.fullmatch(r"decoder\.(.+)\.(weight_g|weight_v|bias|alpha|beta)", name)
        if not m:
            die(f"RsX3Source: unexpected name {name}")
        path, leaf = m.groups()
        leaf = {"weight_g": "parametrizations.weight.original0",
                "weight_v": "parametrizations.weight.original1"}.get(leaf, leaf)
        fixed = {"layers.0": "conv1", "layers.7": "snake1", "layers.8": "conv2"}
        if path in fixed:
            return f"{fixed[path]}.{leaf}"
        p = re.fullmatch(r"pre\.(\d+)\.layers\.([0-3])", path)
        if p:
            return f"pre.{p[1]}.{self.UNIT[int(p[2])]}.{leaf}"
        p = re.fullmatch(r"layers\.([1-6])\.layers\.([01])", path)
        if p:
            return f"blocks.{int(p[1]) - 1}.{('snake1', 'conv_t1')[int(p[2])]}.{leaf}"
        p = re.fullmatch(r"layers\.([1-6])\.layers\.(\d+)\.layers\.([0-3])", path)
        if p:
            return f"blocks.{int(p[1]) - 1}.units.{int(p[2]) - 2}.{self.UNIT[int(p[3])]}.{leaf}"
        die(f"RsX3Source: no mapping for {name}")

    def get(self, name, expect=None, label=None):
        return super().get(self.raw_name(name), expect, label or name)


# ---------------------------------------------------------------------------
# Tensor policy + GGUF writer
# ---------------------------------------------------------------------------

F32 = "f32"                # always F32: norms, biases, alpha/beta, scalars
NATIVE = "native"           # follows --type (bf16/f16/f32), F16-range-guarded
NATIVE_BF16_VERBATIM = "native_bf16_verbatim"   # latent_pos_embed only -- raw bytes, ignores --type

F16_MAX = 65504.0


def to_bf16_rne(arr):
    """F32 -> BF16 bytes, round-to-nearest-even. Ported verbatim from convert-mm3.py.

    Bit-exact for values that started life as BF16 and were losslessly widened
    to F32 by SafeTensorsFile.get() (the low 16 mantissa bits are already zero,
    so the +0x7FFF rounding term never carries past bit 16) -- and standard RNE
    for any other F32 input (e.g. a VAE weight-norm fold result narrowed under
    a non-default --type).
    """
    u = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


class Bundle:
    def __init__(self, arch):
        self.arch = arch
        self.kv = []
        self.tensors = []   # (name, array_or_rawu16, policy)

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr, policy):
        if name in {n for n, _, _ in self.tensors}:
            die(f"duplicate output tensor {name}")
        self.tensors.append((name, arr, policy))

    def write(self, path, type_):
        w = gguf.GGUFWriter(path, self.arch)
        for fn, args in self.kv:
            getattr(w, fn)(*args)
        n = {"f32": 0, "f16": 0, "bf16": 0}
        for name, arr, policy in self.tensors:
            if policy == NATIVE_BF16_VERBATIM:
                w.add_tensor(name, arr, raw_shape=arr.shape,
                             raw_dtype=gguf.GGMLQuantizationType.BF16)
                n["bf16"] += 1
                continue
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if policy == F32 or arr.ndim < 2:
                w.add_tensor(name, arr)
                n["f32"] += 1
                continue
            # NATIVE: follow --type.
            if type_ == "f32":
                w.add_tensor(name, arr)
                n["f32"] += 1
            elif type_ == "bf16":
                w.add_tensor(name, to_bf16_rne(arr), raw_shape=arr.shape,
                             raw_dtype=gguf.GGMLQuantizationType.BF16)
                n["bf16"] += 1
            elif type_ == "f16":
                peak = float(np.abs(arr).max()) if arr.size else 0.0
                if peak > F16_MAX:
                    log(f"  {name}: |w|max={peak:.1f} exceeds f16 range, storing F32 instead")
                    w.add_tensor(name, arr)
                    n["f32"] += 1
                else:
                    w.add_tensor(name, arr.astype(np.float16).view(np.uint16),
                                 raw_shape=arr.shape, raw_dtype=gguf.GGMLQuantizationType.F16)
                    n["f16"] += 1
            else:
                die(f"unknown --type {type_!r}")
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        size = os.path.getsize(path) / 1e9
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors "
            f"({n['f32']} F32, {n['f16']} F16, {n['bf16']} BF16), {size:.2f} GB")


def fold_weight_norm(g, v, label):
    """PyTorch weight_norm (dim=0): w = g * v / ||v||, norm over all axes but 0.

    Ported verbatim from convert-mm3.py's fold_weight_norm -- same formula,
    same float64 accumulation, same shape-mismatch/zero-norm guards. Holds for
    both Conv1d (out,in,k) and ConvTranspose1d (in,out,k) weights."""
    if g.shape[0] != v.shape[0]:
        die(f"{label}: weight_g first dim {g.shape[0]} != weight_v {v.shape[0]}")
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.all(norm > 0):
        die(f"{label}: weight_v has a zero-norm slice; refusing to fold")
    g = g.astype(np.float64).reshape((g.shape[0],) + (1,) * (v.ndim - 1))
    return (g * v.astype(np.float64) / norm).astype(np.float32)


def sha256_matches_pin(manifest_path, pin_sha256):
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        got = manifest["files"]["model.safetensors"]["sha256"]
    except (KeyError, ValueError, OSError):
        return None
    return got == pin_sha256


# ---------------------------------------------------------------------------
# Component: LM  ->  arch "yue2"
# ---------------------------------------------------------------------------

def common_meta(b, type_, attribution=LICENSE_ATTRIBUTION):
    b.meta("add_license", LICENSE_NAME)
    b.meta("add_string", "yue2.license_attribution", attribution)
    b.meta("add_uint32", "yue2.converter_version", CONVERTER_VERSION)
    type_map = {"bf16": gguf.LlamaFileType.MOSTLY_BF16,
                "f16": gguf.LlamaFileType.MOSTLY_F16,
                "f32": gguf.LlamaFileType.ALL_F32}
    b.meta("add_file_type", int(type_map[type_]))


def build_lm(src_dir, bundle, type_, tok_dir):
    cfg_path = os.path.join(src_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    expect_config(cfg, "hidden_size", LM_DIM, "LM")
    expect_config(cfg, "num_hidden_layers", LM_LAYERS, "LM")
    expect_config(cfg, "num_attention_heads", LM_HEADS, "LM")
    expect_config(cfg, "num_key_value_heads", LM_KV_HEADS, "LM")
    expect_config(cfg, "head_dim", LM_HEAD_DIM, "LM")
    expect_config(cfg, "intermediate_size", LM_FF, "LM")
    expect_config(cfg, "vocab_size", LM_VOCAB, "LM")
    expect_config(cfg, "max_position_embeddings", LM_CTX, "LM")
    expect_config(cfg, "latent_dim", LM_LATENT_DIM, "LM")
    expect_config(cfg, "max_latent_frames", LM_MAX_LATENT_FRAMES, "LM")
    expect_config(cfg, "tie_word_embeddings", False, "LM")
    if abs(float(cfg.get("rope_theta", 0)) - LM_ROPE_THETA) > 1.0:
        die(f"LM: config.json rope_theta={cfg.get('rope_theta')!r}, expected {LM_ROPE_THETA}")
    if abs(float(cfg.get("rms_norm_eps", 0)) - LM_RMS_EPS) > 1e-12:
        die(f"LM: config.json rms_norm_eps={cfg.get('rms_norm_eps')!r}, expected {LM_RMS_EPS}")
    timestep_shift = float(cfg.get("timestep_shift", LM_TIMESTEP_SHIFT))

    st_path = os.path.join(src_dir, "model.safetensors")
    src = Source(st_path)

    manifest_path = os.path.join(src_dir, "weights_manifest.json")
    match = sha256_matches_pin(manifest_path, PINS["lm"]["sha256"])
    pin = PINS["lm"] if match else {"repo": "unknown", "revision": "unknown", "sha256": "unknown"}
    if match is False:
        log(f"WARNING: {manifest_path} sha256 does not match the pinned YuE2-3B "
            f"revision -- writing yue2.pinned.* as 'unknown' rather than a wrong pin")
    elif match is None:
        log(f"WARNING: no weights_manifest.json found at {manifest_path} -- "
            f"writing yue2.pinned.* as 'unknown'")

    b = bundle
    b.meta("add_name", "YuE2 LM")
    b.meta("add_description",
           "YuE2 AR+NAR backbone: Mixture-of-Transformers (shared embed/head, "
           "disjoint AR and NAR attention+MLP per layer) plus flow-matching heads "
           "(vae2llm, llm2vae, time_embedder, latent_pos_embed)")
    common_meta(b, type_)

    b.meta("add_uint32", "yue2.context_length", LM_CTX)
    b.meta("add_uint32", "yue2.embedding_length", LM_DIM)
    b.meta("add_uint32", "yue2.block_count", LM_LAYERS)
    b.meta("add_uint32", "yue2.feed_forward_length", LM_FF)
    b.meta("add_uint32", "yue2.attention.head_count", LM_HEADS)
    b.meta("add_uint32", "yue2.attention.head_count_kv", LM_KV_HEADS)
    b.meta("add_uint32", "yue2.attention.key_length", LM_HEAD_DIM)
    b.meta("add_uint32", "yue2.attention.value_length", LM_HEAD_DIM)
    b.meta("add_float32", "yue2.attention.layer_norm_rms_epsilon", LM_RMS_EPS)
    b.meta("add_float32", "yue2.rope.freq_base", LM_ROPE_THETA)
    b.meta("add_uint32", "yue2.vocab_size", LM_VOCAB)
    b.meta("add_uint32", "yue2.latent_dim", LM_LATENT_DIM)
    b.meta("add_uint32", "yue2.max_latent_frames", LM_MAX_LATENT_FRAMES)
    b.meta("add_float32", "yue2.timestep_shift", timestep_shift)
    b.meta("add_float32", "yue2.attention.softmax_scale", float(LM_HEAD_DIM) ** -0.5)

    for k, v in TOK.items():
        b.meta("add_uint32", f"yue2.token.{k}", v)

    for stage, params in SAMPLING_DEFAULTS.items():
        for k, v in params.items():
            fn = "add_float32" if isinstance(v, float) else "add_uint32"
            b.meta(fn, f"yue2.sampling.{stage}.{k}", v)
    b.meta("add_uint32", "yue2.ode.steps", ODE_STEPS)
    b.meta("add_string", "yue2.ode.method", ODE_METHOD)

    b.meta("add_string", "yue2.pinned.lm_repo", pin["repo"])
    b.meta("add_string", "yue2.pinned.lm_revision", pin["revision"])
    b.meta("add_string", "yue2.pinned.lm_sha256", pin["sha256"])
    b.meta("add_string", "yue2.pinned.oracle_runtime", ORACLE_RUNTIME)

    add_tokenizer_kv(b, tok_dir)

    b.put("token_embd.weight", src.get("model.embed_tokens.weight", expect=(LM_VOCAB, LM_DIM)), NATIVE)
    b.put("output_norm.weight", src.get("model.norm.weight", expect=(LM_DIM,)), F32)
    b.put("output.weight", src.get("lm_head.weight", expect=(LM_VOCAB, LM_DIM)), NATIVE)

    q_dim = LM_HEADS * LM_HEAD_DIM
    kv_dim = LM_KV_HEADS * LM_HEAD_DIM
    for i in range(LM_LAYERS):
        p = f"model.layers.{i}"
        d = f"blk.{i}"

        def attn_block(prefix_src, prefix_dst):
            b.put(f"{prefix_dst}_norm.weight",
                  src.get(f"{prefix_src[0]}", expect=(LM_DIM,)), F32)
            b.put(f"{prefix_dst}_q.weight",
                  src.get(f"{prefix_src[1]}.q_proj.weight", expect=(q_dim, LM_DIM)), NATIVE)
            b.put(f"{prefix_dst}_k.weight",
                  src.get(f"{prefix_src[1]}.k_proj.weight", expect=(kv_dim, LM_DIM)), NATIVE)
            b.put(f"{prefix_dst}_v.weight",
                  src.get(f"{prefix_src[1]}.v_proj.weight", expect=(kv_dim, LM_DIM)), NATIVE)
            b.put(f"{prefix_dst}_output.weight",
                  src.get(f"{prefix_src[1]}.o_proj.weight", expect=(LM_DIM, q_dim)), NATIVE)
            b.put(f"{prefix_dst}_q_norm.weight",
                  src.get(f"{prefix_src[1]}.q_norm.weight", expect=(LM_HEAD_DIM,)), F32)
            b.put(f"{prefix_dst}_k_norm.weight",
                  src.get(f"{prefix_src[1]}.k_norm.weight", expect=(LM_HEAD_DIM,)), F32)

        def ffn_block(norm_src, mlp_src, prefix_dst):
            b.put(f"{prefix_dst}_norm.weight", src.get(norm_src, expect=(LM_DIM,)), F32)
            b.put(f"{prefix_dst}_gate.weight",
                  src.get(f"{mlp_src}.gate_proj.weight", expect=(LM_FF, LM_DIM)), NATIVE)
            b.put(f"{prefix_dst}_up.weight",
                  src.get(f"{mlp_src}.up_proj.weight", expect=(LM_FF, LM_DIM)), NATIVE)
            b.put(f"{prefix_dst}_down.weight",
                  src.get(f"{mlp_src}.down_proj.weight", expect=(LM_DIM, LM_FF)), NATIVE)

        # AR path
        attn_block((f"{p}.input_layernorm.weight", f"{p}.self_attn"), f"{d}.attn")
        ffn_block(f"{p}.post_attention_layernorm.weight", f"{p}.mlp", f"{d}.ffn")
        # NAR path
        attn_block((f"{p}.nar_input_layernorm.weight", f"{p}.nar_self_attn"), f"{d}.nar_attn")
        ffn_block(f"{p}.nar_pre_mlp_layernorm.weight", f"{p}.nar_mlp", f"{d}.nar_ffn")

    b.put("vae2llm.weight", src.get("vae2llm.weight", expect=(LM_DIM, LM_LATENT_DIM)), NATIVE)
    b.put("vae2llm.bias", src.get("vae2llm.bias", expect=(LM_DIM,)), F32)
    b.put("llm2vae.weight", src.get("llm2vae.weight", expect=(LM_LATENT_DIM, LM_DIM)), NATIVE)
    b.put("llm2vae.bias", src.get("llm2vae.bias", expect=(LM_LATENT_DIM,)), F32)
    b.put("time_embd.0.weight",
          src.get("time_embedder.mlp.0.weight", expect=(LM_DIM, LM_FREQ_EMBED_SIZE)), NATIVE)
    b.put("time_embd.0.bias", src.get("time_embedder.mlp.0.bias", expect=(LM_DIM,)), F32)
    b.put("time_embd.1.weight",
          src.get("time_embedder.mlp.2.weight", expect=(LM_DIM, LM_DIM)), NATIVE)
    b.put("time_embd.1.bias", src.get("time_embedder.mlp.2.bias", expect=(LM_DIM,)), F32)

    # latent_pos_embed.pe: NATIVE_BF16_VERBATIM, per 05-gguf-layout.md §3.3 -- copied
    # byte-for-byte, never recomputed, never promoted, regardless of --type.
    b.put("latent_pos_embed.weight",
          src.get_bf16_verbatim("latent_pos_embed.pe",
                                 expect=(LM_MAX_LATENT_FRAMES, LM_DIM)),
          NATIVE_BF16_VERBATIM)

    leftover = src.unconsumed(lambda n: True)
    if leftover:
        log(f"ERROR: {len(leftover)} LM source tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n} {src.file.shape(n)} {src.file.dtype(n)}")
        if len(leftover) > 40:
            log(f"    ... and {len(leftover) - 40} more")
        die("the converter does not understand this checkpoint's tensor set -- "
            "refusing to leave a partial model in place")

    src.close()


def add_tokenizer_kv(b, tok_dir):
    vocab_path = os.path.join(tok_dir, "vocab.json")
    merges_path = os.path.join(tok_dir, "merges.txt")
    specials_path = os.path.join(tok_dir, "special_tokens.json")
    pretok_path = os.path.join(tok_dir, "pretokenizer.json")
    for p in (vocab_path, merges_path, specials_path, pretok_path):
        if not os.path.isfile(p):
            die(f"tokenizer file missing: {p} -- run yue2-tokenizer-convert.py first "
                f"(or pass --tokenizer-dir at the directory it wrote to)")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(specials_path, "r", encoding="utf-8") as f:
        specials = json.load(f)
    with open(pretok_path, "r", encoding="utf-8") as f:
        pretok = json.load(f)
    with open(merges_path, "r", encoding="utf-8") as f:
        merge_lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("#version")]

    n_ordinary = specials["base_vocab_size"]
    if n_ordinary != len(vocab):
        die(f"tokenizer: special_tokens.json base_vocab_size={n_ordinary} != "
            f"vocab.json has {len(vocab)} entries")

    id_to_str = [None] * n_ordinary
    for s, i in vocab.items():
        if not (0 <= i < n_ordinary):
            die(f"tokenizer: vocab.json id {i} out of range [0,{n_ordinary})")
        id_to_str[i] = s
    if any(s is None for s in id_to_str):
        die("tokenizer: vocab.json does not cover every id in [0, base_vocab_size)")

    special_entries = specials["tokens"]
    if len(special_entries) != 208:
        die(f"tokenizer: expected 208 specials, special_tokens.json has {len(special_entries)}")
    expected_first_special_id = n_ordinary
    for i, e in enumerate(special_entries):
        if e["id"] != expected_first_special_id + i:
            die(f"tokenizer: special id {e['id']} out of order at index {i}")

    tokens = id_to_str + [e["text"] for e in special_entries]
    n_vocab = len(tokens)   # n_ordinary + 208 == 151851, the tokenizer's own n_vocab
    if n_vocab != TOK["music_start"]:
        die(f"tokenizer: n_vocab={n_vocab} != yue2.token.music_start={TOK['music_start']} "
            f"-- MUSIC_START is defined as exactly n_vocab in protocol.py; this checkpoint's "
            f"tokenizer disagrees with the pinned token ids")

    toktypes = [gguf.TokenType.NORMAL] * n_ordinary + [gguf.TokenType.CONTROL] * len(special_entries)

    b.meta("add_tokenizer_model", "gpt2")
    b.meta("add_token_list", tokens)
    b.meta("add_token_types", toktypes)
    b.meta("add_token_merges", merge_lines)
    b.meta("add_bos_token_id", TOK["eod"])
    b.meta("add_eos_token_id", TOK["eod"])
    b.meta("add_pad_token_id", TOK["eod"])
    b.meta("add_add_bos_token", False)

    b.meta("add_string", "yue2.tokenizer.pretokenize_regex", pretok["regex"])
    b.meta("add_bool", "yue2.tokenizer.normalize_nfc", True)
    b.meta("add_bool", "yue2.tokenizer.add_eos", False)

    log(f"tokenizer: {n_vocab} tokens ({n_ordinary} BPE + {len(special_entries)} specials), "
        f"{len(merge_lines)} merges")


# ---------------------------------------------------------------------------
# Component: VAE  ->  arch "yue2-vae"
# ---------------------------------------------------------------------------

def build_vae(src_dir, bundle, type_, variant, rs_x3_path=None):
    cfg_path = os.path.join(src_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    got_variant = cfg.get("release_variant")
    # rs-x3 takes Standard's encoder and config; only its decoder comes from elsewhere.
    if got_variant != ("standard" if rs_x3_path else variant):
        die(f"{src_dir}: config.json release_variant={got_variant!r}, expected {variant!r} "
            f"-- refusing a standard/legacy VAE mix-up (05-gguf-layout.md §8)")

    expect_config(cfg, "sample_rate", VAE_SAMPLE_RATE, "VAE")
    expect_config(cfg, "downsampling_ratio", VAE_DOWNSAMPLE_RATIO, "VAE")
    expect_config(cfg, "audio_channels", VAE_AUDIO_CHANNELS, "VAE")
    expect_config(cfg, "latent_dim", VAE_LATENT_DIM, "VAE")
    enc_cfg = cfg["encoder_config"]
    dec_cfg = cfg["decoder_config"]
    expect_config(enc_cfg, "latent_dim", VAE_ENCODER_LATENT_DIM, "VAE encoder_config")
    expect_config(enc_cfg, "channels", VAE_BASE_CHANNELS, "VAE encoder_config")
    expect_config(enc_cfg, "in_channels", VAE_AUDIO_CHANNELS, "VAE encoder_config")
    expect_config(enc_cfg, "use_snake", True, "VAE encoder_config")
    if tuple(enc_cfg["c_mults"]) != (1, 2, 4, 8, 16, 32):
        die(f"VAE encoder_config.c_mults = {enc_cfg['c_mults']!r}, expected [1,2,4,8,16,32]")
    if tuple(enc_cfg["strides"]) != VAE_STRIDES:
        die(f"VAE encoder_config.strides = {enc_cfg['strides']!r}, expected {VAE_STRIDES}")
    expect_config(dec_cfg, "latent_dim", VAE_LATENT_DIM, "VAE decoder_config")
    expect_config(dec_cfg, "channels", VAE_BASE_CHANNELS, "VAE decoder_config")
    expect_config(dec_cfg, "out_channels", VAE_AUDIO_CHANNELS, "VAE decoder_config")
    expect_config(dec_cfg, "use_snake", True, "VAE decoder_config")
    expect_config(dec_cfg, "snake_type", "vanilla", "VAE decoder_config")
    expect_config(dec_cfg, "use_filter", False, "VAE decoder_config")
    expect_config(dec_cfg, "final_tanh", VAE_FINAL_TANH, "VAE decoder_config")
    if tuple(dec_cfg["strides"]) != VAE_STRIDES:
        die(f"VAE decoder_config.strides = {dec_cfg['strides']!r}, expected {VAE_STRIDES}")
    # A grown fine-tuned decoder (tools/yue2-vae-train) records its residual
    # units per block; the released ones have none and use 3. More units widen
    # the receptive field, so such a config also carries its own halo.
    dec_res_units = int(dec_cfg.get("res_units", 3))
    required_halo = int(cfg.get("required_halo", VAE_REQUIRED_HALO))
    decode_halo = int(cfg.get("decode_halo_frames", VAE_DECODE_HALO_FRAMES))
    dsrc = None
    if rs_x3_path:
        dsrc = RsX3Source(rs_x3_path)
        dec_res_units, required_halo, decode_halo = RS_X3_RES_UNITS, RS_X3_HALO, RS_X3_HALO
    dec_dilations = [VAE_RES_DILATIONS[i % 3] for i in range(dec_res_units)]
    if decode_halo < required_halo:
        die(f"decode_halo_frames {decode_halo} < required_halo {required_halo}")

    st_path = os.path.join(src_dir, "model.safetensors")
    src = Source(st_path)

    manifest_path = os.path.join(src_dir, "weights_manifest.json")
    pin_key = "standard" if dsrc else variant  # rs-x3 pins the Standard checkpoint its encoder comes from
    match = sha256_matches_pin(manifest_path, PINS[pin_key]["sha256"]) if pin_key in PINS else False
    pin = PINS[pin_key] if match else {"repo": "unknown", "revision": "unknown", "sha256": "unknown"}
    if match is False:
        log(f"WARNING: {manifest_path} sha256 does not match the pinned {variant} VAE "
            f"revision -- writing yue2vae.pinned.* as 'unknown'")
    elif match is None:
        log(f"WARNING: no weights_manifest.json found at {manifest_path} -- "
            f"writing yue2vae.pinned.* as 'unknown'")

    b = bundle
    b.meta("add_name", f"YuE2 VAE ({variant})")
    b.meta("add_description",
           f"YuE2 Oobleck-style VAE, {variant} release variant: SnakeBeta activations, "
           f"weight-norm folded at conversion time. dec.* is what inference decodes with; "
           f"enc.* is carried because the source checkpoint ships a full encoder.")
    common_meta(b, type_, (RS_X3_ATTRIBUTION + LICENSE_ATTRIBUTION) if dsrc else LICENSE_ATTRIBUTION)

    b.meta("add_string", "yue2vae.variant", variant)
    b.meta("add_uint32", "yue2vae.sample_rate", VAE_SAMPLE_RATE)
    b.meta("add_uint32", "yue2vae.downsampling_ratio", VAE_DOWNSAMPLE_RATIO)
    b.meta("add_uint32", "yue2vae.audio_channels", VAE_AUDIO_CHANNELS)
    b.meta("add_uint32", "yue2vae.latent_dim", VAE_LATENT_DIM)
    b.meta("add_uint32", "yue2vae.encoder_latent_dim", VAE_ENCODER_LATENT_DIM)
    b.meta("add_uint32", "yue2vae.channels", VAE_BASE_CHANNELS)
    b.meta("add_array", "yue2vae.strides", list(VAE_STRIDES))
    b.meta("add_array", "yue2vae.res_dilations", list(VAE_RES_DILATIONS))
    if dec_res_units != 3:
        b.meta("add_array", "yue2vae.dec_res_dilations", dec_dilations)
    if dsrc:
        b.meta("add_array", "yue2vae.dec_pre_dilations", list(RS_X3_PRE_DILATIONS))
        b.meta("add_bool", "yue2vae.strict_f32", True)
        b.meta("add_string", "yue2vae.decoder_source",
               f"goldhub/YuE2-VAE-RS-x3 {os.path.basename(rs_x3_path)} (training step {dsrc.step})")
    b.meta("add_bool", "yue2vae.use_snake", True)
    b.meta("add_string", "yue2vae.snake_type", "vanilla")
    b.meta("add_float32", "yue2vae.snake_eps", VAE_SNAKE_EPS)
    b.meta("add_string", "yue2vae.snake_formula",
           "x + 1/(exp(beta)+eps) * sin(x*exp(alpha))^2 -- alpha/beta are BOTH "
           "log-space, exp() both before use")
    b.meta("add_bool", "yue2vae.final_tanh", VAE_FINAL_TANH)
    b.meta("add_float32", "yue2vae.output_clamp_min", VAE_OUTPUT_CLAMP[0])
    b.meta("add_float32", "yue2vae.output_clamp_max", VAE_OUTPUT_CLAMP[1])
    b.meta("add_bool", "yue2vae.weight_norm_folded", True)
    b.meta("add_bool", "yue2vae.has_encoder", True)
    b.meta("add_uint32", "yue2vae.decode_core_frames", VAE_DECODE_CORE_FRAMES)
    b.meta("add_uint32", "yue2vae.decode_halo_frames", decode_halo)
    b.meta("add_uint32", "yue2vae.required_halo", required_halo)
    b.meta("add_string", "yue2vae.pinned.repo", pin["repo"])
    b.meta("add_string", "yue2vae.pinned.revision", pin["revision"])
    b.meta("add_string", "yue2vae.pinned.sha256", pin["sha256"])

    def of(key):  # an rs-x3 decoder comes from its own file, everything else from src
        return dsrc if dsrc and key.startswith("decoder.") else src

    def wn(dst, key, expect_v, expect_bias):
        s = of(key)
        g = s.get(f"{key}.weight_g", expect=(expect_v[0], 1, 1))
        v = s.get(f"{key}.weight_v", expect=expect_v)
        b.put(f"{dst}.weight", fold_weight_norm(g, v, dst), F32)
        if expect_bias is not None:
            b.put(f"{dst}.bias", s.get(f"{key}.bias", expect=(expect_bias,)), F32)

    def alpha_beta(dst, key, ch):
        s = of(key)
        b.put(f"{dst}.alpha", s.get(f"{key}.alpha", expect=(ch,)), F32)
        b.put(f"{dst}.beta", s.get(f"{key}.beta", expect=(ch,)), F32)

    # Channel ladder, shared by both sides (05-gguf-layout.md §4.1/§4.2): the
    # config's own c_mults=[1,2,4,8,16,32] gives an 7-entry width sequence
    # [64,64,128,256,512,1024,2048] -- note the leading 64,64 duplicate (c_mults[0]=1).
    # Encoder walks it forward (block b: widths[b-1] -> widths[b]); decoder walks
    # the REVERSED sequence (block b: dwidths[b-1] -> dwidths[b]), so the decoder's
    # LAST block is 64->64 (mirroring the encoder's FIRST block, also 64->64) --
    # neither is a uniform halve/double at every step, so this is computed from
    # the list, not from ch //= 2 / ch *= 2.
    c_mults = (1, 2, 4, 8, 16, 32)
    widths = [VAE_BASE_CHANNELS] + [VAE_BASE_CHANNELS * m for m in c_mults]   # len 7
    dwidths = list(reversed(widths))
    dstrides = tuple(reversed(VAE_STRIDES))

    # -- decoder: dec.* --------------------------------------------------
    wn("dec.conv_in", "decoder.layers.0", (dwidths[0], VAE_LATENT_DIM, 7), dwidths[0])
    if dsrc:
        c0 = dwidths[0]
        for pi in range(len(RS_X3_PRE_DILATIONS)):
            pbase = f"decoder.pre.{pi}"
            alpha_beta(f"dec.pre.{pi}.snake1", f"{pbase}.layers.0", c0)
            wn(f"dec.pre.{pi}.conv1", f"{pbase}.layers.1", (c0, c0, 7), c0)
            alpha_beta(f"dec.pre.{pi}.snake2", f"{pbase}.layers.2", c0)
            wn(f"dec.pre.{pi}.conv2", f"{pbase}.layers.3", (c0, c0, 1), c0)
    for bi in range(1, 7):
        cin, cout, stride = dwidths[bi - 1], dwidths[bi], dstrides[bi - 1]
        base = f"decoder.layers.{bi}"
        alpha_beta(f"dec.blk.{bi}.snake_pre", f"{base}.layers.0", cin)
        wn(f"dec.blk.{bi}.upsample", f"{base}.layers.1", (cin, cout, 2 * stride), cout)
        for ri in range(dec_res_units):
            rbase = f"{base}.layers.{ri + 2}"
            alpha_beta(f"dec.blk.{bi}.res.{ri}.snake1", f"{rbase}.layers.0", cout)
            wn(f"dec.blk.{bi}.res.{ri}.conv1", f"{rbase}.layers.1", (cout, cout, 7), cout)
            alpha_beta(f"dec.blk.{bi}.res.{ri}.snake2", f"{rbase}.layers.2", cout)
            wn(f"dec.blk.{bi}.res.{ri}.conv2", f"{rbase}.layers.3", (cout, cout, 1), cout)
    alpha_beta("dec.snake_out", "decoder.layers.7", dwidths[6])
    wn("dec.conv_out", "decoder.layers.8", (VAE_AUDIO_CHANNELS, dwidths[6], 7), None)

    # -- encoder: enc.* (source ships one; see 05-gguf-layout.md §4.2) ---
    wn("enc.conv_in", "encoder.layers.0", (widths[0], VAE_AUDIO_CHANNELS, 7), widths[0])
    for bi in range(1, 7):
        cin, cout, stride = widths[bi - 1], widths[bi], VAE_STRIDES[bi - 1]
        base = f"encoder.layers.{bi}"
        for ri in range(3):
            rbase = f"{base}.layers.{ri}"
            alpha_beta(f"enc.blk.{bi}.res.{ri}.snake1", f"{rbase}.layers.0", cin)
            wn(f"enc.blk.{bi}.res.{ri}.conv1", f"{rbase}.layers.1", (cin, cin, 7), cin)
            alpha_beta(f"enc.blk.{bi}.res.{ri}.snake2", f"{rbase}.layers.2", cin)
            wn(f"enc.blk.{bi}.res.{ri}.conv2", f"{rbase}.layers.3", (cin, cin, 1), cin)
        alpha_beta(f"enc.blk.{bi}.snake_post", f"{base}.layers.3", cin)
        wn(f"enc.blk.{bi}.downsample", f"{base}.layers.4", (cout, cin, 2 * stride), cout)
    alpha_beta("enc.snake_out", "encoder.layers.7", widths[6])
    wn("enc.conv_out", "encoder.layers.8",
       (VAE_ENCODER_LATENT_DIM, widths[6], 3), VAE_ENCODER_LATENT_DIM)

    # rs-x3: Standard's own decoder tensors in src are unused by design.
    leftover = src.unconsumed(lambda n: not (dsrc and n.startswith("decoder.")))
    leftover += dsrc.unconsumed(lambda n: True) if dsrc else []
    if leftover:
        log(f"ERROR: {len(leftover)} VAE ({variant}) source tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n} {src.file.shape(n)} {src.file.dtype(n)}")
        if len(leftover) > 40:
            log(f"    ... and {len(leftover) - 40} more")
        die("the converter does not understand this checkpoint's tensor set -- "
            "refusing to leave a partial model in place")

    src.close()
    if dsrc:
        dsrc.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="convert-yue2.py",
        description="YuE2 safetensors -> GGUF for the HOT-Step engine.")
    ap.add_argument("--src-lm", metavar="DIR", help="YuE2-3B model dir")
    ap.add_argument("--src-vae", metavar="DIR", help="YuE2-Vae (standard) model dir")
    ap.add_argument("--src-vae-legacy", metavar="DIR", help="YuE2-Vae-legacy model dir")
    ap.add_argument("--src-vae-dec2x", metavar="DIR",
                    help="fine-tuned grown decoder exported by tools/yue2-vae-train/export.py")
    ap.add_argument("--src-vae-rs-x3", metavar="FILE",
                    help="YuE2-VAE-RS-x3_v1.safetensors (goldhub/YuE2-VAE-RS-x3); writes "
                         "yue2-vae-rs-x3-<type>.gguf with --src-vae's encoder")
    ap.add_argument("--tokenizer-dir", metavar="DIR",
                     help="output dir of yue2-tokenizer-convert.py (vocab.json/merges.txt/"
                          "special_tokens.json/pretokenizer.json); if missing, that script "
                          "is run first (needs --src-lm for qwen.tiktoken)")
    ap.add_argument("--out", required=True, metavar="DIR", help="output directory")
    ap.add_argument("--components", default="all", help="comma list of lm,vae or 'all'")
    ap.add_argument("--type", default=None, choices=("bf16", "f16", "f32"),
                     help="default: bf16 for lm, f32 for vae (each is the lossless-from-"
                          "source baseline for that component)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    want = set(c.strip() for c in args.components.split(",") if c.strip())
    if want == {"all"}:
        want = {"lm", "vae"}
    bad = want - {"lm", "vae"}
    if bad:
        die(f"unknown component(s) {bad}; valid: lm, vae, all")

    os.makedirs(args.out, exist_ok=True)

    if "lm" in want:
        if not args.src_lm:
            die("--components includes lm but --src-lm was not given")
        lm_type = args.type or "bf16"
        lm_path = os.path.join(args.out, f"yue2-lm-{lm_type}.gguf")
        if os.path.exists(lm_path) and not args.force:
            log(f"skip (exists): {lm_path} -- pass --force to overwrite")
        else:
            tok_dir = args.tokenizer_dir
            if not tok_dir:
                tok_dir = os.path.join(args.out, "tokenizer")
            need = not all(os.path.isfile(os.path.join(tok_dir, f))
                           for f in ("vocab.json", "merges.txt", "special_tokens.json",
                                     "pretokenizer.json"))
            if need:
                log(f"tokenizer dir {tok_dir} incomplete -- running yue2-tokenizer-convert.py")
                convert_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "yue2-tokenizer-convert.py")
                tiktoken_path = os.path.join(args.src_lm, "qwen.tiktoken")
                tokenization_src = os.path.join(args.src_lm, "..", "..", "yue2",
                                                 "tokenization_yue2.py")
                # The installed package path is the real source of truth; fall back to it
                # explicitly rather than guessing relative to --src-lm (the model dir does
                # not itself carry tokenization_yue2.py).
                installed_tokenization = "K:/yue2/.venv/Lib/site-packages/yue2/tokenization_yue2.py"
                installed_protocol = "K:/yue2/.venv/Lib/site-packages/yue2/protocol.py"
                src_tok = installed_tokenization if os.path.isfile(installed_tokenization) \
                    else tokenization_src
                cmd = [sys.executable, convert_script,
                       "--tiktoken", tiktoken_path,
                       "--tokenization-src", src_tok,
                       "--out", tok_dir]
                if os.path.isfile(installed_protocol):
                    cmd += ["--protocol-src", installed_protocol]
                log(f"running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            b = Bundle("yue2")
            build_lm(args.src_lm, b, lm_type, tok_dir)
            b.write(lm_path, lm_type)

    if "vae" in want:
        for flag, srcdir, variant in (("--src-vae", args.src_vae, "standard"),
                                       ("--src-vae-legacy", args.src_vae_legacy, "legacy"),
                                       ("--src-vae-dec2x", args.src_vae_dec2x, "dec2x"),
                                       ("--src-vae-rs-x3", args.src_vae_rs_x3, "rs-x3")):
            if not srcdir:
                log(f"skip {variant} VAE: {flag} not given")
                continue
            rs_x3_path = None
            if variant == "rs-x3":
                if not args.src_vae:
                    die("--src-vae-rs-x3 needs --src-vae (Standard's encoder and config)")
                rs_x3_path, srcdir = srcdir, args.src_vae
            vae_type = args.type or "f32"
            vae_path = os.path.join(args.out, f"yue2-vae-{variant}-{vae_type}.gguf")
            if os.path.exists(vae_path) and not args.force:
                log(f"skip (exists): {vae_path} -- pass --force to overwrite")
                continue
            b = Bundle("yue2-vae")
            build_vae(srcdir, b, vae_type, variant, rs_x3_path)
            b.write(vae_path, vae_type)

    log("Done.")


if __name__ == "__main__":
    main()

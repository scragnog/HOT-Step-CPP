#!/usr/bin/env python3
# convert-yue2-tok.py: YuE2 real-audio semantic TOKENIZER -> one GGUF.
#
# Produces exactly one file:
#
#   yue2-tok-<type>.gguf     arch "yue2-tok"
#       mert.*   MERT-v2-FullSong, mel frontend + ConvNext subsampling + Conformer
#                blocks 0..20 ONLY (21/22/23 deliberately dropped)
#       tok.*    Mothersuperior's tokenizer head, all 103 tensors
#
# This is the transport half of the contract pinned in
# docs/plans/yue2/12-tokenizer-oracle-pin.md. READ THAT FIRST. It was produced by
# RUNNING the reference implementation, not by reading it, so where this file and
# that document disagree, the document wins. Section references below (§3.2, §1.5,
# ...) are into that document, and every policy decision here has its reasoning
# there rather than repeated in full.
#
# Modelled on convert-yue2.py: same safetensors reader, same Bundle/put/write
# shape, same "{arch-without-dashes}." KV prefix convention ("yue2-vae" -> yue2vae.,
# so "yue2-tok" -> yue2tok.), same refuse-on-unconsumed-tensors discipline.
#
# THE PIPELINE THIS FILE HAS TO FEED
#   audio -> mono 24 kHz -> mel frontend -> ConvNext subsampling (25 Hz)
#         -> 21 Conformer blocks -> hidden_states[20]
#         -> concat 30 s chunks -> interpolate to T25 -> fp16 store
#         -> per-track per-channel instance norm -> 8-layer head -> argmax
#         -> code in [0,32768) -> YuE2 token id = code + 151853
#
# WHY BLOCK 20 AND NOT 19 (§3.1). MERT2's hidden_states tuple has NO input-embedding
# entry -- 24 entries for 24 Conformer blocks -- so hidden_states[20] is the output
# of layers[20], the 21st block, post its own final_layer_norm. That is unlike
# Wav2Vec2, where hidden_states[k] means "after k blocks". Verified at runtime in
# the pin, not inferred. We therefore keep 21 of 24 blocks; layers.21/22/23 are
# 93 tensors and 0.27 GiB of dead weight, and shipping them mostly invites someone
# to "fix" the layer index later. The drop is asserted below, not assumed.
#
# WHAT DOES *NOT* GO IN THE FILE, ON PURPOSE
#   embed_positions.inv_freq -- registered persistent=False, so it is not in the
#   checkpoint at all and must be BUILT by the loader:
#       inv_freq[i] = 1 / 10000^(2i/64), i in 0..31
#   The rotary base and dimension are written as KV, and
#   yue2tok.mert.rope.inv_freq_in_file=false says so out loud.
#
# WHAT *IS* IN THE FILE THAT A PORT MIGHT TRY TO RECONSTRUCT (§3.2)
#   The STFT window and the mel filterbank SHIP AS CHECKPOINT BUFFERS. Do not
#   build a Hann window; do not reproduce a Slaney/HTK filterbank. They come out
#   of model.safetensors as feature_extractor.spectrogram.window [2048] and
#   feature_extractor.mel_scale.fb [1025,128] and are written verbatim as
#   mert.mel.stft_window / mert.mel.filterbank.
#
# ---------------------------------------------------------------------------
# QUANTIZATION POLICY -- which tensors are FORCED F32, and why
# ---------------------------------------------------------------------------
# The rule is one line: **F16 is applied only to 2-D matmul weights.** Everything
# else is F32. Concretely, these are forced F32 regardless of --outtype:
#
#   * mert.mel.*  (stft_window, filterbank, mean, std) -- the filterbank is 2-D
#     and would otherwise qualify, but the reference frontend runs float32 with
#     autocast EXPLICITLY DISABLED inside it (§3.3/§4.1). Narrowing the one stage
#     the reference refuses to narrow would be a gratuitous divergence, and the
#     whole frontend is 0.5 MB.
#   * every LayerNorm weight and bias (both epsilons, both stacks).
#   * every bias, including the head's fused qkv bias [1536].
#   * every GlobalResponseNorm weight/bias (squeezed [1,1,C] -> [C], so 1-D).
#   * every convolution KERNEL that is not a kernel-1 pointwise: the depthwise
#     7-taps and 31-taps, and the kernel-2 resampling convs. These are consumed by
#     explicit F32 im2col in the port, they total under 2 M parameters, and exact
#     is free at that size.
#   * tok.pos [512,512]. 2-D, but it is an ADDITIVE learned position table, not a
#     matmul weight; it costs 1 MB at F32 and it lands directly on the activation
#     that eight attention layers then read.
#   * anything with ndim < 2, unconditionally, as a backstop.
#
# The kernel-1 pointwise convs in the Conformer conv module ([2048,1024,1] and
# [1024,1024,1]) ARE narrowed: the trailing kernel axis is squeezed away at write
# time, so they are true 2-D matmuls both in the file and in the port (ggml_mul_mat,
# no im2col). They are 63 M parameters across 21 blocks, so exempting them would
# throw away a third of the saving.
#
# At --outtype f16 the result is ~1.2 GB against ~2.4 GB at f32; the F32-forced set
# is about 3 M of 602 M parameters.
#
# TOLERANCE, so nobody gates this wrong (§4). The reference production path runs
# MERT and the head under CUDA bf16 autocast, and fp32-vs-bf16 argmax already
# disagree on 2.34% of frames. Any port-vs-oracle gate is an AGREEMENT RATE with a
# ~98% ceiling, never bit-exactness. F16 storage here lands in the same tolerance
# bucket. Ship --outtype f32 if you want to bisect against the FP32 stage fixtures
# in K:/yue2/fixtures/tokenizer-v1/stages-30s/ without storage noise in the way.
#
# ---------------------------------------------------------------------------
# DEPENDENCIES: numpy + gguf (llama.cpp gguf-py). NO TORCH.
#
# The head is a torch .pt, but K:/yue2/.venv (which has torch) has no gguf and no
# pip to add it, and the system python that has gguf has no torch. Rather than
# bounce the weights through a temp file in two environments, this script carries a
# ~60-line torch-free reader for the zip+pickle .pt container (TorchPtFile below).
# That also matches convert-yue2.py's "no torch dependency" property.
#
# USAGE
#   python engine/tools/convert-yue2-tok.py \
#       --src-mert K:/yue2/models/MERT-v2-FullSong \
#       --src-head <dir-or-file>/tokenizer_head_joint_v4.pt \
#       --out models/yue2 --outtype f16
#
# The default --out is models/yue2, so the shipped path is
# models/yue2/yue2-tok-f16.gguf.

import argparse
import hashlib
import io
import json
import mmap
import os
import pickle
import struct
import sys
import zipfile

import numpy as np
import gguf

CONVERTER_VERSION = 1

# ---------------------------------------------------------------------------
# Model constants. Hardcoded on purpose, same reasoning as convert-yue2.py: a
# converter that only reads shapes from the checkpoint cannot reject a wrong one.
# MERT values are cross-checked against K:/yue2/models/MERT-v2-FullSong/config.json
# by expect_config(); the head values come from the pin §1.3 and are checked
# against the .pt's tensor shapes.
# ---------------------------------------------------------------------------

# -- MERT-v2-FullSong frontend (config.json + pin §3.3) --------------------
MERT_SR = 24000
MERT_N_FFT = 2048
MERT_WIN_LENGTH = 2048
MERT_HOP_LENGTH = 240
MERT_N_MEL = 128
MERT_N_FREQ = MERT_N_FFT // 2 + 1          # 1025, onesided
MERT_FRAME_RATE = 25.0
MERT_SAMPLES_PER_FRAME = 960               # inputs_to_logits_ratio; 24000/25
MERT_MIN_INPUT_SAMPLES = 1025
MERT_SPEC_POWER = 2.0
MERT_DB_AMIN = 1e-10                       # 10*log10(max(x, 1e-10)), top_db=None
MERT_MEL_STD_FLOOR = 1e-5                  # mel_std.clamp_min(1e-5), NOT std+eps

# -- MERT Conformer stack ---------------------------------------------------
MERT_DIM = 1024
MERT_FF = 4096
MERT_HEADS = 16
MERT_HEAD_DIM = 64                         # 1024 / 16
MERT_BLOCKS_FULL = 24                      # what the checkpoint ships
MERT_HIDDEN_STATE_INDEX = 20               # the block we stop at (§3.1)
MERT_BLOCKS_KEPT = MERT_HIDDEN_STATE_INDEX + 1   # 21 -- what we WRITE
MERT_LN_EPS = 1e-5                         # layer_norm_eps: Conformer stack
MERT_CONV_DW_KERNEL = 31                   # conv_depthwise_kernel_size, padding 15
MERT_ROPE_BASE = 10000.0

# -- MERT ConvNext subsampling (pin §3.4) -----------------------------------
MERT_SUB_CHANNELS = (128, 512, 1024)       # config subsampling_channels
MERT_SUB_DEPTHS = (3, 4, 5)                # config subsampling_depths
MERT_SUB_STRIDES = (1, 2, 2)               # hardcoded (1,2,2) in modeling_mert2.py
MERT_SUB_LN_EPS = 1e-6                     # subsampling_layer_norm_eps -- NOT 1e-5
MERT_SUB_KERNEL = 2                        # resampling conv, NO padding
MERT_CNX_KERNEL = 7                        # depthwise conv, padding 3
MERT_GRN_EPS = 1e-6                        # GlobalResponseNorm's own +1e-6

# -- chunking / frame-rate bridge (pin §3.7) --------------------------------
CHUNK_SECONDS = 30
CHUNK_SAMPLES = MERT_SR * CHUNK_SECONDS     # 720,000
MIN_CHUNK_SAMPLES = MERT_SR                 # a tail under 1 second is DROPPED
INSTNORM_EPS = 1e-5                         # (x - mean)/(std + 1e-5), ddof=0

# -- the head (pin §1.2/§1.3) ----------------------------------------------
HEAD_WIN = 512                              # pos dim 1
HEAD_D = 512                                # pos dim 2
HEAD_LAYERS = 8                             # enc.layers.0..7
HEAD_HEADS = 8                              # NOT IN THE FILE -- from train_v2/joint/ar_prep
HEAD_HEAD_DIM = HEAD_D // HEAD_HEADS        # 64
HEAD_FF = 2048                              # linear1.weight rows = 4*D
HEAD_DIN = 1024                             # inp.weight columns = MERT width
HEAD_VOCAB = 32768                          # head.weight rows
HEAD_LN_EPS = 1e-5                          # PyTorch default; never overridden
HEAD_STRIDE = HEAD_WIN // 2                 # 256 -- predict() window advance
HEAD_TRIM = HEAD_WIN // 4                   # 128 -- per-side trim, except true ends
HEAD_QKV_ROWS = 3 * HEAD_D                  # 1536, fused [q;k;v] (§1.5)

CODEC_OFFSET = 151853                       # YuE2 token id = code + CODEC_OFFSET
CODEC_SIZE = 32768

# -- licence. BOTH halves are CC BY-NC 4.0, and an adapter trained through this
# encoder carries the same terms -- which, per the YuE2 authors' 2026-09-15
# clarification, leaves individual creators free to use outputs commercially
# and only obliges companies. MERT: K:/yue2/models/MERT-v2-FullSong/LICENSE. Head:
# HF repo Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4, derived from
# YuE2-3B, also CC BY-NC 4.0. ------------------------------------------------
LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_ATTRIBUTION = (
    "Weights are CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 "
    "International), and anything trained through this encoder carries the same "
    "terms. The YuE2 authors have clarified that individual creators, musicians "
    "and researchers may use the model and its outputs freely, including "
    "commercially; only companies need a commercial licence. "
    "Attribution: MERT2 / MERT-v2-FullSong -- "
    "https://huggingface.co/m-a-p/MERT-v2-FullSong ; tokenizer head -- "
    "https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 "
    "(derived from https://huggingface.co/m-a-p/YuE2-3B)"
)

# -- pinned provenance (12-tokenizer-oracle-pin.md §0). MERT is checked via its
# weights_manifest.json (2.5 GB is not worth hashing every run); the head is
# hashed directly, it is 171 MB. ---------------------------------------------
PINS = {
    "mert": {
        "repo": "m-a-p/MERT-v2-FullSong",
        "sha256": "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d",
        "bytes": 2529812848,
    },
    "head": {
        "repo": "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4",
        "sha256": "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e",
        "bytes": 171305291,
    },
}

# Heads from that repo this converter has been run against. Every one is the
# same 103-tensor layout at the same instnorm regime, so they are drop-in weight
# swaps; only the fixture match rate moves (v4 16.1% top-1 on minted held-out,
# v5-and-later ~18.9%). v5+ ship as safetensors only -- there is no .pt.
KNOWN_HEADS = {
    "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e":
        "tokenizer_head_joint_v4.pt",
    "06440f25605c6c12c4e517ef033d7b70b8d9e961b3b06aa3848b1f8028fd7eb8":
        "tokenizer_head_joint_v9.safetensors",
}

# Preference order when --src-head names a directory: newest known head first.
HEAD_FILE_PREFS = (
    "tokenizer_head_joint_v9.safetensors",
    "tokenizer_head_joint_v8.safetensors",
    "tokenizer_head_v5_30k.safetensors",
    "tokenizer_head_joint_v5.safetensors",
    "tokenizer_head_joint_v4.pt",
)
ORACLE_FIXTURES = "K:/yue2/fixtures/tokenizer-v1 (stages-30s, real-30s, yue2-gen)"


def log(msg):
    print(f"[convert-yue2-tok] {msg}", file=sys.stderr, flush=True)


def die(msg):
    raise SystemExit(f"[convert-yue2-tok] ERROR: {msg}")


def expect_config(cfg, key, want, label="MERT"):
    got = cfg.get(key)
    if isinstance(want, float):
        ok = got is not None and abs(float(got) - want) <= abs(want) * 1e-9
    else:
        ok = got == want
    if not ok:
        die(f"{label}: config.json {key}={got!r}, expected {want!r} -- this does not "
            f"look like the checkpoint this converter understands")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# safetensors reader -- lifted from convert-yue2.py (which lifted it from
# convert-mm3.py). MERT is all F32 so the BF16 path never fires here, but keeping
# the reader identical means one bug surface across the three converters.
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

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def get(self, name):
        """Returns a float32 numpy array (widened losslessly from the native dtype)."""
        spec = self.header[name]
        start, end = spec["data_offsets"]
        buf = self._mm[self._base + start:self._base + end]
        dt = spec["dtype"]
        shape = tuple(spec["shape"])
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        np_dt = {"F64": np.float64, "F32": np.float32, "F16": np.float16}[dt]
        return np.frombuffer(buf, dtype=np_dt).reshape(shape).astype(np.float32)

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


# ---------------------------------------------------------------------------
# torch-free .pt reader
#
# torch.save() with the default (zip) format writes:
#     <prefix>/data.pkl            the pickled object graph
#     <prefix>/data/<key>          one raw, uncompressed storage per tensor
#     <prefix>/byteorder           b"little" | b"big"
# Tensors appear in the pickle as
#     torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
#                                     requires_grad, backward_hooks)
# and `storage` arrives through persistent_load as
#     ("storage", <torch.FloatStorage class>, key, location, numel)
#
# So: intercept find_class so no torch symbol is ever imported, intercept
# persistent_load to note (key, dtype, numel), and materialise from the zip.
# Every tensor in tokenizer_head_joint_v4.pt is float32, offset 0 and C-contiguous;
# all three are ASSERTED rather than assumed, because a silent stride surprise here
# would look exactly like a transposed weight downstream.
# ---------------------------------------------------------------------------

_TORCH_STORAGE_DTYPE = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageRef:
    __slots__ = ("key", "dtype_name", "numel")

    def __init__(self, key, dtype_name, numel):
        self.key = key
        self.dtype_name = dtype_name
        self.numel = numel


class _LazyTensor:
    __slots__ = ("ref", "offset", "size", "stride")

    def __init__(self, ref, offset, size, stride):
        self.ref = ref
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


class TorchPtFile:
    """Minimal reader for a torch.save() zip archive. No torch import."""

    def __init__(self, path):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        names = self._zip.namelist()
        pkls = [n for n in names if n.endswith("/data.pkl")]
        if len(pkls) != 1:
            die(f"{path}: expected exactly one */data.pkl in the archive, found {len(pkls)} "
                f"-- this is not a default-format torch.save() file")
        self._prefix = pkls[0][:-len("data.pkl")]          # e.g. "head_best/"
        bo_name = self._prefix + "byteorder"
        if bo_name in names:
            bo = self._zip.read(bo_name)
            if bo != b"little":
                die(f"{path}: byteorder is {bo!r}; this reader is little-endian only")
        self.obj = self._load_pickle()

    def _load_pickle(self):
        outer = self

        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name in ("_rebuild_tensor_v2",
                                                          "_rebuild_tensor_v3"):
                    # args: (storage, storage_offset, size, stride, requires_grad,
                    #        backward_hooks[, metadata])
                    return lambda *a: _LazyTensor(a[0], a[1], a[2], a[3])
                if module == "torch":
                    # Storage classes: never instantiated, only name-matched.
                    return ("torch", name)
                if module in ("collections", "builtins", "__builtin__"):
                    return super().find_class(module, name)
                die(f"{outer.path}: pickle references {module}.{name}, which this "
                    f"torch-free reader does not implement")

            def persistent_load(self, pid):
                if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
                    die(f"{outer.path}: unexpected persistent id {pid!r}")
                stype = pid[1]
                tname = stype[1] if isinstance(stype, tuple) else getattr(
                    stype, "__name__", str(stype))
                if tname not in _TORCH_STORAGE_DTYPE:
                    die(f"{outer.path}: unsupported storage type {tname}")
                return _StorageRef(str(pid[2]), tname, int(pid[4]))

        return _U(io.BytesIO(self._zip.read(self._prefix + "data.pkl"))).load()

    def tensor(self, lazy):
        if not isinstance(lazy, _LazyTensor):
            die(f"{self.path}: expected a tensor, got {type(lazy).__name__}")
        ref = lazy.ref
        np_dt = _TORCH_STORAGE_DTYPE[ref.dtype_name]
        raw = self._zip.read(self._prefix + "data/" + ref.key)
        flat = np.frombuffer(raw, dtype=np_dt)
        if flat.size != ref.numel:
            die(f"{self.path}: storage {ref.key} holds {flat.size} elements, "
                f"pickle claims {ref.numel}")
        # Contiguity + offset assertions -- see the class comment.
        expect_stride = []
        acc = 1
        for d in reversed(lazy.size):
            expect_stride.append(acc)
            acc *= d
        expect_stride = tuple(reversed(expect_stride))
        if lazy.offset != 0 or lazy.stride != expect_stride:
            die(f"{self.path}: storage {ref.key} is not a C-contiguous view at offset 0 "
                f"(offset={lazy.offset}, stride={lazy.stride}, expected {expect_stride}) "
                f"-- refusing to guess the memory layout")
        return flat.reshape(lazy.size).astype(np.float32)

    def close(self):
        self._zip.close()


class HeadSource:
    """The head's tensors, with expect-checked get().

    Two container formats, one interface. The v4 head is a torch .pt holding
    {"model": state_dict, "cfg": {...}}; v5 and later ship as safetensors only,
    a plain state_dict with the run description in the file metadata. The
    tensor names, shapes and dtypes are identical across both, so everything
    downstream of this class is format-blind.
    """

    def __init__(self, path):
        self.path = path
        self.consumed = set()
        if path.endswith(".safetensors"):
            self.pt = None
            self.st = SafeTensorsFile(path)
            self.state = self.st.header
            meta = self.st.metadata
            # ckpt_io.load_ckpt derives instnorm the same way: the flag lives in
            # the free-text "input" field, not as a key of its own.
            self.cfg = dict(meta)
            self.cfg["instnorm"] = "instnorm=true" in meta.get("input", "")
        else:
            self.st = None
            self.pt = TorchPtFile(path)
            obj = self.pt.obj
            if not isinstance(obj, dict) or "model" not in obj:
                die(f"{path}: expected a dict with a 'model' key, got {type(obj).__name__}")
            self.state = obj["model"]
            self.cfg = obj.get("cfg", {})

    def get(self, name, expect=None):
        if name not in self.state:
            die(f"missing tensor {name} in {self.path}")
        arr = self.st.get(name) if self.st is not None else self.pt.tensor(self.state[name])
        self.consumed.add(name)
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has {tuple(arr.shape)} "
                f"in {self.path} -- not the tokenizer head this converter understands")
        return arr

    def unconsumed(self):
        return sorted(n for n in self.state if n not in self.consumed)

    def close(self):
        (self.st or self.pt).close()


class Source:
    """Thin wrapper over one safetensors file: consumed-set + expect-checked get()."""

    def __init__(self, path):
        self.path = path
        self.file = SafeTensorsFile(path)
        self.consumed = set()

    def get(self, name, expect=None):
        if name not in self.file.header:
            die(f"missing tensor {name} in {self.path}")
        arr = self.file.get(name)
        self.consumed.add(name)
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has {tuple(arr.shape)} "
                f"in {self.path} -- not the MERT checkpoint this converter understands")
        return arr

    def unconsumed(self, predicate=lambda n: True):
        return sorted(n for n in self.file.header if predicate(n) and n not in self.consumed)

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Tensor policy + GGUF writer
# ---------------------------------------------------------------------------

F32 = "f32"          # forced F32 regardless of --outtype (see the header's policy note)
NATIVE = "native"    # follows --outtype; only ever applied to 2-D matmul weights

F16_MAX = 65504.0


class Bundle:
    def __init__(self, arch):
        self.arch = arch
        self.kv = []
        self.tensors = []          # (name, np.ndarray, policy)
        self._names = set()

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr, policy):
        if name in self._names:
            die(f"duplicate output tensor {name}")
        if policy == NATIVE and arr.ndim != 2:
            # Guard-rail for the one-line policy in the file header: NATIVE means
            # "2-D matmul weight". A 3-D tensor reaching here is a mapping bug,
            # not a quantization decision.
            die(f"{name}: NATIVE policy on a {arr.ndim}-D tensor {tuple(arr.shape)} "
                f"-- only 2-D matmul weights may be narrowed")
        self._names.add(name)
        self.tensors.append((name, arr, policy))

    def write(self, path, outtype):
        w = gguf.GGUFWriter(path, self.arch)
        for fn, args in self.kv:
            getattr(w, fn)(*args)
        n = {"f32": 0, "f16": 0}
        for name, arr, policy in self.tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if policy == F32 or arr.ndim < 2 or outtype == "f32":
                w.add_tensor(name, arr)
                n["f32"] += 1
                continue
            peak = float(np.abs(arr).max()) if arr.size else 0.0
            if peak > F16_MAX:
                log(f"  {name}: |w|max={peak:.1f} exceeds f16 range, storing F32 instead")
                w.add_tensor(name, arr)
                n["f32"] += 1
            else:
                w.add_tensor(name, arr.astype(np.float16).view(np.uint16),
                             raw_shape=arr.shape, raw_dtype=gguf.GGMLQuantizationType.F16)
                n["f16"] += 1
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        size = os.path.getsize(path)
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors "
            f"({n['f32']} F32, {n['f16']} F16), {size:,} bytes ({size / 1e9:.3f} GB)")
        return size, n


# ---------------------------------------------------------------------------
# KV block
# ---------------------------------------------------------------------------

def add_meta(b, outtype, mert_pin_ok, head_sha, head_cfg):
    b.meta("add_name", "YuE2 semantic tokenizer")
    b.meta("add_description",
           "Real-audio -> YuE2 semantic codes. MERT-v2-FullSong mel frontend + "
           "ConvNext subsampling + Conformer blocks 0..20 (hidden_states[20]), then "
           "Mothersuperior's 8-layer transformer head to a 32768-way codebook. "
           "YuE2 token id = argmax + 151853.")
    b.meta("add_license", LICENSE_NAME)
    b.meta("add_string", "yue2tok.license_attribution", LICENSE_ATTRIBUTION)
    b.meta("add_uint32", "yue2tok.converter_version", CONVERTER_VERSION)
    # Every tensor keeps its PyTorch row-major shape, which gguf-py writes with the
    # dims reversed: a Linear [out, in] lands as ne = (in, out), exactly what
    # ggml_mul_mat wants. Spelled out because the two 2-D tensors that are NOT
    # Linears read oddly otherwise: mert.mel.filterbank is [n_freq=1025, n_mel=128]
    # -> ne (128, 1025), and tok.pos is [WIN=512, D=512] -> ne (512, 512).
    b.meta("add_string", "yue2tok.tensor_layout",
           "PyTorch row-major shapes, so ggml ne is reversed: Linear [out,in] -> "
           "ne (in,out). mert.mel.filterbank is [n_freq 1025, n_mel 128] (the "
           "orientation spectrum^T @ fb wants). Squeezed from the checkpoint: "
           "tok.pos [1,WIN,D] -> [WIN,D]; GRN [1,1,C] -> [C]; depthwise kernels "
           "[C,1,k] -> [C,k]; kernel-1 pointwise convs [out,in,1] -> [out,in]. "
           "The kernel-2 resampling convs keep their 3-D [out,in,2].")
    b.meta("add_file_type", int({"f16": gguf.LlamaFileType.MOSTLY_F16,
                                 "f32": gguf.LlamaFileType.ALL_F32}[outtype]))

    # ---- audio + mel frontend (§3.3). Every one of these is something a loader
    # would otherwise have to guess or hardcode a second time. ----------------
    b.meta("add_uint32", "yue2tok.mert.sample_rate", MERT_SR)
    b.meta("add_uint32", "yue2tok.mert.n_fft", MERT_N_FFT)
    b.meta("add_uint32", "yue2tok.mert.win_length", MERT_WIN_LENGTH)
    b.meta("add_uint32", "yue2tok.mert.hop_length", MERT_HOP_LENGTH)
    b.meta("add_uint32", "yue2tok.mert.num_mel_bins", MERT_N_MEL)
    b.meta("add_uint32", "yue2tok.mert.num_freq_bins", MERT_N_FREQ)
    b.meta("add_float32", "yue2tok.mert.frame_rate", MERT_FRAME_RATE)
    b.meta("add_uint32", "yue2tok.mert.samples_per_frame", MERT_SAMPLES_PER_FRAME)
    b.meta("add_uint32", "yue2tok.mert.minimum_input_samples", MERT_MIN_INPUT_SAMPLES)
    b.meta("add_float32", "yue2tok.mert.spectrogram_power", MERT_SPEC_POWER)
    b.meta("add_bool", "yue2tok.mert.stft_center", True)
    b.meta("add_string", "yue2tok.mert.stft_pad_mode", "reflect")
    b.meta("add_bool", "yue2tok.mert.stft_onesided", True)
    b.meta("add_bool", "yue2tok.mert.stft_normalized", False)
    b.meta("add_float32", "yue2tok.mert.db_amin", MERT_DB_AMIN)
    b.meta("add_float32", "yue2tok.mert.db_multiplier", 10.0)
    b.meta("add_bool", "yue2tok.mert.db_top_db_clamp", False)
    b.meta("add_bool", "yue2tok.mert.drop_last_mel_frame", True)
    b.meta("add_float32", "yue2tok.mert.mel_std_floor", MERT_MEL_STD_FLOOR)
    b.meta("add_string", "yue2tok.mert.frontend_formula",
           "mel = 10*log10(max(|stft|^2 @ fb, 1e-10)); drop last frame; "
           "(mel - mel_mean) / max(mel_std, 1e-5); output [T_mel, 128] time-major")
    b.meta("add_string", "yue2tok.mert.frontend_precision", "float32 (autocast disabled)")

    # ---- ConvNext subsampling (§3.4) ---------------------------------------
    b.meta("add_uint32", "yue2tok.mert.subsampling.block_count", len(MERT_SUB_CHANNELS))
    b.meta("add_uint32", "yue2tok.mert.subsampling.input_channels", MERT_N_MEL)
    b.meta("add_array", "yue2tok.mert.subsampling.channels", list(MERT_SUB_CHANNELS))
    b.meta("add_array", "yue2tok.mert.subsampling.depths", list(MERT_SUB_DEPTHS))
    b.meta("add_array", "yue2tok.mert.subsampling.strides", list(MERT_SUB_STRIDES))
    b.meta("add_uint32", "yue2tok.mert.subsampling.resample_kernel", MERT_SUB_KERNEL)
    b.meta("add_bool", "yue2tok.mert.subsampling.resample_padding", False)
    b.meta("add_string", "yue2tok.mert.subsampling.resample_out_len",
           "floor((L - 2)/stride) + 1 -- NOT L/2; odd tails lose a frame")
    b.meta("add_uint32", "yue2tok.mert.subsampling.convnext_kernel", MERT_CNX_KERNEL)
    b.meta("add_uint32", "yue2tok.mert.subsampling.convnext_padding", MERT_CNX_KERNEL // 2)
    b.meta("add_float32", "yue2tok.mert.subsampling.layer_norm_epsilon", MERT_SUB_LN_EPS)
    b.meta("add_float32", "yue2tok.mert.subsampling.grn_epsilon", MERT_GRN_EPS)
    b.meta("add_string", "yue2tok.mert.subsampling.grn_formula",
           "magnitude = L2 over TIME (dim=1) of the whole chunk, keepdim; "
           "normalized = magnitude / (mean_over_channels(magnitude) + 1e-6); "
           "out = weight*(h*normalized) + bias + h -- chunk-length dependent, so "
           "padding a short tail into a batch changes the WHOLE tail")

    # ---- Conformer stack (§3.5, §3.6) --------------------------------------
    b.meta("add_uint32", "yue2tok.mert.embedding_length", MERT_DIM)
    b.meta("add_uint32", "yue2tok.mert.feed_forward_length", MERT_FF)
    b.meta("add_uint32", "yue2tok.mert.block_count", MERT_BLOCKS_KEPT)
    b.meta("add_uint32", "yue2tok.mert.block_count_upstream", MERT_BLOCKS_FULL)
    b.meta("add_uint32", "yue2tok.mert.hidden_state_index", MERT_HIDDEN_STATE_INDEX)
    b.meta("add_string", "yue2tok.mert.hidden_state_note",
           "hidden_states[20] is the OUTPUT of layers[20] (the 21st block), post its "
           "own final_layer_norm. MERT2 has no input-embedding entry in hidden_states. "
           "Blocks 21..23 are not in this file.")
    b.meta("add_uint32", "yue2tok.mert.attention.head_count", MERT_HEADS)
    b.meta("add_uint32", "yue2tok.mert.attention.key_length", MERT_HEAD_DIM)
    b.meta("add_uint32", "yue2tok.mert.attention.value_length", MERT_HEAD_DIM)
    b.meta("add_float32", "yue2tok.mert.attention.softmax_scale",
           float(MERT_HEAD_DIM) ** -0.5)
    b.meta("add_bool", "yue2tok.mert.attention.causal", False)
    b.meta("add_bool", "yue2tok.mert.attention.qkv_fused", False)
    b.meta("add_float32", "yue2tok.mert.attention.layer_norm_epsilon", MERT_LN_EPS)
    b.meta("add_uint32", "yue2tok.mert.conv_depthwise_kernel_size", MERT_CONV_DW_KERNEL)
    b.meta("add_uint32", "yue2tok.mert.conv_depthwise_padding", MERT_CONV_DW_KERNEL // 2)
    b.meta("add_string", "yue2tok.mert.conv_glu_order",
           "GLU(dim=1) on [B,2048,T]: FIRST half is the linear branch, SECOND half "
           "is the gate -- a*sigmoid(b)")
    b.meta("add_float32", "yue2tok.mert.ffn_residual_scale", 0.5)
    b.meta("add_string", "yue2tok.mert.block_order",
           "h += 0.5*ffn1(ffn1_norm(h)); h += attn(attn_norm(h)); h += conv(h); "
           "h += 0.5*ffn2(ffn2_norm(h)); h = final_norm(h)")
    b.meta("add_string", "yue2tok.mert.activation", "gelu_erf")

    # ---- RoPE (§3.6). inv_freq is persistent=False and is NOT in this file. --
    b.meta("add_float32", "yue2tok.mert.rope.freq_base", MERT_ROPE_BASE)
    b.meta("add_uint32", "yue2tok.mert.rope.dimension_count", MERT_HEAD_DIM)
    b.meta("add_bool", "yue2tok.mert.rope.inv_freq_in_file", False)
    b.meta("add_string", "yue2tok.mert.rope.style", "half_split_neox")
    b.meta("add_string", "yue2tok.mert.rope.note",
           "BUILD inv_freq[i] = 1/10000^(2i/64) for i in 0..31 in fp32; angles = "
           "cat(freqs, freqs) (duplicated, NOT interleaved); element i pairs with "
           "i+32; applied to q and k only; positions RESTART AT 0 in every chunk")

    # ---- chunking and the 25 Hz bridge (§3.7) ------------------------------
    b.meta("add_uint32", "yue2tok.chunk_seconds", CHUNK_SECONDS)
    b.meta("add_uint32", "yue2tok.chunk_samples", CHUNK_SAMPLES)
    b.meta("add_uint32", "yue2tok.min_chunk_samples", MIN_CHUNK_SAMPLES)
    b.meta("add_bool", "yue2tok.tail_runs_alone", True)
    b.meta("add_string", "yue2tok.interpolate.mode", "linear")
    b.meta("add_bool", "yue2tok.interpolate.align_corners", False)
    b.meta("add_string", "yue2tok.interpolate.target",
           "T25 = round(num_samples / 24000 * 25), applied ONCE to the concatenated "
           "chunk features along time")
    b.meta("add_bool", "yue2tok.feature_store_f16", True)
    b.meta("add_float32", "yue2tok.instance_norm.epsilon", INSTNORM_EPS)
    b.meta("add_string", "yue2tok.instance_norm.formula",
           "per-channel over the WHOLE track: (x - mean_t) / (std_t + 1e-5), "
           "population std (ddof=0), float32")

    # ---- the head (§1) -----------------------------------------------------
    b.meta("add_uint32", "yue2tok.head.window", HEAD_WIN)
    b.meta("add_uint32", "yue2tok.head.window_stride", HEAD_STRIDE)
    b.meta("add_uint32", "yue2tok.head.window_trim", HEAD_TRIM)
    b.meta("add_uint32", "yue2tok.head.embedding_length", HEAD_D)
    b.meta("add_uint32", "yue2tok.head.block_count", HEAD_LAYERS)
    b.meta("add_uint32", "yue2tok.head.attention.head_count", HEAD_HEADS)
    b.meta("add_uint32", "yue2tok.head.attention.key_length", HEAD_HEAD_DIM)
    b.meta("add_uint32", "yue2tok.head.attention.value_length", HEAD_HEAD_DIM)
    b.meta("add_float32", "yue2tok.head.attention.softmax_scale",
           float(HEAD_HEAD_DIM) ** -0.5)
    b.meta("add_bool", "yue2tok.head.attention.causal", False)
    b.meta("add_bool", "yue2tok.head.attention.qkv_fused", True)
    b.meta("add_string", "yue2tok.head.attention.qkv_order",
           "rows [0:512]=Wq, [512:1024]=Wk, [1024:1536]=Wv; same order for the bias")
    b.meta("add_uint32", "yue2tok.head.feed_forward_length", HEAD_FF)
    b.meta("add_uint32", "yue2tok.head.input_length", HEAD_DIN)
    b.meta("add_uint32", "yue2tok.head.vocab_size", HEAD_VOCAB)
    b.meta("add_float32", "yue2tok.head.layer_norm_epsilon", HEAD_LN_EPS)
    b.meta("add_bool", "yue2tok.head.norm_first", True)
    b.meta("add_bool", "yue2tok.head.has_encoder_norm", False)
    b.meta("add_string", "yue2tok.head.activation", "gelu_erf")
    b.meta("add_string", "yue2tok.head.forward",
           "h = inp(x) + pos[:T]; per layer (norm_first): h += attn(norm1(h)); "
           "h += ffn_down(gelu(ffn_up(norm2(h)))); then h = tok.norm(h) "
           "(Tok.norm -- there is NO enc.norm); logits = tok.head(h)")
    b.meta("add_string", "yue2tok.head.window_policy",
           "stride 256 over T frames; starts = range(0, max(1, T-512+1), 256) plus a "
           "flush-right start at T-512 when the last strided window does not reach T; "
           "zero-pad the final window to 512 but keep only its first n outputs; write "
           "[s0 + (0 if s0==0 else 128), s0+n - (0 if s0+n>=T else 128)) in ascending "
           "order, later windows winning any overlap")

    # ---- the codec bridge --------------------------------------------------
    b.meta("add_uint32", "yue2tok.codec_offset", CODEC_OFFSET)
    b.meta("add_uint32", "yue2tok.codec_size", CODEC_SIZE)
    b.meta("add_string", "yue2tok.token_formula",
           "yue2_token_id = argmax(logits) + 151853, argmax in [0, 32768)")

    # ---- provenance --------------------------------------------------------
    b.meta("add_string", "yue2tok.pinned.mert_repo", PINS["mert"]["repo"])
    b.meta("add_string", "yue2tok.pinned.mert_sha256",
           PINS["mert"]["sha256"] if mert_pin_ok else "unknown")
    b.meta("add_string", "yue2tok.pinned.head_repo", PINS["head"]["repo"])
    b.meta("add_string", "yue2tok.pinned.head_sha256", head_sha)
    b.meta("add_string", "yue2tok.pinned.head_cfg", json.dumps(head_cfg, sort_keys=True))
    b.meta("add_string", "yue2tok.pinned.fixtures", ORACLE_FIXTURES)
    b.meta("add_string", "yue2tok.pinned.oracle_doc",
           "docs/plans/yue2/12-tokenizer-oracle-pin.md")
    b.meta("add_float32", "yue2tok.gate.fp32_vs_bf16_argmax_agreement", 0.9765625)
    b.meta("add_string", "yue2tok.gate.note",
           "Port-vs-oracle is an AGREEMENT RATE with a ~98% ceiling, never bit-exact: "
           "the reference runs MERT and the head under CUDA bf16 autocast. End-to-end "
           "chain gate: ~16% exact match against YuE2's own emitted semantic tokens "
           "(random baseline 3.05e-5).")


# ---------------------------------------------------------------------------
# Component: MERT  ->  mert.*
# ---------------------------------------------------------------------------

def build_mert(src_dir, b):
    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(cfg_path):
        die(f"no config.json in {src_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    expect_config(cfg, "model_type", "mert2")
    expect_config(cfg, "sampling_rate", MERT_SR)
    expect_config(cfg, "n_fft", MERT_N_FFT)
    expect_config(cfg, "win_length", MERT_WIN_LENGTH)
    expect_config(cfg, "hop_length", MERT_HOP_LENGTH)
    expect_config(cfg, "num_mel_bins", MERT_N_MEL)
    expect_config(cfg, "frame_rate", MERT_FRAME_RATE)
    expect_config(cfg, "inputs_to_logits_ratio", MERT_SAMPLES_PER_FRAME)
    expect_config(cfg, "minimum_input_samples", MERT_MIN_INPUT_SAMPLES)
    expect_config(cfg, "hidden_size", MERT_DIM)
    expect_config(cfg, "intermediate_size", MERT_FF)
    expect_config(cfg, "num_attention_heads", MERT_HEADS)
    expect_config(cfg, "num_hidden_layers", MERT_BLOCKS_FULL)
    expect_config(cfg, "layer_norm_eps", MERT_LN_EPS)
    expect_config(cfg, "subsampling_layer_norm_eps", MERT_SUB_LN_EPS)
    expect_config(cfg, "conv_depthwise_kernel_size", MERT_CONV_DW_KERNEL)
    expect_config(cfg, "rotary_embedding_base", int(MERT_ROPE_BASE))
    expect_config(cfg, "subsampling_channels", list(MERT_SUB_CHANNELS))
    expect_config(cfg, "subsampling_depths", list(MERT_SUB_DEPTHS))

    st_path = os.path.join(src_dir, "model.safetensors")
    if not os.path.isfile(st_path):
        die(f"no model.safetensors in {src_dir}")
    src = Source(st_path)

    # ---- frontend buffers. All four are FORCED F32 (header policy note). ----
    # The filterbank keeps its checkpoint layout [n_freq=1025, n_mel=128]: the
    # reference does spectrum.transpose(-1,-2) @ fb, so this is the orientation the
    # matmul wants, and flipping it here would only move the confusion.
    b.put("mert.mel.stft_window",
          src.get("feature_extractor.spectrogram.window", expect=(MERT_N_FFT,)), F32)
    b.put("mert.mel.filterbank",
          src.get("feature_extractor.mel_scale.fb", expect=(MERT_N_FREQ, MERT_N_MEL)), F32)
    b.put("mert.mel.mean",
          src.get("feature_extractor.mel_mean", expect=(MERT_N_MEL,)), F32)
    b.put("mert.mel.std",
          src.get("feature_extractor.mel_std", expect=(MERT_N_MEL,)), F32)

    # ---- ConvNext subsampling ---------------------------------------------
    # channels = [num_mel_bins] + subsampling_channels, so block i maps
    # channels[i] -> channels[i+1] at strides (1,2,2). Block 0 is 128->128 stride 1,
    # hence nn.Identity and NO resampling_layer.* tensors at all.
    chans = [MERT_N_MEL] + list(MERT_SUB_CHANNELS)
    for bi in range(3):
        cin, cout = chans[bi], chans[bi + 1]
        stride = MERT_SUB_STRIDES[bi]
        s = f"subsampling_module.{bi}"
        d = f"mert.sub.{bi}"

        identity = (cin == cout and stride == 1)
        if bi == 0 and not identity:
            die("subsampling block 0 is not an identity resample; the channel list "
                "does not match this converter's assumptions")
        if not identity:
            b.put(f"{d}.down_norm.weight",
                  src.get(f"{s}.resampling_layer.0.weight", expect=(cin,)), F32)
            b.put(f"{d}.down_norm.bias",
                  src.get(f"{s}.resampling_layer.0.bias", expect=(cin,)), F32)
            # Conv1d(cin, cout, kernel=2, stride=stride), NO padding. Kept 3-D:
            # the port feeds it through explicit F32 im2col, not ggml_conv_1d.
            b.put(f"{d}.down_conv.weight",
                  src.get(f"{s}.resampling_layer.2.weight",
                          expect=(cout, cin, MERT_SUB_KERNEL)), F32)
            b.put(f"{d}.down_conv.bias",
                  src.get(f"{s}.resampling_layer.2.bias", expect=(cout,)), F32)

        for li in range(MERT_SUB_DEPTHS[bi]):
            ls = f"{s}.convnext_layers.{li}"
            ld = f"{d}.cnx.{li}"
            # depthwise Conv1d(C, C, 7, padding=3, groups=C): [C,1,7] -> [C,7].
            # The middle axis is groups==C's always-1 in-channel; squeezing it makes
            # the ggml layout ne=(7, C) instead of a pointless (7,1,C).
            dw = src.get(f"{ls}.depthwise_block.1.weight",
                         expect=(cout, 1, MERT_CNX_KERNEL))
            b.put(f"{ld}.dw.weight", dw.reshape(cout, MERT_CNX_KERNEL), F32)
            b.put(f"{ld}.dw.bias",
                  src.get(f"{ls}.depthwise_block.1.bias", expect=(cout,)), F32)
            b.put(f"{ld}.norm.weight",
                  src.get(f"{ls}.pointwise_block.0.weight", expect=(cout,)), F32)
            b.put(f"{ld}.norm.bias",
                  src.get(f"{ls}.pointwise_block.0.bias", expect=(cout,)), F32)
            b.put(f"{ld}.pw_up.weight",
                  src.get(f"{ls}.pointwise_block.1.weight", expect=(4 * cout, cout)), NATIVE)
            b.put(f"{ld}.pw_up.bias",
                  src.get(f"{ls}.pointwise_block.1.bias", expect=(4 * cout,)), F32)
            # GlobalResponseNorm params are [1,1,4C]; squeezed to [4C] so they are
            # 1-D and land in the forced-F32 bucket like every other scale/shift.
            grn_w = src.get(f"{ls}.pointwise_block.3.weight", expect=(1, 1, 4 * cout))
            grn_b = src.get(f"{ls}.pointwise_block.3.bias", expect=(1, 1, 4 * cout))
            b.put(f"{ld}.grn.weight", grn_w.reshape(4 * cout), F32)
            b.put(f"{ld}.grn.bias", grn_b.reshape(4 * cout), F32)
            b.put(f"{ld}.pw_down.weight",
                  src.get(f"{ls}.pointwise_block.4.weight", expect=(cout, 4 * cout)), NATIVE)
            b.put(f"{ld}.pw_down.bias",
                  src.get(f"{ls}.pointwise_block.4.bias", expect=(cout,)), F32)

    # ---- Conformer blocks 0..20 -------------------------------------------
    for n in range(MERT_BLOCKS_KEPT):
        s = f"layers.{n}"
        d = f"mert.blk.{n}"

        for tag in ("ffn1", "ffn2"):
            b.put(f"{d}.{tag}_norm.weight",
                  src.get(f"{s}.{tag}_layer_norm.weight", expect=(MERT_DIM,)), F32)
            b.put(f"{d}.{tag}_norm.bias",
                  src.get(f"{s}.{tag}_layer_norm.bias", expect=(MERT_DIM,)), F32)
            b.put(f"{d}.{tag}_up.weight",
                  src.get(f"{s}.{tag}.w_1.weight", expect=(MERT_FF, MERT_DIM)), NATIVE)
            b.put(f"{d}.{tag}_up.bias",
                  src.get(f"{s}.{tag}.w_1.bias", expect=(MERT_FF,)), F32)
            b.put(f"{d}.{tag}_down.weight",
                  src.get(f"{s}.{tag}.w_2.weight", expect=(MERT_DIM, MERT_FF)), NATIVE)
            b.put(f"{d}.{tag}_down.bias",
                  src.get(f"{s}.{tag}.w_2.bias", expect=(MERT_DIM,)), F32)

        b.put(f"{d}.attn_norm.weight",
              src.get(f"{s}.attn_layer_norm.weight", expect=(MERT_DIM,)), F32)
        b.put(f"{d}.attn_norm.bias",
              src.get(f"{s}.attn_layer_norm.bias", expect=(MERT_DIM,)), F32)
        # Four SEPARATE projections with bias -- unlike the head (§1.5), MERT does
        # not fuse qkv. Do not "helpfully" concatenate them here.
        for dst, srcname in (("attn_q", "query_proj"), ("attn_k", "key_proj"),
                             ("attn_v", "value_proj"), ("attn_output", "out_proj")):
            b.put(f"{d}.{dst}.weight",
                  src.get(f"{s}.attn.{srcname}.weight", expect=(MERT_DIM, MERT_DIM)), NATIVE)
            b.put(f"{d}.{dst}.bias",
                  src.get(f"{s}.attn.{srcname}.bias", expect=(MERT_DIM,)), F32)

        # Convolution module. conv_block.{1,3,6} have NO bias.
        b.put(f"{d}.conv_norm.weight",
              src.get(f"{s}.conv_module.layer_norm.weight", expect=(MERT_DIM,)), F32)
        b.put(f"{d}.conv_norm.bias",
              src.get(f"{s}.conv_module.layer_norm.bias", expect=(MERT_DIM,)), F32)
        # conv_block.1: Conv1d(1024, 2048, kernel=1) feeding GLU(dim=1). kernel 1 is
        # a plain matmul, so the trailing axis is squeezed and it narrows like any
        # other 2-D weight. Rows [0:1024] are the LINEAR branch, [1024:2048] the GATE.
        pw1 = src.get(f"{s}.conv_module.conv_block.1.weight",
                      expect=(2 * MERT_DIM, MERT_DIM, 1))
        b.put(f"{d}.conv_pw1.weight", pw1.reshape(2 * MERT_DIM, MERT_DIM), NATIVE)
        # conv_block.3: depthwise Conv1d(1024, 1024, kernel=31, padding=15, groups=1024).
        dw = src.get(f"{s}.conv_module.conv_block.3.weight",
                     expect=(MERT_DIM, 1, MERT_CONV_DW_KERNEL))
        b.put(f"{d}.conv_dw.weight", dw.reshape(MERT_DIM, MERT_CONV_DW_KERNEL), F32)
        b.put(f"{d}.conv_dw_norm.weight",
              src.get(f"{s}.conv_module.conv_block.4.1.weight", expect=(MERT_DIM,)), F32)
        b.put(f"{d}.conv_dw_norm.bias",
              src.get(f"{s}.conv_module.conv_block.4.1.bias", expect=(MERT_DIM,)), F32)
        pw2 = src.get(f"{s}.conv_module.conv_block.6.weight",
                      expect=(MERT_DIM, MERT_DIM, 1))
        b.put(f"{d}.conv_pw2.weight", pw2.reshape(MERT_DIM, MERT_DIM), NATIVE)

        # The block OUTPUT norm. hidden_states[20] is POST this, for block 20.
        b.put(f"{d}.final_norm.weight",
              src.get(f"{s}.final_layer_norm.weight", expect=(MERT_DIM,)), F32)
        b.put(f"{d}.final_norm.bias",
              src.get(f"{s}.final_layer_norm.bias", expect=(MERT_DIM,)), F32)

    # ---- deliberate drops, asserted -----------------------------------------
    dropped_prefixes = tuple(f"layers.{i}." for i in
                             range(MERT_BLOCKS_KEPT, MERT_BLOCKS_FULL))
    dropped = src.unconsumed(lambda n: n.startswith(dropped_prefixes))
    expect_dropped = (MERT_BLOCKS_FULL - MERT_BLOCKS_KEPT) * 31
    if len(dropped) != expect_dropped:
        die(f"expected to drop {expect_dropped} tensors for blocks "
            f"{MERT_BLOCKS_KEPT}..{MERT_BLOCKS_FULL - 1}, found {len(dropped)} "
            f"-- the checkpoint's block layout is not what this converter expects")
    dropped_bytes = sum(
        int(np.prod(src.file.shape(n))) * 4 for n in dropped)
    log(f"MERT: dropping blocks {MERT_BLOCKS_KEPT}..{MERT_BLOCKS_FULL - 1} "
        f"({len(dropped)} tensors, {dropped_bytes / 2**30:.3f} GiB) -- "
        f"hidden_states[{MERT_HIDDEN_STATE_INDEX}] is the output of "
        f"layers[{MERT_HIDDEN_STATE_INDEX}], so they are dead weight")

    leftover = src.unconsumed(lambda n: not n.startswith(dropped_prefixes))
    if leftover:
        log(f"ERROR: {len(leftover)} MERT source tensors were neither consumed nor "
            f"deliberately dropped:")
        for n in leftover[:40]:
            log(f"    {n} {src.file.shape(n)} {src.file.dtype(n)}")
        if len(leftover) > 40:
            log(f"    ... and {len(leftover) - 40} more")
        die("the converter does not understand this checkpoint's tensor set -- "
            "refusing to leave a partial model in place")

    src.close()


# ---------------------------------------------------------------------------
# Component: tokenizer head  ->  tok.*
# ---------------------------------------------------------------------------

def build_head(path, b):
    src = HeadSource(path)
    if len(src.state) != 103:
        die(f"{path}: expected 103 head tensors, found {len(src.state)}")
    log(f"head: cfg={src.cfg!r} (free-text provenance plus the instnorm flag -- "
        f"WIN/D/L/H/VOCAB are NOT in the file; H=8 comes from the training scripts)")

    # pos is (1, WIN, D) in the checkpoint; the leading 1 is a broadcast batch axis.
    # Squeezed to [WIN, D] so the ggml ne is (D, WIN) with no trailing 1.
    # FORCED F32: additive position table, not a matmul weight, and only 1 MB.
    pos = src.get("pos", expect=(1, HEAD_WIN, HEAD_D))
    b.put("tok.pos", pos.reshape(HEAD_WIN, HEAD_D), F32)

    b.put("tok.inp.weight", src.get("inp.weight", expect=(HEAD_D, HEAD_DIN)), NATIVE)
    b.put("tok.inp.bias", src.get("inp.bias", expect=(HEAD_D,)), F32)

    for n in range(HEAD_LAYERS):
        s = f"enc.layers.{n}"
        d = f"tok.blk.{n}"
        b.put(f"{d}.norm1.weight", src.get(f"{s}.norm1.weight", expect=(HEAD_D,)), F32)
        b.put(f"{d}.norm1.bias", src.get(f"{s}.norm1.bias", expect=(HEAD_D,)), F32)
        # KEPT FUSED on purpose. The row order is pinned ([0:512]=q, [512:1024]=k,
        # [1024:1536]=v) and splitting it here would be a third place to get it
        # wrong for no gain -- the port slices views, which is free.
        b.put(f"{d}.attn_qkv.weight",
              src.get(f"{s}.self_attn.in_proj_weight", expect=(HEAD_QKV_ROWS, HEAD_D)),
              NATIVE)
        b.put(f"{d}.attn_qkv.bias",
              src.get(f"{s}.self_attn.in_proj_bias", expect=(HEAD_QKV_ROWS,)), F32)
        b.put(f"{d}.attn_output.weight",
              src.get(f"{s}.self_attn.out_proj.weight", expect=(HEAD_D, HEAD_D)), NATIVE)
        b.put(f"{d}.attn_output.bias",
              src.get(f"{s}.self_attn.out_proj.bias", expect=(HEAD_D,)), F32)
        b.put(f"{d}.norm2.weight", src.get(f"{s}.norm2.weight", expect=(HEAD_D,)), F32)
        b.put(f"{d}.norm2.bias", src.get(f"{s}.norm2.bias", expect=(HEAD_D,)), F32)
        b.put(f"{d}.ffn_up.weight",
              src.get(f"{s}.linear1.weight", expect=(HEAD_FF, HEAD_D)), NATIVE)
        b.put(f"{d}.ffn_up.bias", src.get(f"{s}.linear1.bias", expect=(HEAD_FF,)), F32)
        b.put(f"{d}.ffn_down.weight",
              src.get(f"{s}.linear2.weight", expect=(HEAD_D, HEAD_FF)), NATIVE)
        b.put(f"{d}.ffn_down.bias", src.get(f"{s}.linear2.bias", expect=(HEAD_D,)), F32)

    # Tok.norm, NOT an encoder-level norm. nn.TransformerEncoder was built with
    # norm=None, so there is no enc.norm.* key -- adding one would be wrong by a
    # whole LayerNorm. Asserted below via the unconsumed check.
    b.put("tok.norm.weight", src.get("norm.weight", expect=(HEAD_D,)), F32)
    b.put("tok.norm.bias", src.get("norm.bias", expect=(HEAD_D,)), F32)
    b.put("tok.head.weight",
          src.get("head.weight", expect=(HEAD_VOCAB, HEAD_D)), NATIVE)
    b.put("tok.head.bias", src.get("head.bias", expect=(HEAD_VOCAB,)), F32)

    leftover = src.unconsumed()
    if leftover:
        log(f"ERROR: {len(leftover)} head tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n}")
        die("the converter does not understand this head checkpoint -- refusing to "
            "leave a partial model in place")

    src.close()
    return src.cfg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_head_path(p):
    if os.path.isdir(p):
        for name in HEAD_FILE_PREFS:
            cand = os.path.join(p, name)
            if os.path.isfile(cand):
                return cand
        die(f"{p} is a directory with none of {', '.join(HEAD_FILE_PREFS)} in it")
    if not os.path.isfile(p):
        die(f"head checkpoint not found: {p}")
    return p


def check_mert_pin(src_dir):
    manifest = os.path.join(src_dir, "weights_manifest.json")
    if not os.path.isfile(manifest):
        log(f"WARNING: no weights_manifest.json at {manifest} -- writing "
            f"yue2tok.pinned.mert_sha256 as 'unknown'")
        return False
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            got = json.load(f)["sha256"]
    except (KeyError, ValueError, OSError):
        log(f"WARNING: {manifest} is not readable as expected -- writing "
            f"yue2tok.pinned.mert_sha256 as 'unknown'")
        return False
    if got != PINS["mert"]["sha256"]:
        log(f"WARNING: {manifest} sha256 {got} does not match the pinned "
            f"MERT-v2-FullSong revision -- writing 'unknown' rather than a wrong pin")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        prog="convert-yue2-tok.py",
        description="YuE2 real-audio semantic tokenizer (MERT-v2-FullSong + "
                    "Mothersuperior's head) -> one GGUF, arch 'yue2-tok'.")
    ap.add_argument("--src-mert", default="K:/yue2/models/MERT-v2-FullSong",
                    metavar="DIR", help="MERT-v2-FullSong model dir")
    ap.add_argument("--src-head", required=True, metavar="PATH",
                    help="tokenizer_head_joint_v4.pt, or the ms-tok dir holding it")
    ap.add_argument("--out", default="models/yue2", metavar="DIR",
                    help="output directory (default: models/yue2)")
    ap.add_argument("--outtype", "--type", dest="outtype", default="f16",
                    choices=("f16", "f32"),
                    help="f16 (default) narrows 2-D matmul weights only; f32 keeps "
                         "everything at checkpoint precision")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    head_path = resolve_head_path(args.src_head)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"yue2-tok-{args.outtype}.gguf")
    if os.path.exists(out_path) and not args.force:
        die(f"{out_path} already exists -- pass --force to overwrite")

    mert_pin_ok = check_mert_pin(args.src_mert)

    log(f"hashing head checkpoint {head_path} ...")
    head_sha = sha256_file(head_path)
    known = KNOWN_HEADS.get(head_sha)
    if known is None:
        log(f"WARNING: head sha256 {head_sha} is not a head this converter has been "
            f"run against -- converting anyway, but nothing here has checked it")
    else:
        log(f"head sha256 matches a known head: {known}")
    if head_sha != PINS["head"]["sha256"]:
        log(f"note: this is not the v4 head the oracle fixtures in {ORACLE_FIXTURES} "
            f"were produced with, so expect an agreement rate against them, not parity")

    b = Bundle("yue2-tok")
    build_mert(args.src_mert, b)
    n_mert = len(b.tensors)
    head_cfg = build_head(head_path, b)
    n_head = len(b.tensors) - n_mert
    log(f"assembled {len(b.tensors)} tensors: {n_mert} mert.*, {n_head} tok.*")

    add_meta(b, args.outtype, mert_pin_ok, head_sha, head_cfg)
    b.write(out_path, args.outtype)
    log("Done.")


if __name__ == "__main__":
    main()

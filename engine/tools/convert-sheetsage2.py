#!/usr/bin/env python3
# convert-sheetsage2.py: SheetSage2 (audio -> lead-sheet ABC transcriber) -> one GGUF.
#
# Produces exactly one file:
#
#   sheetsage2-<type>.gguf     arch "sheetsage2"
#       enc.*   MERT-v2-FullSong encoder, ALL 24 Conformer blocks, LoRA already
#               merged into the attention projections (§2.4) -- unlike
#               convert-yue2-tok.py's 21-of-24 cut, nothing is dropped here.
#       dec.*   the 6-layer post-LN BART-style decoder (`sheetsage_decoder.py`'s
#               plain-torch rewrite, NOT transformers.BartDecoder)
#
# This is the transport half of the contract pinned in
# docs/plans/yue2/20-sheetsage2-model-pin.md (encoder/decoder/weights, THIS
# document's primary authority -- READ IT FIRST) and
# docs/plans/yue2/21-sheetsage2-symbolic-pin.md §1 (vocabulary constants and the
# fingerprint self-check below). Both were produced by RUNNING the reference, not
# by reading it; where this file and those documents disagree, the documents win.
# Section references below (§4.2, §1.4, ...) are into doc 20 unless marked "sym"
# for doc 21. Modelled on convert-yue2-tok.py: same safetensors reader, same
# Bundle/put/write shape, same "{arch}." KV prefix convention, same
# refuse-on-unconsumed-tensors discipline.
#
# THE SOURCE FILE IS ALREADY MERGED. Unlike convert-yue2-tok.py (two separate
# checkpoints: MERT body + a torch .pt head), this converter reads exactly one
# safetensors file: the Comfy repack with the encoder's attention LoRA already
# folded in (doc 20 §0's `weights_format="merged"` case). There is no
# `weights_format` key in the safetensors metadata to check directly (measured:
# the file's `__metadata__` is `{base_model, license, source, format, weight_dtype}`
# -- doc 20 §4.5's "refuse if weights_format isn't 'merged'" is therefore enforced
# by the stronger, directly-measurable proxy doc 20 §2.4/§5.1 already establish:
# a merged file has ZERO tensor keys containing "adapter" or "lora" anywhere in
# their name. An unmerged "adapter"-format file would carry an `EncoderAdapters`
# module and fail this check; a merged file passes it. This converter asserts
# that emptiness rather than trusting an absent metadata field.
#
# WHY ALL 24 BLOCKS AND NOT 21 (doc 20 §1's "bottom line" item 1, §2.4). SheetSage2's
# `layer_weight` softmax mixes ALL 24 Conformer block outputs plus the
# post-subsampling embedding (25 states total, doc 20 §2.5) -- there is no single
# "the hidden state we want" the way yue2-tok's hidden_states[20] is. Every block
# is load-bearing, including block 23, which carries the single largest LoRA delta
# in the whole encoder (doc 20 §5.3: out_proj rel-L2 15.2, max|delta| 44.0).
#
# WHAT DOES *NOT* GO IN THE FILE, ON PURPOSE
#   Same RoPE omission as convert-yue2-tok.py: embed_positions.inv_freq is
#   registered persistent=False in the reference, so it is NOT in the checkpoint
#   and must be BUILT by the loader: inv_freq[i] = 1/10000^(2i/64), i in 0..31.
#   `output_projection.weight` is measured bit-identical to `token_embedding.weight`
#   (doc 20 §3.5, §4.1's "1039 of 1040" accounting) -- read and verified here, but
#   deliberately not re-written; `dec.tok_embd` is used for both the embedding
#   lookup and the output projection (transposed matmul) at runtime.
#
# ---------------------------------------------------------------------------
# QUANTIZATION POLICY (doc 20 §4.4) -- same one-line rule as convert-yue2-tok.py:
# **F16 (or bf16, see below) is applied only to 2-D matmul weights.** Everything
# else -- norms, biases, layer_weight, the four F32 mel buffers, any tensor with
# ndim < 2 -- is FORCED F32 regardless of --outtype. The source file is 100% BF16
# except those four mel buffers (measured, doc 20 §4.1); default --outtype is f16,
# a narrowing re-round from the source's own bf16 mantissa (both are 16-bit floats
# with different exponent/mantissa splits -- this is not a lossless copy the way
# "bf16 in, bf16 out" would be, but matches this repo's existing GGUF convention of
# f16 as the shipped default; --outtype f32 is the exact-widen variant with no
# storage narrowing at all).
#
# TOLERANCE (doc 20 §6). The reference decoder runs under CUDA bf16 autocast in
# production; the port's own activations run F32 throughout (more accurate, not
# merely matching). Doc 19's G1 gate is informational (rel-L2 ~1e-3 or better
# expected); the decisive gates are G2 (100% argmax agreement, teacher-forced) and
# G4/G6 (ABC byte-identical) -- this converter cannot and does not gate those; it
# only produces the GGUF and confirms it round-trips.
#
# ---------------------------------------------------------------------------
# DEPENDENCIES: numpy + gguf (llama.cpp gguf-py). NO TORCH -- one safetensors
# file, read via the same mmap'd struct-header reader convert-yue2-tok.py uses.
#
# USAGE
#   K:/yue2/.venv-sheetsage/Scripts/python.exe engine/tools/convert-sheetsage2.py \
#       --src K:/yue2-bakeoff/ai-toolkit/models/audio_encoders/sheetsage2_bf16.safetensors \
#       --out models/yue2 --outtype f16
#
# The default --out is models/yue2, so the shipped path is
# models/yue2/sheetsage2-f16.gguf.

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys

import numpy as np
import gguf

CONVERTER_VERSION = 1

# ---------------------------------------------------------------------------
# Model constants. Hardcoded on purpose, same reasoning as convert-yue2-tok.py: a
# converter that only reads shapes from the checkpoint cannot reject a wrong one.
# All values are cross-cited to doc 20 (model/weights) or doc 21 (vocabulary).
# ---------------------------------------------------------------------------

# -- shared mel frontend / MERT2 encoder body (doc 20 §1.3, identical to
#    yue2-mert.h / convert-yue2-tok.py's own MERT constants) -----------------
SR = 24000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 240
N_MEL = 128
N_FREQ = N_FFT // 2 + 1                    # 1025, onesided
FRAME_RATE = 25.0
SAMPLES_PER_FRAME = 960                    # inputs_to_logits_ratio
MIN_INPUT_SAMPLES = 1025
SPEC_POWER = 2.0
DB_AMIN = 1e-10
MEL_STD_FLOOR = 1e-5

# -- encoder (Conformer stack) -- doc 20 §2, config.json backbone_config -----
ENC_DIM = 1024
ENC_FF = 4096
ENC_HEADS = 16
ENC_HEAD_DIM = 64
ENC_BLOCKS = 24                            # ALL 24 -- doc 20 §1 item 1, unlike yue2-tok
ENC_LN_EPS = 1e-5                          # Conformer stack layer_norm_eps
ENC_CONV_DW_KERNEL = 31
ENC_ROPE_BASE = 10000.0

# -- ConvNext subsampling -- doc 20 §2.1, identical constants to yue2-mert.h -
SUB_CHANNELS = (128, 512, 1024)
SUB_DEPTHS = (3, 4, 5)
SUB_STRIDES = (1, 2, 2)
SUB_LN_EPS = 1e-6
SUB_KERNEL = 2
CNX_KERNEL = 7
GRN_EPS = 1e-6

# -- encoder head: layer_weight softmax + projection -- doc 20 §2.5 ---------
N_HIDDEN_STATES = ENC_BLOCKS + 1           # 25: embedding + 24 block outputs
PROJ_OUT = 512                             # encoder_projection: 1024 -> 512

# -- LoRA merge provenance (doc 20 §2.4) -- already folded into the weights,
#    recorded here for informational KVs only, never re-applied. -------------
LORA_RANK = 64
LORA_ALPHA = 128.0

# -- decoder (doc 20 §3) -----------------------------------------------------
DEC_DIM = 512
DEC_FF = 2048
DEC_HEADS = 8
DEC_HEAD_DIM = 64
DEC_BLOCKS = 6
DEC_POSITION_OFFSET = 2                    # doc 20 §3.1
DEC_MAX_POSITIONS = 5122                   # [max_output_seq_len(5120) + 2, 512]
DEC_LN_EPS = 1e-5                          # PyTorch default; no config override

# -- vocabulary / tokens (doc 21 §0, §1) -------------------------------------
VOCAB_SIZE = 31678
TIME_HZ = 100
AUDIO_LENGTH_SECONDS = 300.0               # window length == config input_audio_length
MAX_OUTPUT_SEQ_LEN = 5120
TOKEN_PAD, TOKEN_SOS, TOKEN_EOS, TOKEN_OUT = 0, 1, 2, 3
PROMPT_CAPACITY = 256
MAX_SUBBEAT_SHIFT = 256
N_EIGHTH_POSITIONS = 256
N_PITCH_TOKENS = 256
N_KEY_TOKENS = 24
SCHEMA_VERSION = "v1"
VOCAB_FINGERPRINT = "5ba3325af0344c7f"     # doc 21 §0 / §1.8, config.json's own pin

CHROMATIC_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# V1_TASKS, schema order (doc 21 §1.3) -- ids 4..11
PROMPT_NAMES = ("timestamp", "downbeat_meter", "structure", "key",
                "chord_majmin", "chord_full", "melody_vocal", "melody_full")
# FULL_TASK_PROMPTS -- the only prompt set the port needs (doc 21 §2)
DEFAULT_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key",
                   "chord_full", "melody_full")
EVENT_FIELD_ORDER = ("timestamp", "rhythm", "structure", "key", "chord", "melody")

# STRUCTURE_LABELS -- the SECOND (surviving) definition, 23 entries, in id order
# (doc 21 §1.4; labels_sheetsage2.py defines this list twice, only the second
# survives Python's plain name rebinding -- reading only the first gets 22, not
# 23, and shifts every id after this block by one).
STRUCTURE_LABELS = (
    "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus",
    "post-chorus", "interlude", "fade-out", "loop", "rap", "preshot",
    "irregular", "instrumental", "intro and verse", "pre-chorus and chorus",
    "verse and pre-chorus", "solo", "theme", "development", "variation",
    "pre-outro",
)
assert len(STRUCTURE_LABELS) == 23, "structure label count regressed"

# meter: numerator outer (1..32), denominator inner (doc 21 §1.2)
METER_NUMERATORS = tuple(range(1, 33))               # 32
METER_DENOMINATORS = (1, 2, 4, 8, 16, 32)            # 6

# FULL_CHORD_QUALITIES / FULL_CHORD_INVERSIONS / build order -- doc 21 §1.5,
# quality outer, root middle, inversion inner, plain-root last.
FULL_CHORD_QUALITIES = (
    "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7",
    "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6",
)
FULL_CHORD_INVERSIONS = {
    "maj": ("/2", "/3", "/5"),
    "min": ("/2", "/b3", "/5"),
    "maj7": ("/3", "/5", "/7"),
    "min7": ("/b3", "/5", "/b7"),
    "7": ("/3", "/5", "/b7"),
}

# duration templates, in subbeat units (doc 21 §1.6)
DURATION_TEMPLATES = (
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
)
assert len(DURATION_TEMPLATES) == 24

APPENDED_TOKEN_BLOCKS = ()                 # v1 declares none (doc 21 §1.2)

# -- licence. Same terms and the same 2026-09-15 authors' clarification as
# convert-yue2-tok.py's head/MERT: CC BY-NC 4.0, individual/non-commercial use
# is unrestricted, only companies need a commercial licence. --------------
LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_ATTRIBUTION = (
    "Weights are CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 "
    "International), and anything trained through this model carries the same "
    "terms. The YuE2 authors have clarified that individual creators, musicians "
    "and researchers may use the model and its outputs freely, including "
    "commercially; only companies need a commercial licence. "
    "Attribution: m-a-p/SheetSage2 (https://huggingface.co/m-a-p/SheetSage2), "
    "built on m-a-p/MERT-v2-FullSong "
    "(https://huggingface.co/m-a-p/MERT-v2-FullSong)."
)

# -- pinned provenance (doc 20 §0) -------------------------------------------
BASE_REPO = "m-a-p/MERT-v2-FullSong"
BASE_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
BASE_SHA256 = "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"
SHEETSAGE2_REPO = "m-a-p/SheetSage2"
ORACLE_FIXTURES = "_experiments/yue2-sheetsage/fixtures/{short-a,short-b,long-a,long-b}"
MODEL_PIN_DOC = "docs/plans/yue2/20-sheetsage2-model-pin.md"
SYMBOLIC_PIN_DOC = "docs/plans/yue2/21-sheetsage2-symbolic-pin.md"

# Reference Python files this converter's KV constants were transcribed from
# (doc 20 §0's file inventory, doc 21's citations) -- hashed for provenance,
# never imported or executed.
PYTHON_REF_DIR = ("K:/yue2/.cache/huggingface-sheetsage/modules/"
                   "transformers_modules/SheetSage2")
PYTHON_REF_FILES = (
    "audio_sheetsage2.py", "configuration_mert2.py", "configuration_sheetsage2.py",
    "durations_sheetsage2.py", "exports_sheetsage2.py", "generation_sheetsage2.py",
    "io_sheetsage2.py", "labels_sheetsage2.py", "midi_sheetsage2.py",
    "modeling_mert2.py", "modeling_sheetsage2.py", "notation_sheetsage2.py",
    "pipeline_sheetsage2.py", "processing_sheetsage2.py", "rendering_sheetsage2.py",
    "schema_sheetsage2.py", "tensors_sheetsage2.py", "tokenization_sheetsage2.py",
)


def log(msg):
    print(f"[convert-sheetsage2] {msg}", file=sys.stderr, flush=True)


def die(msg):
    raise SystemExit(f"[convert-sheetsage2] ERROR: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Vocabulary self-check: build the exact 13-key fingerprint payload doc 21
# §1.8 / §4.6 describes, from the SAME literal constants above that also drive
# the KVs, and assert it hashes to the reference's own baked fingerprint. This
# is a build-time proof that the constants transcribed into this file (not the
# Python module) are right, independent of any tensor or fixture.
# ---------------------------------------------------------------------------

def build_full_chord_labels():
    labels = ["N"]
    for quality in FULL_CHORD_QUALITIES:
        inversions = FULL_CHORD_INVERSIONS.get(quality, ())
        for root in CHROMATIC_SHARPS:
            for inversion in (*inversions, ""):
                labels.append(f"{root}:{quality}{inversion}")
    return tuple(labels)


def build_majmin_chord_labels():
    # "N" + 12 major + 12 minor (doc 20 §4.6 block-size table; doc 21 §1's
    # chord_majmin block is dead in this port -- built only for the fingerprint
    # payload and the block-size arithmetic, never emitted or decoded).
    labels = ["N"]
    labels += [f"{r}:maj" for r in CHROMATIC_SHARPS]
    labels += [f"{r}:min" for r in CHROMATIC_SHARPS]
    return tuple(labels)


def build_meter_pairs():
    return tuple((n, d) for n in METER_NUMERATORS for d in METER_DENOMINATORS)


def compute_vocab_fingerprint():
    full_chord_labels = build_full_chord_labels()
    majmin_chord_labels = build_majmin_chord_labels()
    meter_pairs = build_meter_pairs()
    n_tokens = vocab_block_table()[-1][2]  # end of the last block == n_tokens
    payload = {
        "schema_version": SCHEMA_VERSION,
        "audio_length_seconds": AUDIO_LENGTH_SECONDS,
        "time_hz": TIME_HZ,
        "prompt_capacity": PROMPT_CAPACITY,
        "prompt_names": list(PROMPT_NAMES),
        "event_field_order": list(EVENT_FIELD_ORDER),
        "meter_pairs": [list(p) for p in meter_pairs],
        "structure_labels": list(STRUCTURE_LABELS),
        "majmin_chord_labels": list(majmin_chord_labels),
        "full_chord_labels": list(full_chord_labels),
        "duration_templates": [int(x) for x in DURATION_TEMPLATES],
        "appended_token_blocks": list(APPENDED_TOKEN_BLOCKS),
        "n_tokens": n_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return digest, full_chord_labels, majmin_chord_labels, meter_pairs


def vocab_block_table():
    """[(name, start, end), ...] in id order -- doc 21 §1.2's table, recomputed
    from the same constants rather than transcribed a second time as literals."""
    blocks = []
    pos = 0
    blocks.append(("specials", pos, pos + 4)); pos += 4
    blocks.append(("prompt", pos, pos + PROMPT_CAPACITY)); pos += PROMPT_CAPACITY
    n_shift = MAX_SUBBEAT_SHIFT + 1
    blocks.append(("subbeat_shift", pos, pos + n_shift)); pos += n_shift
    n_time = round(AUDIO_LENGTH_SECONDS * TIME_HZ)
    blocks.append(("time", pos, pos + n_time)); pos += n_time
    n_meter = len(METER_NUMERATORS) * len(METER_DENOMINATORS)
    blocks.append(("meter", pos, pos + n_meter)); pos += n_meter
    blocks.append(("eighth_position", pos, pos + N_EIGHTH_POSITIONS)); pos += N_EIGHTH_POSITIONS
    blocks.append(("structure", pos, pos + len(STRUCTURE_LABELS))); pos += len(STRUCTURE_LABELS)
    blocks.append(("key", pos, pos + N_KEY_TOKENS)); pos += N_KEY_TOKENS
    n_majmin = 1 + 2 * len(CHROMATIC_SHARPS)
    blocks.append(("chord_majmin", pos, pos + n_majmin)); pos += n_majmin
    n_full_chord = len(build_full_chord_labels())
    blocks.append(("chord_full", pos, pos + n_full_chord)); pos += n_full_chord
    blocks.append(("pitch", pos, pos + N_PITCH_TOKENS)); pos += N_PITCH_TOKENS
    blocks.append(("duration", pos, pos + len(DURATION_TEMPLATES))); pos += len(DURATION_TEMPLATES)
    blocks.append(("appended", pos, pos + 0))
    return blocks


def verify_vocab_arithmetic():
    table = vocab_block_table()
    n_tokens = table[-1][2]
    if n_tokens != VOCAB_SIZE:
        die(f"vocab arithmetic closes at {n_tokens}, expected {VOCAB_SIZE} -- a "
            f"block-size constant above is wrong (doc 21 §1.2/§4.6)")
    fp, full_chord_labels, majmin_chord_labels, meter_pairs = compute_vocab_fingerprint()
    if fp != VOCAB_FINGERPRINT:
        die(f"computed vocab fingerprint {fp} != pinned {VOCAB_FINGERPRINT} -- the "
            f"vocabulary constants transcribed into this converter do not match "
            f"doc 21's reference derivation")
    log(f"vocab self-check OK: n_tokens={n_tokens}, fingerprint={fp}, "
        f"full_chord_labels={len(full_chord_labels)}, "
        f"majmin_chord_labels={len(majmin_chord_labels)}, "
        f"meter_pairs={len(meter_pairs)}")
    return table, full_chord_labels


# ---------------------------------------------------------------------------
# safetensors reader -- lifted verbatim from convert-yue2-tok.py.
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
                f"in {self.path} -- not the SheetSage2 checkpoint this converter understands")
        return arr

    def unconsumed(self, predicate=lambda n: True):
        return sorted(n for n in self.file.header if predicate(n) and n not in self.consumed)

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Tensor policy + GGUF writer -- identical shape to convert-yue2-tok.py's Bundle.
# ---------------------------------------------------------------------------

F32 = "f32"          # forced F32 regardless of --outtype
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

def add_meta(b, outtype, src_sha256, python_sha256, python_files_hashed, vocab_table):
    b.meta("add_name", "SheetSage2 lead-sheet transcriber")
    b.meta("add_description",
           "Audio -> ABC lead sheet (structure/key/chord/melody/rhythm), used as the "
           "YuE2 AR adapters' chain-of-thought sheet under cot=full. MERT-v2-FullSong "
           "encoder (all 24 blocks, LoRA-merged attention) + a 6-layer post-LN BART-style "
           "decoder, greedy grammar-masked decode.")
    b.meta("add_license", LICENSE_NAME)
    b.meta("add_string", "sheetsage2.license_attribution", LICENSE_ATTRIBUTION)
    b.meta("add_uint32", "sheetsage2.converter_version", CONVERTER_VERSION)
    b.meta("add_string", "sheetsage2.tensor_layout",
           "PyTorch row-major shapes, so ggml ne is reversed: Linear [out,in] -> "
           "ne (in,out). enc.mel.filterbank is [n_freq 1025, n_mel 128] -> ne (128,1025) "
           "(the orientation spectrum^T @ fb wants). Squeezed from the checkpoint: "
           "GRN [1,1,4C] -> [4C]; depthwise kernels [C,1,k] -> [C,k]; kernel-1 pointwise "
           "convs [out,in,1] -> [out,in]. The kernel-2 resampling convs keep their 3-D "
           "[out,in,2] shape and stay F32 (im2col path, no ggml_conv_1d).")
    b.meta("add_file_type", int({"f16": gguf.LlamaFileType.MOSTLY_F16,
                                 "f32": gguf.LlamaFileType.ALL_F32}[outtype]))

    # ---- audio + mel frontend (doc 20 §1.3, identical to yue2tok.mert.mel.*) --
    b.meta("add_uint32", "sheetsage2.sample_rate", SR)
    b.meta("add_float32", "sheetsage2.window_seconds", AUDIO_LENGTH_SECONDS)
    b.meta("add_uint32", "sheetsage2.mel.n_fft", N_FFT)
    b.meta("add_uint32", "sheetsage2.mel.win_length", WIN_LENGTH)
    b.meta("add_uint32", "sheetsage2.mel.hop_length", HOP_LENGTH)
    b.meta("add_uint32", "sheetsage2.mel.num_mel_bins", N_MEL)
    b.meta("add_uint32", "sheetsage2.mel.num_freq_bins", N_FREQ)
    b.meta("add_float32", "sheetsage2.mel.frame_rate", FRAME_RATE)
    b.meta("add_uint32", "sheetsage2.mel.samples_per_frame", SAMPLES_PER_FRAME)
    b.meta("add_uint32", "sheetsage2.mel.minimum_input_samples", MIN_INPUT_SAMPLES)
    b.meta("add_float32", "sheetsage2.mel.spectrogram_power", SPEC_POWER)
    b.meta("add_bool", "sheetsage2.mel.stft_center", True)
    b.meta("add_string", "sheetsage2.mel.stft_pad_mode", "reflect")
    b.meta("add_bool", "sheetsage2.mel.stft_onesided", True)
    b.meta("add_bool", "sheetsage2.mel.stft_normalized", False)
    b.meta("add_float32", "sheetsage2.mel.db_amin", DB_AMIN)
    b.meta("add_float32", "sheetsage2.mel.db_multiplier", 10.0)
    b.meta("add_bool", "sheetsage2.mel.db_top_db_clamp", False)
    b.meta("add_bool", "sheetsage2.mel.drop_last_mel_frame", True)
    b.meta("add_float32", "sheetsage2.mel.mel_std_floor", MEL_STD_FLOOR)
    b.meta("add_string", "sheetsage2.mel.frontend_formula",
           "mel = 10*log10(max(|stft|^2 @ fb, 1e-10)); drop last frame; "
           "(mel - mel_mean) / max(mel_std, 1e-5); output [T_mel, 128] time-major")
    b.meta("add_string", "sheetsage2.mel.frontend_precision", "float32 (autocast disabled)")

    # ---- window plan -- always the full padded window, never masked (doc 20 §1.2/§1.4)
    b.meta("add_bool", "sheetsage2.window.always_full_padded", True)
    b.meta("add_string", "sheetsage2.window.padding_side", "right (zero-fill)")
    b.meta("add_float32", "sheetsage2.window.overlap_seconds_default", 200.0)
    b.meta("add_float32", "sheetsage2.window.lookahead_seconds_default", 100.0)

    # ---- ConvNext subsampling (doc 20 §2.1) --------------------------------
    b.meta("add_uint32", "sheetsage2.subsampling.block_count", len(SUB_CHANNELS))
    b.meta("add_uint32", "sheetsage2.subsampling.input_channels", N_MEL)
    b.meta("add_array", "sheetsage2.subsampling.channels", list(SUB_CHANNELS))
    b.meta("add_array", "sheetsage2.subsampling.depths", list(SUB_DEPTHS))
    b.meta("add_array", "sheetsage2.subsampling.strides", list(SUB_STRIDES))
    b.meta("add_uint32", "sheetsage2.subsampling.resample_kernel", SUB_KERNEL)
    b.meta("add_bool", "sheetsage2.subsampling.resample_padding", False)
    b.meta("add_uint32", "sheetsage2.subsampling.convnext_kernel", CNX_KERNEL)
    b.meta("add_uint32", "sheetsage2.subsampling.convnext_padding", CNX_KERNEL // 2)
    b.meta("add_float32", "sheetsage2.subsampling.layer_norm_epsilon", SUB_LN_EPS)
    b.meta("add_float32", "sheetsage2.subsampling.grn_epsilon", GRN_EPS)

    # ---- Conformer encoder stack (doc 20 §2.2/§2.3) ------------------------
    b.meta("add_uint32", "sheetsage2.encoder.block_count", ENC_BLOCKS)
    b.meta("add_uint32", "sheetsage2.encoder.embedding_length", ENC_DIM)
    b.meta("add_uint32", "sheetsage2.encoder.feed_forward_length", ENC_FF)
    b.meta("add_uint32", "sheetsage2.encoder.attention.head_count", ENC_HEADS)
    b.meta("add_uint32", "sheetsage2.encoder.attention.key_length", ENC_HEAD_DIM)
    b.meta("add_uint32", "sheetsage2.encoder.attention.value_length", ENC_HEAD_DIM)
    b.meta("add_float32", "sheetsage2.encoder.attention.softmax_scale",
           float(ENC_HEAD_DIM) ** -0.5)
    b.meta("add_bool", "sheetsage2.encoder.attention.causal", False)
    b.meta("add_bool", "sheetsage2.encoder.attention.qkv_fused", False)
    b.meta("add_float32", "sheetsage2.encoder.layer_norm_epsilon", ENC_LN_EPS)
    b.meta("add_uint32", "sheetsage2.encoder.conv_depthwise_kernel_size", ENC_CONV_DW_KERNEL)
    b.meta("add_uint32", "sheetsage2.encoder.conv_depthwise_padding", ENC_CONV_DW_KERNEL // 2)
    b.meta("add_string", "sheetsage2.encoder.conv_glu_order",
           "GLU(dim=1) on [B,2048,T]: FIRST half is the linear branch, SECOND half "
           "is the gate -- a*sigmoid(b)")
    b.meta("add_float32", "sheetsage2.encoder.ffn_residual_scale", 0.5)
    b.meta("add_string", "sheetsage2.encoder.block_order",
           "h += 0.5*ffn1(ffn1_norm(h)); h += attn(attn_norm(h)) [no mask, full "
           "bidirectional over all 7500 frames incl. padding]; h += conv(h); "
           "h += 0.5*ffn2(ffn2_norm(h)); h = final_norm(h)")
    b.meta("add_string", "sheetsage2.encoder.activation", "gelu_erf")
    b.meta("add_bool", "sheetsage2.encoder.attention_lora_merged", True)
    b.meta("add_uint32", "sheetsage2.encoder.attention_lora_rank", LORA_RANK)
    b.meta("add_float32", "sheetsage2.encoder.attention_lora_alpha", LORA_ALPHA)

    # ---- RoPE (doc 20 §2.2). inv_freq is persistent=False -- NOT in this file.
    b.meta("add_float32", "sheetsage2.rope.freq_base", ENC_ROPE_BASE)
    b.meta("add_uint32", "sheetsage2.rope.dimension_count", ENC_HEAD_DIM)
    b.meta("add_bool", "sheetsage2.rope.inv_freq_in_file", False)
    b.meta("add_string", "sheetsage2.rope.style", "half_split_neox")
    b.meta("add_string", "sheetsage2.rope.note",
           "BUILD inv_freq[i] = 1/10000^(2i/64) for i in 0..31 in fp32; angles = "
           "cat(freqs, freqs) (duplicated, NOT interleaved); applied to q and k only; "
           "positions RESTART AT 0 every 300 s window, always covering 0..7499 "
           "(the window is always exactly 7500 frames, doc 20 §1.2/§2.2)")

    # ---- encoder head: layer_weight softmax + projection (doc 20 §2.5) ----
    b.meta("add_uint32", "sheetsage2.encoder.hidden_state_count", N_HIDDEN_STATES)
    b.meta("add_string", "sheetsage2.encoder.output_formula",
           "weights = softmax(layer_weight); mixed = hidden0*weights[0]; for each "
           "block N in 0..23: hidden = block(hidden); mixed += hidden*weights[N+1]; "
           "memory = encoder_projection(mixed). hidden0 is the post-subsampling "
           "embedding, BEFORE any Conformer block runs.")
    b.meta("add_uint32", "sheetsage2.encoder.projection_out_length", PROJ_OUT)

    # ---- decoder (doc 20 §3) ------------------------------------------------
    b.meta("add_uint32", "sheetsage2.decoder.block_count", DEC_BLOCKS)
    b.meta("add_uint32", "sheetsage2.decoder.embedding_length", DEC_DIM)
    b.meta("add_uint32", "sheetsage2.decoder.feed_forward_length", DEC_FF)
    b.meta("add_uint32", "sheetsage2.decoder.attention.head_count", DEC_HEADS)
    b.meta("add_uint32", "sheetsage2.decoder.attention.key_length", DEC_HEAD_DIM)
    b.meta("add_uint32", "sheetsage2.decoder.attention.value_length", DEC_HEAD_DIM)
    b.meta("add_bool", "sheetsage2.decoder.attention.qkv_fused", False)
    b.meta("add_string", "sheetsage2.decoder.norm_style", "post")
    b.meta("add_string", "sheetsage2.decoder.activation", "gelu_erf")
    b.meta("add_float32", "sheetsage2.decoder.layer_norm_epsilon", DEC_LN_EPS)
    b.meta("add_uint32", "sheetsage2.decoder.position_offset", DEC_POSITION_OFFSET)
    b.meta("add_uint32", "sheetsage2.decoder.max_position_embeddings", DEC_MAX_POSITIONS)
    b.meta("add_float32", "sheetsage2.decoder.embed_scale", 1.0)
    b.meta("add_string", "sheetsage2.decoder.layer_order",
           "h,self_kv = self_attn(x, causal_mask, past=self_cache); "
           "x = self_attn_norm(x+h); h,cross_kv = cross_attn(x, memory, "
           "past=cross_cache, cache_static=True); x = cross_attn_norm(x+h); "
           "x = final_norm(x + fc2(gelu(fc1(x))))")
    b.meta("add_bool", "sheetsage2.decoder.cross_attn_kv_static", True)
    b.meta("add_bool", "sheetsage2.decoder.cross_attn_masked", False)
    b.meta("add_string", "sheetsage2.decoder.tied_embedding_note",
           "output_projection.weight is bit-identical to token_embedding.weight "
           "(measured, doc 20 §3.5/§4.1) -- this GGUF stores dec.tok_embd once and "
           "the engine must use it for both the embedding lookup and the "
           "(transposed) output projection.")

    # ---- vocab / tokens (doc 21 §0, §1, §4.6) ------------------------------
    b.meta("add_uint32", "sheetsage2.vocab_size", VOCAB_SIZE)
    b.meta("add_string", "sheetsage2.vocab.schema_version", SCHEMA_VERSION)
    b.meta("add_uint32", "sheetsage2.time_hz", TIME_HZ)
    b.meta("add_uint32", "sheetsage2.max_output_seq_len", MAX_OUTPUT_SEQ_LEN)
    b.meta("add_uint32", "sheetsage2.token.pad", TOKEN_PAD)
    b.meta("add_uint32", "sheetsage2.token.sos", TOKEN_SOS)
    b.meta("add_uint32", "sheetsage2.token.eos", TOKEN_EOS)
    b.meta("add_uint32", "sheetsage2.token.out", TOKEN_OUT)
    b.meta("add_uint32", "sheetsage2.vocab.prompt_capacity", PROMPT_CAPACITY)
    b.meta("add_uint32", "sheetsage2.vocab.max_subbeat_shift", MAX_SUBBEAT_SHIFT)
    b.meta("add_uint32", "sheetsage2.vocab.n_eighth_positions", N_EIGHTH_POSITIONS)
    b.meta("add_uint32", "sheetsage2.vocab.n_pitch_tokens", N_PITCH_TOKENS)
    b.meta("add_uint32", "sheetsage2.vocab.n_key_tokens", N_KEY_TOKENS)
    b.meta("add_array", "sheetsage2.vocab.prompt_names", list(PROMPT_NAMES))
    b.meta("add_array", "sheetsage2.generation.default_prompts", list(DEFAULT_PROMPTS))
    b.meta("add_array", "sheetsage2.vocab.event_field_order", list(EVENT_FIELD_ORDER))
    b.meta("add_uint32", "sheetsage2.vocab.appended_token_blocks_count",
           len(APPENDED_TOKEN_BLOCKS))
    b.meta("add_array", "sheetsage2.vocab.chromatic_sharps", list(CHROMATIC_SHARPS))
    b.meta("add_array", "sheetsage2.vocab.meter_numerators", list(METER_NUMERATORS))
    b.meta("add_array", "sheetsage2.vocab.meter_denominators", list(METER_DENOMINATORS))
    b.meta("add_string", "sheetsage2.vocab.meter_pair_order",
           "numerator OUTER, denominator INNER: pair i = (i//6 + 1, "
           "meter_denominators[i % 6])")
    b.meta("add_array", "sheetsage2.vocab.structure_labels", list(STRUCTURE_LABELS))
    b.meta("add_array", "sheetsage2.vocab.full_chord_qualities", list(FULL_CHORD_QUALITIES))
    b.meta("add_string", "sheetsage2.vocab.full_chord_build_order",
           "labels = [\"N\"]; for quality in full_chord_qualities: for root in "
           "chromatic_sharps: for inversion in (*inversions_for(quality), \"\"): "
           "labels.append(f\"{root}:{quality}{inversion}\") -- quality outer, root "
           "middle, inversion inner, plain-root last; only maj/min/maj7/min7/7 carry "
           "inversions (doc 21 §1.5)")
    for q, invs in FULL_CHORD_INVERSIONS.items():
        b.meta("add_array", f"sheetsage2.vocab.full_chord_inversions.{q}", list(invs))
    b.meta("add_array", "sheetsage2.vocab.duration_templates", list(DURATION_TEMPLATES))
    b.meta("add_string", "sheetsage2.vocab.fingerprint_serialization",
           "json.dumps(payload, sort_keys=True, separators=(\",\",\":\")).encode(\"utf-8\"); "
           "sha256(...).hexdigest()[:16]. Payload keys (sorted): appended_token_blocks, "
           "audio_length_seconds, duration_templates, event_field_order, "
           "full_chord_labels, majmin_chord_labels, meter_pairs, n_tokens, "
           "prompt_capacity, prompt_names, schema_version, structure_labels, time_hz")
    b.meta("add_string", "sheetsage2.vocab_fingerprint", VOCAB_FINGERPRINT)

    # ---- vocab block boundaries [start, end) in id order (doc 21 §1.2) -----
    for name, start, end in vocab_table:
        b.meta("add_uint32", f"sheetsage2.vocab.block.{name}.start", start)
        b.meta("add_uint32", f"sheetsage2.vocab.block.{name}.count", end - start)

    # ---- provenance ----------------------------------------------------------
    b.meta("add_string", "sheetsage2.pinned.base_repo", BASE_REPO)
    b.meta("add_string", "sheetsage2.pinned.base_revision", BASE_REVISION)
    b.meta("add_string", "sheetsage2.pinned.base_sha256", BASE_SHA256)
    b.meta("add_string", "sheetsage2.pinned.repo", SHEETSAGE2_REPO)
    b.meta("add_string", "sheetsage2.pinned.source_sha256", src_sha256)
    b.meta("add_string", "sheetsage2.pinned.python_sha256", python_sha256)
    b.meta("add_array", "sheetsage2.pinned.python_files", list(python_files_hashed))
    b.meta("add_string", "sheetsage2.pinned.fixtures", ORACLE_FIXTURES)
    b.meta("add_string", "sheetsage2.pinned.model_pin_doc", MODEL_PIN_DOC)
    b.meta("add_string", "sheetsage2.pinned.symbolic_pin_doc", SYMBOLIC_PIN_DOC)
    b.meta("add_string", "sheetsage2.gate.note",
           "G1 (encoder vs FP32 CPU oracle) is informational, rel-L2 ~1e-3 or better "
           "expected; decisive gates are G2 (100% masked-argmax agreement, "
           "teacher-forced) and G4/G6 (ABC byte-identical) -- see doc 19.")


# ---------------------------------------------------------------------------
# Component: encoder  ->  enc.*
# ---------------------------------------------------------------------------

def build_encoder(src, b, samples):
    # ---- frontend buffers. All four FORCED F32 (doc 20 §1.3/§4.4). The
    # filterbank keeps its checkpoint layout [n_freq=1025, n_mel=128]: the
    # reference does spectrum.transpose(-1,-2) @ fb, the orientation the matmul
    # wants. ---------------------------------------------------------------
    b.put("enc.mel.stft_window",
          src.get("encoder.feature_extractor.spectrogram.window", expect=(N_FFT,)), F32)
    fb = src.get("encoder.feature_extractor.mel_scale.fb", expect=(N_FREQ, N_MEL))
    samples["enc.mel.filterbank"] = fb.copy()
    b.put("enc.mel.filterbank", fb, F32)
    b.put("enc.mel.mean",
          src.get("encoder.feature_extractor.mel_mean", expect=(N_MEL,)), F32)
    b.put("enc.mel.std",
          src.get("encoder.feature_extractor.mel_std", expect=(N_MEL,)), F32)

    # ---- ConvNext subsampling ----------------------------------------------
    chans = [N_MEL] + list(SUB_CHANNELS)
    for bi in range(3):
        cin, cout = chans[bi], chans[bi + 1]
        stride = SUB_STRIDES[bi]
        s = f"encoder.subsampling_module.{bi}"
        d = f"enc.sub.{bi}"

        identity = (cin == cout and stride == 1)
        if bi == 0 and not identity:
            die("subsampling block 0 is not an identity resample; the channel list "
                "does not match this converter's assumptions")
        if not identity:
            b.put(f"{d}.down_norm.weight",
                  src.get(f"{s}.resampling_layer.0.weight", expect=(cin,)), F32)
            b.put(f"{d}.down_norm.bias",
                  src.get(f"{s}.resampling_layer.0.bias", expect=(cin,)), F32)
            b.put(f"{d}.down_conv.weight",
                  src.get(f"{s}.resampling_layer.2.weight",
                          expect=(cout, cin, SUB_KERNEL)), F32)
            b.put(f"{d}.down_conv.bias",
                  src.get(f"{s}.resampling_layer.2.bias", expect=(cout,)), F32)

        for li in range(SUB_DEPTHS[bi]):
            ls = f"{s}.convnext_layers.{li}"
            ld = f"{d}.cnx.{li}"
            dw = src.get(f"{ls}.depthwise_block.1.weight", expect=(cout, 1, CNX_KERNEL))
            b.put(f"{ld}.dw.weight", dw.reshape(cout, CNX_KERNEL), F32)
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
            grn_w = src.get(f"{ls}.pointwise_block.3.weight", expect=(1, 1, 4 * cout))
            grn_b = src.get(f"{ls}.pointwise_block.3.bias", expect=(1, 1, 4 * cout))
            b.put(f"{ld}.grn.weight", grn_w.reshape(4 * cout), F32)
            b.put(f"{ld}.grn.bias", grn_b.reshape(4 * cout), F32)
            b.put(f"{ld}.pw_down.weight",
                  src.get(f"{ls}.pointwise_block.4.weight", expect=(cout, 4 * cout)), NATIVE)
            b.put(f"{ld}.pw_down.bias",
                  src.get(f"{ls}.pointwise_block.4.bias", expect=(cout,)), F32)

    # ---- Conformer blocks 0..23 -- ALL 24, doc 20 §1 item 1 ----------------
    for n in range(ENC_BLOCKS):
        s = f"encoder.layers.{n}"
        d = f"enc.blk.{n}"

        for tag in ("ffn1", "ffn2"):
            b.put(f"{d}.{tag}_norm.weight",
                  src.get(f"{s}.{tag}_layer_norm.weight", expect=(ENC_DIM,)), F32)
            b.put(f"{d}.{tag}_norm.bias",
                  src.get(f"{s}.{tag}_layer_norm.bias", expect=(ENC_DIM,)), F32)
            b.put(f"{d}.{tag}_up.weight",
                  src.get(f"{s}.{tag}.w_1.weight", expect=(ENC_FF, ENC_DIM)), NATIVE)
            b.put(f"{d}.{tag}_up.bias",
                  src.get(f"{s}.{tag}.w_1.bias", expect=(ENC_FF,)), F32)
            b.put(f"{d}.{tag}_down.weight",
                  src.get(f"{s}.{tag}.w_2.weight", expect=(ENC_DIM, ENC_FF)), NATIVE)
            b.put(f"{d}.{tag}_down.bias",
                  src.get(f"{s}.{tag}.w_2.bias", expect=(ENC_DIM,)), F32)

        b.put(f"{d}.attn_norm.weight",
              src.get(f"{s}.attn_layer_norm.weight", expect=(ENC_DIM,)), F32)
        b.put(f"{d}.attn_norm.bias",
              src.get(f"{s}.attn_layer_norm.bias", expect=(ENC_DIM,)), F32)
        # Four SEPARATE projections with bias, LoRA already merged into the
        # weight (doc 20 §2.4) -- loaded and stored as plain native weights,
        # the merge math itself is never run here.
        for dst, srcname in (("attn_q", "query_proj"), ("attn_k", "key_proj"),
                             ("attn_v", "value_proj"), ("attn_output", "out_proj")):
            w = src.get(f"{s}.attn.{srcname}.weight", expect=(ENC_DIM, ENC_DIM))
            if n == ENC_BLOCKS - 1 and dst == "attn_output":
                samples[f"{d}.{dst}.weight"] = w.copy()
            b.put(f"{d}.{dst}.weight", w, NATIVE)
            b.put(f"{d}.{dst}.bias",
                  src.get(f"{s}.attn.{srcname}.bias", expect=(ENC_DIM,)), F32)

        b.put(f"{d}.conv_norm.weight",
              src.get(f"{s}.conv_module.layer_norm.weight", expect=(ENC_DIM,)), F32)
        b.put(f"{d}.conv_norm.bias",
              src.get(f"{s}.conv_module.layer_norm.bias", expect=(ENC_DIM,)), F32)
        pw1 = src.get(f"{s}.conv_module.conv_block.1.weight",
                      expect=(2 * ENC_DIM, ENC_DIM, 1))
        b.put(f"{d}.conv_pw1.weight", pw1.reshape(2 * ENC_DIM, ENC_DIM), NATIVE)
        dw = src.get(f"{s}.conv_module.conv_block.3.weight",
                     expect=(ENC_DIM, 1, ENC_CONV_DW_KERNEL))
        b.put(f"{d}.conv_dw.weight", dw.reshape(ENC_DIM, ENC_CONV_DW_KERNEL), F32)
        b.put(f"{d}.conv_dw_norm.weight",
              src.get(f"{s}.conv_module.conv_block.4.1.weight", expect=(ENC_DIM,)), F32)
        b.put(f"{d}.conv_dw_norm.bias",
              src.get(f"{s}.conv_module.conv_block.4.1.bias", expect=(ENC_DIM,)), F32)
        pw2 = src.get(f"{s}.conv_module.conv_block.6.weight",
                      expect=(ENC_DIM, ENC_DIM, 1))
        b.put(f"{d}.conv_pw2.weight", pw2.reshape(ENC_DIM, ENC_DIM), NATIVE)

        b.put(f"{d}.final_norm.weight",
              src.get(f"{s}.final_layer_norm.weight", expect=(ENC_DIM,)), F32)
        b.put(f"{d}.final_norm.bias",
              src.get(f"{s}.final_layer_norm.bias", expect=(ENC_DIM,)), F32)

    # ---- layer_weight + encoder_projection (doc 20 §2.5) -------------------
    # Source is BF16 but this is a 25-element softmax input -- cheap and
    # precision-sensitive, so it is promoted to F32 (doc 20 §4.2's table).
    b.put("enc.layer_w", src.get("layer_weight", expect=(N_HIDDEN_STATES,)), F32)
    b.put("enc.proj.weight",
          src.get("encoder_projection.weight", expect=(PROJ_OUT, ENC_DIM)), NATIVE)
    b.put("enc.proj.bias",
          src.get("encoder_projection.bias", expect=(PROJ_OUT,)), F32)

    leftover = src.unconsumed(lambda n: n.startswith("encoder.") or n == "layer_weight"
                               or n.startswith("encoder_projection."))
    if leftover:
        log(f"ERROR: {len(leftover)} encoder-side source tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n} {src.file.shape(n)} {src.file.dtype(n)}")
        if len(leftover) > 40:
            log(f"    ... and {len(leftover) - 40} more")
        die("the converter does not understand this checkpoint's encoder tensor set "
            "-- refusing to leave a partial model in place")


# ---------------------------------------------------------------------------
# Component: decoder  ->  dec.*
# ---------------------------------------------------------------------------

def build_decoder(src, b, samples):
    tok_embd = src.get("token_embedding.weight", expect=(VOCAB_SIZE, DEC_DIM))
    out_proj = src.get("output_projection.weight", expect=(VOCAB_SIZE, DEC_DIM))
    if not np.array_equal(tok_embd, out_proj):
        die("token_embedding.weight and output_projection.weight are NOT "
            "bit-identical in this checkpoint (doc 20 §3.5/§4.1 asserts they are, "
            "tied weights) -- refusing to store only one of them silently")
    log("verified: token_embedding.weight == output_projection.weight (tied, "
        "storing once as dec.tok_embd, per doc 20 §3.5/§4.2)")
    samples["dec.tok_embd"] = tok_embd.copy()
    b.put("dec.tok_embd", tok_embd, NATIVE)

    b.put("dec.pos_embd",
          src.get("decoder.embed_positions.weight", expect=(DEC_MAX_POSITIONS, DEC_DIM)),
          NATIVE)
    b.put("dec.norm_embd.weight",
          src.get("decoder.layernorm_embedding.weight", expect=(DEC_DIM,)), F32)
    b.put("dec.norm_embd.bias",
          src.get("decoder.layernorm_embedding.bias", expect=(DEC_DIM,)), F32)

    for m in range(DEC_BLOCKS):
        s = f"decoder.layers.{m}"
        d = f"dec.blk.{m}"

        b.put(f"{d}.self_attn_norm.weight",
              src.get(f"{s}.self_attn_layer_norm.weight", expect=(DEC_DIM,)), F32)
        b.put(f"{d}.self_attn_norm.bias",
              src.get(f"{s}.self_attn_layer_norm.bias", expect=(DEC_DIM,)), F32)
        for dst, srcname in (("self_attn_q", "q_proj"), ("self_attn_k", "k_proj"),
                             ("self_attn_v", "v_proj"), ("self_attn_output", "out_proj")):
            b.put(f"{d}.{dst}.weight",
                  src.get(f"{s}.self_attn.{srcname}.weight", expect=(DEC_DIM, DEC_DIM)),
                  NATIVE)
            b.put(f"{d}.{dst}.bias",
                  src.get(f"{s}.self_attn.{srcname}.bias", expect=(DEC_DIM,)), F32)

        b.put(f"{d}.cross_attn_norm.weight",
              src.get(f"{s}.encoder_attn_layer_norm.weight", expect=(DEC_DIM,)), F32)
        b.put(f"{d}.cross_attn_norm.bias",
              src.get(f"{s}.encoder_attn_layer_norm.bias", expect=(DEC_DIM,)), F32)
        for dst, srcname in (("cross_attn_q", "q_proj"), ("cross_attn_k", "k_proj"),
                             ("cross_attn_v", "v_proj"), ("cross_attn_output", "out_proj")):
            b.put(f"{d}.{dst}.weight",
                  src.get(f"{s}.encoder_attn.{srcname}.weight", expect=(DEC_DIM, DEC_DIM)),
                  NATIVE)
            b.put(f"{d}.{dst}.bias",
                  src.get(f"{s}.encoder_attn.{srcname}.bias", expect=(DEC_DIM,)), F32)

        b.put(f"{d}.ffn_up.weight",
              src.get(f"{s}.fc1.weight", expect=(DEC_FF, DEC_DIM)), NATIVE)
        b.put(f"{d}.ffn_up.bias", src.get(f"{s}.fc1.bias", expect=(DEC_FF,)), F32)
        b.put(f"{d}.ffn_down.weight",
              src.get(f"{s}.fc2.weight", expect=(DEC_DIM, DEC_FF)), NATIVE)
        b.put(f"{d}.ffn_down.bias", src.get(f"{s}.fc2.bias", expect=(DEC_DIM,)), F32)

        b.put(f"{d}.final_norm.weight",
              src.get(f"{s}.final_layer_norm.weight", expect=(DEC_DIM,)), F32)
        b.put(f"{d}.final_norm.bias",
              src.get(f"{s}.final_layer_norm.bias", expect=(DEC_DIM,)), F32)

    leftover = src.unconsumed(lambda n: n.startswith("decoder.")
                               or n in ("token_embedding.weight", "output_projection.weight"))
    if leftover:
        log(f"ERROR: {len(leftover)} decoder-side source tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n} {src.file.shape(n)} {src.file.dtype(n)}")
        die("the converter does not understand this checkpoint's decoder tensor set "
            "-- refusing to leave a partial model in place")


# ---------------------------------------------------------------------------
# Verify: reload the written GGUF and spot-check a few tensors round-trip.
# ---------------------------------------------------------------------------

def verify_gguf(path, outtype, expect_tensor_count, original_arrays):
    reader = gguf.GGUFReader(path)
    n_tensors = len(reader.tensors)
    if n_tensors != expect_tensor_count:
        die(f"reload check: GGUF has {n_tensors} tensors, expected {expect_tensor_count}")

    by_name = {t.name: t for t in reader.tensors}
    for key in ("sheetsage2.vocab_fingerprint", "sheetsage2.vocab_size",
                "sheetsage2.encoder.block_count", "sheetsage2.decoder.block_count"):
        field = reader.get_field(key)
        if field is None:
            die(f"reload check: KV {key!r} missing from written GGUF")

    fp_val = reader.get_field("sheetsage2.vocab_fingerprint").contents()
    if fp_val != VOCAB_FINGERPRINT:
        die(f"reload check: sheetsage2.vocab_fingerprint reads back as {fp_val!r}, "
            f"expected {VOCAB_FINGERPRINT!r}")
    vocab_size = reader.get_field("sheetsage2.vocab_size").contents()
    if vocab_size != VOCAB_SIZE:
        die(f"reload check: sheetsage2.vocab_size reads back as {vocab_size!r}, "
            f"expected {VOCAB_SIZE}")
    enc_blocks = reader.get_field("sheetsage2.encoder.block_count").contents()
    dec_blocks = reader.get_field("sheetsage2.decoder.block_count").contents()
    if enc_blocks != ENC_BLOCKS or dec_blocks != DEC_BLOCKS:
        die(f"reload check: block counts read back as enc={enc_blocks}, "
            f"dec={dec_blocks}, expected enc={ENC_BLOCKS}, dec={DEC_BLOCKS}")
    log(f"reload check: {n_tensors} tensors present, KVs round-trip "
        f"(vocab_fingerprint={fp_val}, vocab_size={vocab_size}, "
        f"enc_blocks={enc_blocks}, dec_blocks={dec_blocks})")

    # Sample three tensors across the model (encoder frontend F32 buffer, a
    # deep encoder block's LoRA-merged attention weight, the decoder's tied
    # embedding) and check they round-trip against the exact array this
    # process wrote, within the f16 error bound (or bit-exact at f32).
    for name in ("enc.mel.filterbank", "enc.blk.23.attn_output.weight", "dec.tok_embd"):
        if name not in by_name:
            die(f"reload check: expected sample tensor {name!r} not found in GGUF")
        if name not in original_arrays:
            die(f"reload check: no original array recorded for sample {name!r}")
        t = by_name[name]
        # GGUFReader hands back t.data already in the natural (non-reversed)
        # numpy shape and native dtype (float16/float32) -- no manual
        # unpacking needed, unlike the raw uint16 view the writer takes in.
        got = np.asarray(t.data, dtype=np.float32)
        want = original_arrays[name]
        if got.shape != want.shape:
            die(f"reload check: {name} shape {got.shape} != original {want.shape}")
        max_abs_err = float(np.max(np.abs(got - want)))
        bound = 4.0 if t.tensor_type.name == "F32" else max(1e-2, 5e-3 * float(np.max(np.abs(want))))
        ok = max_abs_err <= bound if t.tensor_type.name != "F32" else max_abs_err == 0.0
        log(f"  sample {name}: gguf_dtype={t.tensor_type.name}, shape={tuple(t.shape)}, "
            f"max_abs_err={max_abs_err:.6g}, bound={bound:.6g}, ok={ok}")
        if not ok:
            die(f"reload check: {name} round-trip error {max_abs_err:.6g} exceeds "
                f"bound {bound:.6g}")
    log(f"reload check OK ({outtype})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def hash_python_reference():
    if not os.path.isdir(PYTHON_REF_DIR):
        log(f"WARNING: reference Python dir {PYTHON_REF_DIR} not found -- "
            f"writing python_sha256 as 'unknown' (this does not affect the "
            f"converted weights, only provenance metadata)")
        return "unknown", []
    lines = []
    missing = []
    for fname in PYTHON_REF_FILES:
        fpath = os.path.join(PYTHON_REF_DIR, fname)
        if not os.path.isfile(fpath):
            missing.append(fname)
            continue
        lines.append(f"{fname}:{sha256_file(fpath)}")
    if missing:
        log(f"WARNING: {len(missing)} reference Python files missing from "
            f"{PYTHON_REF_DIR}: {missing} -- writing python_sha256 as 'unknown'")
        return "unknown", lines
    manifest = "\n".join(sorted(lines)) + "\n"
    combined = sha256_bytes(manifest.encode("utf-8"))
    return combined, lines


def main():
    ap = argparse.ArgumentParser(
        prog="convert-sheetsage2.py",
        description="SheetSage2 (audio -> ABC lead sheet transcriber) -> one GGUF, "
                    "arch 'sheetsage2'.")
    ap.add_argument("--src", default=("K:/yue2-bakeoff/ai-toolkit/models/audio_encoders/"
                                       "sheetsage2_bf16.safetensors"),
                    metavar="PATH", help="merged SheetSage2 safetensors checkpoint")
    ap.add_argument("--out", default="models/yue2", metavar="DIR",
                    help="output directory (default: models/yue2)")
    ap.add_argument("--outtype", "--type", dest="outtype", default="f16",
                    choices=("f16", "f32"),
                    help="f16 (default) narrows 2-D matmul weights only; f32 keeps "
                         "everything at checkpoint precision")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        die(f"source checkpoint not found: {args.src}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"sheetsage2-{args.outtype}.gguf")
    if os.path.exists(out_path) and not args.force:
        die(f"{out_path} already exists -- pass --force to overwrite")

    # ---- vocabulary self-check FIRST -- before touching a single tensor ----
    vocab_table, full_chord_labels = verify_vocab_arithmetic()

    log(f"hashing source checkpoint {args.src} ...")
    src_sha256 = sha256_file(args.src)
    log(f"source sha256: {src_sha256}")

    python_sha256, python_files_hashed = hash_python_reference()
    log(f"reference python sha256 (combined manifest): {python_sha256}")

    st = SafeTensorsFile(args.src)
    # Defensive check standing in for doc 20 §4.5's "refuse an unmerged adapter
    # file": a merged file has zero tensor keys mentioning adapter/lora
    # anywhere (measured fact, doc 20 §2.4/§5.1). There is no `weights_format`
    # key in this file's own `__metadata__` to check directly (measured: it is
    # {base_model, license, source, format, weight_dtype}).
    suspect = [n for n in st.header if "adapter" in n.lower() or "lora" in n.lower()]
    if suspect:
        die(f"source checkpoint has {len(suspect)} adapter/lora-named tensors "
            f"({suspect[:5]}...) -- this looks like an UNMERGED 'adapter' format "
            f"file, not the merged encoder this converter expects (doc 20 §0/§4.5)")
    n_total = len(st.header)
    if n_total != 1040:
        die(f"source checkpoint has {n_total} tensors, expected 1040 (doc 20 §4.1) "
            f"-- this does not look like the pinned SheetSage2 merged checkpoint")
    st.close()

    src = Source(args.src)
    b = Bundle("sheetsage2")
    samples = {}
    build_encoder(src, b, samples)
    n_enc = len(b.tensors)
    build_decoder(src, b, samples)
    n_dec = len(b.tensors) - n_enc
    log(f"assembled {len(b.tensors)} tensors: {n_enc} enc.*, {n_dec} dec.*")

    # Doc 20 §4.2: 1039 of 1040 source tensors are written by name; the 1040th
    # (output_projection.weight) is read (for the tie assertion) but
    # intentionally not re-written -- confirm that accounting exactly.
    leftover = src.unconsumed()
    if leftover:
        log(f"ERROR: {len(leftover)} source tensors were neither written nor the "
            f"intentional output_projection.weight dedupe: {leftover[:20]}")
        die("tensor accounting does not match doc 20 §4.2 -- refusing to ship")
    if len(src.consumed) != 1040:
        die(f"consumed {len(src.consumed)} of 1040 source tensors, expected all 1040 "
            f"(1039 written + output_projection.weight read-and-discarded)")
    if len(b.tensors) != 1039:
        die(f"wrote {len(b.tensors)} output tensors, expected 1039 (doc 20 §4.2)")
    log("tensor accounting OK: 1040 source tensors consumed, 1039 written "
        "(output_projection.weight deduped against token_embedding.weight)")

    add_meta(b, args.outtype, src_sha256, python_sha256, python_files_hashed, vocab_table)
    src.close()

    size, counts = b.write(out_path, args.outtype)
    verify_gguf(out_path, args.outtype, expect_tensor_count=1039,
                original_arrays=samples)

    log(f"SUMMARY: {out_path}")
    log(f"  tensors: {len(b.tensors)} (enc.*={n_enc}, dec.*={n_dec})")
    log(f"  dtype mix: {counts['f16']} F16, {counts['f32']} F32")
    log(f"  size: {size:,} bytes ({size / 1e9:.3f} GB)")
    log(f"  vocab_fingerprint: {VOCAB_FINGERPRINT} (self-verified against doc 21 §1.8)")
    log(f"  full_chord_labels: {len(full_chord_labels)} entries")
    log("Done.")


if __name__ == "__main__":
    main()

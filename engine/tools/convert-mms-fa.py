#!/usr/bin/env python3
# convert-mms-fa.py: torchaudio MMS_FA (Meta MMS forced-aligner) -> one GGUF,
# plus a --verify round-trip against the source state dict.
#
# This is the transport half of the contract pinned in
# docs/plans/yue2/17-mms-fa-port.md. READ THAT FIRST, especially section 1
# (architecture, the weight-norm baking rule) and section 5 (shape of the
# port). Where this file and that document disagree, the document wins.
#
# Modelled on convert-yue2-tok.py: same Bundle/put/write shape, same
# "{arch-without-dashes}." KV prefix convention ("mms-fa" -> "mmsfa."), same
# refuse-on-unconsumed-tensors discipline, same plain/no-emoji header tone.
# Unlike that converter this one DOES use torch (and torchaudio): the
# environment that holds the fixture (K:/yue2/.venv) also holds both, and
# torchaudio's own `torch._weight_norm` is the only way to bake the pos_conv
# parametrization without risking a hand-rolled norm that disagrees with the
# library in the last ULP (measured: a naive per-tap L2 norm in float32 was
# 6.2e-6 off the live module's weight; `torch._weight_norm(v, g, dim=2)`,
# the exact function PyTorch's own parametrization calls, is 0.0 off).
#
# THE MODEL (pin.json, 315.5M params, all F32; with_star=False so the CTC
# head has 28 classes, index 0 = blank):
#   7x (Conv1d -> LayerNorm(channel) -> GELU) feature extractor, stride 320
#   -> LayerNorm(512) -> Linear 512->1024 feature projection
#   -> pos_conv: Conv1d 1024->1024 k=128 pad=64 groups=16, weight-norm baked,
#      GELU, drop last frame (k even + pad k/2 yields T+1)
#   -> enc.ln: LayerNorm(x + pos(x)), the transformer's actual input
#   -> 24x post-LN encoder layer:
#        a = x + attn(x); a = ln1(a); x = ln2(a + ffn(a))
#      attention: 16 heads x 64, q/k/v/out all with bias, scale 64^-0.5
#      ffn: Linear 1024->4096, GELU, Linear 4096->1024
#   -> aux: Linear 1024->28 (the CTC head; log_softmax is applied by the
#      caller at inference time, not baked into the weights)
#
# WHAT IS **NOT** IN THE FILE, ON PURPOSE
#   No rotary/positional tables beyond the pos_conv: this architecture's only
#   position signal is the pos_conv term, there is no RoPE here (that is the
#   MERT port's architecture, not this one -- do not carry that assumption
#   over). No `star` token (with_star=False upstream, nothing to drop).
#
# ---------------------------------------------------------------------------
# QUANTIZATION POLICY: every tensor is stored F32. The pin's own gates (conv
# <=1e-4, layer23 <=2e-3, emissions <=2e-3) are tight enough that narrowing is
# a question for a later pass, once the gates hold at F32 -- see §5 of the
# plan doc ("F16 later if the gates hold -- measure, since LN-heavy stacks
# are sensitive"). This converter does not implement --outtype at all.
#
# TENSOR LAYOUT. Two kinds of tensor leave this converter, both written with
# their PyTorch row-major shape (gguf-py reverses dims on write, so the ggml
# `ne` a C++ loader sees is the torch shape backwards):
#   * Linear [out, in]               -> ne = (in, out)   (ggml_mul_mat-ready)
#   * Conv1d [out, in/groups, k]     -> ne = (k, in/groups, out), kept 3-D,
#     for explicit F32 im2col in the port (ggml_conv_1d forces F16 staging,
#     which this feature extractor's gates are too tight for -- same call
#     yue2-mert.h made for its own depthwise/resampling convs).
#   * LayerNorm weight/bias          -> ne = (C,), unchanged (1-D)
# This matches the convention read out of engine/src/yue2/yue2-mert.h's
# loader section: Linear rows stay rows, conv weights keep their full
# [out, in, k] shape for the sibling loader's im2col path, LayerNorm stays flat.
#
# DEPENDENCIES: numpy, torch, torchaudio (for the state dict, and for
# --verify's live weight-norm cross-check), gguf (llama.cpp gguf-py; `uv pip
# install --python <venv> gguf` if missing -- it is not in K:/yue2/.venv by
# default).
#
# USAGE
#   python engine/tools/convert-mms-fa.py \
#       --src K:/yue2/fixtures/mms-fa/mms_fa_state_dict.pt \
#       --out K:/yue2/models/mms-fa --verify
#
# Output: K:/yue2/models/mms-fa/mms-fa-f32.gguf

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import gguf

CONVERTER_VERSION = 1

# ---------------------------------------------------------------------------
# Model constants, from K:/yue2/fixtures/mms-fa/pin.json. Hardcoded (not read
# from a config file) for the same reason as the MERT/MM3 converters: a
# converter that only reads shapes from the checkpoint cannot reject a wrong
# one. Every shape below is also passed as `expect=` on the matching get(),
# so a checkpoint that does not match pin.json fails loudly, not silently.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000

# (in, out, kernel, stride) per feature-extractor conv layer, all bias=True,
# all followed by LayerNorm(out) + GELU ("LayerNormConvLayer" in torchaudio).
CONV_LAYERS = [
    (1, 512, 10, 5),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 3, 2),
    (512, 512, 2, 2),
    (512, 512, 2, 2),
]
N_CONV = len(CONV_LAYERS)
FEAT_DIM = 512                   # feature-extractor output channels

EMBED = 1024                     # encoder width (feature projection output)
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = EMBED // N_HEADS      # 64
FFN = 4096
LN_EPS = 1e-5
LAYER_NORM_FIRST = False         # post-LN encoder layers

POS_CONV_KERNEL = 128
POS_CONV_GROUPS = 16
POS_CONV_IN_PER_GROUP = EMBED // POS_CONV_GROUPS   # 64
POS_CONV_PAD = POS_CONV_KERNEL // 2                # 64
POS_CONV_NUM_REMOVE = 1          # k even + pad k/2 -> T+1; drop the last frame

# CTC label set, index = class id, 0 = blank. From pin.json, authoritative
# (this IS the order forced_align/argmax index into).
LABELS = ["-", "a", "i", "e", "n", "o", "u", "t", "s", "r", "m", "k", "l",
          "d", "g", "h", "y", "b", "p", "w", "c", "v", "j", "z", "f", "'",
          "q", "x"]
N_LABELS = len(LABELS)           # 28
assert N_LABELS == 28

LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_ATTRIBUTION = (
    "Weights are CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 "
    "International), NON-COMMERCIAL USE ONLY. Meta MMS forced-aligner "
    "(MMS_FA), served via torchaudio.pipelines.MMS_FA -- "
    "https://pytorch.org/audio/stable/pipelines.html#torchaudio.pipelines.MMS_FA"
)


def log(msg):
    print(f"[convert-mms-fa] {msg}", file=sys.stderr, flush=True)


def die(msg):
    raise SystemExit(f"[convert-mms-fa] ERROR: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Source: the state dict, with a consumed-set and expect-checked get(), same
# discipline as convert-yue2-tok.py's Source/HeadSource.
# ---------------------------------------------------------------------------

class Source:
    def __init__(self, path):
        self.path = path
        raw = torch.load(path, map_location="cpu")
        if not isinstance(raw, dict):
            die(f"{path}: expected a state-dict-shaped object, got {type(raw).__name__}")
        self.state = raw
        self.consumed = set()

    def get(self, name, expect=None):
        if name not in self.state:
            die(f"missing tensor {name} in {self.path}")
        t = self.state[name]
        if not torch.is_tensor(t):
            die(f"{name}: expected a tensor, got {type(t).__name__}")
        if t.dtype != torch.float32:
            die(f"{name}: expected float32, checkpoint has {t.dtype} -- this converter "
                f"assumes the whole state dict is F32 (pin.json says it is)")
        self.consumed.add(name)
        arr = t.detach().cpu().numpy()
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has {tuple(arr.shape)} "
                f"-- not the MMS_FA checkpoint this converter understands")
        return arr

    def unconsumed(self, predicate=lambda n: True):
        return sorted(n for n in self.state if predicate(n) and n not in self.consumed)


# ---------------------------------------------------------------------------
# GGUF writer: every tensor F32, no narrowing policy to speak of -- see the
# header's quantization-policy note.
# ---------------------------------------------------------------------------

class Bundle:
    def __init__(self, arch):
        self.arch = arch
        self.kv = []
        self.tensors = []          # (name, np.ndarray)
        self._names = set()

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr):
        if name in self._names:
            die(f"duplicate output tensor {name}")
        self._names.add(name)
        self.tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32)))

    def write(self, path):
        w = gguf.GGUFWriter(path, self.arch)
        for fn, args in self.kv:
            getattr(w, fn)(*args)
        for name, arr in self.tensors:
            w.add_tensor(name, arr)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        size = os.path.getsize(path)
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors, "
            f"{size:,} bytes ({size / 1e9:.3f} GB)")
        return size


# ---------------------------------------------------------------------------
# KV metadata
# ---------------------------------------------------------------------------

def add_meta(b, src_sha256):
    b.meta("add_name", "MMS forced aligner")
    b.meta("add_description",
           "Meta MMS forced-alignment acoustic model (torchaudio MMS_FA, "
           "with_star=False): wav2vec2-large CTC over a 28-symbol romanised "
           "character set, 16 kHz in, 50 Hz log-probabilities out.")
    b.meta("add_license", LICENSE_NAME)
    b.meta("add_string", "mmsfa.license_attribution", LICENSE_ATTRIBUTION)
    b.meta("add_uint32", "mmsfa.converter_version", CONVERTER_VERSION)
    b.meta("add_file_type", int(gguf.LlamaFileType.ALL_F32))
    b.meta("add_string", "mmsfa.tensor_layout",
           "PyTorch row-major shapes, gguf ne reversed: Linear [out,in] -> "
           "ne (in,out); Conv1d [out,in/groups,k] kept 3-D -> ne (k,in/groups,out); "
           "LayerNorm weight/bias stay 1-D. pos_conv.weight is the BAKED "
           "weight-norm tensor (g*v/||v||), stored as a plain (1024,64,128) "
           "torch-shaped tensor, same rule.")

    b.meta("add_uint32", "mmsfa.sample_rate", SAMPLE_RATE)
    b.meta("add_uint32", "mmsfa.n_conv", N_CONV)
    b.meta("add_array", "mmsfa.conv_in", [c[0] for c in CONV_LAYERS])
    b.meta("add_array", "mmsfa.conv_out", [c[1] for c in CONV_LAYERS])
    b.meta("add_array", "mmsfa.conv_kernel", [c[2] for c in CONV_LAYERS])
    b.meta("add_array", "mmsfa.conv_stride", [c[3] for c in CONV_LAYERS])
    b.meta("add_uint32", "mmsfa.feat_dim", FEAT_DIM)
    b.meta("add_uint32", "mmsfa.embed", EMBED)
    b.meta("add_uint32", "mmsfa.n_layers", N_LAYERS)
    b.meta("add_uint32", "mmsfa.n_heads", N_HEADS)
    b.meta("add_uint32", "mmsfa.head_dim", HEAD_DIM)
    b.meta("add_uint32", "mmsfa.ffn", FFN)
    b.meta("add_float32", "mmsfa.attention_scale", float(HEAD_DIM) ** -0.5)
    b.meta("add_uint32", "mmsfa.pos_conv_kernel", POS_CONV_KERNEL)
    b.meta("add_uint32", "mmsfa.pos_conv_groups", POS_CONV_GROUPS)
    b.meta("add_uint32", "mmsfa.pos_conv_padding", POS_CONV_PAD)
    b.meta("add_uint32", "mmsfa.pos_conv_num_remove", POS_CONV_NUM_REMOVE)
    b.meta("add_float32", "mmsfa.ln_eps", LN_EPS)
    b.meta("add_bool", "mmsfa.layer_norm_first", LAYER_NORM_FIRST)
    b.meta("add_string", "mmsfa.block_order",
           "a = x + attn(x); a = ln1(a); x = ln2(a + ffn(a))  (post-LN)")
    b.meta("add_string", "mmsfa.activation", "gelu")
    b.meta("add_uint32", "mmsfa.n_labels", N_LABELS)
    b.meta("add_array", "mmsfa.labels", LABELS)
    b.meta("add_string", "mmsfa.ctc_blank_label", LABELS[0])
    b.meta("add_string", "mmsfa.head_note",
           "mmsfa.head is the raw Linear(1024,28) (model.aux); log_softmax is "
           "applied by the caller, not baked into the weights")

    b.meta("add_string", "mmsfa.pinned.source_repo",
           "torchaudio.pipelines.MMS_FA.get_model(with_star=False)")
    b.meta("add_string", "mmsfa.pinned.source_sha256", src_sha256)
    b.meta("add_string", "mmsfa.pinned.fixtures", "K:/yue2/fixtures/mms-fa")
    b.meta("add_string", "mmsfa.pinned.oracle_doc",
           "docs/plans/yue2/17-mms-fa-port.md")


# ---------------------------------------------------------------------------
# Weight-norm baking (plan doc §1): w = g * v / ||v||, norm over every axis
# except the last (the kernel axis, dim=2). Done via torch._weight_norm, the
# exact function torch.nn.utils.parametrizations.weight_norm's forward calls
# -- see the header comment on why a hand-rolled norm was rejected.
# ---------------------------------------------------------------------------

def bake_pos_conv_weight(src):
    g = src.get("model.encoder.transformer.pos_conv_embed.conv.parametrizations."
                 "weight.original0", expect=(1, 1, POS_CONV_KERNEL))
    v = src.get("model.encoder.transformer.pos_conv_embed.conv.parametrizations."
                 "weight.original1",
                 expect=(EMBED, POS_CONV_IN_PER_GROUP, POS_CONV_KERNEL))
    baked = torch._weight_norm(torch.from_numpy(v), torch.from_numpy(g), 2).numpy()
    if baked.shape != (EMBED, POS_CONV_IN_PER_GROUP, POS_CONV_KERNEL):
        die(f"pos_conv weight-norm bake produced shape {baked.shape}, expected "
            f"{(EMBED, POS_CONV_IN_PER_GROUP, POS_CONV_KERNEL)}")
    return baked


def verify_pos_conv_bake(baked):
    """Cross-check the baked weight against the LIVE torchaudio module, not
    just against our own bake -- this is the assertion the plan doc asks for
    before writing anything."""
    import torchaudio
    log("loading torchaudio.pipelines.MMS_FA.get_model(with_star=False) to "
        "cross-check the pos_conv bake against the live module...")
    m = torchaudio.pipelines.MMS_FA.get_model(with_star=False)
    live = m.model.encoder.transformer.pos_conv_embed.conv.weight.detach().cpu().numpy()
    diff = float(np.abs(baked - live).max())
    log(f"pos_conv weight-norm bake vs live module: max-abs diff = {diff:.3e}")
    if diff >= 1e-6:
        die(f"pos_conv weight-norm bake does not match the live module "
            f"(max-abs diff {diff:.3e} >= 1e-6) -- refusing to ship a miscomputed "
            f"weight-norm tensor")
    return diff


# ---------------------------------------------------------------------------
# Mapping: state dict -> mmsfa.* tensors
# ---------------------------------------------------------------------------

MAPPING_TABLE = [
    ("mmsfa.conv.{i}.weight / .bias",
     "model.feature_extractor.conv_layers.{i}.conv.weight / .bias"),
    ("mmsfa.conv.{i}.ln.weight / .bias",
     "model.feature_extractor.conv_layers.{i}.layer_norm.weight / .bias"),
    ("mmsfa.feat_proj.ln.weight / .bias",
     "model.encoder.feature_projection.layer_norm.weight / .bias"),
    ("mmsfa.feat_proj.weight / .bias",
     "model.encoder.feature_projection.projection.weight / .bias"),
    ("mmsfa.pos_conv.weight",
     "BAKED: parametrizations.weight.original0 (g) * original1 (v) / ||v|| "
     "(dim=2), via torch._weight_norm"),
    ("mmsfa.pos_conv.bias",
     "model.encoder.transformer.pos_conv_embed.conv.bias"),
    ("mmsfa.enc.ln.weight / .bias",
     "model.encoder.transformer.layer_norm.weight / .bias"),
    ("mmsfa.blk.{l}.attn_q.weight / .bias",
     "model.encoder.transformer.layers.{l}.attention.q_proj.weight / .bias"),
    ("mmsfa.blk.{l}.attn_k.weight / .bias",
     "model.encoder.transformer.layers.{l}.attention.k_proj.weight / .bias"),
    ("mmsfa.blk.{l}.attn_v.weight / .bias",
     "model.encoder.transformer.layers.{l}.attention.v_proj.weight / .bias"),
    ("mmsfa.blk.{l}.attn_o.weight / .bias",
     "model.encoder.transformer.layers.{l}.attention.out_proj.weight / .bias"),
    ("mmsfa.blk.{l}.ln1.weight / .bias",
     "model.encoder.transformer.layers.{l}.layer_norm.weight / .bias "
     "(applied AFTER the attention residual)"),
    ("mmsfa.blk.{l}.ffn_up.weight / .bias",
     "model.encoder.transformer.layers.{l}.feed_forward.intermediate_dense.weight / .bias"),
    ("mmsfa.blk.{l}.ffn_down.weight / .bias",
     "model.encoder.transformer.layers.{l}.feed_forward.output_dense.weight / .bias"),
    ("mmsfa.blk.{l}.ln2.weight / .bias",
     "model.encoder.transformer.layers.{l}.final_layer_norm.weight / .bias"),
    ("mmsfa.head.weight / .bias", "model.aux.weight / .bias"),
]


def print_mapping_table():
    log("tensor name mapping (GGUF <- torchaudio state dict):")
    width = max(len(dst) for dst, _ in MAPPING_TABLE)
    for dst, srcname in MAPPING_TABLE:
        log(f"    {dst:<{width}}  <-  {srcname}")


def build(src, b):
    # ---- feature extractor: 7x (conv -> layer_norm -> gelu) ----------------
    for i, (cin, cout, k, _stride) in enumerate(CONV_LAYERS):
        s = f"model.feature_extractor.conv_layers.{i}"
        d = f"mmsfa.conv.{i}"
        b.put(f"{d}.weight", src.get(f"{s}.conv.weight", expect=(cout, cin, k)))
        b.put(f"{d}.bias", src.get(f"{s}.conv.bias", expect=(cout,)))
        b.put(f"{d}.ln.weight", src.get(f"{s}.layer_norm.weight", expect=(cout,)))
        b.put(f"{d}.ln.bias", src.get(f"{s}.layer_norm.bias", expect=(cout,)))

    # ---- feature projection -------------------------------------------------
    b.put("mmsfa.feat_proj.ln.weight",
          src.get("model.encoder.feature_projection.layer_norm.weight", expect=(FEAT_DIM,)))
    b.put("mmsfa.feat_proj.ln.bias",
          src.get("model.encoder.feature_projection.layer_norm.bias", expect=(FEAT_DIM,)))
    b.put("mmsfa.feat_proj.weight",
          src.get("model.encoder.feature_projection.projection.weight",
                  expect=(EMBED, FEAT_DIM)))
    b.put("mmsfa.feat_proj.bias",
          src.get("model.encoder.feature_projection.projection.bias", expect=(EMBED,)))

    # ---- positional conv: weight-norm BAKED, bias passed through -----------
    baked = bake_pos_conv_weight(src)
    b.put("mmsfa.pos_conv.weight", baked)
    b.put("mmsfa.pos_conv.bias",
          src.get("model.encoder.transformer.pos_conv_embed.conv.bias", expect=(EMBED,)))

    # ---- the LayerNorm(x + pos(x)) that feeds the transformer stack --------
    b.put("mmsfa.enc.ln.weight",
          src.get("model.encoder.transformer.layer_norm.weight", expect=(EMBED,)))
    b.put("mmsfa.enc.ln.bias",
          src.get("model.encoder.transformer.layer_norm.bias", expect=(EMBED,)))

    # ---- 24 post-LN encoder layers ------------------------------------------
    for l in range(N_LAYERS):
        s = f"model.encoder.transformer.layers.{l}"
        d = f"mmsfa.blk.{l}"
        for dst, srcname in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                              ("attn_v", "v_proj"), ("attn_o", "out_proj")):
            b.put(f"{d}.{dst}.weight",
                  src.get(f"{s}.attention.{srcname}.weight", expect=(EMBED, EMBED)))
            b.put(f"{d}.{dst}.bias",
                  src.get(f"{s}.attention.{srcname}.bias", expect=(EMBED,)))
        b.put(f"{d}.ln1.weight", src.get(f"{s}.layer_norm.weight", expect=(EMBED,)))
        b.put(f"{d}.ln1.bias", src.get(f"{s}.layer_norm.bias", expect=(EMBED,)))
        b.put(f"{d}.ffn_up.weight",
              src.get(f"{s}.feed_forward.intermediate_dense.weight", expect=(FFN, EMBED)))
        b.put(f"{d}.ffn_up.bias",
              src.get(f"{s}.feed_forward.intermediate_dense.bias", expect=(FFN,)))
        b.put(f"{d}.ffn_down.weight",
              src.get(f"{s}.feed_forward.output_dense.weight", expect=(EMBED, FFN)))
        b.put(f"{d}.ffn_down.bias",
              src.get(f"{s}.feed_forward.output_dense.bias", expect=(EMBED,)))
        b.put(f"{d}.ln2.weight", src.get(f"{s}.final_layer_norm.weight", expect=(EMBED,)))
        b.put(f"{d}.ln2.bias", src.get(f"{s}.final_layer_norm.bias", expect=(EMBED,)))

    # ---- the CTC head --------------------------------------------------------
    b.put("mmsfa.head.weight", src.get("model.aux.weight", expect=(N_LABELS, EMBED)))
    b.put("mmsfa.head.bias", src.get("model.aux.bias", expect=(N_LABELS,)))

    leftover = src.unconsumed()
    if leftover:
        log(f"ERROR: {len(leftover)} source tensors were not consumed:")
        for n in leftover[:40]:
            log(f"    {n} {tuple(src.state[n].shape)}")
        if len(leftover) > 40:
            log(f"    ... and {len(leftover) - 40} more")
        die("the converter does not understand this checkpoint's tensor set -- "
            "refusing to leave a partial model in place")
    log(f"all {len(src.consumed)} source tensors consumed, nothing left over")


# ---------------------------------------------------------------------------
# --verify: reopen the GGUF, check structure, and compare every tensor's
# bytes (after the layout transform) back to a FRESH load of the source .pt.
# ---------------------------------------------------------------------------

def expected_tensor_names():
    names = []
    for i in range(N_CONV):
        names += [f"mmsfa.conv.{i}.weight", f"mmsfa.conv.{i}.bias",
                  f"mmsfa.conv.{i}.ln.weight", f"mmsfa.conv.{i}.ln.bias"]
    names += ["mmsfa.feat_proj.ln.weight", "mmsfa.feat_proj.ln.bias",
              "mmsfa.feat_proj.weight", "mmsfa.feat_proj.bias",
              "mmsfa.pos_conv.weight", "mmsfa.pos_conv.bias",
              "mmsfa.enc.ln.weight", "mmsfa.enc.ln.bias"]
    for l in range(N_LAYERS):
        d = f"mmsfa.blk.{l}"
        for part in ("attn_q", "attn_k", "attn_v", "attn_o"):
            names += [f"{d}.{part}.weight", f"{d}.{part}.bias"]
        names += [f"{d}.ln1.weight", f"{d}.ln1.bias",
                  f"{d}.ffn_up.weight", f"{d}.ffn_up.bias",
                  f"{d}.ffn_down.weight", f"{d}.ffn_down.bias",
                  f"{d}.ln2.weight", f"{d}.ln2.bias"]
    names += ["mmsfa.head.weight", "mmsfa.head.bias"]
    return names


def source_array_for(name, src):
    """Recompute, from the source state dict, the exact array this converter
    would have written for `name` -- used by --verify to diff against the
    GGUF's bytes without trusting this process's own in-memory copy."""
    if name == "mmsfa.pos_conv.weight":
        return bake_pos_conv_weight(src)
    if name == "mmsfa.pos_conv.bias":
        return src.get("model.encoder.transformer.pos_conv_embed.conv.bias")
    if name == "mmsfa.head.weight":
        return src.get("model.aux.weight")
    if name == "mmsfa.head.bias":
        return src.get("model.aux.bias")
    if name == "mmsfa.enc.ln.weight":
        return src.get("model.encoder.transformer.layer_norm.weight")
    if name == "mmsfa.enc.ln.bias":
        return src.get("model.encoder.transformer.layer_norm.bias")
    if name == "mmsfa.feat_proj.ln.weight":
        return src.get("model.encoder.feature_projection.layer_norm.weight")
    if name == "mmsfa.feat_proj.ln.bias":
        return src.get("model.encoder.feature_projection.layer_norm.bias")
    if name == "mmsfa.feat_proj.weight":
        return src.get("model.encoder.feature_projection.projection.weight")
    if name == "mmsfa.feat_proj.bias":
        return src.get("model.encoder.feature_projection.projection.bias")
    if name.startswith("mmsfa.conv."):
        rest = name[len("mmsfa.conv."):]
        i_str, field = rest.split(".", 1)
        i = int(i_str)
        s = f"model.feature_extractor.conv_layers.{i}"
        if field == "weight":
            return src.get(f"{s}.conv.weight")
        if field == "bias":
            return src.get(f"{s}.conv.bias")
        if field == "ln.weight":
            return src.get(f"{s}.layer_norm.weight")
        if field == "ln.bias":
            return src.get(f"{s}.layer_norm.bias")
    if name.startswith("mmsfa.blk."):
        rest = name[len("mmsfa.blk."):]
        l_str, field = rest.split(".", 1)
        l = int(l_str)
        s = f"model.encoder.transformer.layers.{l}"
        proj = {"attn_q": "q_proj", "attn_k": "k_proj",
                "attn_v": "v_proj", "attn_o": "out_proj"}
        for tag, projname in proj.items():
            if field == f"{tag}.weight":
                return src.get(f"{s}.attention.{projname}.weight")
            if field == f"{tag}.bias":
                return src.get(f"{s}.attention.{projname}.bias")
        if field == "ln1.weight":
            return src.get(f"{s}.layer_norm.weight")
        if field == "ln1.bias":
            return src.get(f"{s}.layer_norm.bias")
        if field == "ffn_up.weight":
            return src.get(f"{s}.feed_forward.intermediate_dense.weight")
        if field == "ffn_up.bias":
            return src.get(f"{s}.feed_forward.intermediate_dense.bias")
        if field == "ffn_down.weight":
            return src.get(f"{s}.feed_forward.output_dense.weight")
        if field == "ffn_down.bias":
            return src.get(f"{s}.feed_forward.output_dense.bias")
        if field == "ln2.weight":
            return src.get(f"{s}.final_layer_norm.weight")
        if field == "ln2.bias":
            return src.get(f"{s}.final_layer_norm.bias")
    die(f"--verify: no mapping rule for GGUF tensor {name}")


def group_of(name):
    if name.startswith("mmsfa.conv."):
        return "feature_extractor"
    if name.startswith("mmsfa.feat_proj."):
        return "feature_projection"
    if name.startswith("mmsfa.pos_conv."):
        return "pos_conv (baked)"
    if name.startswith("mmsfa.enc.ln."):
        return "enc.ln"
    if name.startswith("mmsfa.blk."):
        return "encoder layers"
    if name.startswith("mmsfa.head."):
        return "head (aux)"
    return "other"


def verify(gguf_path, src_path):
    log(f"--- verify: {gguf_path} ---")
    size = os.path.getsize(gguf_path)
    log(f"file size: {size:,} bytes ({size / 1e9:.3f} GB)")

    r = gguf.GGUFReader(gguf_path)
    by_name = {t.name: t for t in r.tensors}

    expect_names = expected_tensor_names()
    ok = True

    def check(cond, msg):
        nonlocal ok
        log(f"  {'OK  ' if cond else 'FAIL'} {msg}")
        ok = ok and bool(cond)

    check(len(r.tensors) == len(expect_names),
          f"tensor count == {len(expect_names)} (got {len(r.tensors)})")
    missing = [n for n in expect_names if n not in by_name]
    extra = [n for n in by_name if n not in expect_names]
    check(not missing, f"no missing tensors (missing: {missing[:10]})")
    check(not extra, f"no unexpected tensors (extra: {extra[:10]})")

    n_labels_kv = None
    for f in r.fields.values():
        if f.name == "mmsfa.n_labels":
            n_labels_kv = int(f.contents())
        if f.name == "mmsfa.labels":
            labels_kv = list(f.contents())
    check(n_labels_kv == N_LABELS, f"mmsfa.n_labels == {N_LABELS} (got {n_labels_kv})")
    check(labels_kv == LABELS, "mmsfa.labels matches pin.json order exactly")

    for t in r.tensors:
        dt = gguf.GGMLQuantizationType(t.tensor_type).name
        check(dt == "F32", f"{t.name} is F32 (got {dt})")

    # Fresh source load -- do not reuse anything the build() pass cached.
    src = Source(src_path)
    max_diff_by_group = {}
    checked = 0
    for name in expect_names:
        if name not in by_name:
            continue
        t = by_name[name]
        gguf_arr = np.array(t.data, copy=False)
        # gguf-py stores the reversed (ne-order) shape; reverse it back to the
        # torch-row-major shape this converter wrote it from.
        gguf_arr = gguf_arr.reshape(tuple(reversed(t.shape.tolist())))
        want = source_array_for(name, src)
        if gguf_arr.shape != want.shape:
            check(False, f"{name}: GGUF shape {gguf_arr.shape} != source shape {want.shape}")
            continue
        diff = float(np.abs(gguf_arr.astype(np.float64) - want.astype(np.float64)).max())
        g = group_of(name)
        max_diff_by_group[g] = max(max_diff_by_group.get(g, 0.0), diff)
        checked += 1

    leftover = src.unconsumed()
    check(not leftover, f"verify re-consumed every source tensor (leftover: {leftover[:10]})")

    log(f"byte-for-byte comparison against a fresh load of {src_path}:")
    for g, d in sorted(max_diff_by_group.items()):
        log(f"    {g:<22} max-abs diff = {d:.3e}")
    all_exact = all(d == 0.0 for d in max_diff_by_group.values())
    check(all_exact, f"every one of {checked} tensors is byte-exact against the source "
          f"(after the layout transform)")

    # The one piece of this file that is NOT a pass-through of a checkpoint
    # tensor: re-derive it and cross-check against the live torchaudio module.
    baked = bake_pos_conv_weight(src)
    diff = verify_pos_conv_bake(baked)
    check(diff < 1e-6, f"pos_conv weight-norm bake vs live torchaudio module < 1e-6 "
          f"(got {diff:.3e})")

    log("VERIFY " + ("PASSED" if ok else "FAILED"))
    return ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="convert-mms-fa.py",
        description="torchaudio MMS_FA (Meta MMS forced aligner) -> one GGUF, "
                     "arch 'mmsfa'.")
    ap.add_argument("--src", default="K:/yue2/fixtures/mms-fa/mms_fa_state_dict.pt",
                     metavar="PATH", help="the wrap.state_dict() .pt file")
    ap.add_argument("--out", default="K:/yue2/models/mms-fa", metavar="DIR",
                     help="output directory (default: K:/yue2/models/mms-fa)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--verify", action="store_true",
                     help="after writing, reopen the GGUF and verify it against the "
                          "source .pt and the live torchaudio module")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        die(f"source state dict not found: {args.src}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "mms-fa-f32.gguf")
    if os.path.exists(out_path) and not args.force:
        die(f"{out_path} already exists -- pass --force to overwrite")

    log(f"hashing source {args.src} ...")
    src_sha256 = sha256_file(args.src)
    log(f"source sha256: {src_sha256}")

    print_mapping_table()

    src = Source(args.src)
    log(f"source state dict: {len(src.state)} tensors")

    log("baking pos_conv weight-norm and cross-checking against the live "
        "torchaudio module before writing anything...")
    baked_preview = bake_pos_conv_weight(src)
    verify_pos_conv_bake(baked_preview)
    src.consumed.discard("model.encoder.transformer.pos_conv_embed.conv."
                          "parametrizations.weight.original0")
    src.consumed.discard("model.encoder.transformer.pos_conv_embed.conv."
                          "parametrizations.weight.original1")

    b = Bundle("mmsfa")
    build(src, b)
    add_meta(b, src_sha256)
    b.write(out_path)
    log("Done.")

    if args.verify:
        ok = verify(out_path, args.src)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

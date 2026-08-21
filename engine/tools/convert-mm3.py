#!/usr/bin/env python3
# convert-mm3.py: MiniMax-Music3 safetensors -> GGUF for the HOT-Step MM3 backend.
#
# Produces up to two files into --out:
#
#   mm3-lm-<quant>.gguf     arch "qwen3"  -- the 8B global LM, written to
#                                            llama.cpp Qwen3 conventions so a
#                                            llama.cpp-grade loader can consume
#                                            it structurally (token_embd.weight,
#                                            blk.N.attn_q.weight, ...). Music
#                                            specifics ride along as mm3.* KV.
#   mm3-synth-<quant>.gguf  arch "mm3"    -- depth decoder + condition encoder +
#                                            flow DiT + vocoder, tensor-prefixed
#                                            depth. / cond. / dit. / voc.
#
# WHY TWO FILES (not one, not five)
# ---------------------------------
# The LM file is kept structurally *pure* Qwen3: llama.cpp's loader rejects
# unknown tensors, so nothing music-specific may live in it. Everything else is
# ours to shape, and lives in one synth file because the four modules share a
# conversion pass and are always deployed together.
#
# Runtime residency caveat for the future C++ loader: the RVQ depth decoder runs
# *inside* the AR loop (it consumes the LM's last_hidden_state every frame), so
# stage 1 needs the LM file AND the depth.* half of the synth file resident at
# once. `model.audio_extra_embedding` (the 7168-row acoustic embedding the AR
# feedback path reads) is likewise in the synth file as depth.audio_embd.weight.
# See docs/plans/mm3-gguf-layout.md.
#
# SOURCES
# -------
# Preferred: Comfy-Org/MiniMax-Music-3 (repackaged single-file safetensors)
#   text_encoders/minimax_music3_text_encoder_bf16.safetensors  -> lm + depth
#   diffusion_models/minimax_music3_dit_fp{16,32}.safetensors   -> cond + dit
#   vae/minimax_music3_dav.safetensors                          -> vocoder
# Fallback: MiniMaxAI/MiniMax-Music3 (diffusers subfolder layout)
#   language_model/ rvq_depth_decoder/ condition_encoder/ transformer/ vocoder/
#
# Both name schemes are accepted; every tensor is looked up through an alias
# list and validated against a config-derived expected shape. Anything missing,
# mis-shaped or left over is reported as a diff and the run exits nonzero --
# a partial model is never emitted.
#
# NOT accepted: *_int8_convrot.safetensors (Comfy's packed int8 + rotation
# format -- not dequantizable here) and *_pruned_*.safetensors (fused qkv /
# gate_up and split/pruned embeddings -- a different model surface). Both are
# detected and refused with an explanatory message.
#
# DEPENDENCIES: numpy + gguf (llama.cpp's gguf-py). torch only for .pth inputs.
# safetensors is read with a small built-in mmap reader so BF16 tensors work
# without a torch dependency.
#
# USAGE
#   python engine/tools/convert-mm3.py --src D:/models/MiniMax-Music-3 \
#          --out engine/models --components lm,depth,cond,dit,vocoder --quant f16

import argparse
import json
import mmap
import os
import re
import struct
import sys

import numpy as np
import gguf

CONVERTER_VERSION = 1

# ---------------------------------------------------------------------------
# Model constants. Sourced from the MiniMaxAI/MiniMax-Music3 config.json files
# (language_model/, transformer/, rvq_depth_decoder/, condition_encoder/,
# vocoder/, scheduler/) and from ComfyUI comfy/ldm/minimax_music/*.py for the
# behaviours that are code, not config. Hardcoded on purpose: the converter must
# be able to reject a mismatched checkpoint, which it cannot do if it takes the
# checkpoint's word for the shapes.
# ---------------------------------------------------------------------------

# -- global LM (Qwen3-8B with extended audio vocab) -------------------------
# LM_LAYERS is the only one of these a legitimate variant may change: the
# depth-pruned distilled students (e.g. Mothersuperior/minimax-music3-composer-
# 5.7b-distilled, 21 layers) keep every other dimension bit-for-bit, because
# the stock depth decoder consumes the LM's 4096-wide hidden state and the
# 200000-row audio vocabulary is what makes it a music model at all. So it is
# overridable via --lm-layers while the rest stays hardcoded (see the note
# above). The leftover-tensor diff at the end of main() is what makes this
# safe: a wrong --lm-layers leaves whole `model.layers.N.*` groups unconsumed
# and the run dies rather than emitting a truncated model.
LM_LAYERS = 36
LM_DIM = 4096
LM_FF = 12288
LM_HEADS = 32
LM_KV_HEADS = 8
LM_HEAD_DIM = 128
LM_VOCAB = 200000
LM_CTX = 10240
LM_RMS_EPS = 1e-6
LM_ROPE_THETA = 1000000.0

# -- audio vocab layout (comfy/ldm/minimax_music/prompt.py + ar.py) ---------
AUDIO_CODE_OFFSET = 151675          # semantic (codebook 0) codes start here
SEMANTIC_VOCAB = 16384              # ... and run to 168058 inclusive
ACOUSTIC_VOCAB = 1024               # codebooks 1..7
NUM_CODEBOOKS = 8
FRAME_RATE = 25                     # audio frames per second
MAX_AUDIO_FRAMES = 9000             # 6 min ceiling in the reference impl
MAX_PROMPT_TOKENS = 5000
AR_CFG_SCALE = 1.5
AR_TOP_K = 50
SPECIAL_TOKEN_IDS = {
    "im_start": 151644,
    "im_end": 151645,
    "audio_cfg": 151654,
    "audio_start": 151669,
    "audio_end": 151670,            # the AR stop token
    "caption_start": 151671,
    "caption_end": 151672,
    "lyrics_start": 151673,
    "lyrics_end": 151674,
}

# -- RVQ depth decoder ------------------------------------------------------
DEPTH_LAYERS = 4
DEPTH_DIM = 4096
DEPTH_FF = 6144
DEPTH_HEADS = 16
DEPTH_HEAD_DIM = DEPTH_DIM // DEPTH_HEADS   # 256
DEPTH_MAX_POS = 16
DEPTH_RMS_EPS = 1e-6
DEPTH_AUDIO_EMBD_ROWS = ACOUSTIC_VOCAB * (NUM_CODEBOOKS - 1)   # 7168

# -- condition encoder ------------------------------------------------------
COND_LAYERS = 8                     # LM hidden layers mixed by softmax(logits)
COND_HIDDEN = 4096
COND_OUT = 2048
COND_KERNEL = 3
COND_PADDING = 1
COND_IN_SR = 24000
COND_IN_HOP = 960                   # 24000/960 = 25 fps, matches FRAME_RATE
COND_OUT_SR = 44100
COND_OUT_HOP = 512                  # 44100/512 = 86.13 latent fps

# -- flow DiT ---------------------------------------------------------------
DIT_LAYERS = 36
DIT_DIM = 2048
DIT_HEADS = 32
DIT_HEAD_DIM = 64
DIT_FF_INNER = 8192
DIT_IN_CH = 128
DIT_COND_DIM = 2048
DIT_CONCAT_CH = DIT_IN_CH * 2 + DIT_COND_DIM        # 2304: [x, zeros_like(x), cond]
DIT_ROPE_DIM = 32                                   # of DIT_HEAD_DIM = 64
DIT_ROPE_THETA = 10000.0
DIT_FOURIER_DIM = 256                               # cat(cos, sin) of 128 freqs
DIT_LN_EPS = 1e-5                                   # torch layer_norm default
DIT_WINDOW_FRAMES = 200
DIT_HOP_FRAMES = 100
DIT_STEPS = 30
DIT_CFG_SCALE = 1.7

# -- vocoder (DAC-style decoder) --------------------------------------------
# -- DAV encoder (audio -> latents; TRAINING ONLY) --------------------------
# The mirror of the vocoder. Published ONLY in the original dav.pth: the
# diffusers layout exposes the decoder half as vocoder/ and drops the encoder,
# because inference never needs it. Training does — these are the flow DiT's
# target latents. Written to its own small mm3-enc-*.gguf rather than into
# mm3-synth-*.gguf, so an existing 6.4 GB synth file stays valid and ace-server
# never loads a tensor it cannot use.
ENC_IN_DIM = 64                     # channels after the stem conv
ENC_RATES = (2, 4, 8, 8)            # product 512 -> 44100/512 = 86.13 latent fps
ENC_RES_DILATIONS = (1, 3, 9)
ENC_LATENT_DIM = 1024               # encoder trunk output width
ENC_FOLD_CH = 64                    # per audio channel; 2 x 64 = 128 latent ch
ENC_SR = 44100
ENC_SNAKE_EPS = 1e-9

VOC_LATENT_CH = 128
VOC_FOLD_CH = 64                    # stereo folded as 2 x 64ch through shared weights
VOC_DEC_IN_DIM = 1024
VOC_HIDDEN = 1536
VOC_UPSAMPLE = (8, 8, 4, 2)         # product 512
VOC_RES_DILATIONS = (1, 3, 9)
VOC_SR = 44100
VOC_SNAKE_EPS = 1e-9


def latent_length(frames):
    """AR frames (25 fps) -> flow latent frames (86.13 fps). ComfyUI dit.py."""
    return max(1, int(frames * COND_OUT_SR / COND_IN_SR * COND_IN_HOP / COND_OUT_HOP))


def log(msg):
    print(f"[convert-mm3] {msg}", file=sys.stderr, flush=True)


def die(msg):
    raise SystemExit(f"[convert-mm3] ERROR: {msg}")


# ---------------------------------------------------------------------------
# safetensors reader
#
# Deliberately built-in rather than via the safetensors package: BF16 is the
# native dtype of the LM/depth weights and the package's numpy framework cannot
# express it, which would force a torch dependency on the common path. The
# format is trivial -- u64 header length, JSON header, then a data blob whose
# offsets are relative to the end of the header.
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

    def get(self, name):
        """Returns a float32 numpy array (or the natural dtype for int/byte)."""
        spec = self.header[name]
        buf = self.raw(name)
        dt = spec["dtype"]
        shape = tuple(spec["shape"])
        if dt == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        np_dt = {
            "F64": np.float64, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16,
            "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
        }[dt]
        arr = np.frombuffer(buf, dtype=np_dt).reshape(shape)
        if np_dt in (np.float64, np.float32, np.float16):
            return arr.astype(np.float32)
        return arr

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


class TorchFile:
    """.pth fallback (dav.pth / flowmatching_vae.pth in the official repo)."""

    def __init__(self, path):
        try:
            import torch
        except ImportError:
            die(f"{path} is a .pth checkpoint; install torch or use the "
                f"safetensors sources instead")
        self.path = path
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        if not isinstance(sd, dict):
            die(f"{path}: not a state dict")
        self._sd = sd

    def keys(self):
        return self._sd.keys()

    def shape(self, name):
        return tuple(self._sd[name].shape)

    def dtype(self, name):
        return str(self._sd[name].dtype).replace("torch.", "").upper()

    def get(self, name):
        import torch
        t = self._sd[name].detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.to(torch.float32).contiguous().numpy()

    def close(self):
        self._sd = {}


# ---------------------------------------------------------------------------
# Source index: every tensor name across every checkpoint under --src.
# ---------------------------------------------------------------------------

REFUSE_PATTERNS = [
    (re.compile(r"int8|convrot|fp8|nf4", re.I),
     "packed int8/convrot/fp8 weights cannot be dequantized by this converter; "
     "use the fp16 or fp32 variant"),
    (re.compile(r"pruned", re.I),
     "the '_pruned_' repack fuses qkv/gate_up and splits the embedding into "
     "embed_tokens_prefill/embed_tokens_audio + lm_head_pruned; it is a "
     "different model surface. Use the plain bf16 text encoder"),
]

# Higher score wins when two files define the same tensor name.
def _file_score(path):
    b = os.path.basename(path).lower()
    score = 0
    if "fp32" in b or "f32" in b:
        score += 30
    elif "bf16" in b:
        score += 20
    elif "fp16" in b or "f16" in b:
        score += 10
    if path.lower().endswith(".pth"):
        score -= 5
    return score


class Source:
    def __init__(self, paths, verbose=False):
        self.files = []
        self.index = {}          # tensor name -> file object
        self.consumed = set()
        self.verbose = verbose
        found = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                for root, _dirs, names in os.walk(p):
                    for n in sorted(names):
                        if n.endswith(".safetensors") or n.endswith(".pth"):
                            found.append(os.path.join(root, n))
            elif os.path.isfile(p):
                found.append(p)
            else:
                die(f"--src path does not exist: {p}")
        if not found:
            die(f"no .safetensors or .pth files found under: {', '.join(paths)}")

        keep = []
        for path in found:
            base = os.path.basename(path)
            refused = None
            for pat, why in REFUSE_PATTERNS:
                if pat.search(base):
                    refused = why
                    break
            if refused:
                log(f"skipping {base}: {refused}")
                continue
            keep.append(path)
        if not keep:
            die("every candidate file was refused; see the messages above")

        for path in sorted(keep, key=_file_score):
            f = SafeTensorsFile(path) if path.endswith(".safetensors") else TorchFile(path)
            self.files.append(f)
            n_new = n_over = 0
            for name in f.keys():
                if name in self.index:
                    n_over += 1
                else:
                    n_new += 1
                self.index[name] = f            # later (higher score) wins
            note = f", {n_over} overriding earlier files" if n_over else ""
            log(f"indexed {short(path)}: {n_new + n_over} tensors{note}")

        self.layout = self._detect_layout()
        log(f"source layout: {self.layout}")

    def _detect_layout(self):
        if "diffusion_transformer.preprocess_conv.weight" in self.index:
            return "comfy"
        if "preprocess_conv.weight" in self.index or "transformer_blocks.0.attn.to_q.weight" in self.index:
            return "diffusers"
        if "model.audio_decoder.norm.weight" in self.index:
            return "comfy"
        if "audio_embeddings.weight" in self.index:
            return "diffusers"
        return "unknown"

    def has(self, *names):
        return any(n in self.index for n in names)

    def resolve(self, *names):
        for n in names:
            if n in self.index:
                return n
        return None

    def get(self, *names, expect=None, label=None):
        name = self.resolve(*names)
        if name is None:
            die(f"missing tensor {label or names[0]}; tried: {', '.join(names)}")
        arr = self.index[name].get(name)
        self.consumed.add(name)
        if expect is not None and tuple(arr.shape) != tuple(expect):
            die(f"{name}: expected shape {tuple(expect)}, checkpoint has "
                f"{tuple(arr.shape)} -- this is not a MiniMax-Music3 checkpoint "
                f"this converter understands")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def get_bytes(self, name):
        if name not in self.index:
            return None
        f = self.index[name]
        self.consumed.add(name)
        if isinstance(f, SafeTensorsFile):
            return bytes(f.raw(name))
        return bytes(f.get(name).astype(np.uint8).tobytes())

    def unconsumed(self, predicate):
        return sorted(n for n in self.index if predicate(n) and n not in self.consumed)

    def close(self):
        for f in self.files:
            f.close()


# ---------------------------------------------------------------------------
# Tensor policy + writer
# ---------------------------------------------------------------------------

F32 = "f32"        # never quantized: norms, biases, alphas, scales, inv_freq
HALF = "half"      # f16 even under --quant q8_0 (ne0 not a multiple of 32)
QUANT = "quant"    # q8_0 under --quant q8_0, f16 otherwise

F16_MAX = 65504.0


class Bundle:
    """Accumulates (name, array, policy) and writes one GGUF."""

    def __init__(self, arch):
        self.arch = arch
        self.kv = []
        self.tensors = []

    def meta(self, fn, *args):
        self.kv.append((fn, args))

    def put(self, name, arr, policy):
        if name in {n for n, _, _ in self.tensors}:
            die(f"duplicate output tensor {name}")
        self.tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32), policy))

    def write(self, path, quant):
        w = gguf.GGUFWriter(path, self.arch)
        for fn, args in self.kv:
            getattr(w, fn)(*args)
        n = {"f32": 0, "f16": 0, "q8_0": 0}
        for name, arr, policy in self.tensors:
            ne0 = arr.shape[-1] if arr.ndim else 1
            if policy == F32 or arr.ndim < 2:
                w.add_tensor(name, arr)
                n["f32"] += 1
                continue
            peak = float(np.abs(arr).max()) if arr.size else 0.0
            if peak > F16_MAX:
                log(f"  {name}: |w|max={peak:.1f} exceeds f16 range, storing F32")
                w.add_tensor(name, arr)
                n["f32"] += 1
                continue
            if quant == "q8_0" and policy == QUANT and ne0 % 32 == 0:
                # Pass no raw_shape: gguf derives the logical shape from the
                # quantized byte rows, and would misread a logical one.
                w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0),
                             raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                n["q8_0"] += 1
            else:
                # raw_dtype labels, it does not convert -- hand it real f16 bytes.
                w.add_tensor(name, arr.astype(np.float16).view(np.uint16),
                             raw_shape=arr.shape,
                             raw_dtype=gguf.GGMLQuantizationType.F16)
                n["f16"] += 1
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        size = os.path.getsize(path) / 1e9
        log(f"wrote {os.path.basename(path)}: {len(self.tensors)} tensors "
            f"({n['f32']} F32, {n['f16']} F16, {n['q8_0']} Q8_0), {size:.2f} GB")


def short(path):
    """Display path: relative when that is meaningful, absolute otherwise.

    os.path.relpath RAISES on Windows when `path` and the cwd sit on different
    drives, which is the normal case here -- bulk weights live on a data drive
    and the repo does not. Cosmetic logging must never abort a conversion."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return os.path.abspath(path)


def fold_weight_norm(g, v, label):
    """PyTorch weight_norm (dim=0): w = g * v / ||v||, norm over all axes but 0.

    Holds for Conv1d weights (out, in, k) and ConvTranspose1d weights
    (in, out, k) alike -- torch normalises over dim 0 in both cases, which is
    why the ConvTranspose g vectors are sized by *input* channels."""
    if g.shape[0] != v.shape[0]:
        die(f"{label}: weight_g first dim {g.shape[0]} != weight_v {v.shape[0]}")
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.all(norm > 0):
        die(f"{label}: weight_v has a zero-norm slice; refusing to fold")
    g = g.astype(np.float64).reshape((g.shape[0],) + (1,) * (v.ndim - 1))
    return (g * v.astype(np.float64) / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Component: global LM  ->  arch "qwen3", llama.cpp tensor names
# ---------------------------------------------------------------------------

def build_lm(src, bundle, tokenizer, embed_tokenizer_json, n_layers=LM_LAYERS,
             ar_cfg_scale=AR_CFG_SCALE):
    if src.has("model.embed_tokens_prefill.weight", "model.lm_head_pruned.weight",
               "model.layers.0.self_attn.qkv_proj.weight"):
        die("this text encoder is the '_pruned_' repack (fused qkv / split "
            "embeddings). Convert from the plain bf16 text encoder or from "
            "MiniMaxAI/MiniMax-Music3 language_model/ instead")

    b = bundle
    b.meta("add_name", "MiniMax-Music3 LM")
    b.meta("add_description",
           "MiniMax-Music3 global LM: Qwen3-8B with an extended audio vocabulary"
           + ("" if n_layers == LM_LAYERS else
              f" (depth-pruned to {n_layers} of {LM_LAYERS} blocks)"))
    b.meta("add_context_length", LM_CTX)
    b.meta("add_embedding_length", LM_DIM)
    b.meta("add_block_count", n_layers)
    b.meta("add_feed_forward_length", LM_FF)
    b.meta("add_head_count", LM_HEADS)
    b.meta("add_head_count_kv", LM_KV_HEADS)
    b.meta("add_key_length", LM_HEAD_DIM)
    b.meta("add_value_length", LM_HEAD_DIM)
    b.meta("add_layer_norm_rms_eps", LM_RMS_EPS)
    b.meta("add_rope_freq_base", LM_ROPE_THETA)
    b.meta("add_vocab_size", LM_VOCAB)

    # -- music specifics, namespaced so the qwen3 loader ignores them --------
    b.meta("add_uint32", "mm3.semantic_vocab_offset", AUDIO_CODE_OFFSET)
    b.meta("add_uint32", "mm3.semantic_vocab_size", SEMANTIC_VOCAB)
    b.meta("add_uint32", "mm3.acoustic_vocab_size", ACOUSTIC_VOCAB)
    b.meta("add_uint32", "mm3.num_codebooks", NUM_CODEBOOKS)
    b.meta("add_uint32", "mm3.eos_audio", SPECIAL_TOKEN_IDS["audio_end"])
    b.meta("add_uint32", "mm3.frame_rate", FRAME_RATE)
    b.meta("add_uint32", "mm3.max_audio_frames", MAX_AUDIO_FRAMES)
    b.meta("add_uint32", "mm3.max_prompt_tokens", MAX_PROMPT_TOKENS)
    b.meta("add_float32", "mm3.ar.cfg_scale", ar_cfg_scale)
    b.meta("add_uint32", "mm3.ar.top_k", AR_TOP_K)
    b.meta("add_float32", "mm3.ar.embedding_scale", float(NUM_CODEBOOKS) ** -0.5)
    for k, v in SPECIAL_TOKEN_IDS.items():
        b.meta("add_uint32", f"mm3.token.{k}", v)

    if tokenizer is not None:
        add_tokenizer_kv(b, tokenizer, embed_tokenizer_json)
    else:
        log("WARNING: no tokenizer.json found -- the LM gguf will carry no "
            "vocab. Pass --tokenizer, or point --src at a tree containing "
            "tokenizer/tokenizer.json")

    b.put("token_embd.weight",
          src.get("model.embed_tokens.weight", expect=(LM_VOCAB, LM_DIM)), QUANT)
    b.put("output_norm.weight",
          src.get("model.norm.weight", expect=(LM_DIM,)), F32)
    b.put("output.weight",
          src.get("model.lm_head.weight", "lm_head.weight",
                  expect=(LM_VOCAB, LM_DIM)), QUANT)

    q_dim = LM_HEADS * LM_HEAD_DIM
    kv_dim = LM_KV_HEADS * LM_HEAD_DIM
    for i in range(n_layers):
        p = f"model.layers.{i}"
        d = f"blk.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(f"{p}.input_layernorm.weight", expect=(LM_DIM,)), F32)
        b.put(f"{d}.attn_q.weight",
              src.get(f"{p}.self_attn.q_proj.weight", expect=(q_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_k.weight",
              src.get(f"{p}.self_attn.k_proj.weight", expect=(kv_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_v.weight",
              src.get(f"{p}.self_attn.v_proj.weight", expect=(kv_dim, LM_DIM)), QUANT)
        b.put(f"{d}.attn_output.weight",
              src.get(f"{p}.self_attn.o_proj.weight", expect=(LM_DIM, q_dim)), QUANT)
        b.put(f"{d}.attn_q_norm.weight",
              src.get(f"{p}.self_attn.q_norm.weight", expect=(LM_HEAD_DIM,)), F32)
        b.put(f"{d}.attn_k_norm.weight",
              src.get(f"{p}.self_attn.k_norm.weight", expect=(LM_HEAD_DIM,)), F32)
        b.put(f"{d}.ffn_norm.weight",
              src.get(f"{p}.post_attention_layernorm.weight", expect=(LM_DIM,)), F32)
        b.put(f"{d}.ffn_gate.weight",
              src.get(f"{p}.mlp.gate_proj.weight", expect=(LM_FF, LM_DIM)), QUANT)
        b.put(f"{d}.ffn_up.weight",
              src.get(f"{p}.mlp.up_proj.weight", expect=(LM_FF, LM_DIM)), QUANT)
        b.put(f"{d}.ffn_down.weight",
              src.get(f"{p}.mlp.down_proj.weight", expect=(LM_DIM, LM_FF)), QUANT)

    def belongs(n):
        return (n.startswith("model.layers.")
                or n in ("model.embed_tokens.weight", "model.norm.weight",
                         "model.lm_head.weight", "lm_head.weight"))
    return belongs


# ---------------------------------------------------------------------------
# Tokenizer (GPT-2/BPE, llama.cpp conventions)
# ---------------------------------------------------------------------------

def add_tokenizer_kv(b, tok, embed_raw):
    model = tok.get("model") or {}
    if model.get("type") != "BPE":
        die(f"tokenizer.json model.type is {model.get('type')!r}, expected 'BPE'")
    vocab = model.get("vocab") or {}
    added = tok.get("added_tokens") or []

    reverse = {i: s for s, i in vocab.items()}
    special_ids, user_ids = set(), set()
    for a in added:
        reverse[a["id"]] = a["content"]
        (special_ids if a.get("special") else user_ids).add(a["id"])

    tokens, toktypes = [], []
    n_pad = n_code = 0
    for i in range(LM_VOCAB):
        if i in reverse:
            tokens.append(reverse[i])
            if i in special_ids:
                toktypes.append(gguf.TokenType.CONTROL)
            elif i in user_ids:
                toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
        elif AUDIO_CODE_OFFSET <= i < AUDIO_CODE_OFFSET + SEMANTIC_VOCAB:
            # The 16384 semantic codes. Never produced by text tokenisation --
            # the AR loop maps code -> id arithmetically -- but named so that
            # detokenising generated ids is legible.
            tokens.append(f"<|audio_code_{i - AUDIO_CODE_OFFSET}|>")
            toktypes.append(gguf.TokenType.USER_DEFINED)
            n_code += 1
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            n_pad += 1

    merges = []
    for m in model.get("merges") or []:
        merges.append(" ".join(m) if isinstance(m, (list, tuple)) else m)

    b.meta("add_tokenizer_model", "gpt2")
    b.meta("add_tokenizer_pre", "qwen2")
    b.meta("add_token_list", tokens)
    b.meta("add_token_types", toktypes)
    b.meta("add_token_merges", merges)
    b.meta("add_eos_token_id", SPECIAL_TOKEN_IDS["im_end"])
    b.meta("add_pad_token_id", 151643)          # <|endoftext|>
    b.meta("add_add_bos_token", False)
    if embed_raw is not None:
        b.meta("add_string", "mm3.tokenizer_json", embed_raw)
    log(f"tokenizer: {len(tokens)} tokens ({len(vocab)} BPE + {len(added)} added "
        f"+ {n_code} audio codes + {n_pad} unused), {len(merges)} merges")


def find_tokenizer(src, explicit, search_paths):
    """Returns (parsed_json, raw_text) or (None, None)."""
    if explicit:
        with open(explicit, "r", encoding="utf-8") as f:
            raw = f.read()
        return json.loads(raw), raw
    # The Comfy text-encoder repack embeds tokenizer.json as a U8 tensor.
    blob = src.get_bytes("tokenizer_json")
    if blob:
        raw = blob.decode("utf-8")
        log("tokenizer: taken from the 'tokenizer_json' tensor in the source")
        return json.loads(raw), raw
    for p in search_paths:
        p = os.path.abspath(p)
        roots = [p] if os.path.isdir(p) else [os.path.dirname(p), os.path.dirname(os.path.dirname(p))]
        for root in roots:
            for cand in ("tokenizer.json",
                         os.path.join("tokenizer", "tokenizer.json"),
                         os.path.join("qwen_7B", "qwen3-8B-tokenizer-music", "tokenizer.json")):
                f = os.path.join(root, cand)
                if os.path.isfile(f):
                    with open(f, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                    log(f"tokenizer: {short(f)}")
                    return json.loads(raw), raw
    return None, None


# ---------------------------------------------------------------------------
# Component: RVQ depth decoder  ->  depth.*
# ---------------------------------------------------------------------------

def build_depth(src, b):
    b.meta("add_uint32", "mm3.depth.block_count", DEPTH_LAYERS)
    b.meta("add_uint32", "mm3.depth.embedding_length", DEPTH_DIM)
    b.meta("add_uint32", "mm3.depth.feed_forward_length", DEPTH_FF)
    b.meta("add_uint32", "mm3.depth.head_count", DEPTH_HEADS)
    b.meta("add_uint32", "mm3.depth.head_dim", DEPTH_HEAD_DIM)
    b.meta("add_uint32", "mm3.depth.max_position", DEPTH_MAX_POS)
    b.meta("add_float32", "mm3.depth.rms_eps", DEPTH_RMS_EPS)
    b.meta("add_uint32", "mm3.depth.num_codebooks", NUM_CODEBOOKS)
    b.meta("add_uint32", "mm3.depth.audio_vocab_size", ACOUSTIC_VOCAB)
    b.meta("add_uint32", "mm3.depth.audio_embd_rows", DEPTH_AUDIO_EMBD_ROWS)
    b.meta("add_bool", "mm3.depth.causal", True)
    b.meta("add_bool", "mm3.depth.rope", False)

    P = ["model.audio_decoder.", ""]     # comfy prefix, diffusers prefix

    def a(suffix):
        return tuple(p + suffix for p in P)

    b.put("depth.proj.weight",
          src.get(*a("projection.weight"), expect=(DEPTH_DIM, DEPTH_DIM)), QUANT)
    b.put("depth.pos_embd.weight",
          src.get(*a("pos_embedding.weight"), expect=(DEPTH_MAX_POS, DEPTH_DIM)), F32)
    b.put("depth.output_norm.weight",
          src.get(*a("norm.weight"), expect=(DEPTH_DIM,)), F32)
    b.put("depth.audio_embd.weight",
          src.get("model.audio_extra_embedding.weight", "audio_embeddings.weight",
                  expect=(DEPTH_AUDIO_EMBD_ROWS, DEPTH_DIM)), QUANT)
    for c in range(NUM_CODEBOOKS - 1):
        b.put(f"depth.head.{c}.weight",
              src.get(*a(f"audio_heads.{c}.weight"),
                      expect=(ACOUSTIC_VOCAB, DEPTH_DIM)), QUANT)

    for i in range(DEPTH_LAYERS):
        d = f"depth.blk.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(*a(f"layers.{i}.input_layernorm.weight"), expect=(DEPTH_DIM,)), F32)
        for tag, comfy, diff in (("attn_q", "self_attn.q_proj", "attn.to_q"),
                                 ("attn_k", "self_attn.k_proj", "attn.to_k"),
                                 ("attn_v", "self_attn.v_proj", "attn.to_v"),
                                 ("attn_output", "self_attn.o_proj", "attn.to_out")):
            b.put(f"{d}.{tag}.weight",
                  src.get(f"model.audio_decoder.layers.{i}.{comfy}.weight",
                          f"layers.{i}.{diff}.weight",
                          expect=(DEPTH_DIM, DEPTH_DIM)), QUANT)
        b.put(f"{d}.ffn_norm.weight",
              src.get(*a(f"layers.{i}.post_attention_layernorm.weight"),
                      expect=(DEPTH_DIM,)), F32)
        for tag, comfy, diff, shape in (
                ("ffn_gate", "mlp.gate_proj", "gate_proj", (DEPTH_FF, DEPTH_DIM)),
                ("ffn_up", "mlp.up_proj", "up_proj", (DEPTH_FF, DEPTH_DIM)),
                ("ffn_down", "mlp.down_proj", "down_proj", (DEPTH_DIM, DEPTH_FF))):
            b.put(f"{d}.{tag}.weight",
                  src.get(f"model.audio_decoder.layers.{i}.{comfy}.weight",
                          f"layers.{i}.{diff}.weight", expect=shape), QUANT)

    def belongs(n):
        return (n.startswith("model.audio_decoder.")
                or n == "model.audio_extra_embedding.weight"
                or n == "audio_embeddings.weight"
                or (re.match(r"layers\.\d+\.", n) and not n.startswith("model."))
                or n in ("projection.weight", "pos_embedding.weight", "norm.weight")
                or n.startswith("audio_heads."))
    return belongs


# ---------------------------------------------------------------------------
# Component: condition encoder  ->  cond.*
# ---------------------------------------------------------------------------

def build_cond(src, b):
    b.meta("add_uint32", "mm3.cond.num_layers", COND_LAYERS)
    b.meta("add_uint32", "mm3.cond.hidden_dim", COND_HIDDEN)
    b.meta("add_uint32", "mm3.cond.out_dim", COND_OUT)
    b.meta("add_uint32", "mm3.cond.kernel_size", COND_KERNEL)
    b.meta("add_uint32", "mm3.cond.padding", COND_PADDING)
    b.meta("add_uint32", "mm3.cond.input_sampling_rate", COND_IN_SR)
    b.meta("add_uint32", "mm3.cond.input_hop_length", COND_IN_HOP)
    b.meta("add_uint32", "mm3.cond.output_sampling_rate", COND_OUT_SR)
    b.meta("add_uint32", "mm3.cond.output_hop_length", COND_OUT_HOP)
    b.meta("add_string", "mm3.cond.interpolation", "nearest")
    b.meta("add_string", "mm3.cond.layer_mix", "softmax")

    b.put("cond.layer_logits",
          src.get("cond_layer_logits", "layer_weight_logits", expect=(COND_LAYERS,)), F32)
    b.put("cond.layer_scale",
          src.get("cond_layer_scale", "layer_scale", expect=(1,)), F32)
    b.put("cond.proj.weight",
          src.get("latent_conditioners.0.weight", "proj.weight",
                  expect=(COND_OUT, COND_HIDDEN, COND_KERNEL)), HALF)
    b.put("cond.proj.bias",
          src.get("latent_conditioners.0.bias", "proj.bias", expect=(COND_OUT,)), F32)

    def belongs(n):
        return n in ("cond_layer_logits", "cond_layer_scale", "layer_weight_logits",
                     "layer_scale", "proj.weight", "proj.bias") \
            or n.startswith("latent_conditioners.")
    return belongs


# ---------------------------------------------------------------------------
# Component: flow DiT  ->  dit.*
# ---------------------------------------------------------------------------

def build_dit(src, b):
    b.meta("add_uint32", "mm3.dit.block_count", DIT_LAYERS)
    b.meta("add_uint32", "mm3.dit.embedding_length", DIT_DIM)
    b.meta("add_uint32", "mm3.dit.head_count", DIT_HEADS)
    b.meta("add_uint32", "mm3.dit.head_dim", DIT_HEAD_DIM)
    b.meta("add_uint32", "mm3.dit.ff_inner", DIT_FF_INNER)
    b.meta("add_uint32", "mm3.dit.in_channels", DIT_IN_CH)
    b.meta("add_uint32", "mm3.dit.condition_dim", DIT_COND_DIM)
    b.meta("add_uint32", "mm3.dit.concat_channels", DIT_CONCAT_CH)
    b.meta("add_uint32", "mm3.dit.rope_dim", DIT_ROPE_DIM)
    b.meta("add_float32", "mm3.dit.rope_theta", DIT_ROPE_THETA)
    b.meta("add_string", "mm3.dit.rope_type", "neox")
    b.meta("add_uint32", "mm3.dit.fourier_dim", DIT_FOURIER_DIM)
    b.meta("add_float32", "mm3.dit.layer_norm_eps", DIT_LN_EPS)
    b.meta("add_string", "mm3.dit.glu_order", "value_gate")
    b.meta("add_bool", "mm3.dit.output_negated", True)
    b.meta("add_bool", "mm3.dit.timestep_token_prepended", True)
    b.meta("add_bool", "mm3.dit.pre_post_conv_residual", True)
    b.meta("add_bool", "mm3.dit.attn_bias", False)
    b.meta("add_uint32", "mm3.dit.window_frames", DIT_WINDOW_FRAMES)
    b.meta("add_uint32", "mm3.dit.hop_frames", DIT_HOP_FRAMES)
    b.meta("add_uint32", "mm3.dit.window_latents", latent_length(DIT_WINDOW_FRAMES))
    b.meta("add_uint32", "mm3.dit.hop_latents", latent_length(DIT_HOP_FRAMES))
    b.meta("add_string", "mm3.flow.scheduler", "FlowMatchEulerDiscrete")
    b.meta("add_uint32", "mm3.flow.steps", DIT_STEPS)
    b.meta("add_float32", "mm3.flow.cfg_scale", DIT_CFG_SCALE)
    b.meta("add_bool", "mm3.flow.invert_sigmas", True)
    b.meta("add_float32", "mm3.flow.shift", 1.0)
    b.meta("add_uint32", "mm3.flow.num_train_timesteps", 1)

    C = "diffusion_transformer."
    b.put("dit.preprocess_conv.weight",
          src.get(C + "preprocess_conv.weight", "preprocess_conv.weight",
                  expect=(DIT_CONCAT_CH, DIT_CONCAT_CH, 1)), HALF)
    b.put("dit.postprocess_conv.weight",
          src.get(C + "postprocess_conv.weight", "postprocess_conv.weight",
                  expect=(DIT_IN_CH, DIT_IN_CH, 1)), HALF)
    # Random-Fourier timestep features: features = 2*pi * t @ W.T, then
    # cat(cos, sin). W is (fourier_dim/2, 1) -- kept F32, it is a frequency
    # basis and half precision visibly quantises the timestep embedding.
    b.put("dit.time_fourier.weight",
          src.get(C + "timestep_features.weight", "time_proj.weight",
                  expect=(DIT_FOURIER_DIM // 2, 1)), F32)
    b.put("dit.time_embd.0.weight",
          src.get(C + "to_timestep_embed.0.weight", "time_embed.linear_1.weight",
                  expect=(DIT_DIM, DIT_FOURIER_DIM)), QUANT)
    b.put("dit.time_embd.0.bias",
          src.get(C + "to_timestep_embed.0.bias", "time_embed.linear_1.bias",
                  expect=(DIT_DIM,)), F32)
    b.put("dit.time_embd.1.weight",
          src.get(C + "to_timestep_embed.2.weight", "time_embed.linear_2.weight",
                  expect=(DIT_DIM, DIT_DIM)), QUANT)
    b.put("dit.time_embd.1.bias",
          src.get(C + "to_timestep_embed.2.bias", "time_embed.linear_2.bias",
                  expect=(DIT_DIM,)), F32)
    b.put("dit.proj_in.weight",
          src.get(C + "transformer.project_in.weight", "proj_in.weight",
                  expect=(DIT_DIM, DIT_CONCAT_CH)), QUANT)
    b.put("dit.proj_out.weight",
          src.get(C + "transformer.project_out.weight", "proj_out.weight",
                  expect=(DIT_IN_CH, DIT_DIM)), QUANT)

    # RoPE inverse frequencies. Synthesised at F32 from theta rather than taken
    # from the checkpoint: Comfy stores the buffer at F16, which is lossy for a
    # frequency basis. The stored copy, when present, is used as a check.
    inv_freq = 1.0 / (DIT_ROPE_THETA ** (
        np.arange(0, DIT_ROPE_DIM, 2, dtype=np.float64) / DIT_ROPE_DIM))
    stored = None
    if src.has(C + "transformer.rotary_pos_emb.inv_freq"):
        stored = src.get(C + "transformer.rotary_pos_emb.inv_freq",
                         expect=(DIT_ROPE_DIM // 2,))
        rel = np.abs(stored.astype(np.float64) - inv_freq) / np.maximum(inv_freq, 1e-12)
        if rel.max() > 1e-2:
            die(f"dit rope inv_freq mismatch: checkpoint disagrees with "
                f"theta={DIT_ROPE_THETA} by up to {rel.max():.3%}. The DiT rope "
                f"base is not 10000 -- fix DIT_ROPE_THETA before converting")
        log(f"dit rope inv_freq: checkpoint matches theta={DIT_ROPE_THETA:.0f} "
            f"(max rel dev {rel.max():.2e}), using the exact F32 basis")
    b.put("dit.rope_inv_freq", inv_freq.astype(np.float32), F32)

    for i in range(DIT_LAYERS):
        d = f"dit.blk.{i}"
        cp = f"{C}transformer.layers.{i}"
        dp = f"transformer_blocks.{i}"
        b.put(f"{d}.attn_norm.weight",
              src.get(f"{cp}.pre_norm.gamma", f"{dp}.norm1.weight", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.attn_norm.bias",
              src.get(f"{cp}.pre_norm.beta", f"{dp}.norm1.bias", expect=(DIT_DIM,)), F32)
        # Canonicalise to a fused qkv: Comfy ships it fused and chunks q,k,v in
        # that order; diffusers ships three matrices. One GGUF layout either way.
        if src.has(f"{cp}.self_attn.to_qkv.weight"):
            qkv = src.get(f"{cp}.self_attn.to_qkv.weight", expect=(3 * DIT_DIM, DIT_DIM))
        else:
            qkv = np.concatenate([
                src.get(f"{dp}.attn.to_q.weight", expect=(DIT_DIM, DIT_DIM)),
                src.get(f"{dp}.attn.to_k.weight", expect=(DIT_DIM, DIT_DIM)),
                src.get(f"{dp}.attn.to_v.weight", expect=(DIT_DIM, DIT_DIM)),
            ], axis=0)
        b.put(f"{d}.attn_qkv.weight", qkv, QUANT)
        b.put(f"{d}.attn_output.weight",
              src.get(f"{cp}.self_attn.to_out.weight", f"{dp}.attn.to_out.0.weight",
                      expect=(DIT_DIM, DIT_DIM)), QUANT)
        b.put(f"{d}.ffn_norm.weight",
              src.get(f"{cp}.ff_norm.gamma", f"{dp}.norm2.weight", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.ffn_norm.bias",
              src.get(f"{cp}.ff_norm.beta", f"{dp}.norm2.bias", expect=(DIT_DIM,)), F32)
        b.put(f"{d}.ffn_in.weight",
              src.get(f"{cp}.ff.ff.0.proj.weight", f"{dp}.ff_in.weight",
                      expect=(2 * DIT_FF_INNER, DIT_DIM)), QUANT)
        b.put(f"{d}.ffn_in.bias",
              src.get(f"{cp}.ff.ff.0.proj.bias", f"{dp}.ff_in.bias",
                      expect=(2 * DIT_FF_INNER,)), F32)
        b.put(f"{d}.ffn_out.weight",
              src.get(f"{cp}.ff.ff.2.weight", f"{dp}.ff_out.weight",
                      expect=(DIT_DIM, DIT_FF_INNER)), QUANT)
        b.put(f"{d}.ffn_out.bias",
              src.get(f"{cp}.ff.ff.2.bias", f"{dp}.ff_out.bias", expect=(DIT_DIM,)), F32)

    def belongs(n):
        return (n.startswith("diffusion_transformer.")
                or n.startswith("transformer_blocks.")
                or n in ("preprocess_conv.weight", "postprocess_conv.weight",
                         "proj_in.weight", "proj_out.weight", "time_proj.weight")
                or n.startswith("time_embed."))
    return belongs


# ---------------------------------------------------------------------------
# Component: vocoder  ->  voc.*   (weight-norm folded here, once)
# ---------------------------------------------------------------------------

def build_vocoder(src, b):
    b.meta("add_uint32", "mm3.voc.latent_channels", VOC_LATENT_CH)
    b.meta("add_uint32", "mm3.voc.fold_channels", VOC_FOLD_CH)
    b.meta("add_uint32", "mm3.voc.dec_in_dim", VOC_DEC_IN_DIM)
    b.meta("add_uint32", "mm3.voc.hidden_dim", VOC_HIDDEN)
    b.meta("add_array", "mm3.voc.upsample_rates", list(VOC_UPSAMPLE))
    b.meta("add_array", "mm3.voc.res_dilations", list(VOC_RES_DILATIONS))
    b.meta("add_uint32", "mm3.voc.total_upsample", int(np.prod(VOC_UPSAMPLE)))
    b.meta("add_uint32", "mm3.voc.sampling_rate", VOC_SR)
    b.meta("add_uint32", "mm3.voc.channels", 2)
    b.meta("add_float32", "mm3.voc.snake_eps", VOC_SNAKE_EPS)
    b.meta("add_bool", "mm3.voc.final_tanh", True)
    b.meta("add_bool", "mm3.voc.weight_norm_folded", True)
    b.meta("add_string", "mm3.voc.snake",
           "x + 1/(alpha+eps) * sin(alpha*x)^2")

    def wn(dst, comfy, diffusers, expect_v, expect_bias):
        g = src.get(f"{comfy}.weight_g", f"{diffusers}.weight_g",
                    expect=(expect_v[0], 1, 1))
        v = src.get(f"{comfy}.weight_v", f"{diffusers}.weight_v", expect=expect_v)
        b.put(f"{dst}.weight", fold_weight_norm(g, v, dst), F32)
        b.put(f"{dst}.bias", src.get(f"{comfy}.bias", f"{diffusers}.bias",
                                     expect=(expect_bias,)), F32)

    def alpha(dst, comfy, diffusers, ch):
        b.put(dst, src.get(comfy, diffusers, expect=(1, ch, 1)), F32)

    b.put("voc.dec_in.weight",
          src.get("dec_in_proj.weight", expect=(VOC_DEC_IN_DIM, VOC_FOLD_CH, 1)), F32)
    b.put("voc.dec_in.bias",
          src.get("dec_in_proj.bias", expect=(VOC_DEC_IN_DIM,)), F32)
    wn("voc.conv_in", "decoder.model.0", "conv_in",
       (VOC_HIDDEN, VOC_DEC_IN_DIM, 7), VOC_HIDDEN)

    ch = VOC_HIDDEN
    for bi, stride in enumerate(VOC_UPSAMPLE):
        out_ch = ch // 2
        m = bi + 1                                  # comfy decoder.model index
        alpha(f"voc.blk.{bi}.snake.alpha",
              f"decoder.model.{m}.block.0.alpha", f"blocks.{bi}.snake1.alpha", ch)
        # ConvTranspose1d weight is (in, out, k); torch weight_norm dim=0 means
        # the g vector is sized by *input* channels.
        wn(f"voc.blk.{bi}.convt", f"decoder.model.{m}.block.1", f"blocks.{bi}.conv_t1",
           (ch, out_ch, 2 * stride), out_ch)
        for ri, dil in enumerate(VOC_RES_DILATIONS):
            u = ri + 2                              # comfy .block.{2,3,4}
            alpha(f"voc.blk.{bi}.res.{ri}.snake1.alpha",
                  f"decoder.model.{m}.block.{u}.block.0.alpha",
                  f"blocks.{bi}.res_unit{ri + 1}.snake1.alpha", out_ch)
            wn(f"voc.blk.{bi}.res.{ri}.conv1",
               f"decoder.model.{m}.block.{u}.block.1",
               f"blocks.{bi}.res_unit{ri + 1}.conv1",
               (out_ch, out_ch, 7), out_ch)
            alpha(f"voc.blk.{bi}.res.{ri}.snake2.alpha",
                  f"decoder.model.{m}.block.{u}.block.2.alpha",
                  f"blocks.{bi}.res_unit{ri + 1}.snake2.alpha", out_ch)
            wn(f"voc.blk.{bi}.res.{ri}.conv2",
               f"decoder.model.{m}.block.{u}.block.3",
               f"blocks.{bi}.res_unit{ri + 1}.conv2",
               (out_ch, out_ch, 1), out_ch)
        ch = out_ch

    n_blk = len(VOC_UPSAMPLE)
    alpha("voc.snake_out.alpha",
          f"decoder.model.{n_blk + 1}.alpha", "snake_out.alpha", ch)
    wn("voc.conv_out", f"decoder.model.{n_blk + 2}", "conv_out", (1, ch, 7), 1)

    def belongs(n):
        return (n.startswith("decoder.model.") or n.startswith("blocks.")
                or n.startswith("dec_in_proj.") or n.startswith("conv_in.")
                or n.startswith("conv_out.") or n.startswith("snake_out."))
    return belongs


# ---------------------------------------------------------------------------
# Component: DAV encoder  ->  enc.*      (source: dav.pth ONLY)
# ---------------------------------------------------------------------------

def build_enc(src, b):
    """Audio -> 128-channel latents at 86.13 Hz, the flow DiT's target space.

    Structure (transcribed from the only public implementation, SimpleTuner's
    helpers/models/minimaxmusic/vocoder.py, whose from_original_dav loads this
    exact checkpoint):

        encoder.block.0                Conv1d(1 -> 64, k7, p3)
        encoder.block.{1..4}           EncoderBlock, strides (2,4,8,8)
            .block.{0,1,2}                 ResidualUnit, dilations (1,3,9)
                .block.0.alpha                 snake1
                .block.1                       Conv1d(c,c,k7,dil,p=3*dil)
                .block.2.alpha                 snake2
                .block.3                       Conv1d(c,c,k1)
            .block.3.alpha                 snake
            .block.4                       Conv1d(c -> 2c, k=2s, stride=s, p=ceil(s/2))
        encoder.block.5.alpha          snake_out
        encoder.block.6                Conv1d(1024 -> 1024, k3, p1)
        mean_proj                      Conv1d(1024 -> 64, k1)   [NO weight-norm]

    Two traps worth stating:
      - The checkpoint names the two halves INCONSISTENTLY: `encoder.block.*`
        but `decoder.model.*`. build_vocoder above reads the latter.
      - `logs_proj` is deliberately NOT converted. encode() returns the
        posterior MEAN, so a training target carries no sampling noise. Loading
        logs_proj would imply a reparameterised sample that the reference does
        not take.
    """
    b.meta("add_uint32", "mm3.enc.in_dim", ENC_IN_DIM)
    b.meta("add_array", "mm3.enc.rates", list(ENC_RATES))
    b.meta("add_array", "mm3.enc.res_dilations", list(ENC_RES_DILATIONS))
    b.meta("add_uint32", "mm3.enc.total_downsample", int(np.prod(ENC_RATES)))
    b.meta("add_uint32", "mm3.enc.latent_dim", ENC_LATENT_DIM)
    b.meta("add_uint32", "mm3.enc.fold_channels", ENC_FOLD_CH)
    b.meta("add_uint32", "mm3.enc.latent_channels", ENC_FOLD_CH * 2)
    b.meta("add_uint32", "mm3.enc.sampling_rate", ENC_SR)
    b.meta("add_uint32", "mm3.enc.channels", 2)
    b.meta("add_float32", "mm3.enc.snake_eps", ENC_SNAKE_EPS)
    b.meta("add_bool", "mm3.enc.weight_norm_folded", True)
    b.meta("add_bool", "mm3.enc.posterior_mean_only", True)
    b.meta("add_string", "mm3.enc.snake", "x + 1/(alpha+eps) * sin(alpha*x)^2")

    def wn(dst, key, expect_v, expect_bias):
        g = src.get(f"{key}.weight_g", expect=(expect_v[0], 1, 1))
        v = src.get(f"{key}.weight_v", expect=expect_v)
        b.put(f"{dst}.weight", fold_weight_norm(g, v, dst), F32)
        b.put(f"{dst}.bias", src.get(f"{key}.bias", expect=(expect_bias,)), F32)

    def alpha(dst, key, ch):
        b.put(dst, src.get(key, expect=(1, ch, 1)), F32)

    wn("enc.conv_in", "encoder.block.0", (ENC_IN_DIM, 1, 7), ENC_IN_DIM)

    ch = ENC_IN_DIM
    for bi, stride in enumerate(ENC_RATES):
        out_ch = ch * 2
        m      = bi + 1
        for ri, dil in enumerate(ENC_RES_DILATIONS):
            u = f"encoder.block.{m}.block.{ri}"
            alpha(f"enc.blk.{bi}.res.{ri}.snake1.alpha", f"{u}.block.0.alpha", ch)
            wn(f"enc.blk.{bi}.res.{ri}.conv1", f"{u}.block.1", (ch, ch, 7), ch)
            alpha(f"enc.blk.{bi}.res.{ri}.snake2.alpha", f"{u}.block.2.alpha", ch)
            wn(f"enc.blk.{bi}.res.{ri}.conv2", f"{u}.block.3", (ch, ch, 1), ch)
        alpha(f"enc.blk.{bi}.snake.alpha", f"encoder.block.{m}.block.3.alpha", ch)
        wn(f"enc.blk.{bi}.conv", f"encoder.block.{m}.block.4",
           (out_ch, ch, 2 * stride), out_ch)
        ch = out_ch

    n_blk = len(ENC_RATES)
    alpha("enc.snake_out.alpha", f"encoder.block.{n_blk + 1}.alpha", ch)
    wn("enc.conv_out", f"encoder.block.{n_blk + 2}", (ENC_LATENT_DIM, ch, 3), ENC_LATENT_DIM)

    # mean_proj is a bare Conv1d — no weight_g/weight_v to fold.
    b.put("enc.mean_proj.weight",
          src.get("mean_proj.weight", expect=(ENC_FOLD_CH, ENC_LATENT_DIM, 1)), F32)
    b.put("enc.mean_proj.bias",
          src.get("mean_proj.bias", expect=(ENC_FOLD_CH,)), F32)

    def belongs(n):
        # logs_proj and the decoder/flow halves of dav.pth are intentionally
        # unconsumed, so they must not be claimed here or the leftover report
        # will flag them.
        return n.startswith("encoder.") or n.startswith("mean_proj.")
    return belongs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_COMPONENTS = ("lm", "depth", "cond", "dit", "vocoder", "enc")
SYNTH_COMPONENTS = ("depth", "cond", "dit", "vocoder")

EPILOG = """\
components:
  lm       global LM (Qwen3-8B + extended audio vocab)   -> mm3-lm-<quant>.gguf
  depth    RVQ depth decoder (0.6B)          \\
  cond     condition encoder (~25M)           }-> mm3-synth-<quant>.gguf
  dit      flow-matching transformer (2.4B)   |
  vocoder  DAC-style vocoder decoder (54M)   /

quantisation:
  --quant f16   (default) 2-D weights F16, norms/biases/alphas/scales F32.
  --quant q8_0  additionally Q8_0 for 2-D weights whose innermost dim is a
                multiple of 32. The vocoder is ALWAYS F32 (54M params; the
                Snake + folded weight-norm path is quality-critical) and the
                DiT timestep Fourier basis is always F32.

  You do not have to quantise here. Emitting f16 and running the existing
  `quantize` tool over mm3-lm-f16.gguf is the better-trodden path for the LM,
  and it gives access to the k-quants this script does not implement.

sources (--src may be repeated; directories are scanned recursively):
  preferred  Comfy-Org/MiniMax-Music-3     (single-file safetensors)
  fallback   MiniMaxAI/MiniMax-Music3      (diffusers subfolder layout)
  refused    *_int8_convrot.safetensors, *_pruned_*.safetensors

Tensor-name mapping and every mm3.* key: docs/plans/mm3-gguf-layout.md
"""


def main():
    ap = argparse.ArgumentParser(
        prog="convert-mm3.py",
        description="MiniMax-Music3 safetensors -> GGUF for the HOT-Step engine.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, metavar="PATH",
                    help="model dir or .safetensors/.pth file (repeatable)")
    ap.add_argument("--out", required=True, metavar="DIR", help="output directory")
    ap.add_argument("--components", default=",".join(ALL_COMPONENTS),
                    help=f"comma list of {','.join(ALL_COMPONENTS)} (default: all)")
    ap.add_argument("--quant", default="f16", choices=("f16", "q8_0"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files (default: skip)")
    ap.add_argument("--tokenizer", metavar="PATH",
                    help="tokenizer.json (default: found in --src, or read from "
                         "the source's embedded tokenizer_json tensor)")
    ap.add_argument("--embed-tokenizer-json", action="store_true",
                    help="also store tokenizer.json verbatim as mm3.tokenizer_json "
                         "(+11 MB)")
    ap.add_argument("--lm-layers", type=int, default=LM_LAYERS, metavar="N",
                    help=f"LM block count (default {LM_LAYERS}). Set this for a "
                         "depth-pruned student such as the 5.7B distilled "
                         "composer (--lm-layers 21). A wrong value is caught by "
                         "the leftover-tensor check, not silently emitted.")
    ap.add_argument("--ar-cfg-scale", type=float, default=AR_CFG_SCALE,
                    metavar="S",
                    help=f"mm3.ar.cfg_scale (default {AR_CFG_SCALE}). Pass 1.0 "
                         "for a CFG-free / guidance-distilled composer: the "
                         "engine then runs a SINGLE AR row instead of the "
                         "cond+uncond pair, halving the LM KV cache.")
    ap.add_argument("--suffix", default="", metavar="TAG",
                    help="appended to the quant token in the output filename, "
                         "e.g. --suffix -d21-lr6e5 -> mm3-lm-q8_0-d21-lr6e5.gguf. "
                         "The engine treats the whole token as the variant name, "
                         "so tagged files sit alongside the stock ones in the "
                         "model picker instead of overwriting them.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.lm_layers < 1:
        die("--lm-layers must be positive")
    if args.suffix and not all(ch.isalnum() or ch in "-_." for ch in args.suffix):
        die("--suffix may only contain alphanumerics, '-', '_' and '.'")

    want = [c.strip() for c in args.components.split(",") if c.strip()]
    bad = [c for c in want if c not in ALL_COMPONENTS]
    if bad:
        die(f"unknown component(s) {bad}; valid: {', '.join(ALL_COMPONENTS)}")
    if not want:
        die("--components selected nothing")

    os.makedirs(args.out, exist_ok=True)
    tag = args.quant + args.suffix
    lm_path = os.path.join(args.out, f"mm3-lm-{tag}.gguf")
    synth_path = os.path.join(args.out, f"mm3-synth-{tag}.gguf")
    enc_path = os.path.join(args.out, f"mm3-enc-{tag}.gguf")

    want_lm = "lm" in want
    want_synth = [c for c in SYNTH_COMPONENTS if c in want]
    want_enc = "enc" in want
    if want_lm and os.path.exists(lm_path) and not args.force:
        log(f"skip (exists): {lm_path} -- pass --force to overwrite")
        want_lm = False
    if want_synth and os.path.exists(synth_path) and not args.force:
        log(f"skip (exists): {synth_path} -- pass --force to overwrite")
        want_synth = []
    if want_enc and os.path.exists(enc_path) and not args.force:
        log(f"skip (exists): {enc_path} -- pass --force to overwrite")
        want_enc = False
    if not want_lm and not want_synth and not want_enc:
        log("nothing to do")
        return

    src = Source(args.src, verbose=args.verbose)
    try:
        predicates = []

        if want_lm:
            tok, raw = find_tokenizer(src, args.tokenizer, args.src)
            b = Bundle("qwen3")
            common_meta(b, src, args, ["lm"])
            predicates.append(("lm", build_lm(
                src, b, tok, raw if args.embed_tokenizer_json else None,
                n_layers=args.lm_layers, ar_cfg_scale=args.ar_cfg_scale)))
            b.write(lm_path, args.quant)

        if want_synth:
            if len(want_synth) != len(SYNTH_COMPONENTS):
                log(f"WARNING: partial synth file -- only {want_synth} included. "
                    f"The engine loader expects all of {list(SYNTH_COMPONENTS)}")
            b = Bundle("mm3")
            b.meta("add_name", "MiniMax-Music3 synth")
            b.meta("add_description",
                   "MiniMax-Music3 synthesis stack: RVQ depth decoder, condition "
                   "encoder, flow-matching transformer, DAC-style vocoder")
            common_meta(b, src, args, want_synth)
            b.meta("add_array", "mm3.synth.components", list(want_synth))
            builders = {"depth": build_depth, "cond": build_cond,
                        "dit": build_dit, "vocoder": build_vocoder}
            for c in want_synth:
                predicates.append((c, builders[c](src, b)))
            b.write(synth_path, args.quant)

        if want_enc:
            # Training-only, and its own file: ace-train loads it for preprocess,
            # ace-server never does.
            b = Bundle("mm3enc")
            b.meta("add_name", "MiniMax-Music3 DAV encoder")
            b.meta("add_description",
                   "MiniMax-Music3 DAV audio encoder: 44.1 kHz stereo -> 128-channel "
                   "flow latents at 86.13 Hz. Training only; source is dav.pth.")
            common_meta(b, src, args, ["enc"])
            predicates.append(("enc", build_enc(src, b)))
            b.write(enc_path, args.quant)

        # -- leftover report: source tensors that look like they belong to a
        #    converted component but were never consumed.
        leftover = []
        for comp, pred in predicates:
            for n in src.unconsumed(pred):
                leftover.append((comp, n))
        if leftover:
            log(f"ERROR: {len(leftover)} source tensors matched a converted "
                f"component but were not written:")
            for comp, n in leftover[:40]:
                f = src.index[n]
                log(f"    [{comp}] {n} {tuple(f.shape(n))} {f.dtype(n)}")
            if len(leftover) > 40:
                log(f"    ... and {len(leftover) - 40} more")
            die("the converter does not understand this checkpoint layout -- "
                "refusing to leave a partial model in place. Delete the output "
                "files and report the list above.")

        if args.verbose:
            others = src.unconsumed(lambda n: True)
            if others:
                log(f"note: {len(others)} source tensors outside the converted "
                    f"components were ignored, e.g. {others[:5]}")
        log("Done.")
    finally:
        src.close()


def common_meta(b, src, args, components):
    # The MiniMax-Music3 Community License requires notices to travel with
    # copies -- keep this on every emitted file, and ship the LICENSE text
    # alongside the GGUFs.
    b.meta("add_license", "MiniMax-Music3 Community License")
    b.meta("add_string", "mm3.model", "MiniMax-Music3")
    b.meta("add_uint32", "mm3.converter_version", CONVERTER_VERSION)
    b.meta("add_string", "mm3.source_layout", src.layout)
    b.meta("add_file_type",
           int(gguf.LlamaFileType.MOSTLY_Q8_0 if args.quant == "q8_0"
               else gguf.LlamaFileType.MOSTLY_F16))
    b.meta("add_quantization_version", gguf.GGML_QUANT_VERSION)


if __name__ == "__main__":
    main()

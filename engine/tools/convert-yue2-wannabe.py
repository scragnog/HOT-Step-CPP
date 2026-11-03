#!/usr/bin/env python3
# convert-yue2-wannabe.py: WannabeTokenizer (ManyOtherFunctions) -> one yue2-tok GGUF.
#
# WannabeTokenizer is a second real-audio -> YuE2 semantic-code tokenizer, by
# ManyOtherFunctions: https://huggingface.co/ManyOtherFunctions/Yue2-WannabeTokenizer
# One unified safetensors holds a fine-tuned MERT2 encoder ("encoder.*", same
# architecture and tensor names as MERT-v2-FullSong) and a cosine-VQ head
# ("head.*"). This script writes the same arch as convert-yue2-tok.py, so the
# engine's yue2-tok loader, MERT port and windowed head driver all apply, with
# what differs carried as KV:
#
#   * MERT blocks 0..13 only, CAUSAL self-attention (is_causal=True in SDPA).
#   * The whole track in ONE forward: no 30 s chunks (source_spec chunk=0), no
#     interpolation to T25 (the head reads the raw subsampled frames), no
#     instance norm (head.instnorm=0). The fp16 feature store stays ON: the
#     reference casts features to half before the head.
#   * The head is per-frame, no attention:
#         h = x + down(silu(gate(LN(x))) * up(LN(x)))       SwiGLU adapter, 1024 -> 8192
#         z = normalize(proj(h))                             1024 -> 256
#         logits = z @ normalize(codebook).T / tau           32768 codes
#     The codebook is written PRE-normalised and PRE-divided by tau, so the
#     engine's logits are the reference's logits and its log-softmax is the
#     reference's too (yue2-guide keeps top-K log-probs).
#   * A one-frame delay: token t comes from feature frame t - offset
#     (to_token_clock), frame 0 repeated.
#
# The reference runs encoder and head under CUDA bf16 autocast, so a port gate is
# an agreement rate, never bit-exactness (same posture as convert-yue2-tok.py).
#
# DEPENDENCIES: numpy + gguf. NO TORCH. Reuses convert-yue2-tok.py's readers,
# Bundle and MERT tensor mapping.
#
# USAGE
#   python engine/tools/convert-yue2-wannabe.py --src WannabeTokenizerV2.safetensors --name v2 \
#       --out models/yue2 --outtype f16
#   -> models/yue2/yue2-wannabe-v2-f16.gguf

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("convert_yue2_tok", os.path.join(_here, "convert-yue2-tok.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)
log, die = ct.log, ct.die

CONVERTER_VERSION = 1
UPSTREAM_REPO = "ManyOtherFunctions/Yue2-WannabeTokenizer"
# The upstream repo states no licence. Until its author settles one, the file says
# exactly that instead of guessing; set this before publishing.
LICENSE_NAME = "unspecified upstream (see " + "https://huggingface.co/" + UPSTREAM_REPO + ")"
ATTRIBUTION = ("WannabeTokenizer by ManyOtherFunctions (" + "https://huggingface.co/" + UPSTREAM_REPO +
               "). MERT2 encoder fine-tuned end to end with a cosine-VQ head onto YuE2's 32768-code "
               "semantic codebook. Converted to GGUF for HOT-Step; weights unchanged apart from "
               "f16 storage of the 2-D matmul weights and a pre-normalised codebook.")

CVQ_DIM = 256
CVQ_HIDDEN = 8192
HEAD_WIN, HEAD_STRIDE, HEAD_TRIM = 512, 256, 128   # the reference's own _predict() windows


class Prefixed(ct.Source):
    """One safetensors file seen through a name prefix ("encoder." / "head.")."""

    def __init__(self, path, prefix):
        super().__init__(path)
        self.prefix = prefix

    def get(self, name, expect=None):
        return super().get(self.prefix + name, expect)

    def unconsumed(self, predicate=lambda n: True):
        p = self.prefix
        return sorted(n[len(p):] for n in self.file.header
                      if n.startswith(p) and n not in self.consumed and predicate(n[len(p):]))


def read_meta(path):
    f = ct.SafeTensorsFile(path)
    meta = dict(f.metadata)
    f.close()
    if meta.get("format") != "yue2sem-tokenizer-1":
        die(f"{path}: format is {meta.get('format')!r}, expected 'yue2sem-tokenizer-1'")
    if meta.get("head.kind") != "cvq":
        die(f"{path}: head.kind is {meta.get('head.kind')!r}, expected 'cvq'")
    spec = json.loads(meta["head.source_spec"])
    cfg = json.loads(meta["encoder.config"])
    if spec.get("causal") is not True or float(spec.get("chunk", 0)) != 0:
        die(f"{path}: source_spec {spec} is not causal whole-track; this converter only writes that layout")
    if meta.get("head.instnorm") != "0":
        die(f"{path}: head.instnorm={meta.get('head.instnorm')!r}; only 0 is supported")
    if int(meta.get("head.dim", CVQ_DIM)) != CVQ_DIM or int(meta.get("head.hidden", CVQ_HIDDEN)) != CVQ_HIDDEN:
        die(f"{path}: head dim/hidden {meta.get('head.dim')}/{meta.get('head.hidden')} != {CVQ_DIM}/{CVQ_HIDDEN}")
    return meta, spec, cfg


def build_head(src, b, tau):
    D = ct.MERT_DIM
    b.put("cvq.adapter.norm.weight", src.get("adapter.norm.weight", expect=(D,)), ct.F32)
    b.put("cvq.adapter.norm.bias", src.get("adapter.norm.bias", expect=(D,)), ct.F32)
    b.put("cvq.adapter.gate.weight", src.get("adapter.gate.weight", expect=(CVQ_HIDDEN, D)), ct.NATIVE)
    b.put("cvq.adapter.up.weight", src.get("adapter.up.weight", expect=(CVQ_HIDDEN, D)), ct.NATIVE)
    b.put("cvq.adapter.down.weight", src.get("adapter.down.weight", expect=(D, CVQ_HIDDEN)), ct.NATIVE)
    b.put("cvq.proj.weight", src.get("proj.weight", expect=(CVQ_DIM, D)), ct.F32)
    b.put("cvq.proj.bias", src.get("proj.bias", expect=(CVQ_DIM,)), ct.F32)
    # F.normalize(codebook, dim=-1) / tau, done once here. Kept F32: it is the
    # argmax's right-hand side and only 32 MB.
    cb = src.get("codebook", expect=(ct.HEAD_VOCAB, CVQ_DIM)).astype(np.float64)
    cb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12) / tau
    b.put("cvq.codebook", cb.astype(np.float32), ct.F32)
    left = src.unconsumed()
    if left:
        die(f"head tensors not consumed: {left}")


def add_meta(b, outtype, name, meta, spec, sha):
    # Start from the MERT tokenizer's KV block, then drop what describes its
    # transformer head, chunking, instance norm and provenance, and say what is
    # true here instead.
    ct.add_meta(b, outtype, False, sha, {})
    drop_fns = {"add_name", "add_description", "add_license"}
    drop_keys = {"yue2tok.license_attribution", "yue2tok.converter_version", "yue2tok.mert.attention.causal",
                 "yue2tok.mert.hidden_state_note", "yue2tok.mert.rope.note", "yue2tok.chunk_seconds",
                 "yue2tok.chunk_samples", "yue2tok.min_chunk_samples", "yue2tok.tail_runs_alone",
                 "yue2tok.interpolate.mode", "yue2tok.interpolate.target", "yue2tok.instance_norm.epsilon",
                 "yue2tok.instance_norm.formula"}
    b.kv = [(fn, a) for fn, a in b.kv
            if fn not in drop_fns and not (a and isinstance(a[0], str) and
                                           (a[0] in drop_keys or a[0].startswith(("yue2tok.head.", "yue2tok.pinned.",
                                                                                  "yue2tok.gate."))))]
    L = ct.MERT_HIDDEN_STATE_INDEX
    b.meta("add_name", f"YuE2 WannabeTokenizer {name}")
    b.meta("add_description",
           f"Real-audio -> YuE2 semantic codes. WannabeTokenizer {name} by ManyOtherFunctions: causal MERT2 "
           f"blocks 0..{L} over the whole track, then a SwiGLU + cosine-VQ head to a 32768-way codebook. "
           "YuE2 token id = argmax + 151853.")
    b.meta("add_license", LICENSE_NAME)
    b.meta("add_string", "yue2tok.license_attribution", ATTRIBUTION)
    b.meta("add_uint32", "yue2tok.converter_version", CONVERTER_VERSION)
    b.meta("add_bool", "yue2tok.mert.attention.causal", True)
    b.meta("add_string", "yue2tok.mert.hidden_state_note",
           f"the OUTPUT of layers[{L}], post its final_layer_norm; the checkpoint carries blocks 0..{L} only")
    b.meta("add_string", "yue2tok.mert.rope.note",
           "BUILD inv_freq[i] = 1/10000^(2i/64); angles = cat(freqs, freqs); positions 0..T-1 over the WHOLE track")
    b.meta("add_uint32", "yue2tok.chunk_seconds", 0)
    b.meta("add_string", "yue2tok.interpolate.mode", "none")
    b.meta("add_bool", "yue2tok.instance_norm.enabled", False)
    b.meta("add_bool", "yue2tok.feature_store_f16.apply", True)   # features.half() before the head
    # Its codes render WITHOUT nar_lora_joint_v9 (the v9 head's companion): the
    # author's own advice, and a blind ear test agreed. yue2-tokenize copies
    # this into the cache manifest; trainers and the renderer follow it.
    b.meta("add_bool", "yue2tok.companion", False)

    b.meta("add_string", "yue2tok.head.kind", "cvq")
    b.meta("add_uint32", "yue2tok.head.window", HEAD_WIN)
    b.meta("add_uint32", "yue2tok.head.window_stride", HEAD_STRIDE)
    b.meta("add_uint32", "yue2tok.head.window_trim", HEAD_TRIM)
    b.meta("add_uint32", "yue2tok.head.input_length", ct.MERT_DIM)
    b.meta("add_uint32", "yue2tok.head.vocab_size", ct.HEAD_VOCAB)
    b.meta("add_float32", "yue2tok.head.layer_norm_epsilon", 1e-5)
    b.meta("add_uint32", "yue2tok.cvq.dim", CVQ_DIM)
    b.meta("add_uint32", "yue2tok.cvq.hidden", CVQ_HIDDEN)
    b.meta("add_float32", "yue2tok.cvq.tau", float(meta.get("head.tau", 0.07)))
    b.meta("add_uint32", "yue2tok.cvq.offset", int(meta["head.offset"]))
    b.meta("add_string", "yue2tok.head.forward",
           "h = x + down(silu(gate(LN(x))) * up(LN(x))); z = l2norm(proj(h)); logits = cvq.codebook @ z "
           "(codebook stored l2-normalised and divided by tau); code[t] = argmax[max(t - offset, 0)]")

    b.meta("add_string", "yue2tok.pinned.source_repo", UPSTREAM_REPO)
    b.meta("add_string", "yue2tok.pinned.source_sha256", sha)
    b.meta("add_string", "yue2tok.pinned.source_head", meta.get("head.source", ""))
    b.meta("add_string", "yue2tok.pinned.source_step", meta.get("step", ""))
    b.meta("add_string", "yue2tok.pinned.source_spec", json.dumps(spec, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(prog="convert-yue2-wannabe.py",
                                 description="WannabeTokenizer unified safetensors -> yue2-tok GGUF")
    ap.add_argument("--src", required=True, metavar="FILE", help="WannabeTokenizer*.safetensors")
    ap.add_argument("--name", required=True, metavar="TAG", help="version tag in the file name, e.g. v1 or v2")
    ap.add_argument("--out", default="models/yue2", metavar="DIR")
    ap.add_argument("--outtype", "--type", dest="outtype", default="f16", choices=("f16", "f32"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta, spec, cfg = read_meta(args.src)
    layer = int(spec["layer"])
    if int(cfg.get("num_hidden_layers", 0)) != layer + 1:
        die(f"encoder has {cfg.get('num_hidden_layers')} blocks but the head reads block {layer}; "
            "this converter expects the checkpoint to stop at the head's block")
    ct.MERT_BLOCKS_FULL = layer + 1
    ct.MERT_HIDDEN_STATE_INDEX = layer
    ct.MERT_BLOCKS_KEPT = layer + 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"yue2-wannabe-{args.name}-{args.outtype}.gguf")
    if os.path.exists(out_path) and not args.force:
        die(f"{out_path} already exists -- pass --force to overwrite")
    log(f"hashing {args.src} ...")
    sha = ct.sha256_file(args.src)

    b = ct.Bundle("yue2-tok")
    ct.build_mert_tensors(cfg, Prefixed(args.src, "encoder."), b)
    n_mert = len(b.tensors)
    head = Prefixed(args.src, "head.")
    build_head(head, b, float(meta.get("head.tau", 0.07)))
    head.close()
    log(f"assembled {len(b.tensors)} tensors: {n_mert} mert.*, {len(b.tensors) - n_mert} cvq.*")
    add_meta(b, args.outtype, args.name, meta, spec, sha)
    b.write(out_path, args.outtype)
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())

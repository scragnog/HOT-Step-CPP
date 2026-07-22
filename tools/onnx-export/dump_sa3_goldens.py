#!/usr/bin/env python3
"""Dump golden input/output tensor pairs for the StableStep GGML port.

Runs the validated ONNX graphs (the numerical reference — cosine 0.99995 vs
PyTorch) on fixed-seed inputs and writes raw little-endian f32/i64 .bin files
plus a manifest.json describing shapes. The C++ GGML modules replay these in
unit tests: load input.bin -> forward -> compare against output.bin
(target cosine > 0.999 for BF16 weights).

Components dumped:
  text_enc:  input_ids [1,256] i64, attention_mask [1,256] u8 -> embeddings [1,256,768]
  seconds:   seconds [1] f32 -> embed [1,768]
  same_enc:  audio [1,2,524288] f32 -> latents [1,256,128]
  same_dec:  latents [1,256,128] f32 -> audio [1,2,524288]
  dit:       x [1,256,64], t [1], cross [1,257,768], glob [1,768],
             local [1,257,64], pad [1,64] -> v [1,256,64]   (small T=64 for speed)

Runs in the StableAudio3 uv venv:
  uv run --with onnx --with onnxruntime python dump_sa3_goldens.py \
      --onnx-dir <dir> --out-dir <dir>
"""

import argparse
import json
import os

import numpy as np


def save(out_dir, name, arr):
    path = os.path.join(out_dir, name + ".bin")
    np.ascontiguousarray(arr).tofile(path)
    return {"file": name + ".bin", "shape": list(arr.shape), "dtype": str(arr.dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import onnxruntime as ort
    load = lambda n: ort.InferenceSession(
        os.path.join(args.onnx_dir, n), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    manifest = {}

    # ── text encoder ────────────────────────────────────────────────────
    # Realistic ids: the validation prompt's 26 tokens + pad, from tokens_csv
    # if present, else synthetic small ids.
    ids = np.zeros((1, 256), dtype=np.int64)
    n_tok = 26
    tok_csv = os.path.join(os.path.dirname(args.out_dir), "tokens_csv.txt")
    if os.path.exists(tok_csv):
        vals = [int(x) for x in open(tok_csv).read().strip().split(",")]
        ids[0, :len(vals)] = vals[:256]
        n_tok = sum(1 for v in vals if v != 0) or 26
    else:
        ids[0, :n_tok] = rng.integers(3, 50000, n_tok)
    mask = np.zeros((1, 256), dtype=np.bool_)
    mask[0, :n_tok] = True
    s = load("sa3-text_encoder.onnx")
    emb = s.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    manifest["text_enc"] = {
        "inputs": {"input_ids": save(args.out_dir, "text_enc.input_ids", ids),
                   "attention_mask": save(args.out_dir, "text_enc.attention_mask",
                                          mask.astype(np.uint8))},
        "outputs": {"embeddings": save(args.out_dir, "text_enc.embeddings", emb)},
        "n_tokens": n_tok,
    }
    del s

    # ── seconds embedder ────────────────────────────────────────────────
    sec = np.array([203.8], dtype=np.float32)
    s = load("sa3-seconds_embedder.onnx")
    sec_emb = s.run(None, {"seconds": sec})[0]
    manifest["seconds"] = {
        "inputs": {"seconds": save(args.out_dir, "seconds.in", sec)},
        "outputs": {"embed": save(args.out_dir, "seconds.embed", sec_emb)},
    }
    del s

    # ── SAME encoder / decoder (one static chunk each) ──────────────────
    audio = (rng.standard_normal((1, 2, 524288)) * 0.1).astype(np.float32)
    s = load("sa3-same_encoder.onnx")
    latents = s.run(None, {"audio": audio})[0]
    manifest["same_enc"] = {
        "inputs": {"audio": save(args.out_dir, "same_enc.audio", audio)},
        "outputs": {"latents": save(args.out_dir, "same_enc.latents", latents)},
    }
    del s

    s = load("sa3-same_decoder.onnx")
    dec_audio = s.run(None, {"latents": latents})[0]
    manifest["same_dec"] = {
        "inputs": {"latents": save(args.out_dir, "same_dec.latents", latents)},
        "outputs": {"audio": save(args.out_dir, "same_dec.audio", dec_audio)},
    }
    del s

    # ── DiT single forward at small T ───────────────────────────────────
    T = 64
    x = rng.standard_normal((1, 256, T)).astype(np.float32)
    t = np.array([0.3], dtype=np.float32)
    cross = np.concatenate([emb, sec_emb[:, None, :]], axis=1).astype(np.float32)
    glob = sec_emb.astype(np.float32)
    local = np.zeros((1, 257, T), dtype=np.float32)
    pad = np.ones((1, T), dtype=np.bool_)
    pad[0, 48:] = False  # exercise the padding-mask path
    s = load("sa3-dit.onnx")
    v = s.run(None, {"x": x, "t": t, "cross_attn_cond": cross,
                     "global_embed": glob, "local_add_cond": local,
                     "padding_mask": pad})[0]
    manifest["dit"] = {
        "inputs": {"x": save(args.out_dir, "dit.x", x),
                   "t": save(args.out_dir, "dit.t", t),
                   "cross_attn_cond": save(args.out_dir, "dit.cross", cross),
                   "global_embed": save(args.out_dir, "dit.glob", glob),
                   "local_add_cond": save(args.out_dir, "dit.local", local),
                   "padding_mask": save(args.out_dir, "dit.pad", pad.astype(np.uint8))},
        "outputs": {"v": save(args.out_dir, "dit.v", v)},
    }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Goldens written to {args.out_dir}")


if __name__ == "__main__":
    main()

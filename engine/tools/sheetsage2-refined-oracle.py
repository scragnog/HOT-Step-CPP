#!/usr/bin/env python3
# sheetsage2-refined-oracle.py: reference dumps for the SheetSage2-Refined port.
#
# Runs the MIT reference package (github.com/anime-song/SheetSage2-Refined,
# put it on PYTHONPATH) in fp32, no autocast, on one clip encoded at its own
# length (the refined decoder's pad_encoder=False), and writes raw
# little-endian f32 files plus manifest.json:
#
#   wave.f32                      mono 24 kHz clip
#   memory_base.f32  [T,512]      layer mix + projection, before bar features
#   memory.f32       [T,512]      memory + BarFeatures (what the decoder sees)
#   bar_facts.f32    [T,43]       mean of the two bar heads' facts (chunked GRU)
#   mix_<head>_<j>.f32 [T,1024]   head <head>'s j-th layer mix, before its LayerNorm
#   out_<head>_<name>.f32         head outputs, unchunked GRU (what rescoring reads)
#   outc_<head>_<name>.f32        bar heads only, chunked GRU (what BarFeatures reads)
#   table_<term>.f32              transcribe.frame_table() per rescoring term
#
# USAGE
#   PYTHONPATH=<ref checkout> python engine/tools/sheetsage2-refined-oracle.py \
#       --model <snapshot dir> --audio song.wav --start 30 --seconds 120 --out <dir>

import argparse
import json
import os

import numpy as np
import torch

from sheetsage2_refined.heads import CHUNK, BarFeatures, HeadMean, load_head
from sheetsage2_refined.model import SheetSage
from sheetsage2_refined.transcribe import frame_table, load_waveform

HEADS = {"bar0": "heads/bar-v5-f0.safetensors", "bar1": "heads/bar-v5-f1.safetensors",
         "chord": "heads/chord-v1.safetensors", "detail": "heads/chord-detail-c3.safetensors",
         "multi": "heads/multi-v2.safetensors"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    manifest = {}

    def dump(name, t):
        a = t.detach().float().cpu().numpy()
        a = a[0] if a.ndim == 3 and a.shape[0] == 1 else a
        a.astype("<f4").tofile(os.path.join(args.out, name + ".f32"))
        manifest[name] = list(a.shape)

    cfg = json.load(open(os.path.join(args.model, "config.json"), encoding="utf-8"))
    model = SheetSage.load(os.path.join(args.model, cfg["base_weights"]), args.device,
                           decoder=os.path.join(args.model, cfg["decoder"]))
    heads = {k: load_head(os.path.join(args.model, p), args.device) for k, p in HEADS.items()}
    bar = BarFeatures([heads["bar0"], heads["bar1"]]).to(args.device).eval()
    bar.load_projection(os.path.join(args.model, cfg["bar_features"]["projection"]))
    bar_mean = HeadMean([heads["bar0"], heads["bar1"]]).eval()

    wave = load_waveform(args.audio, model.sample_rate)
    a = round(args.start * model.sample_rate)
    wave = wave[a:a + round(args.seconds * model.sample_rate)].contiguous()
    dump("wave", wave)

    with torch.inference_mode():
        states = model.encoder(wave[None].to(args.device))
        memory = model.memory(states)
        dump("memory_base", memory)
        dump("bar_facts", bar.probabilities(states))
        dump("memory", bar(memory, states))
        for k, head in heads.items():
            w = torch.softmax(head.mix, -1).reshape(-1, len(states))
            for j, row in enumerate(w):
                dump(f"mix_{k}_{j}", sum(wt * s.float() for wt, s in zip(row, states)))
            for name, t in head(states).items():
                dump(f"out_{k}_{name}", t)
            if k.startswith("bar"):
                for name, t in head(states, CHUNK).items():
                    dump(f"outc_{k}_{name}", t)
        outs = {"bar": bar_mean(states), "chord": heads["chord"](states),
                "detail": heads["detail"](states), "multi": heads["multi"](states)}
        for term, (src, head_key) in {"downbeat": ("downbeat", "bar"), "meter": ("bar_beats", "bar"),
                                      "root": ("root", "chord"), "bass": ("bass", "chord"),
                                      "quality": ("detail", "detail"), "key": ("key", "multi"),
                                      "key_boundary": ("key_boundary", "multi")}.items():
            dump(f"table_{term}", frame_table(outs[head_key], term))

    manifest["frames"] = int(memory.shape[1])
    manifest["args"] = vars(args)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(manifest) - 2} arrays, T={manifest['frames']} to {args.out}")


if __name__ == "__main__":
    main()

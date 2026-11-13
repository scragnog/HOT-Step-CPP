#!/usr/bin/env python3
"""ChordNet (five folds) to one unquantised f32 GGUF, plus the chord templates.

ChordNet is Music X Lab's ISMIR 2019 large-vocabulary chord recogniser as
bundled by MuLaCover (MIT). Requires numpy, torch and gguf. No model code is
imported for the weights; the templates come from the upstream vocabulary code.

  python engine/tools/convert-chordnet.py --weights <dir with the five .sdict> \
      --inventory tools/lead-sheet-reference/inventory.json --out chordnet-F32.gguf
  python engine/tools/convert-chordnet.py --vocab-out chordnet-templates.tsv \
      --upstream <MuLaCover>/src/mulacover/_symbolic_transcription/chord

Every fold must match inventory.json exactly (names, shapes, f32, 500,772
parameters). Tensors are written as fold<i>.<name>. condition_linear is left
out: upstream keeps it only for strict loading and never calls it.

The checkpoints' terms are unconfirmed: keep the GGUF out of the repository.
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

FOLD_NAME = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{}.best.sdict"
PARAMS_PER_FOLD = 500772
UNUSED = ("condition_linear.",)


def convert(weights: Path, inventory: Path, out: Path) -> None:
    import gguf
    import numpy as np
    import torch

    inv = json.loads(inventory.read_text(encoding="utf-8"))["models"]
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    w = gguf.GGUFWriter(str(out), "chordnet")
    w.add_uint32("chordnet.folds", 5)
    w.add_string("chordnet.source", "Music X Lab ISMIR2019 large-vocabulary chord recognition (MIT code)")
    written = 0
    for fold in range(5):
        expect = inv[f"chord-{fold}"]
        state = torch.load(weights / FOLD_NAME.format(fold), map_location="cpu", weights_only=True)["net"]
        got = [(k, list(v.shape), str(v.dtype)) for k, v in state.items()]
        want = [(t["name"], t["shape"], t["dtype"]) for t in expect["tensors"]]
        if got != want:
            sys.exit(f"fold {fold}: tensor inventory differs from {inventory}")
        n = sum(v.numel() for v in state.values())
        if n != PARAMS_PER_FOLD or expect["parameters"] != PARAMS_PER_FOLD:
            sys.exit(f"fold {fold}: {n} parameters, expected {PARAMS_PER_FOLD}")
        for k, v in state.items():
            if k.startswith(UNUSED):
                continue
            a = v.detach().to(torch.float32).contiguous().numpy()
            if not np.isfinite(a).all():
                sys.exit(f"fold {fold}: {k} is not finite")
            w.add_tensor(f"fold{fold}.{k}", a)
            written += a.size
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: 5 folds, {written} values (condition_linear left out)")


def templates(upstream: Path, out: Path) -> None:
    # decoder.py imports .complex_chord, so load both under a stand-in package.
    pkg = types.ModuleType("chordpkg")
    pkg.__path__ = [str(upstream)]
    sys.modules["chordpkg"] = pkg
    for name in ("complex_chord", "decoder"):
        spec = importlib.util.spec_from_file_location(f"chordpkg.{name}", upstream / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    chords = sys.modules["chordpkg.decoder"].ChordDecoder().chords
    lines = [
        "# ChordNet chord templates: label, triad, bass, seventh, ninth, eleventh, thirteenth.",
        "# Expanded from chord_vocabulary.txt by Music X Lab's complex_chord.py (MIT); see LICENSE.",
    ]
    lines += ["\t".join([name] + [str(int(v)) for v in chord]) for chord, name in chords]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(chords)} templates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--inventory", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--vocab-out", type=Path)
    ap.add_argument("--upstream", type=Path)
    a = ap.parse_args()
    if a.vocab_out:
        if not a.upstream:
            ap.error("--vocab-out needs --upstream")
        templates(a.upstream, a.vocab_out)
    elif a.weights and a.inventory and a.out:
        convert(a.weights, a.inventory, a.out)
    else:
        ap.error("give --weights, --inventory and --out, or --vocab-out and --upstream")


if __name__ == "__main__":
    main()

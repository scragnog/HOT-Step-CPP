#!/usr/bin/env python3
"""YourMT3+ (YPTF.MoE+Multi, noPS) checkpoint to one unquantised f32 GGUF.

The checkpoint is the one MuLaCover loads for melody transcription. Requires
numpy, torch and gguf; no model code is imported.

  python engine/tools/convert-yourmt3.py --ckpt <last.ckpt> \
      --inventory tools/lead-sheet-reference/inventory.json --out yourmt3-F32.gguf

The state dictionary must match inventory.json exactly (655 names, shapes and
dtypes, 47,078,022 parameters). Every tensor is written under its torch name,
except:
  - *.num_batches_tracked (6): BatchNorm training counters, unused in eval.
  - the 15 per-layer *.rotary_emb.freqs: each is the same shared Parameter as
    encoder.rotary_emb_{sca,latent,temporal}.freqs (checked equal), so only
    those three are written.
The buffers spectrogram.stft.window and the BatchNorm running statistics are
written; they are state, not parameters.

Only the state dictionary is read. The checkpoint also holds training state
(optimizer states, schedulers, loops, callbacks, hyper-parameters, tokenizer
objects): none of it reaches the GGUF. Loading follows MuLaCover's restricted
loader: weights_only, with the pickled tokenizer classes mapped to an inert
holder and discarded.

The checkpoint's redistribution is not cleared: keep the GGUF out of the
repository and off any download entry.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PARAMETERS = 47078022
SHARED_FREQS = {"spectral_cross_attention": "sca", "local_transformer": "latent", "temporal_transformer": "temporal"}


class _Inert:
    """Holder for the pickled tokenizer and task objects; their state is dropped."""


def load_state(path: Path):
    import numpy as np
    import torch

    aliases = [
        (_Inert, name)
        for name in (
            "utils.task_manager.TaskManager",
            "utils.tokenizer.NoteEventTokenizer",
            "utils.note_event_dataclasses.EventRange",
            "utils.event_codec.FastCodec",
            "utils.note_event_dataclasses.Event",
        )
    ]
    core = np._core if hasattr(np, "_core") else np.core
    aliases += [
        core.multiarray._reconstruct,
        core.multiarray.scalar,
        (core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.ndarray,
        np.dtype,
        type(np.dtype("int64")),
        type(np.dtype("float64")),
        type(np.dtype("int32")),
        type(np.dtype("float32")),
    ]
    with torch.serialization.safe_globals(aliases):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict) or not state:
        sys.exit(f"{path}: no state_dict")
    # MuLaCover's loader discards pitch-shift augmentation entries; the
    # inventory check below fails if any are present instead.
    return state


def shared_freq_name(key: str):
    m = re.match(r"encoder\.blocks\.\d+\.(\w+?)(\.\d+)?\.(?:attention\.)?self\.rotary_emb\.freqs$", key)
    return f"encoder.rotary_emb_{SHARED_FREQS[m.group(1)]}.freqs" if m else None


def convert(ckpt: Path, inventory: Path, out: Path) -> None:
    import gguf
    import numpy as np
    import torch

    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    expect = json.loads(inventory.read_text(encoding="utf-8"))["models"]["yourmt3"]
    state = load_state(ckpt)
    got = [(k, list(v.shape), str(v.dtype)) for k, v in state.items()]
    want = [(t["name"], t["shape"], t["dtype"]) for t in expect["tensors"]]
    if got != want:
        extra = sorted({g[0] for g in got} - {w[0] for w in want})
        missing = sorted({w[0] for w in want} - {g[0] for g in got})
        sys.exit(f"state dict differs from {inventory}: extra {extra[:5]}, missing {missing[:5]}, "
                 f"or shapes/dtypes/order differ")

    dropped, aliased, written, params = [], [], 0, 0
    w = gguf.GGUFWriter(str(out), "yourmt3")
    w.add_string("yourmt3.source", "YourMT3+ YPTF.MoE+Multi (noPS) checkpoint as used by MuLaCover (Apache-2.0 code)")
    w.add_uint32("yourmt3.parameters", PARAMETERS)
    for k, v in state.items():
        if k.endswith("num_batches_tracked"):
            dropped.append(k)
            continue
        alias = shared_freq_name(k)
        if alias:
            if not torch.equal(v, state[alias]):
                sys.exit(f"{k} differs from {alias}; the shared-frequency assumption is wrong")
            aliased.append(k)
            continue
        a = v.detach().to(torch.float32).contiguous().numpy()
        if not np.isfinite(a).all():
            sys.exit(f"{k} is not finite")
        w.add_tensor(k, a)
        written += a.size
        buffer = k == "spectrogram.stft.window" or ".running_" in k
        params += 0 if buffer else a.size
    if len(dropped) != 6 or len(aliased) != 15:
        sys.exit(f"expected 6 counters and 15 frequency aliases, found {len(dropped)} and {len(aliased)}")
    if params != PARAMETERS or expect["parameters"] != PARAMETERS:
        sys.exit(f"{params} parameters written, expected {PARAMETERS}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: {len(state) - len(dropped) - len(aliased)} tensors, {written} values, "
          f"{params} parameters; left out {len(dropped)} num_batches_tracked and {len(aliased)} shared rotary aliases")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    convert(a.ckpt, a.inventory, a.out)


if __name__ == "__main__":
    main()

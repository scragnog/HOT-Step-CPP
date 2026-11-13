"""Convert the YourMT3+ checkpoint tensors the GGML spike needs into an f32 GGUF.

Spike tool for HOTSTEP-1178. Unquantised; writes the Perceiver-TF encoder,
the multi-channel projection, the token embedding, the T5 decoder and the LM
head. Front-end (spectrogram, conv pre-encoder) tensors are left out because
the spike feeds the encoder from the captured pre-encoder output.

The checkpoint and the GGUF are not redistributable: keep both outside git.

Usage: python convert.py <last.ckpt> <out.gguf>
Needs torch, numpy and gguf.
"""

import pickle
import sys

import numpy as np
import torch
import gguf

KEEP = ("encoder.", "pre_decoder.", "embed_tokens.", "decoder.", "lm_head.")


class _Inert:
    """Stand-in for the tokenizer/task objects pickled beside the weights."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    # Only torch's own storage/rebuild helpers and numpy scalar plumbing are
    # resolved; every other global becomes an inert object and is discarded.
    def find_class(self, module, name):
        if module.startswith("torch") or module.startswith("collections"):
            return super().find_class(module, name)
        if module.startswith("numpy") or module in ("builtins", "_codecs"):
            return super().find_class(module, name)
        return _Inert


class _PickleModule:
    Unpickler = _Unpickler
    load = pickle.load
    __name__ = "spike_pickle"


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False, pickle_module=_PickleModule)
    state = ckpt.get("state_dict", ckpt)
    w = gguf.GGUFWriter(dst, "yourmt3-spike")
    n = 0
    params = 0
    for key, t in state.items():
        if not isinstance(t, torch.Tensor) or not key.startswith(KEEP):
            continue
        if key.endswith("num_batches_tracked"):
            continue
        a = t.detach().to(torch.float32).contiguous().numpy()
        w.add_tensor(key, np.ascontiguousarray(a))
        n += 1
        params += a.size
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {n} tensors, {params} values, to {dst}")


if __name__ == "__main__":
    main()

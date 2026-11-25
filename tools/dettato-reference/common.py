"""Shared helpers for the dettato reference scripts: upstream import, tensor dumps, capture reader."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

UPSTREAM_COMMIT = '51ade0f09f2272f4674166026582185d26baaeaf'   # kobimusic/dettato, version 2.0.0
STAGES = ('transcriber', 'evidence', 'grouper', 'drums')
REPOS = {'transcriber': 'dettato', 'evidence': 'dettato-evidence', 'grouper': 'dettato-grouper', 'drums': 'dettato-drums'}


def load_upstream(upstream, weights, device):
    """Import dettato from the pinned clone and point it at local weights. cpu = the reference run (fp32, deterministic);
    cuda = upstream as it ships (bfloat16 autocast), used only to measure the CPU/CUDA spread."""
    upstream = Path(upstream).resolve()
    os.environ['DETTATO_WEIGHTS'] = str(Path(weights).resolve())
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(upstream / 'src'))
    import torch
    if device == 'cpu':
        if torch.version.cuda is not None:
            raise RuntimeError('The CPU reference needs a CPU-only PyTorch wheel (venv-cpu)')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    elif not torch.cuda.is_available():
        raise RuntimeError('The CUDA capture needs the venv-cuda environment and a visible GPU')
    torch.manual_seed(0)
    return torch


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


_NATIVE = {'float16': '<f2', 'int8': '|i1', 'int16': '<i2', 'int32': '<i4', 'int64': '<i8', 'uint8': '|u1', 'bool': '|u1'}


class Dumps:
    """Raw little-endian dumps. float16 and int8 keep their own dtype (upstream's rounding is part of the function);
    everything else floating is stored as float32, integers as their own width. tensors.json holds shape, dtype, size, sha256."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}

    def put(self, name, value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind in 'fc' and not np.isfinite(array).all():
            raise ValueError(f'Nonfinite capture: {name}')
        src = str(array.dtype)
        code = _NATIVE.get(src, '<f4')
        array = np.ascontiguousarray(array, dtype=code)
        path = self.root / (name + '.bin')
        array.tofile(path)
        self.entries[name] = dict(file=path.name, shape=list(array.shape), dtype=code, source_dtype=src,
                                  bytes=path.stat().st_size, sha256=digest(path))
        write_json(self.root / 'tensors.json', self.entries)


def load_tensors(root):
    """-> {name: ndarray} read lazily (memory-mapped) from a capture's tensors.json."""
    root = Path(root)
    entries = json.loads((root / 'tensors.json').read_text(encoding='utf-8'))
    return {k: np.memmap(root / e['file'], dtype=e['dtype'], mode='r', shape=tuple(e['shape'])) if e['bytes'] else
            np.zeros(e['shape'], e['dtype']) for k, e in entries.items()}


def load_capture(root):
    root = Path(root)
    return load_tensors(root), json.loads((root / 'result.json').read_text(encoding='utf-8'))

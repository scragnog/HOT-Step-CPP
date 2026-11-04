"""CPU-only imports and portable tensor receipts for the reference scripts."""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'


def load_upstream(root):
    root = Path(root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError('Run downloaded code from its own upstream directory')
    # Avoid the generator-only __init__ imports (torchtune/torchao/codec).
    # All transcriber implementation files remain unmodified.
    package = types.ModuleType('mulacover')
    package.__path__ = [str(root / 'src' / 'mulacover')]
    sys.modules['mulacover'] = package
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError('This oracle requires a CPU-only PyTorch wheel')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return torch


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Dumps:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}

    def put(self, name, value):
        import numpy as np
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if np.iscomplexobj(array):
            self.put(name + '.real', array.real)
            self.put(name + '.imag', array.imag)
            return
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite capture: {name}')
        original_dtype = str(array.dtype)
        array = np.ascontiguousarray(array, dtype='<f4')
        path = self.root / (name + '.f32')
        array.tofile(path)
        self.entries[name] = dict(file=path.name, shape=list(array.shape),
                                  dtype='float32', source_dtype=original_dtype,
                                  byte_order='little', bytes=path.stat().st_size,
                                  sha256=digest(path))
        write_json(self.root / 'tensors.json', self.entries)

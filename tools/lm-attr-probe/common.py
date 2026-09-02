"""Shared helpers for the LM attribute-probe study (docs/plans/2026-09-02-lm-flash-attn.md, Stream A).

Everything here is render-free: 5 Hz planner-LM codes on one side, audio attributes
computed from the source track on the other. No DiT, no VAE, no adapter renders.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TENSORS_ROOT = REPO / "server" / "data" / "training" / "tensors"
CODE_HZ = 5.0
FRAME_S = 1.0 / CODE_HZ

# ACE-Step 1.5 FSQ tokenizer (engine/src/fsq-quant.h). One flat index -> 6 grid values in [-1, 1].
FSQ_LEVELS = np.array([8, 8, 8, 5, 5, 5], dtype=np.int64)
FSQ_STRIDE = np.concatenate([[1], np.cumprod(FSQ_LEVELS[:-1])])  # [1, 8, 64, 512, 2560, 12800]
FSQ_VOCAB = int(np.prod(FSQ_LEVELS))  # 64000


def fsq_decode(codes: np.ndarray) -> np.ndarray:
    """[T] int codes -> [T, 6] float32, bit-compatible with fsq_decode_index()."""
    codes = np.asarray(codes, dtype=np.int64)
    bad = (codes < 0) | (codes >= FSQ_VOCAB)
    safe = np.where(bad, 0, codes)
    level = (safe[:, None] // FSQ_STRIDE[None, :]) % FSQ_LEVELS[None, :]
    step = (2.0 / (FSQ_LEVELS - 1)).astype(np.float32)
    out = (level.astype(np.float32) * step[None, :] - 1.0).astype(np.float32)
    out[bad] = 0.0
    return out


def fsq_digits(codes: np.ndarray) -> np.ndarray:
    """[T] -> [T, 6] int level indices (for categorical/one-hot inputs)."""
    codes = np.asarray(codes, dtype=np.int64)
    safe = np.clip(codes, 0, FSQ_VOCAB - 1)
    return (safe[:, None] // FSQ_STRIDE[None, :]) % FSQ_LEVELS[None, :]


@dataclass
class Variant:
    slug: str
    key: str
    dir: Path
    max_duration: int

    @property
    def codes_path(self) -> Path:
        return self.dir / "lm_codes.jsonl"

    @property
    def attr_dir(self) -> Path:
        return self.dir / "lm-attr"


def list_variants(root: Path = TENSORS_ROOT, max_duration: int | None = 600) -> list[Variant]:
    out: list[Variant] = []
    for slug_dir in sorted(root.iterdir()):
        if not slug_dir.is_dir():
            continue
        for vdir in sorted(slug_dir.iterdir()):
            meta = vdir / "preprocess_meta.json"
            codes = vdir / "lm_codes.jsonl"
            if not (meta.exists() and codes.exists()):
                continue
            md = int(json.load(open(meta, encoding="utf-8")).get("max_duration") or 0)
            if max_duration is not None and md != max_duration:
                continue
            out.append(Variant(slug_dir.name, vdir.name, vdir, md))
    return out


@dataclass
class CodeRow:
    file: str            # <stem>.safetensors
    caption: str
    lyrics: str
    duration: float
    codes: np.ndarray    # [T] int
    truncated: bool
    raw: dict


def load_codes(v: Variant) -> list[CodeRow]:
    rows: list[CodeRow] = []
    for line in open(v.codes_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        codes = np.asarray(r.get("codes") or [], dtype=np.int64)
        if codes.size == 0:
            continue
        rows.append(CodeRow(
            file=str(r.get("file", "")), caption=str(r.get("caption", "")),
            lyrics=str(r.get("lyrics", "")), duration=float(r.get("duration") or 0.0),
            codes=codes, truncated=bool(r.get("truncated")), raw=r,
        ))
    return rows


def safetensors_meta(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header.get("__metadata__", {})


def audio_path_for(v: Variant, row: CodeRow) -> Path | None:
    st = v.dir / row.file
    if not st.exists():
        return None
    ap = safetensors_meta(st).get("audio_path", "")
    return Path(ap) if ap and os.path.exists(ap) else None


def attr_path_for(v: Variant, row: CodeRow) -> Path:
    return v.attr_dir / (Path(row.file).stem + ".npz")


def st_tensor(path: Path, name: str) -> np.ndarray:
    """Read one F32 tensor out of a safetensors file without the safetensors package."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        info = header[name]
        a, b = info["data_offsets"]
        fh.seek(8 + n + a)
        buf = fh.read(b - a)
    assert info["dtype"] == "F32", info["dtype"]
    return np.frombuffer(buf, dtype=np.float32).reshape(info["shape"])


# ── code representations ─────────────────────────────────────────────────────
#
# The FSQ detokenizer (engine/src/fsq-detok.h) is PER TOKEN: one code -> 5 latent
# frames x 64 channels, with no cross-token context. So "what the model thinks a
# code means" is a fixed table, dumped once by `ace-train detok-table`.
DETOK_TABLE_PATH = REPO / "docs" / "plans" / "lm-attr-probe" / "detok-table.f32"
_DETOK: np.ndarray | None = None


def detok_table() -> np.ndarray:
    global _DETOK
    if _DETOK is None:
        if not DETOK_TABLE_PATH.exists():
            raise FileNotFoundError(f"{DETOK_TABLE_PATH} — run: ace-train detok-table --dit <dit.gguf> --out {DETOK_TABLE_PATH}")
        arr = np.fromfile(DETOK_TABLE_PATH, dtype=np.float32)
        _DETOK = arr.reshape(FSQ_VOCAB, 5, 64)
    return _DETOK


def onehot(digits: np.ndarray) -> np.ndarray:
    T = digits.shape[0]
    cols = []
    for d in range(6):
        L = int(FSQ_LEVELS[d])
        oh = np.zeros((T, L), dtype=np.float32)
        oh[np.arange(T), digits[:, d]] = 1.0
        cols.append(oh)
    return np.concatenate(cols, axis=1)


def stack_ctx(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x
    T = x.shape[0]
    pad = np.pad(x, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([pad[i : i + T] for i in range(2 * k + 1)], axis=1)


def rep_codes(codes: np.ndarray, rep: str) -> np.ndarray:
    """codes [T] -> features [T, F] for a representation name like 'detok64-ctx5'."""
    base, _, ctx = rep.partition("-ctx")
    k = int(ctx) if ctx else 0
    codes = np.clip(np.asarray(codes, dtype=np.int64), 0, FSQ_VOCAB - 1)
    if base == "fsq6":
        x = fsq_decode(codes)
    elif base == "onehot":
        x = onehot(fsq_digits(codes))
    elif base == "detok64":
        x = detok_table()[codes].mean(axis=1)
    elif base == "detok320":
        x = detok_table()[codes].reshape(codes.size, 320)
    else:
        raise ValueError(f"unknown representation {rep}")
    return stack_ctx(x.astype(np.float32), k)

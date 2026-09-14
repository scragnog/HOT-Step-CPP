#!/usr/bin/env python3
"""Convert Mothersuperior's minted_regularizer_pack.pt into the torch-free form
`ace-train yue2-ar-train --minted` reads.

The pack is a torch pickle: a flat list of 4,732 dicts, each
{name, src, style, lyrics, codec}, where `codec` is an np.int32 array of RAW
YuE2 semantic codes in [0, 32768) -- NOT pre-offset token ids. The AR trainer
adds YUE2_CODEC_OFFSET itself and nowhere else, so the codes are written out
raw and stay raw.

Output (format "yue2-minted-pack-v1", spec: docs/plans/yue2/15-minted-pack.md):

  <out>/minted_codes.i32      every song's codes concatenated, little-endian i32
  <out>/minted_manifest.json  a yue2-preprocess sources[]-shaped index, with
                              per-source codec_ids_offset / codec_ids_frames
                              (both in FRAMES, not bytes) into that blob

usage:
  python convert-yue2-minted.py <minted_regularizer_pack.pt> <out_dir>
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src_pt, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    h = hashlib.sha256()
    with open(src_pt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    pack_sha = h.hexdigest()
    pack_bytes = os.path.getsize(src_pt)

    recs = torch.load(src_pt, map_location="cpu", weights_only=False)
    if not isinstance(recs, list):
        print(f"error: {src_pt} is a {type(recs).__name__}, expected a list of records")
        return 1

    blob_name = "minted_codes.i32"
    blob_path = os.path.join(out_dir, blob_name)
    sources = []
    offset = 0          # in FRAMES
    n_val = 0
    lyric_colon = 0

    with open(blob_path, "wb") as blob:
        for i, r in enumerate(recs):
            missing = {"name", "src", "style", "lyrics", "codec"} - set(r)
            if missing:
                print(f"error: record {i} is missing {sorted(missing)}")
                return 1
            c = r["codec"]
            if isinstance(c, torch.Tensor):
                c = c.numpy()
            c = np.ascontiguousarray(np.asarray(c), dtype="<i4")
            if c.ndim != 1:
                print(f"error: record {i} ({r['name']}) codec has shape {c.shape}, expected 1-D")
                return 1
            if c.size and (int(c.min()) < 0 or int(c.max()) >= 32768):
                # A pre-offset stream would land at >= 151853 and double-offset
                # inside the trainer: in-vocabulary, finite, and completely wrong.
                print(f"error: record {i} ({r['name']}) codes out of [0, 32768): "
                      f"[{int(c.min())}, {int(c.max())}]")
                return 1
            src = r["src"]
            if src not in ("minted", "minted_val"):
                print(f"error: record {i} ({r['name']}) has src {src!r}, expected minted/minted_val")
                return 1
            n_val += src == "minted_val"
            style = " ".join(str(r["style"]).split())
            lyrics = str(r["lyrics"])
            # The AR trainer refuses a caption holding a `lyrics:` line unless
            # caption_format says the manifest was built from ACE sidecars --
            # a whole sidecar fed in as a style string is the one mistake
            # nothing downstream can catch. Count them so the refusal is never
            # a surprise at startup.
            lyric_colon += "lyrics:" in style

            blob.write(c.tobytes())
            sources.append({
                "name": r["name"],
                "src": src,
                "caption": style,
                "lyrics": lyrics,
                "codec_ids": blob_name,
                "codec_ids_offset": offset,
                "codec_ids_frames": int(c.size),
            })
            offset += int(c.size)

    if lyric_colon:
        print(f"warning: {lyric_colon} style string(s) contain a `lyrics:` line; the AR trainer "
              f"will refuse this manifest")

    manifest = {
        "format": "yue2-minted-pack-v1",
        "caption_format": "plain",
        "codec_ids_present": True,
        "codec_ids_blob": blob_name,
        "codec_ids_dtype": "int32-le",
        "codec_ids_raw": True,
        "frame_rate": 25.0,
        "source_repo": "Mothersuperior/yue2-minted-corpus",
        "source_file": os.path.basename(src_pt),
        "source_bytes": pack_bytes,
        "source_sha256": pack_sha,
        "n_sources": len(sources),
        "n_minted_val": n_val,
        "total_frames": offset,
        "sources": sources,
    }
    man_path = os.path.join(out_dir, "minted_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    lens = np.array([s["codec_ids_frames"] for s in sources])
    print(f"{len(sources)} songs ({n_val} minted_val), {offset} frames "
          f"({offset / 25 / 3600:.1f} h at 25 Hz)")
    print(f"  frames min {lens.min()} median {int(np.median(lens))} mean {lens.mean():.0f} "
          f"max {lens.max()}")
    print(f"  {blob_path}  {os.path.getsize(blob_path)} bytes")
    print(f"  {man_path}  {os.path.getsize(man_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

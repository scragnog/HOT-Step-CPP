#!/usr/bin/env python3
"""Put forced-alignment word spans into a YuE2 preprocess manifest.

`ace-train yue2-ar-train` trains upstream's lyric-cursor auxiliary loss from
per-word spans named per source as `cursor_words`: little-endian f32 [n, 5] =
(start_s, end_s, score, char0, char1), upstream cursor_prep.py's layout, with
char offsets in CODEPOINTS into the manifest's own `lyrics` string.

This is the BRIDGE from a Python producer to that seam. Today the producer is
upstream's own cursor_prep.py (Demucs vocal stem -> torchaudio MMS_FA forced
alignment of the full sheet), run in K:/yue2/.venv; it writes
<prep>/<name>/cursor_words.npy. A native aligner is the follow-up, and it will
write the same five columns straight into the cache without this script.

THE ONE THING THIS MUST GET RIGHT: the char offsets point into the lyrics the
aligner was given. If the manifest's `lyrics` differs from that string by a
single character - a collapsed double space, a stripped trailing newline, a
smart quote - every word after the difference lands on the wrong token, and
nothing downstream can tell. So the aligner's lyrics file is read back and
compared BYTE FOR BYTE to the manifest field, and a mismatch refuses the track
rather than writing a plausible-looking file.

usage:
  python engine/tools/yue2-cursor-bridge.py <manifest.json> <prep_dir> <artist_lyrics_dir>

    <manifest.json>      the cache's yue2_preprocess.json (rewritten in place,
                         a .bak is left beside it)
    <prep_dir>           upstream's RP: <prep_dir>/<name>/cursor_words.npy
    <artist_lyrics_dir>  upstream's LYD: <dir>/<name>.lyrics.txt, the exact
                         lyrics the aligner used

Sources are matched by the audio file's stem, which is what both layouts key on.
"""

import json
import os
import shutil
import sys

import numpy as np


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    man_path, prep, lyd = sys.argv[1:4]
    man = json.load(open(man_path, encoding="utf-8"))
    sources = man.get("sources") or []
    if not sources:
        print(f"{man_path}: no sources[]")
        sys.exit(1)
    cache_dir = os.path.dirname(os.path.abspath(man_path))
    out_dir = os.path.join(cache_dir, "cursor")
    os.makedirs(out_dir, exist_ok=True)

    n_ok, n_refused = 0, 0
    for s in sources:
        name = s.get("name") or ""
        stem = os.path.splitext(os.path.basename(name))[0]
        npy = os.path.join(prep, stem, "cursor_words.npy")
        lyr_file = os.path.join(lyd, stem + ".lyrics.txt")
        if not os.path.exists(npy):
            print(f"SKIP    {stem}: no {npy}")
            continue
        if not os.path.exists(lyr_file):
            print(f"REFUSED {stem}: aligner lyrics file missing: {lyr_file}")
            n_refused += 1
            continue

        # The aligner's string: upstream's prep reads it with .strip(), and
        # ar_prep.py stores THAT as item["lyrics"]; the char offsets are into
        # it. Our manifest's `lyrics` must be that string exactly.
        aligned = open(lyr_file, encoding="utf-8").read().strip()
        ours = (s.get("lyrics") or "").strip()
        if aligned != ours:
            # Say where, in codepoints, so the fix is findable.
            i = next((k for k, (a, b) in enumerate(zip(aligned, ours)) if a != b),
                     min(len(aligned), len(ours)))
            print(f"REFUSED {stem}: manifest lyrics differ from the aligner's at char {i} "
                  f"(aligner {len(aligned)} chars, manifest {len(ours)}): "
                  f"{aligned[max(0,i-20):i+20]!r} vs {ours[max(0,i-20):i+20]!r}")
            n_refused += 1
            continue

        w = np.load(npy)
        if w.ndim != 2 or w.shape[1] != 5 or len(w) == 0:
            print(f"REFUSED {stem}: {npy} has shape {w.shape}, want [n, 5]")
            n_refused += 1
            continue
        w = w.astype("<f4")
        if not (np.diff(w[:, 0]) >= 0).all() or (w[:, 4] > len(ours)).any() or (w[:, 3] < 0).any():
            print(f"REFUSED {stem}: spans are not monotonic or run past the lyrics")
            n_refused += 1
            continue

        rel = f"cursor/{stem}.f32"
        w.tofile(os.path.join(cache_dir, rel))
        s["cursor_words"] = rel
        n_ok += 1
        print(f"OK      {stem}: {len(w)} words, {w[:,0].min():.1f}s..{w[:,1].max():.1f}s, "
              f"{len(ours)} lyric chars -> {rel}")

    if n_ok:
        bak = man_path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(man_path, bak)
        json.dump(man, open(man_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n{n_ok} bound, {n_refused} refused, of {len(sources)} sources; manifest "
          f"{'rewritten' if n_ok else 'untouched'}")
    sys.exit(1 if n_refused else 0)


if __name__ == "__main__":
    main()

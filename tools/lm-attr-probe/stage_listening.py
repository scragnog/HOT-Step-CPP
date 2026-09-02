"""Stage a numbered listening set from the render harness into the _LISTENING hub (copies, never moves).

    python tools/lm-attr-probe/stage_listening.py --out "D:\\...\\_LISTENING\\<date>_<name>" \
        --slugs kinks_somethingelse,nas_illmatic --sides gt,base,ctrl,pp --songs 2 [--originals] [--note "..."]

Sides are manifest side labels in <variant>/lm-attr/renders/renders.json ('adapter' = the shipped
final adapter). Files: NN_<slug>_<songstem>_<side>.wav, originals as NNa_..._ORIGINAL.<ext>.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import list_variants, safetensors_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slugs", required=True)
    ap.add_argument("--sides", default="gt,base,adapter")
    ap.add_argument("--songs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--originals", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sides = [s for s in a.sides.split(",") if s]
    slugs = [s for s in a.slugs.split(",") if s]
    variants = {v.slug: v for v in list_variants(max_duration=None)}
    n = 1
    lines = [a.note or "Same caption, lyrics, seed and base DiT (no DiT adapter) for every file; only the plan differs.",
             "sides: " + ", ".join(sides), ""]
    for slug in slugs:
        v = variants.get(slug)
        if v is None:
            print("no variant for", slug)
            continue
        rd = v.attr_dir / "renders"
        man = json.load(open(rd / "renders.json", encoding="utf-8"))
        gt_files = []
        for m in man:
            if m["side"] == "gt" and m["seed"] == a.seed and m["file"] not in gt_files:
                gt_files.append(m["file"])
        for song in gt_files[: a.songs]:
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", song.replace(".safetensors", ""))[:28]
            first = n
            for side in sides:
                m = next((x for x in man if x["file"] == song and x["side"] == side and x["seed"] == a.seed), None)
                if m is None:
                    lines.append(f"(missing {side} for {slug}/{stem})")
                    continue
                label = "final" if side == "adapter" else side
                dst = out / f"{n:02d}_{slug}_{stem}_{label}.wav"
                shutil.copy2(rd / m["wav"], dst)
                lines.append(dst.name)
                n += 1
            if a.originals:
                st = v.dir / song
                src = Path(safetensors_meta(st).get("audio_path", "")) if st.exists() else None
                if src and src.exists():
                    dst = out / f"{first:02d}a_{slug}_{stem}_ORIGINAL{src.suffix.lower()}"
                    shutil.copy2(src, dst)
                    lines.append(dst.name)
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{n - 1} renders staged in {out}")


if __name__ == "__main__":
    main()

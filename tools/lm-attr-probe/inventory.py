"""A-1 inventory: which 600 s variants exist, which LM adapter each artist ships, and which runs are on disk.

    python tools/lm-attr-probe/inventory.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from common import REPO, list_variants, load_codes

LM4B_ROOT = Path(r"M:\HOT-Step-CPP\Adapters\lm-4b")
DB = REPO / "server" / "data" / "hotstep.db"
SEP = chr(92)


def has_weights(d: Path) -> bool:
    return (d / "adapter_model.safetensors").exists() or (d / "lokr_weights.safetensors").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    presets = {}
    c = sqlite3.connect(DB)
    for pid, p, scale in c.execute(
        "select id, lm_adapter_path, lm_adapter_scale from album_presets "
        "where lm_adapter_path is not null and lm_adapter_path != ''"
    ):
        parts = p.replace("/", SEP).split(SEP)
        # <root>\lm-4b\<slug>\<run>  or a bare artist dir
        slug = parts[-2] if len(parts) >= 2 and parts[-3].lower().startswith("lm-") else parts[-1]
        presets.setdefault(slug, []).append({"preset_id": pid, "path": p, "scale": scale, "exists": os.path.isdir(p)})

    out = []
    for v in list_variants():
        rows = load_codes(v)
        frames = int(sum(r.codes.size for r in rows))
        artist_dir = LM4B_ROOT / v.slug
        runs = []
        if artist_dir.is_dir():
            runs = sorted(d.name for d in artist_dir.iterdir() if d.is_dir() and has_weights(d))
        pr = presets.get(v.slug, [])
        shipped = next((x for x in pr if x["exists"]), None)
        out.append({
            "slug": v.slug, "variant": v.key, "songs": len(rows), "frames": frames,
            "hours": round(frames / 5 / 3600, 2), "truncated": int(sum(r.truncated for r in rows)),
            "preset_adapter": shipped["path"] if shipped else "",
            "preset_scale": shipped["scale"] if shipped else None,
            "runs_on_disk": runs,
        })

    print(f"{len(out)} variants at 600 s; {sum(1 for o in out if o['preset_adapter'])} with a shipped LM adapter")
    for o in out:
        print(f"{o['slug']:<32} songs={o['songs']:>3} h={o['hours']:>5} trunc={o['truncated']} "
              f"preset={'Y' if o['preset_adapter'] else '-'} scale={o['preset_scale']} runs={len(o['runs_on_disk'])} "
              f"newest={o['runs_on_disk'][-1] if o['runs_on_disk'] else '-'}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

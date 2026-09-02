"""summarize_arms.py — per-arm corpus summary for the caption-dropout / prior-preservation
experiment (docs/plans/lm-attr-probe/HANDOFF.md), read off score_renders.py's output.

For each arm side rendered with `lm-plan-render.ts --label <arm>` (ctrl/pp/cd/both — any side
name that is not gt/base/adapter counts as an arm), prints the corpus median ratio per axis
(< 1 = that arm's plans render closer to the artist than the base-LM plan; see score_renders.py's
own docstring for the exact distance) and the median degenerate-plan share.

    python tools/lm-attr-probe/summarize_arms.py [--scores docs/plans/lm-attr-probe/render_scores.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_probes import AXES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = REPO / "docs" / "plans" / "lm-attr-probe" / "render_scores.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    a = ap.parse_args()

    p = Path(a.scores)
    if not p.exists():
        raise SystemExit(f"no {p} — run score_renders.py first")
    data = json.loads(p.read_text(encoding="utf-8"))
    artists = data["artists"]

    arms = sorted({s for art in artists for s in art.get("extra_sides", [])})
    if not arms:
        raise SystemExit(f"no extra (--label) sides found in {p} — did lm-plan-render.ts run with --label?")

    print(f"{len(artists)} artist(s), arms: {', '.join(arms)}\n")
    header = f"{'arm':<8}" + "".join(f"{ax[:14]:>16}" for ax in AXES) + f"{'degenerate':>12}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        cells = []
        for ax in AXES:
            vals = [art["axes"][ax][f"median_ratio_{arm}"] for art in artists if f"median_ratio_{arm}" in art["axes"][ax]]
            cells.append(f"{np.median(vals):>16.3f}" if vals else f"{'-':>16}")
        deg = [art["degenerate_frac"][arm] for art in artists if arm in art.get("degenerate_frac", {})]
        deg_cell = f"{np.median(deg):>12.2f}" if deg else f"{'-':>12}"
        print(f"{arm:<8}" + "".join(cells) + deg_cell)

    print("\nratio < 1: that arm's plans render closer to the artist than the base-LM plan, per axis "
          "(same base DiT, same caption/lyrics/seed as the gt/base sides). degenerate: median share of "
          "plans flagged looping (>55% of the worst 40s) or short (<60% of expected length).")


if __name__ == "__main__":
    main()

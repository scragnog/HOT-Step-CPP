"""A-4 (rendered): do LM-adapter plans land closer to the artist than base-LM plans, per axis,
measured on AUDIO ATTRIBUTES of renders through the BASE DiT (no DiT adapter)?

Per artist, three render sides exist (server/scripts/lm-plan-render.ts): `gt` (the song's own
ground-truth codes), `base` (bare-LM plan), `adapter` (LM-adapter plan) — same caption, lyrics,
seed and decoder. Attributes come from extract_attrs.py --renders (same 56 features as the
source tracks, so the real audio is a second reference).

For each attribute (pooled over frames, z-scored on the real-audio corpus):
    d_base    = W1(base,    gt-render)      d_adapter = W1(adapter, gt-render)
    floor     = W1(gt-render half A, half B)
    ratio     = d_adapter / d_base          (< 1: the adapter's plans sit closer to the artist)
    real_*    = the same against the artist's REAL audio attributes (decoder offset included)
Matched-song variant: plan render vs the gt-render of the SAME song, averaged.
Degeneracy screen on the plan codes (loop share, length) is reported alongside.

    python tools/lm-attr-probe/score_renders.py [--min-songs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, attr_path_for, list_variants, load_codes  # noqa: E402
from fit_probes import AXES, rolling_mean  # noqa: E402
from score_plans import degeneracy, w1  # noqa: E402

OUT_DEFAULT = REPO / "docs" / "plans" / "lm-attr-probe"


def load_attr(p: Path, T: int | None = None) -> tuple[np.ndarray, list[str]]:
    d = np.load(p, allow_pickle=False)
    X = d["X"].astype(np.float32)
    names = [str(n) for n in d["names"]]
    dens = rolling_mean(X[:, names.index("onset_count")], 10).astype(np.float32)
    X = np.concatenate([X, dens[:, None]], axis=1)
    X[:, names.index("rms_db")] = np.maximum(X[:, names.index("rms_db")], -60.0)
    if T is not None:
        X = X[:T]
    return X, names + ["onset_density_2s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--min-songs", type=int, default=4)
    a = ap.parse_args()

    # corpus scaler from the real audio (all 46 artists), so distances are in comparable z units
    allX = []
    names = None
    for v in list_variants():
        for r in load_codes(v):
            p = attr_path_for(v, r)
            if p.exists():
                X, names = load_attr(p)
                allX.append(X)
    A = np.concatenate(allX)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    del allX, A

    per_artist = []
    for v in list_variants():
        rd = v.attr_dir / "renders"
        man = rd / "renders.json"
        rp = v.attr_dir / "plans" / "runs.json"
        if not (man.exists() and rp.exists()):
            continue
        manifest = json.load(open(man, encoding="utf-8"))
        runs = json.load(open(rp, encoding="utf-8"))
        rows = {r["id"]: r for r in runs["rows"]}
        gens = {(g["rowId"], g["side"], g["seed"]): np.asarray(g["codes"], dtype=np.int64) for g in runs["gens"]}
        sides: dict[str, list[np.ndarray]] = {"gt": [], "base": [], "adapter": []}
        by_song: dict[str, dict[str, np.ndarray]] = {}
        deg = {"base": [], "adapter": []}
        for m in manifest:
            npz = (rd / m["wav"]).with_suffix(".npz")
            if not npz.exists() or m["side"] not in sides:
                continue
            X, _ = load_attr(npz)
            Xz = (X - mu) / sd
            sides[m["side"]].append(Xz)
            by_song.setdefault(m["rowId"], {})[m["side"]] = Xz
            if m["side"] in deg:
                codes = gens.get((m["rowId"], m["side"], m["seed"]))
                if codes is not None:
                    deg[m["side"]].append(degeneracy(codes, int(rows[m["rowId"]]["durUsed"] * 5)))
        if len(sides["gt"]) < a.min_songs or not sides["base"] or not sides["adapter"]:
            continue
        # real audio, windowed to durUsed
        real = []
        for r in load_codes(v):
            rid = str(r.raw.get("_id", "")) or r.file[-20:]
            row = rows.get(rid)
            p = attr_path_for(v, r)
            if row is None or not p.exists():
                continue
            X, _ = load_attr(p, int(row["durUsed"] * 5))
            real.append((X - mu) / sd)
        GT = np.concatenate(sides["gt"])
        half = len(sides["gt"]) // 2
        GA, GB = np.concatenate(sides["gt"][:half]), np.concatenate(sides["gt"][half:])
        B, AD = np.concatenate(sides["base"]), np.concatenate(sides["adapter"])
        RL = np.concatenate(real) if real else None
        attrs = {}
        for j, name in enumerate(names):
            d_b, d_a, fl = w1(B[:, j], GT[:, j]), w1(AD[:, j], GT[:, j]), w1(GA[:, j], GB[:, j])
            mb = [w1(s["base"][:, j], s["gt"][:, j]) for s in by_song.values() if "base" in s and "gt" in s]
            ma = [w1(s["adapter"][:, j], s["gt"][:, j]) for s in by_song.values() if "adapter" in s and "gt" in s]
            rec = {"d_base": d_b, "d_adapter": d_a, "floor": fl, "ratio": d_a / max(d_b, 1e-9),
                   "matched_base": float(np.mean(mb)) if mb else float("nan"),
                   "matched_adapter": float(np.mean(ma)) if ma else float("nan"),
                   "matched_ratio": (float(np.mean(ma)) / max(float(np.mean(mb)), 1e-9)) if mb and ma else float("nan"),
                   "mean_shift_base": float(B[:, j].mean() - GT[:, j].mean()),
                   "mean_shift_adapter": float(AD[:, j].mean() - GT[:, j].mean())}
            if RL is not None:
                rb, ra, rg = w1(B[:, j], RL[:, j]), w1(AD[:, j], RL[:, j]), w1(GT[:, j], RL[:, j])
                rec.update({"real_d_base": rb, "real_d_adapter": ra, "real_d_gtrender": rg,
                            "real_ratio": ra / max(rb, 1e-9)})
            attrs[name] = rec
        axes = {}
        for ax, cols in AXES.items():
            cols = [c for c in cols if c in attrs]
            ratios = np.array([attrs[c]["ratio"] for c in cols])
            mrat = np.array([attrs[c]["matched_ratio"] for c in cols])
            rrat = np.array([attrs[c].get("real_ratio", np.nan) for c in cols])
            axes[ax] = {"n": len(cols), "median_ratio": float(np.median(ratios)),
                        "closer_frac": float(np.mean(ratios < 1.0)),
                        "median_matched_ratio": float(np.nanmedian(mrat)),
                        "median_real_ratio": float(np.nanmedian(rrat)),
                        "d_base": float(np.mean([attrs[c]["d_base"] for c in cols])),
                        "d_adapter": float(np.mean([attrs[c]["d_adapter"] for c in cols])),
                        "floor": float(np.mean([attrs[c]["floor"] for c in cols]))}
        degs_flag = {s: float(np.mean([(d["loop_worst40s"] > 0.55) or (d["len_ratio"] < 0.6) for d in deg[s]]))
                     for s in deg if deg[s]}
        per_artist.append({"slug": v.slug, "adapter": runs["adapterPath"], "n_gt": len(sides["gt"]),
                           "n_base": len(sides["base"]), "n_adapter": len(sides["adapter"]),
                           "axes": axes, "attrs": attrs, "degenerate_frac": degs_flag})
        print(f"{v.slug:<32} " + "  ".join(f"{ax[:8]}={axes[ax]['median_ratio']:.2f}" for ax in AXES) +
              f"  degen(adapter)={degs_flag.get('adapter', 0):.2f}", flush=True)

    if not per_artist:
        raise SystemExit("no rendered artists found — run lm-plan-render.ts + extract_attrs.py --renders first")
    corpus = {}
    for ax in AXES:
        vals = [p["axes"][ax] for p in per_artist]
        med = np.array([x["median_ratio"] for x in vals])
        mm = np.array([x["median_matched_ratio"] for x in vals])
        rr = np.array([x["median_real_ratio"] for x in vals])
        corpus[ax] = {"artists": len(vals), "n_attrs": vals[0]["n"],
                      "artists_adapter_closer": float(np.mean(med < 1.0)), "median_ratio": float(np.median(med)),
                      "matched_artists_adapter_closer": float(np.nanmean(mm < 1.0)), "matched_median_ratio": float(np.nanmedian(mm)),
                      "real_artists_adapter_closer": float(np.nanmean(rr < 1.0)), "real_median_ratio": float(np.nanmedian(rr)),
                      "mean_d_base": float(np.mean([x["d_base"] for x in vals])),
                      "mean_d_adapter": float(np.mean([x["d_adapter"] for x in vals])),
                      "mean_floor": float(np.mean([x["floor"] for x in vals]))}
    out = Path(a.out)
    (out / "render_scores.json").write_text(json.dumps({"corpus": corpus, "artists": per_artist}, indent=1), encoding="utf-8")
    lines = [f"# Base-LM vs adapter-LM plans rendered through the BASE DiT, per axis ({len(per_artist)} artists)\n",
             "ratio = W1(adapter render, gt render) / W1(base render, gt render); < 1 = adapter closer to the artist's "
             "own plan rendered by the same decoder. `real` = same against the artist's real audio. "
             "floor = W1 between two halves of the gt renders.\n",
             "| axis | attrs | artists | closer | median ratio | matched closer | matched median | real closer | real median | d_base | d_adapter | floor |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ax, c in corpus.items():
        lines.append(f"| {ax} | {c['n_attrs']} | {c['artists']} | {c['artists_adapter_closer']:.2f} | {c['median_ratio']:.3f} | "
                     f"{c['matched_artists_adapter_closer']:.2f} | {c['matched_median_ratio']:.3f} | "
                     f"{c['real_artists_adapter_closer']:.2f} | {c['real_median_ratio']:.3f} | "
                     f"{c['mean_d_base']:.3f} | {c['mean_d_adapter']:.3f} | {c['mean_floor']:.3f} |")
    lines.append("\n## Per artist (median ratio per axis; degenerate = share of adapter plans flagged loop/short)\n")
    lines.append("| artist | " + " | ".join(AXES) + " | degenerate |")
    lines.append("|---|" + "---|" * (len(AXES) + 1))
    for p in per_artist:
        lines.append(f"| {p['slug']} | " + " | ".join(f"{p['axes'][ax]['median_ratio']:.2f}" for ax in AXES) +
                     f" | {p['degenerate_frac'].get('adapter', 0):.2f} |")
    (out / "render_scores.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'render_scores.md'}")
    for ax, c in corpus.items():
        print(f"  {ax:<20} closer={c['artists_adapter_closer']:.2f} median ratio={c['median_ratio']:.3f} "
              f"(matched {c['matched_median_ratio']:.3f}, vs real {c['real_median_ratio']:.3f})")


if __name__ == "__main__":
    main()

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
        # real audio, windowed to durUsed, keyed by row so gt-render and real pair per song
        real = []
        real_by_row: dict[str, np.ndarray] = {}
        for r in load_codes(v):
            rid = str(r.raw.get("_id", "")) or r.file[-20:]
            row = rows.get(rid)
            p = attr_path_for(v, r)
            if row is None or not p.exists():
                continue
            X, _ = load_attr(p, int(row["durUsed"] * 5))
            real.append((X - mu) / sd)
            real_by_row[rid] = (X - mu) / sd
        # ── what the codes express through the base DiT: gt-render vs the song's real audio ──
        # per-song Pearson r per attribute (frame level and 10 s pooled) against a null that pairs
        # the gt-render with a DIFFERENT song's real audio (same artist, length-matched).
        paired = [(rid, by_song[rid]["gt"], real_by_row[rid]) for rid in by_song if "gt" in by_song[rid] and rid in real_by_row]
        express: dict[str, dict[str, float]] = {}
        if len(paired) >= 2:
            def corr(a: np.ndarray, b: np.ndarray) -> float:
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
                if n < 10 or a.std() < 1e-6 or b.std() < 1e-6:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])

            def pool10(x: np.ndarray) -> np.ndarray:
                n = len(x) // 50
                return x[: n * 50].reshape(n, 50).mean(1) if n >= 2 else x[:0]

            for j, name in enumerate(names):
                rs, rs10, nulls = [], [], []
                for k, (rid, g, rl) in enumerate(paired):
                    rs.append(corr(g[:, j], rl[:, j]))
                    rs10.append(corr(pool10(g[:, j]), pool10(rl[:, j])))
                    other = paired[(k + 1) % len(paired)][2]
                    nulls.append(corr(g[:, j], other[:, j]))
                express[name] = {"r": float(np.nanmedian(rs)), "r10": float(np.nanmedian(rs10)),
                                 "null_r": float(np.nanmedian(nulls)), "n": len(paired)}
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
            if name in express:
                rec.update({"express_r": express[name]["r"], "express_r10": express[name]["r10"],
                            "express_null_r": express[name]["null_r"]})
            attrs[name] = rec
        axes = {}
        for ax, cols in AXES.items():
            cols = [c for c in cols if c in attrs]
            ratios = np.array([attrs[c]["ratio"] for c in cols])
            mrat = np.array([attrs[c]["matched_ratio"] for c in cols])
            rrat = np.array([attrs[c].get("real_ratio", np.nan) for c in cols])
            ex = np.array([attrs[c].get("express_r", np.nan) for c in cols])
            ex10 = np.array([attrs[c].get("express_r10", np.nan) for c in cols])
            exn = np.array([attrs[c].get("express_null_r", np.nan) for c in cols])
            axes[ax] = {"n": len(cols), "median_ratio": float(np.median(ratios)),
                        "closer_frac": float(np.mean(ratios < 1.0)),
                        "median_matched_ratio": float(np.nanmedian(mrat)),
                        "median_real_ratio": float(np.nanmedian(rrat)),
                        "express_r": float(np.nanmedian(ex)), "express_r10": float(np.nanmedian(ex10)),
                        "express_null_r": float(np.nanmedian(exn)),
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
                      "express_r": float(np.nanmedian([x["express_r"] for x in vals])),
                      "express_r10": float(np.nanmedian([x["express_r10"] for x in vals])),
                      "express_null_r": float(np.nanmedian([x["express_null_r"] for x in vals])),
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
    lines.append("\n## What the codes express through the base DiT\n")
    lines.append("Per-song Pearson r between the render of the song's OWN codes and the song's real audio, "
                 "per attribute (median over songs, then artists). `null` pairs the render with a different song of "
                 "the same artist, so it is the genre/artist prior; the gap to `r` is what the codes add.\n")
    lines.append("| axis | frame r | 10 s r | null r |")
    lines.append("|---|---|---|---|")
    for ax, c in corpus.items():
        lines.append(f"| {ax} | {c['express_r']:.3f} | {c['express_r10']:.3f} | {c['express_null_r']:.3f} |")
    lines.append("")
    lines.append("| attribute | frame r | 10 s r | null r |")
    lines.append("|---|---|---|---|")
    attr_names = list(per_artist[0]["attrs"].keys())
    ex_med = {n: float(np.nanmedian([p["attrs"][n].get("express_r", np.nan) for p in per_artist])) for n in attr_names}
    for n in sorted(attr_names, key=lambda k: -np.nan_to_num(ex_med[k], nan=-9)):
        r10 = float(np.nanmedian([p["attrs"][n].get("express_r10", np.nan) for p in per_artist]))
        nl = float(np.nanmedian([p["attrs"][n].get("express_null_r", np.nan) for p in per_artist]))
        lines.append(f"| {n} | {ex_med[n]:.3f} | {r10:.3f} | {nl:.3f} |")
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

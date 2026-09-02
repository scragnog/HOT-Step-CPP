"""A-4: do LM adapters bring plans closer to the artist than the bare base LM? Per axis, render-free.

Input per artist: a runs.json written by server/scripts/lm-adapter-eval.ts generate
(base and adapter plans on the artist's own captions/lyrics, identical seeds), found at
<variant>/lm-attr/plans/runs.json (see gen_plans.ps1).

Every plan and every ground-truth sequence goes through the SAME fitted probe
(fit_probes.py models/<rep>.pkl), so probe bias cancels in the comparison. For each
attribute:
    d_base    = W1( probe(base plans),    probe(GT codes) )   pooled over frames, z units
    d_adapter = W1( probe(adapter plans), probe(GT codes) )
    floor     = W1( probe(GT half A),     probe(GT half B) )  the artist's own spread
    ratio     = d_adapter / d_base                             (< 1: adapter closer)
Also a MATCHED variant: plan vs the GT of the same song (same caption), averaged.
GT is windowed to the first `durUsed` seconds of each song so form/position bias is equal.

Plans are also screened for the degeneracies the 2026-08-29 RCA found (loop share,
unique share, length vs requested), reported alongside — a "closer" adapter whose plans
loop is not a win.

    python tools/lm-attr-probe/score_plans.py [--rep onehot-ctx5] [--min-r2 0.2]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, list_variants, rep_codes  # noqa: E402
from fit_probes import AXES  # noqa: E402

OUT_DEFAULT = REPO / "docs" / "plans" / "lm-attr-probe"


def w1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein-1 between two samples."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def degeneracy(codes: np.ndarray, requested_frames: int) -> dict:
    T = codes.size
    uniq = len(np.unique(codes)) / max(T, 1)
    loop = np.zeros(T, dtype=bool)
    for p in range(1, 26):  # periods up to 5 s
        if T > p:
            loop[p:] |= codes[p:] == codes[:-p]
    # worst 40 s window of loop share (the RCA's metric)
    k = min(200, T)
    ls = loop.astype(np.float32)
    worst = float(np.max(np.convolve(ls, np.ones(k), mode="valid") / k)) if T >= k else float(ls.mean())
    return {"uniq": float(uniq), "loop": float(loop.mean()), "loop_worst40s": worst,
            "len_ratio": float(T / max(requested_frames, 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="nn", choices=["nn", "ridge"])
    ap.add_argument("--rep", default="detok64-ctx5", help="ridge probe representation")
    ap.add_argument("--nn-tag", default="nn", help="CNN probe tag (fit_probes_nn.py --tag)")
    ap.add_argument("--models", default=str(OUT_DEFAULT / "models"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--min-r2", type=float, default=0.2, help="trust attributes whose probe frame R² >= this")
    a = ap.parse_args()

    if a.probe == "ridge":
        with open(Path(a.models) / f"{a.rep}.pkl", "rb") as fh:
            M = pickle.load(fh)
        probes = json.load(open(Path(a.out) / "probes.json", encoding="utf-8"))
        r2 = probes["reps"][a.rep]["frame_r2"]
        ynames: list[str] = M["ynames"]
        probe_label = f"ridge:{a.rep}"

        def predict(codes: np.ndarray, slug: str) -> np.ndarray:
            X = M["xsc"].transform(rep_codes(codes, a.rep)).astype(np.float32)
            return M["ridge"].predict(X)  # z-scored attribute space
    else:
        import torch
        from fit_probes_nn import build_model

        meta = json.load(open(Path(a.models) / f"{a.nn_tag}_meta.json", encoding="utf-8"))
        probes = json.load(open(Path(a.out) / f"probes_{a.nn_tag}.json", encoding="utf-8"))
        r2 = probes["frame_r2"]
        ynames = meta["ynames"]
        fold_of = meta["fold_of"]
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for f in sorted({v for v in fold_of.values()}):
            ck = torch.load(Path(a.models) / f"{a.nn_tag}_fold{f}.pt", map_location=dev)
            m = build_model(ck["n_out"], None, False, ck["width"]).to(dev)
            m.load_state_dict(ck["state"])
            m.eval()
            models[f] = m
        probe_label = f"cnn:{a.nn_tag} (held-out fold per artist)"

        def predict(codes: np.ndarray, slug: str) -> np.ndarray:
            f = fold_of.get(slug, 0)  # an artist outside the fit corpus falls back to fold 0
            with torch.no_grad():
                c = torch.from_numpy(np.clip(codes, 0, 63999)[None]).to(dev)
                return models[f](c)[0].cpu().numpy()

    per_artist = []
    for v in list_variants():
        rp = v.attr_dir / "plans" / "runs.json"
        if not rp.exists():
            continue
        runs = json.load(open(rp, encoding="utf-8"))
        rows = {r["id"]: r for r in runs["rows"]}
        gens = runs["gens"]
        sides = {"base": [], "adapter": []}
        gt_pred, gt_pred_song = [], {}
        for rid, r in rows.items():
            n = int(r["durUsed"] * 5)
            gt = np.asarray(r["gtCodes"], dtype=np.int64)[:n]
            if gt.size < 25:
                continue
            p = predict(gt, v.slug)
            gt_pred.append(p)
            gt_pred_song[rid] = p
        if len(gt_pred) < 4:
            continue
        deg = {"base": [], "adapter": []}
        matched = {"base": [], "adapter": []}
        for g in gens:
            rid = g["rowId"]
            if rid not in gt_pred_song:
                continue
            codes = np.asarray(g["codes"], dtype=np.int64)
            if codes.size < 25:
                continue
            p = predict(codes, v.slug)
            sides[g["side"]].append(p)
            deg[g["side"]].append(degeneracy(codes, int(rows[rid]["durUsed"] * 5)))
            matched[g["side"]].append((p, gt_pred_song[rid]))
        if not sides["base"] or not sides["adapter"]:
            continue
        GT = np.concatenate(gt_pred)
        half = len(gt_pred) // 2
        GA, GB = np.concatenate(gt_pred[:half]), np.concatenate(gt_pred[half:])
        B, A = np.concatenate(sides["base"]), np.concatenate(sides["adapter"])
        attrs = {}
        for j, name in enumerate(ynames):
            d_b, d_a, fl = w1(B[:, j], GT[:, j]), w1(A[:, j], GT[:, j]), w1(GA[:, j], GB[:, j])
            m_b = float(np.mean([w1(p[:, j], q[:, j]) for p, q in matched["base"]]))
            m_a = float(np.mean([w1(p[:, j], q[:, j]) for p, q in matched["adapter"]]))
            attrs[name] = {"d_base": d_b, "d_adapter": d_a, "floor": fl, "ratio": d_a / max(d_b, 1e-9),
                           "matched_base": m_b, "matched_adapter": m_a,
                           "matched_ratio": m_a / max(m_b, 1e-9), "r2": r2.get(name, float("nan")),
                           "mean_shift_base": float(B[:, j].mean() - GT[:, j].mean()),
                           "mean_shift_adapter": float(A[:, j].mean() - GT[:, j].mean())}
        axes = {}
        for ax, cols in AXES.items():
            cols = [c for c in cols if c in attrs and attrs[c]["r2"] >= a.min_r2]
            if not cols:
                axes[ax] = {"n": 0}
                continue
            ratios = np.array([attrs[c]["ratio"] for c in cols])
            mratios = np.array([attrs[c]["matched_ratio"] for c in cols])
            axes[ax] = {"n": len(cols), "median_ratio": float(np.median(ratios)),
                        "closer_frac": float(np.mean(ratios < 1.0)),
                        "median_matched_ratio": float(np.median(mratios)),
                        "matched_closer_frac": float(np.mean(mratios < 1.0)),
                        "d_base": float(np.mean([attrs[c]["d_base"] for c in cols])),
                        "d_adapter": float(np.mean([attrs[c]["d_adapter"] for c in cols])),
                        "floor": float(np.mean([attrs[c]["floor"] for c in cols]))}
        degs = {s: {k: float(np.mean([d[k] for d in deg[s]])) for k in deg[s][0]} for s in deg if deg[s]}
        degs_flag = {s: float(np.mean([(d["loop_worst40s"] > 0.55) or (d["len_ratio"] < 0.6) for d in deg[s]]))
                     for s in deg if deg[s]}
        per_artist.append({"slug": v.slug, "adapter": runs["adapterPath"], "lm": runs["lmModel"],
                           "n_songs": len(gt_pred), "n_gens": {s: len(sides[s]) for s in sides},
                           "axes": axes, "attrs": attrs, "degeneracy": degs, "degenerate_frac": degs_flag})
        print(f"{v.slug:<32} " + "  ".join(f"{ax[:8]}={axes[ax].get('median_ratio', float('nan')):.2f}"
                                          for ax in AXES) +
              f"  degen(adapter)={degs_flag.get('adapter', 0):.2f}", flush=True)

    if not per_artist:
        raise SystemExit("no plans found — run gen_plans.ps1 first")

    corpus = {}
    for ax in AXES:
        vals = [p["axes"][ax] for p in per_artist if p["axes"][ax].get("n", 0) > 0]
        if not vals:
            corpus[ax] = {"artists": 0}
            continue
        med = np.array([x["median_ratio"] for x in vals])
        mmed = np.array([x["median_matched_ratio"] for x in vals])
        corpus[ax] = {"artists": len(vals), "n_attrs": vals[0]["n"],
                      "artists_adapter_closer": float(np.mean(med < 1.0)),
                      "median_ratio": float(np.median(med)),
                      "matched_artists_adapter_closer": float(np.mean(mmed < 1.0)),
                      "matched_median_ratio": float(np.median(mmed)),
                      "mean_d_base": float(np.mean([x["d_base"] for x in vals])),
                      "mean_d_adapter": float(np.mean([x["d_adapter"] for x in vals])),
                      "mean_floor": float(np.mean([x["floor"] for x in vals]))}
    out = Path(a.out)
    (out / "scores.json").write_text(json.dumps({"probe": probe_label, "min_r2": a.min_r2, "corpus": corpus,
                                                  "artists": per_artist}, indent=1), encoding="utf-8")
    lines = [f"# Base vs adapter plans, per axis ({len(per_artist)} artists, probe {probe_label}, "
             f"attributes with frame R² >= {a.min_r2})\n",
             "ratio = W1(adapter, GT) / W1(base, GT); < 1 means the adapter's plans sit closer to the artist's own "
             "codes than the bare base LM's. `closer` = share of artists with median ratio < 1. "
             "floor = W1 between two halves of the artist's own songs.\n",
             "| axis | attrs | artists | closer (pooled) | median ratio | closer (matched song) | median matched ratio | d_base | d_adapter | floor |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ax, c in corpus.items():
        if not c.get("artists"):
            lines.append(f"| {ax} | 0 | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {ax} | {c['n_attrs']} | {c['artists']} | {c['artists_adapter_closer']:.2f} | "
                     f"{c['median_ratio']:.3f} | {c['matched_artists_adapter_closer']:.2f} | "
                     f"{c['matched_median_ratio']:.3f} | {c['mean_d_base']:.3f} | {c['mean_d_adapter']:.3f} | "
                     f"{c['mean_floor']:.3f} |")
    lines.append("\n## Per artist (median ratio per axis; degenerate = share of adapter plans flagged loop/short)\n")
    lines.append("| artist | " + " | ".join(AXES) + " | degenerate |")
    lines.append("|---|" + "---|" * (len(AXES) + 1))
    for p in per_artist:
        cells = [f"{p['axes'][ax]['median_ratio']:.2f}" if p["axes"][ax].get("n") else "-" for ax in AXES]
        lines.append(f"| {p['slug']} | " + " | ".join(cells) + f" | {p['degenerate_frac'].get('adapter', 0):.2f} |")
    (out / "scores.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'scores.md'}")
    for ax, c in corpus.items():
        if c.get("artists"):
            print(f"  {ax:<20} closer={c['artists_adapter_closer']:.2f} median ratio={c['median_ratio']:.3f} "
                  f"(matched {c['matched_artists_adapter_closer']:.2f} / {c['matched_median_ratio']:.3f})")


if __name__ == "__main__":
    main()

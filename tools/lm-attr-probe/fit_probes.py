"""A-3: what is linearly predictable from 5 Hz planner-LM codes?

Fits ridge probes representation -> audio attribute, leave-ARTIST-out (GroupKFold),
and a code -> artist logistic probe, leave-SONG-out. Reports R² per attribute per
representation (frame level and 10 s-window level) and artist-ID accuracy vs chance.

Representations (common.rep_codes):
    fsq6            the 6 FSQ grid values in [-1, 1]                          (6 dims)
    onehot          one-hot of the 6 FSQ digits (8+8+8+5+5+5)                 (39 dims)
    detok64         the FSQ detokenizer's output for the code, 5 frames averaged (64 dims)
    detok64-ctxK    detok64 stacked over +-K frames
    vae64           NOT a function of codes: the VAE latent the code was made from,
                    mean-pooled 25 -> 5 Hz. The ceiling: what the latent carries.

Outputs (default <repo>/docs/plans/lm-attr-probe/):
    probes.json, probes.md, models/<rep>.pkl (fitted ridge for score_plans.py)

    python tools/lm-attr-probe/fit_probes.py [--reps fsq6,onehot,detok64,detok64-ctx5,vae64] [--max-frames 400000]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, attr_path_for, list_variants, load_codes, rep_codes, st_tensor, stack_ctx  # noqa: E402

OUT_DEFAULT = REPO / "docs" / "plans" / "lm-attr-probe"

# Attributes grouped into the axes the study reports on.
AXES = {
    "loudness/dynamics": ["rms_db", "rms_delta", "crest_2s"],
    "rhythm": ["onset_strength", "onset_count", "onset_density_2s", "spec_flux"],
    "timbre": ["spec_centroid", "spec_bandwidth", "spec_flatness", "spec_rolloff", "zcr", "hp_ratio_db"]
              + [f"mfcc{i}" for i in range(13)],
    "harmony": ["chroma_entropy"] + [f"chroma_rel{i}" for i in range(12)] + [f"tonnetz{i}" for i in range(6)],
    "harmony(absolute)": [f"chroma{i}" for i in range(12)],
}


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    pad = np.pad(x, (k // 2, k - k // 2 - 1), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, k).mean(axis=1)


def load_corpus(reps: list[str], max_frames: int, seed: int):
    rng = np.random.default_rng(seed)
    songs = []
    want_vae = any(r.startswith("vae64") for r in reps)
    for v in list_variants():
        for r in load_codes(v):
            p = attr_path_for(v, r)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=False)
            X = d["X"]
            names = [str(n) for n in d["names"]]
            T = min(X.shape[0], r.codes.size)
            vae5 = None
            if want_vae:
                st = v.dir / r.file
                if not st.exists():
                    continue
                lat = st_tensor(st, "target_latents")  # [T25, 64]
                T5 = lat.shape[0] // 5
                vae5 = lat[: T5 * 5].reshape(T5, 5, 64).mean(axis=1)
                T = min(T, T5)
            if T < 25:
                continue
            Y = X[:T].astype(np.float32)
            oc = names.index("onset_count")
            dens = rolling_mean(Y[:, oc], 10).astype(np.float32)
            Y = np.concatenate([Y, dens[:, None]], axis=1)
            ynames = names + ["onset_density_2s"]
            Y[:, names.index("rms_db")] = np.maximum(Y[:, names.index("rms_db")], -60.0)
            songs.append({"slug": v.slug, "file": r.file, "codes": r.codes[:T], "Y": Y, "ynames": ynames,
                          "vae5": None if vae5 is None else vae5[:T]})
    if not songs:
        raise SystemExit("no attribute files found — run extract_attrs.py first")
    ynames = songs[0]["ynames"]
    artists = sorted({s["slug"] for s in songs})

    total = sum(s["Y"].shape[0] for s in songs)
    keep_frac = min(1.0, max_frames / total) if max_frames > 0 else 1.0
    Xs, Ys, g_art, g_song = [], [], [], []
    for si, s in enumerate(songs):
        T = s["Y"].shape[0]
        idx = np.arange(T) if keep_frac >= 1.0 else np.sort(rng.choice(T, size=max(1, int(T * keep_frac)), replace=False))
        rep = {}
        for name in reps:
            if name.startswith("vae64"):
                _, _, ctx = name.partition("-ctx")
                rep[name] = stack_ctx(s["vae5"], int(ctx) if ctx else 0)[idx]
            else:
                rep[name] = rep_codes(s["codes"], name)[idx]
        Xs.append(rep)
        Ys.append(s["Y"][idx])
        g_art.append(np.full(idx.size, artists.index(s["slug"])))
        g_song.append(np.full(idx.size, si))
    reps_x = {k: np.concatenate([x[k] for x in Xs], axis=0) for k in reps}
    Y = np.concatenate(Ys, axis=0)
    return reps_x, Y, ynames, np.concatenate(g_art), np.concatenate(g_song), artists, songs


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="fsq6,onehot,detok64,detok64-ctx5,vae64")
    ap.add_argument("--max-frames", type=int, default=400000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", default="0.1,1,10,100,1000")
    ap.add_argument("--artist-rep", default="detok64-ctx5")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    reps_wanted = [r for r in a.reps.split(",") if r]
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reps, Y, ynames, g_art, g_song, artists, songs = load_corpus(reps_wanted, a.max_frames, a.seed)
    print(f"corpus: {len(songs)} songs, {len(artists)} artists, {Y.shape[0]} frames used, "
          f"{Y.shape[1]} targets, reps={ {k: v.shape[1] for k, v in reps.items()} } ({time.time() - t0:.0f}s)", flush=True)

    ysc = StandardScaler().fit(Y)
    Yz = ysc.transform(Y).astype(np.float32)
    alphas = [float(x) for x in a.alphas.split(",")]
    gkf = GroupKFold(n_splits=a.folds)
    results: dict = {"frames": int(Y.shape[0]), "songs": len(songs), "artists": artists,
                     "targets": ynames, "reps": {}, "artist_id": {}}

    for rep_name, X in reps.items():
        print(f"\n== {rep_name} [{X.shape[1]} dims]", flush=True)
        xsc = StandardScaler().fit(X)
        Xz = xsc.transform(X).astype(np.float32)
        tr, te = next(gkf.split(Xz, Yz, g_art))
        best = (None, -1e9)
        for al in alphas:
            m = Ridge(alpha=al).fit(Xz[tr], Yz[tr])
            P = m.predict(Xz[te])
            score = float(np.nanmean([r2(Yz[te, j], P[:, j]) for j in range(Yz.shape[1])]))
            if score > best[1]:
                best = (al, score)
        alpha = best[0]
        P_all = np.zeros_like(Yz)
        for tr, te in gkf.split(Xz, Yz, g_art):
            m = Ridge(alpha=alpha).fit(Xz[tr], Yz[tr])
            P_all[te] = m.predict(Xz[te])
        frame_r2 = {ynames[j]: r2(Yz[:, j], P_all[:, j]) for j in range(Yz.shape[1])}
        yt, yp = [], []
        for si in np.unique(g_song):
            sel = np.where(g_song == si)[0]
            n = sel.size // 50
            if n == 0:
                continue
            sel = sel[: n * 50]
            yt.append(Yz[sel].reshape(n, 50, -1).mean(axis=1))
            yp.append(P_all[sel].reshape(n, 50, -1).mean(axis=1))
        YT, YP = np.concatenate(yt), np.concatenate(yp)
        win_r2 = {ynames[j]: r2(YT[:, j], YP[:, j]) for j in range(Yz.shape[1])}
        axis_summary = {}
        for ax, cols in AXES.items():
            cols = [c for c in cols if c in frame_r2]
            axis_summary[ax] = {
                "frame_r2_mean": float(np.mean([frame_r2[c] for c in cols])),
                "win10s_r2_mean": float(np.mean([win_r2[c] for c in cols])),
                "n": len(cols),
            }
        results["reps"][rep_name] = {"dims": int(X.shape[1]), "alpha": alpha, "frame_r2": frame_r2,
                                     "win10s_r2": win_r2, "axes": axis_summary}
        for ax, s in axis_summary.items():
            print(f"  {ax:<20} frame R2={s['frame_r2_mean']:6.3f}   10s R2={s['win10s_r2_mean']:6.3f}", flush=True)
        if not rep_name.startswith("vae64"):
            final = Ridge(alpha=alpha).fit(Xz, Yz)
            with open(out / "models" / f"{rep_name}.pkl", "wb") as fh:
                pickle.dump({"xsc": xsc, "ysc": ysc, "ridge": final, "ynames": ynames, "rep": rep_name}, fh)

    # ── artist identity from codes, leave-SONG-out ──────────────────────
    rep_name = a.artist_rep if a.artist_rep in reps else next(k for k in reps if not k.startswith("vae64"))
    X = reps[rep_name]
    xsc = StandardScaler().fit(X)
    Xz = xsc.transform(X).astype(np.float32)
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(Xz.shape[0], size=min(Xz.shape[0], 150000), replace=False))
    gkf_s = GroupKFold(n_splits=a.folds)
    top1 = top5 = n = 0
    win_top1 = win_n = 0
    for tr, te in gkf_s.split(Xz[sub], g_art[sub], g_song[sub]):
        clf = LogisticRegression(max_iter=200, C=0.5)
        clf.fit(Xz[sub][tr], g_art[sub][tr])
        lp = clf.predict_log_proba(Xz[sub][te])
        y = g_art[sub][te]
        order = np.argsort(-lp, axis=1)
        top1 += int(np.sum(order[:, 0] == y))
        top5 += int(np.sum(np.any(order[:, :5] == y[:, None], axis=1)))
        n += y.size
        gs = g_song[sub][te]
        for si in np.unique(gs):
            sel = np.where(gs == si)[0]
            nw = sel.size // 50
            if nw == 0:
                continue
            agg = lp[sel[: nw * 50]].reshape(nw, 50, -1).sum(axis=1)
            win_top1 += int(np.sum(np.argmax(agg, axis=1) == y[sel[0]]))
            win_n += nw
    results["artist_id"] = {"rep": rep_name, "n_artists": len(artists), "chance_top1": 1.0 / len(artists),
                            "frame_top1": top1 / n, "frame_top5": top5 / n,
                            "win10s_top1": win_top1 / max(win_n, 1), "frames_used": int(n)}
    print(f"\nartist-ID from codes ({rep_name}, leave-song-out, {len(artists)} artists, chance {1 / len(artists):.3f}): "
          f"frame top1={top1 / n:.3f} top5={top5 / n:.3f}  10s-window top1={win_top1 / max(win_n, 1):.3f}", flush=True)

    (out / "probes.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = [f"# Representation -> attribute probes ({len(songs)} songs / {len(artists)} artists, "
             f"{Y.shape[0]} frames, leave-artist-out {a.folds}-fold ridge)\n",
             "Frame R² / 10 s-window R², z-scored targets. `vae64` is the VAE latent itself (ceiling); "
             "the others are functions of the code.\n"]
    lines.append("| axis | " + " | ".join(results["reps"]) + " |")
    lines.append("|---|" + "---|" * len(results["reps"]))
    for ax in AXES:
        cells = []
        for r in results["reps"].values():
            s = r["axes"][ax]
            cells.append(f"{s['frame_r2_mean']:.3f} / {s['win10s_r2_mean']:.3f}")
        lines.append(f"| {ax} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per attribute (frame R² / 10 s R²)\n")
    lines.append("| attribute | " + " | ".join(results["reps"]) + " |")
    lines.append("|---|" + "---|" * len(results["reps"]))
    order = sorted(ynames, key=lambda k: -max(r["frame_r2"][k] for r in results["reps"].values()))
    for name in order:
        cells = [f"{r['frame_r2'][name]:.3f} / {r['win10s_r2'][name]:.3f}" for r in results["reps"].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    ai = results["artist_id"]
    lines.append(f"\n## Artist identity from codes ({ai['rep']})\n\n{ai['n_artists']} artists, chance {ai['chance_top1']:.3f}: "
                 f"frame top-1 {ai['frame_top1']:.3f}, top-5 {ai['frame_top5']:.3f}, 10 s-window top-1 {ai['win10s_top1']:.3f}\n")
    (out / "probes.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'probes.md'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

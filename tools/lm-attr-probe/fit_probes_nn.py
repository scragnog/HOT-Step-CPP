"""A-3b: a NON-linear, CONTEXTUAL probe from 5 Hz codes to audio attributes.

The linear probes (fit_probes.py) are weak because the FSQ code is a plan token the
DiT decodes with context, not a latent. This is the honest "what is predictable from
codes" bound: a small dilated 1-D CNN over the code sequence (receptive field ~+-6 s),
code embedding initialised from the detokenizer table, trained leave-ARTIST-out
(5 folds by artist, so every artist has a model that never saw it).

Outputs (default <repo>/docs/plans/lm-attr-probe/):
    probes_nn.json / probes_nn.md         R² per attribute / axis, frame + 10 s window
    models/nn_fold{k}.pt, models/nn_meta.json   fold models + artist->fold map + target scaler,
                                          consumed by score_plans.py --probe nn

    python tools/lm-attr-probe/fit_probes_nn.py [--epochs 12] [--embed-init detok|random] [--freeze-embed]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FSQ_VOCAB, REPO, attr_path_for, detok_table, list_variants, load_codes  # noqa: E402
from fit_probes import AXES, rolling_mean  # noqa: E402

OUT_DEFAULT = REPO / "docs" / "plans" / "lm-attr-probe"


def load_songs():
    songs = []
    for v in list_variants():
        for r in load_codes(v):
            p = attr_path_for(v, r)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=False)
            X = d["X"]
            names = [str(n) for n in d["names"]]
            T = min(X.shape[0], r.codes.size)
            if T < 50:
                continue
            Y = X[:T].astype(np.float32)
            dens = rolling_mean(Y[:, names.index("onset_count")], 10).astype(np.float32)
            Y = np.concatenate([Y, dens[:, None]], axis=1)
            Y[:, names.index("rms_db")] = np.maximum(Y[:, names.index("rms_db")], -60.0)
            songs.append({"slug": v.slug, "file": r.file, "codes": r.codes[:T].astype(np.int64), "Y": Y,
                          "ynames": names + ["onset_density_2s"]})
    if not songs:
        raise SystemExit("no attribute files — run extract_attrs.py first")
    return songs


def build_model(n_out: int, embed_init: np.ndarray | None, freeze: bool, width: int = 128, dropout: float = 0.0):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c: int, dil: int):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
            self.norm = nn.GroupNorm(8, c)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.act(self.norm(self.conv(x))))

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(FSQ_VOCAB, 64)
            if embed_init is not None:
                self.emb.weight.data.copy_(torch.from_numpy(embed_init))
            self.emb.weight.requires_grad = not freeze
            self.inp = nn.Conv1d(64, width, 5, padding=2)
            self.blocks = nn.Sequential(Block(width, 1), Block(width, 2), Block(width, 4), Block(width, 8))
            self.out = nn.Conv1d(width, n_out, 1)

        def forward(self, codes):  # [B, T] -> [B, T, n_out]
            x = self.emb(codes).transpose(1, 2)  # [B, 64, T]
            x = torch.nn.functional.gelu(self.inp(x))
            x = self.blocks(x)
            return self.out(x).transpose(1, 2)

    return Probe()


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=256, help="training crop, frames (256 = 51 s)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--embed-init", default="detok", choices=["detok", "random"])
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-artists", type=int, default=4,
                    help="training artists held back per fold for early stopping (0 = none)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    songs = load_songs()
    ynames = songs[0]["ynames"]
    artists = sorted({s["slug"] for s in songs})
    fold_of = {art: i % a.folds for i, art in enumerate(artists)}
    Yall = np.concatenate([s["Y"] for s in songs])
    ymu, ysd = Yall.mean(0), Yall.std(0) + 1e-6
    for s in songs:
        s["Yz"] = ((s["Y"] - ymu) / ysd).astype(np.float32)
    n_out = len(ynames)
    print(f"{len(songs)} songs, {len(artists)} artists, {Yall.shape[0]} frames, {n_out} targets, device {dev} "
          f"({time.time() - t0:.0f}s)", flush=True)

    embed_init = None
    if a.embed_init == "detok":
        tab = detok_table().mean(axis=1)  # [64000, 64]
        embed_init = ((tab - tab.mean(0)) / (tab.std(0) + 1e-6)).astype(np.float32)

    preds = {}  # song index -> [T, n_out] held-out prediction
    for fold in range(a.folds):
        tr_all = [s for s in songs if fold_of[s["slug"]] != fold]
        te = [(i, s) for i, s in enumerate(songs) if fold_of[s["slug"]] == fold]
        tr_arts = sorted({s["slug"] for s in tr_all})
        val_arts = set(tr_arts[-a.val_artists:]) if a.val_artists > 0 else set()
        tr = [s for s in tr_all if s["slug"] not in val_arts]
        va = [s for s in tr_all if s["slug"] in val_arts]
        model = build_model(n_out, embed_init, a.freeze_embed, a.width, a.dropout).to(dev)
        best_state, best_val = None, -1e9

        def val_score() -> float:
            model.eval()
            with torch.no_grad():
                yt = np.concatenate([s["Yz"] for s in va])
                yp = np.concatenate([model(torch.from_numpy(s["codes"][None]).to(dev))[0].cpu().numpy() for s in va])
            model.train()
            return float(np.mean([r2(yt[:, j], yp[:, j]) for j in range(n_out)]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-2)
        n_chunks = sum(max(1, s["codes"].size // a.chunk) for s in tr)
        steps_per_epoch = max(1, n_chunks // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
        lengths = np.array([s["codes"].size for s in tr], dtype=np.float64)
        p_song = lengths / lengths.sum()
        for ep in range(a.epochs):
            model.train()
            tot = 0.0
            for _ in range(steps_per_epoch):
                idx = rng.choice(len(tr), size=a.batch, p=p_song)
                cb = np.zeros((a.batch, a.chunk), dtype=np.int64)
                yb = np.zeros((a.batch, a.chunk, n_out), dtype=np.float32)
                mb = np.zeros((a.batch, a.chunk), dtype=np.float32)
                for b, si in enumerate(idx):
                    s = tr[si]
                    T = s["codes"].size
                    if T <= a.chunk:
                        cb[b, :T] = s["codes"]
                        yb[b, :T] = s["Yz"]
                        mb[b, :T] = 1.0
                    else:
                        st = rng.integers(0, T - a.chunk + 1)
                        cb[b] = s["codes"][st : st + a.chunk]
                        yb[b] = s["Yz"][st : st + a.chunk]
                        mb[b] = 1.0
                c = torch.from_numpy(cb).to(dev)
                y = torch.from_numpy(yb).to(dev)
                m = torch.from_numpy(mb).to(dev)
                p = model(c)
                loss = (((p - y) ** 2).mean(-1) * m).sum() / m.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            msg = f"  fold {fold} epoch {ep + 1}/{a.epochs} train mse {tot / steps_per_epoch:.4f}"
            if va and ((ep + 1) % 2 == 0 or ep + 1 == a.epochs):
                vs = val_score()
                msg += f"  val R2 {vs:.4f}"
                if vs > best_val:
                    best_val, best_state = vs, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(msg, flush=True)
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  fold {fold}: restored best val R2 {best_val:.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            for i, s in te:
                c = torch.from_numpy(s["codes"][None]).to(dev)
                preds[i] = model(c)[0].cpu().numpy()
        torch.save({"state": model.state_dict(), "width": a.width, "n_out": n_out, "fold": fold},
                   out / "models" / f"{a.tag}_fold{fold}.pt")
        # fold R² snapshot
        yt = np.concatenate([songs[i]["Yz"] for i, _ in te])
        yp = np.concatenate([preds[i] for i, _ in te])
        fr = {ynames[j]: r2(yt[:, j], yp[:, j]) for j in range(n_out)}
        print(f"fold {fold} held-out ({len(te)} songs): " + "  ".join(
            f"{ax[:8]}={np.mean([fr[c] for c in cols if c in fr]):.3f}" for ax, cols in AXES.items()), flush=True)

    # aggregate over all held-out predictions
    YT = np.concatenate([s["Yz"] for s in songs])
    YP = np.concatenate([preds[i] for i in range(len(songs))])
    frame_r2 = {ynames[j]: r2(YT[:, j], YP[:, j]) for j in range(n_out)}
    wt, wp = [], []
    for i, s in enumerate(songs):
        n = s["Yz"].shape[0] // 50
        if n == 0:
            continue
        wt.append(s["Yz"][: n * 50].reshape(n, 50, -1).mean(1))
        wp.append(preds[i][: n * 50].reshape(n, 50, -1).mean(1))
    WT, WP = np.concatenate(wt), np.concatenate(wp)
    win_r2 = {ynames[j]: r2(WT[:, j], WP[:, j]) for j in range(n_out)}
    axes = {}
    for ax, cols in AXES.items():
        cols = [c for c in cols if c in frame_r2]
        axes[ax] = {"frame_r2_mean": float(np.mean([frame_r2[c] for c in cols])),
                    "win10s_r2_mean": float(np.mean([win_r2[c] for c in cols])), "n": len(cols)}
    res = {"tag": a.tag, "epochs": a.epochs, "folds": a.folds, "embed_init": a.embed_init,
           "freeze_embed": a.freeze_embed, "width": a.width, "songs": len(songs), "artists": artists,
           "frames": int(YT.shape[0]), "targets": ynames, "frame_r2": frame_r2, "win10s_r2": win_r2, "axes": axes}
    (out / f"probes_{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    json.dump({"fold_of": fold_of, "ymu": ymu.tolist(), "ysd": ysd.tolist(), "ynames": ynames, "width": a.width,
               "embed_init": a.embed_init, "tag": a.tag}, open(out / "models" / f"{a.tag}_meta.json", "w"), indent=1)
    lines = [f"# CNN probe ({a.tag}): codes -> attributes, leave-artist-out {a.folds}-fold, "
             f"{len(songs)} songs / {len(artists)} artists, {YT.shape[0]} frames, embed {a.embed_init}"
             f"{' frozen' if a.freeze_embed else ''}, {a.epochs} epochs\n",
             "| axis | frame R² | 10 s R² |", "|---|---|---|"]
    for ax, s in axes.items():
        lines.append(f"| {ax} | {s['frame_r2_mean']:.3f} | {s['win10s_r2_mean']:.3f} |")
    lines += ["", "| attribute | frame R² | 10 s R² |", "|---|---|---|"]
    for name in sorted(frame_r2, key=lambda k: -frame_r2[k]):
        lines.append(f"| {name} | {frame_r2[name]:.3f} | {win_r2[name]:.3f} |")
    (out / f"probes_{a.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:9]))
    print(f"\nwrote {out / f'probes_{a.tag}.md'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

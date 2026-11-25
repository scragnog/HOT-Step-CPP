"""Upstream's grouper on the CPU with PCEN computed exactly: the reference for the port's grouper network.

Usage (venv-cpu): python exact_grouper.py --upstream <clone> --weights <dir> --captures <root> [--fixtures a,b,...]

PCEN smooths every channel with a 200-tap filter. Upstream applies it through a float32 FFT, so wherever the power sits far
below the row's peak the result is that FFT's rounding noise, and on the CPU the FFT is MKL's. Any other FFT (numpy's
pocketfft, MKL at another length, an exact sum) moves the window probabilities by up to 0.1 and, on t1, a fifth of the
notes to another track. Here the filter runs in float64, where its error is around 1e-12: a value the port can reproduce.
Everything else is upstream's code, fed the CPU capture's notes, features and retuned audio. Writes <fixture>/cpu-exact/
(grouper.probs, .emb, .h, .win.* in the capture layout, result.json with the groups) and <fixture>/grouper-exact.json: the
gaps and the track agreement against the CPU oracle, i.e. the oracle's sensitivity to the FFT's rounding."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import common

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--captures', required=True)
ap.add_argument('--fixtures')
a = ap.parse_args()
torch = common.load_upstream(a.upstream, a.weights, 'cpu')

from dettato.grouper import core as gcore, front as FR       # noqa: E402
from dettato.grouper.instruments import INSTRUMENTS           # noqa: E402


def pcen_exact(X, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, K=200):
    """front.pcen with the smoothing in float64 (FFT error ~1e-12); exp and the output stay float32 like upstream."""
    T = X.shape[2]
    E = torch.exp(X.float()).permute(0, 1, 3, 2).double()
    k = s * (1 - s) ** torch.arange(K, dtype=torch.float64)
    n = 1 << int(np.ceil(np.log2(T + K)))
    M = torch.fft.irfft(torch.fft.rfft(E, n) * torch.fft.rfft(k, n), n)[..., :T].clamp(min=0)
    out = (E / (eps + M) ** alpha + delta) ** r - delta ** r
    return torch.log(out + 1e-4).float().permute(0, 1, 3, 2)


FR.pcen = pcen_exact
gr = gcore.Grouper.from_pretrained('cpu')
names = a.fixtures.split(',') if a.fixtures else sorted(p.name for p in Path(a.captures).iterdir() if (p / 'cpu').is_dir())
for name in names:
    T, r = common.load_capture(Path(a.captures) / name / 'cpu')
    meta = np.asarray(T['grouper.meta'])
    n = len(meta)
    out_dir = Path(a.captures) / name / 'cpu-exact'
    dump = common.Dumps(out_dir)
    if n == 0:
        common.write_json(out_dir / 'result.json', dict(groups=[], grouper_window_s=gr.window_s))
        common.write_json(Path(a.captures) / name / 'grouper-exact.json', dict(notes=0, same_track=1.0, names_ok=True))
        print(name, 'no notes')
        continue
    y = np.asarray(T['x16_retuned'] if 'x16_retuned' in T else T['x16'], np.float32)
    P = gr.predict(gcore._Song(y, meta, np.asarray(T['grouper.feat']), np.asarray(T['grouper.fsc'])))
    for k in ('probs', 'emb', 'h'):
        dump.put(f'grouper.{k}', P[k])
    wins = P['wins']
    dump.put('grouper.win.t0', np.array([w[0] for w in wins], np.float64))
    dump.put('grouper.win.count', np.array([len(w[1]) for w in wins], np.int64))
    dump.put('grouper.win.sel', np.concatenate([w[1] for w in wins]).astype(np.int64))
    dump.put('grouper.win.emb', np.concatenate([w[2] for w in wins]))
    dump.put('grouper.win.probs', np.concatenate([w[3] for w in wins]))
    on, off = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64)
    pitch = meta[:, 2].round().astype(np.int64)
    groups = gcore.decode(P, (on, off, pitch.astype(np.float64)), gr.th, **gr.params)
    common.write_json(out_dir / 'result.json', dict(groups=[[int(k), np.asarray(ids).astype(int).tolist()] for k, ids in groups],
                                                    grouper_window_s=gr.window_s))
    # agreement with the oracle (MKL's FFT): same bookkeeping as replay_grouper.py
    ref = r['groups']
    la, lb = -np.ones(n, np.int64), -np.ones(n, np.int64)
    for i, (_, ids) in enumerate(ref):
        la[ids] = i
    for i, (_, ids) in enumerate(groups):
        lb[ids] = i
    both = (la >= 0) & (lb >= 0)
    M = np.zeros((max(len(ref), 1), max(len(groups), 1)))
    np.add.at(M, (la[both], lb[both]), 1)
    ri, ci = linear_sum_assignment(-M)
    mp = dict(zip(ri.tolist(), ci.tolist()))
    in_ref = int((la >= 0).sum())
    bad = []
    for i, (k, ids) in enumerate(ref):
        if len(ids) / max(in_ref, 1) >= 0.10:
            j = mp.get(i)
            other = INSTRUMENTS[groups[j][0]] if j is not None and j < len(groups) else None
            if other != INSTRUMENTS[k]:
                bad.append(dict(oracle=INSTRUMENTS[k], other=other))
    gaps = {k: float(np.abs(np.asarray(T[f'grouper.{k}'], np.float32) - P[k].astype(np.float32)).max()) for k in ('probs', 'emb', 'h')}
    res = dict(notes=n, gaps=gaps, coverage=float(both.sum()) / max(in_ref, 1), same_track=float(M[ri, ci].sum()) / max(in_ref, 1),
               big_track_name_mismatches=bad, names_ok=not bad)
    common.write_json(Path(a.captures) / name / 'grouper-exact.json', res)
    print(name, {k: round(v, 5) for k, v in gaps.items()}, 'same_track', round(res['same_track'], 4), 'names_ok', not bad, flush=True)

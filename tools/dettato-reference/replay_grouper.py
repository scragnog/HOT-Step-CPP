"""Grouper spread on identical inputs: feed the CPU capture's note features to upstream's grouper on CUDA (bfloat16).

Usage (venv-cuda): python replay_grouper.py --upstream <clone> --weights <dir> --captures <root> [--fixtures a,b,...]
Independent CPU and CUDA audio runs can produce different note rows, which mixes transcriber spread into the grouper's. Here the
CUDA grouper gets exactly the CPU run's `meta`, `feat`, `fsc` and retuned audio, so every gap is the grouper's own. Writes
<root>/<fixture>/grouper-replay.json: per-tensor max abs error against the CPU capture, and the host decode of the CUDA
predictions against the CPU groups (note ids are identical, so track agreement is exact bookkeeping)."""
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
torch = common.load_upstream(a.upstream, a.weights, 'cuda')

from dettato.grouper import core as gcore                     # noqa: E402
from dettato.grouper.instruments import INSTRUMENTS           # noqa: E402

gr = gcore.Grouper.from_pretrained('cuda')
names = a.fixtures.split(',') if a.fixtures else sorted(p.name for p in Path(a.captures).iterdir() if (p / 'cpu').is_dir())
for name in names:
    T, r = common.load_capture(Path(a.captures) / name / 'cpu')
    n = T['grouper.meta'].shape[0]
    out = dict(notes=int(n))
    if n == 0:
        out.update(gaps={}, same_track=1.0, coverage=1.0, names_ok=True)
        common.write_json(Path(a.captures) / name / 'grouper-replay.json', out)
        continue
    y = np.asarray(T['x16_retuned'] if 'x16_retuned' in T else T['x16'], np.float32)
    meta = np.asarray(T['grouper.meta'])
    song = gcore._Song(y, meta, np.asarray(T['grouper.feat']), np.asarray(T['grouper.fsc']))
    P = gr.predict(song)
    gaps = {}
    for k, v in (('probs', P['probs']), ('emb', P['emb']), ('h', P['h'])):
        gaps[k] = float(np.abs(np.asarray(T[f'grouper.{k}'], np.float32) - v.astype(np.float32)).max())
    cnt = np.array([len(w[1]) for w in P['wins']])
    same_layout = len(cnt) == len(T['grouper.win.count']) and (cnt == np.asarray(T['grouper.win.count'])).all()
    out['window_layout_equal'] = bool(same_layout)
    if same_layout:
        gaps['win.emb'] = float(np.abs(np.asarray(T['grouper.win.emb'], np.float32) - np.concatenate([w[2] for w in P['wins']]).astype(np.float32)).max())
        gaps['win.probs'] = float(np.abs(np.asarray(T['grouper.win.probs'], np.float32) - np.concatenate([w[3] for w in P['wins']]).astype(np.float32)).max())
    out['gaps'] = gaps
    on, off = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64)
    pitch = meta[:, 2].round().astype(np.int64)
    groups = gcore.decode(P, (on, off, pitch.astype(np.float64)), gr.th, **gr.params)
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
    out.update(notes_grouped_cpu=in_ref, notes_grouped_cuda=int((lb >= 0).sum()),
               coverage=float(both.sum()) / max(in_ref, 1), same_track=float(M[ri, ci].sum()) / max(in_ref, 1))
    bad = []
    for i, (k, ids) in enumerate(ref):
        if len(ids) / max(in_ref, 1) >= 0.10:
            j = mp.get(i)
            other = INSTRUMENTS[groups[j][0]] if j is not None and j < len(groups) else None
            if other != INSTRUMENTS[k]:
                bad.append(dict(oracle=INSTRUMENTS[k], other=other))
    out.update(big_track_name_mismatches=bad, names_ok=not bad)
    common.write_json(Path(a.captures) / name / 'grouper-replay.json', out)
    print(name, {k: round(v, 5) for k, v in gaps.items()}, 'same_track', round(out['same_track'], 4), 'names_ok', not bad)

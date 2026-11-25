"""Repairer spread on identical inputs: feed a CPU repairer capture's notes and evidence to upstream's repairer on CUDA.

Usage (venv-cuda): python replay_repairer.py --upstream <clone> --weights <dir> --captures <root> [--fixtures a,b,...]
Reads <root>/<fixture>/cpu-repair (capture.py --repairer). The CUDA repairer (float32: upstream has no autocast in this stage)
gets exactly the CPU run's transcriber notes and evidence rows, so every gap is the repairer's own. Writes
<root>/<fixture>/repairer-replay.json: max abs error of P(keep) and the insertion scores, how many onsets and ends moved to
another bin, and how many repaired notes differ."""
import argparse
from pathlib import Path

import numpy as np

import common

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--captures', required=True)
ap.add_argument('--fixtures')
a = ap.parse_args()
torch = common.load_upstream(a.upstream, a.weights, 'cuda')

from dettato.evidence.repair import EvidenceRepairer, input_notes  # noqa: E402

rp = EvidenceRepairer.from_pretrained('cuda')
names = a.fixtures.split(',') if a.fixtures else sorted(p.name for p in Path(a.captures).iterdir() if (p / 'cpu-repair').is_dir())
for name in names:
    T, r = common.load_capture(Path(a.captures) / name / 'cpu-repair')
    res = {k: np.asarray(v) for k, v in r['repair']['input_notes'].items()}
    k, o = input_notes(res)
    pk, on_r, off_r, ins = rp.edits(k, np.asarray(T['repair.ev'], np.float32)[o])
    ran = np.asarray(T['repair.window_ran'], bool)
    same_ran = bool(((~np.isnan(pk)) == ran).all())
    out: dict = dict(notes=int(len(k)), same_windows=same_ran)
    if same_ran and ran.any():
        out['pk_max_abs'] = float(np.abs(pk[ran] - np.asarray(T['repair.pk'])[ran]).max())
        out['onsets_moved'] = int((np.abs(on_r[ran] - np.asarray(T['repair.on_r'])[ran]) > 1e-4).sum())
        out['ends_moved'] = int((np.abs(off_r[ran] - np.asarray(T['repair.off_r'])[ran]) > 1e-4).sum())
    ref_ins = r['repair']['insertions']
    if len(ref_ins) == len(ins):
        x, y = np.asarray(ref_ins, np.float64).reshape(-1, 4), np.asarray(ins, np.float64).reshape(-1, 4)
        out['insertion_score_max_abs'] = float(np.abs(x[:, 3] - y[:, 3]).max()) if len(x) else 0.0
        out['insertions_placed_differently'] = int(((np.abs(x[:, :2] - y[:, :2]) > 1e-4).any(1) | (x[:, 2] != y[:, 2])).sum())
    out['insertion_proposals'] = [len(ref_ins), len(ins)]
    rep = rp.apply(k, (pk, on_r, off_r, ins))
    ref = r['repair']['output']
    same = len(ref['onset']) == len(rep['onset']) and all(
        a_ == b_ for a_, b_ in zip(zip(ref['onset'], ref['offset'], ref['pitch']), zip(rep['onset'], rep['offset'], rep['pitch'])))
    out.update(output_notes=[len(ref['onset']), int(len(rep['onset']))], output_identical=bool(same))
    common.write_json(Path(a.captures) / name / 'repairer-replay.json', out)
    print(name, out)

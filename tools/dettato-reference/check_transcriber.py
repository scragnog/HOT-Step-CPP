"""Gate the transcriber port against the CPU oracle with compare.py's metrics and tolerances.json.

Usage: python check_transcriber.py --captures <oracle root> --port <port root> --run <name> [--gpu]
       [--tap-steps k] [--spread spread.json] [--json out.json]

<port root>/<fixture>/<run> is a capture written by
  ace-midi --dettato-model <weights>/dettato (--dettato-fed <oracle capture> | --transcribe <wav>) --dettato-capture <dir>
and <oracle root>/<fixture>/cpu the matching oracle capture. Gates (tolerances.json):
  fed tensors   maps.* and sub.maps.* max abs < 1e-3; sub.tap.* |a - b| <= 1e-3 + k 2^-10 |a| elementwise (k = 1 as
                declared; --tap-steps sets k). Only for a fed capture (result.json "input": "fed"): a capture from
                audio went through the port's own resampler and tuning, so its tensors are reported, not gated
  notes         F1 >= 0.99, median onset error <= 1 ms, offsets within 20 ms >= 99%, unmatched oracle onset prob <= 0.35
  seconds       |a - b| < 1e-6
  non-finite    fails
--gpu: each gate is the looser of the CPU gate and upstream's own CPU/CUDA spread on that fixture (spread.json).
The taps only enter when the port capture has them. Exit 1 when any gate fails."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import load_capture
from compare import tensor_gap, transcriber_notes

HEADS = ('onset', 'frame', 'offset', 'onset2', 'frame2', 'offset2')

ap = argparse.ArgumentParser()
ap.add_argument('--captures', required=True)
ap.add_argument('--port', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--gpu', action='store_true')
ap.add_argument('--spread', default=str(Path(__file__).with_name('spread.json')))
ap.add_argument('--json', help='write the per-fixture metrics here')
ap.add_argument('--tap-steps', type=float, default=1.0, help='float16 rounding steps the tap gate allows (declared: 1)')
a = ap.parse_args()
spread = json.loads(Path(a.spread).read_text(encoding='utf-8'))

failed, report = False, {}
for fx_dir in sorted(Path(a.port).iterdir()):
    port_dir, ref_dir = fx_dir / a.run, Path(a.captures) / fx_dir.name / 'cpu'
    if not (port_dir / 'result.json').exists() or not (ref_dir / 'result.json').exists():
        continue
    (A, ra), (B, rb) = load_capture(ref_dir), load_capture(port_dir)
    sp = spread.get(fx_dir.name, {})
    rows, ok = [], True
    fed = rb.get('input') == 'fed'

    def gate(name, value, limit, good, tensor=False):
        global ok
        passed = bool(good)
        if tensor and not fed:
            name += ' [info: audio input]'
        else:
            ok &= passed
        rows.append((name, value, limit, passed))

    for group, prefix in (('maps', 'maps.'), ('sub_maps', 'sub.maps.')):
        names = [prefix + h for h in HEADS]
        gaps = tensor_gap(A, B, names)
        for n in names:
            g = gaps[n]
            if 'max_abs' not in g:
                gate(n, 'shape ' + str(g.get('shape_mismatch')), '-', False)
                continue
            lim = 1e-3
            if a.gpu:
                lim = max(lim, sp.get(group, {}).get(n, {}).get('max_abs', 0.0))
            finite = bool(np.isfinite(np.asarray(B[n], np.float32)).all())
            gate(n, g['max_abs'], lim, finite and (g['max_abs'] < lim if not a.gpu else g['max_abs'] <= lim), True)
    for n in ('sub.tap.hidden', 'sub.tap.stem'):
        if n not in B:
            continue
        x, y = np.asarray(A[n], np.float32), np.asarray(B[n], np.float32)
        if x.shape != y.shape:
            gate(n, f'shape {list(y.shape)} vs {list(x.shape)}', '-', False, True)
            continue
        d = np.abs(x - y)
        excess = d - (1e-3 + a.tap_steps * 2.0 ** -10 * np.abs(x))
        worst = float(excess.max()) if d.size else 0.0
        n_over = int((excess > 0).sum())
        mx = float(d.max()) if d.size else 0.0
        big = np.abs(x) >= 1.0                # ulp distance where the relative term, not the 1e-3 floor, binds
        ulps = int(np.abs(np.asarray(A[n]).view(np.int16).astype(np.int32) - np.asarray(B[n]).view(np.int16).astype(np.int32))[big & (np.sign(x) == np.sign(y))].max()) if big.any() else 0
        good = np.isfinite(y).all() and worst <= 0
        if a.gpu and not good:
            good = np.isfinite(y).all() and mx <= sp.get('taps', {}).get(n, {}).get('max_abs', 0.0)
        gate(n + f' (max abs; ulps; elems over 1e-3 + {a.tap_steps:g} 2^-10|ref|)', f'{mx:.3g}; {ulps} ulp; {n_over}/{d.size}',
             'none over', good, True)
    tn = transcriber_notes(ra, rb)
    st = sp.get('transcriber_notes', {})
    lo = lambda g, k: min(g, st[k]) if a.gpu and k in st else g
    hi = lambda g, k: max(g, st[k]) if a.gpu and k in st else g
    gate('notes F1', tn['f1'], lo(0.99, 'f1'), tn['f1'] >= lo(0.99, 'f1'))
    if 'median_onset_err_ms' in tn:
        gate('median onset err ms', tn['median_onset_err_ms'], hi(1.0, 'median_onset_err_ms'),
             tn['median_onset_err_ms'] <= hi(1.0, 'median_onset_err_ms'))
        gate('offsets within 20 ms', tn['offsets_within_20ms'], lo(0.99, 'offsets_within_20ms'),
             tn['offsets_within_20ms'] >= lo(0.99, 'offsets_within_20ms'))
    gate('max unmatched oracle onset prob', tn['max_unmatched_oracle_onset_prob'], 0.35, tn['max_unmatched_oracle_onset_prob'] <= 0.35)
    gate('seconds', abs(ra['seconds'] - rb['seconds']), 1e-6, abs(ra['seconds'] - rb['seconds']) < 1e-6)
    tu_a, tu_b = ra['tuning'], rb['tuning']
    gate('retuned (same decision)', f"{tu_b['estimate_cents']:+.3f} vs {tu_a['estimate_cents']:+.3f} cents",
         'same', bool(tu_a['shift_cents']) == bool(tu_b['shift_cents']))
    print(f"{'PASS' if ok else 'FAIL'} {fx_dir.name} ({a.run}, {'gpu' if a.gpu else 'cpu'} gates): "
          f"{tn['oracle_notes']} oracle notes, {tn['other_notes']} port, {tn['matched']} matched")
    for name, v, lim, p in rows:
        vs = f'{v:.6g}' if isinstance(v, float) else str(v)
        ls = f'{lim:.6g}' if isinstance(lim, float) else str(lim)
        if not p or name.startswith('notes') or name.startswith('sub.tap') or name == 'maps.onset':
            print(f"   {'ok ' if p else 'BAD'} {name:<58} {vs:>24}  limit {ls}")
    report[fx_dir.name] = dict(passed=ok, gates=[dict(name=n, value=v, limit=l, passed=p) for n, v, l, p in rows], notes=tn,
                               timing=rb.get('timing'))
    failed |= not ok
if not report:
    sys.exit('no port captures found')
if a.json:
    Path(a.json).write_text(json.dumps(report, indent=1, default=str) + '\n', encoding='utf-8')
print('ALL PASS' if not failed else 'SOME FAIL', f'({len(report)} fixtures)')
sys.exit(1 if failed else 0)

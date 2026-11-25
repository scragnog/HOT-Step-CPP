"""Stage-by-stage comparison of two captures of one fixture, with the gate metrics from tolerances.json.

Usage:
  python compare.py <oracle capture dir> <other capture dir>              one fixture, JSON to stdout
  python compare.py --captures <root> --a cpu --b cuda --out spread.json  every fixture under <root>/<name>/<a|b>

With cpu and cuda this measures the upstream spread: the gap between upstream's own float32 CPU run and its bfloat16 CUDA run.
A port is judged with the same metrics, oracle = the cpu capture. Pitched notes are matched one to one by key (pitch, and the
voice slot for the transcriber's notes); a pair is a match when the onsets differ by at most ONSET_TOL."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import load_capture, write_json

ONSET_TOL = 0.010          # s, half a 20 ms frame
OFFSET_TOL = 0.020
FRAME = 0.02
BIG_TRACK = 0.10           # a track holding this share of the song's notes must keep its instrument name


def match(a_on, a_key, b_on, b_key, tol=ONSET_TOL):
    """One-to-one nearest match by key. -> (index pairs (ia, ib)) sorted by ia."""
    pairs = []
    for key in np.unique(a_key):
        ia = np.where(a_key == key)[0]
        ib = np.where(b_key == key)[0]
        if not len(ib):
            continue
        ia = ia[np.argsort(a_on[ia])]
        ib = ib[np.argsort(b_on[ib])]
        used = np.zeros(len(ib), bool)
        bo = b_on[ib]
        for i in ia:
            j = int(np.searchsorted(bo, a_on[i]))
            best, bd = None, tol + 1e-9
            for c in (j - 1, j, j + 1):
                if 0 <= c < len(ib) and not used[c] and abs(bo[c] - a_on[i]) < bd:
                    best, bd = c, abs(bo[c] - a_on[i])
            if best is not None:
                used[best] = True
                pairs.append((i, int(ib[best])))
    pairs.sort()
    return np.array(pairs, np.int64).reshape(-1, 2)


def f1(n_match, n_a, n_b):
    if n_a == 0 and n_b == 0:
        return 1.0
    return 2.0 * n_match / max(n_a + n_b, 1)


def tensor_gap(A, B, names):
    out = {}
    for n in names:
        if n in A and n in B and A[n].shape == B[n].shape:
            d = np.abs(np.asarray(A[n], np.float32) - np.asarray(B[n], np.float32))
            out[n] = dict(max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0)
        else:
            out[n] = dict(shape_mismatch=[list(A[n].shape) if n in A else None, list(B[n].shape) if n in B else None])
    return out


def transcriber_notes(ra, rb):
    na, nb = ra['transcriber_notes'], rb['transcriber_notes']
    a_on, b_on = np.array(na['onset']), np.array(nb['onset'])
    a_key = np.array(na['pitch'], np.int64) * 4 + np.array(na['slot'], np.int64)
    b_key = np.array(nb['pitch'], np.int64) * 4 + np.array(nb['slot'], np.int64)
    p = match(a_on, a_key, b_on, b_key)
    out = dict(oracle_notes=len(a_on), other_notes=len(b_on), matched=len(p), f1=f1(len(p), len(a_on), len(b_on)))
    if len(p):
        on_err = np.abs(a_on[p[:, 0]] - b_on[p[:, 1]])
        off_err = np.abs(np.array(na['offset'])[p[:, 0]] - np.array(nb['offset'])[p[:, 1]])
        out.update(median_onset_err_ms=float(np.median(on_err) * 1000), max_onset_err_ms=float(on_err.max() * 1000),
                   offsets_within_20ms=float((off_err <= OFFSET_TOL + 1e-9).mean()))
    seen = np.zeros(len(a_on), bool)
    seen[p[:, 0]] = True
    conf = np.array(na['conf'])
    out['max_unmatched_oracle_onset_prob'] = float(conf[~seen].max()) if (~seen).any() else 0.0
    return out


def pitched(r):
    rows = [(on, off, pitch, vel, ti, t['name']) for ti, t in enumerate(r['tracks']) if not t['is_drum']
            for on, off, pitch, vel in t['notes']]
    return rows


def grouping(ra, rb):
    A, B = pitched(ra), pitched(rb)
    out = dict(oracle_notes=len(A), other_notes=len(B))
    if not A or not B:
        out.update(matched=0, same_track=1.0 if not A and not B else 0.0, names_ok=not A and not B)
        return out
    a_on, b_on = np.array([r[0] for r in A]), np.array([r[0] for r in B])
    a_p, b_p = np.array([r[2] for r in A], np.int64), np.array([r[2] for r in B], np.int64)
    p = match(a_on, a_p, b_on, b_p)
    ta, tb = np.array([r[4] for r in A]), np.array([r[4] for r in B])
    n_ta, n_tb = ta.max() + 1, tb.max() + 1
    M = np.zeros((n_ta, n_tb))
    np.add.at(M, (ta[p[:, 0]], tb[p[:, 1]]), 1)
    ri, ci = linear_sum_assignment(-M)
    mapping = dict(zip(ri.tolist(), ci.tolist()))
    same = float(M[ri, ci].sum()) / max(len(p), 1)
    big = [t for t in range(n_ta) if (ta == t).mean() >= BIG_TRACK]
    names_a = {t: ra['tracks'][t]['name'] for t in range(n_ta)}
    names_b = {t: rb['tracks'][t]['name'] for t in range(n_tb)}
    bad = [dict(oracle=names_a[t], other=names_b.get(mapping.get(t))) for t in big if names_b.get(mapping.get(t)) != names_a[t]]
    vel = np.abs(np.array([A[i][3] for i in p[:, 0]]) - np.array([B[j][3] for j in p[:, 1]])) if len(p) else np.zeros(0)
    out.update(matched=len(p), f1=f1(len(p), len(A), len(B)), same_track=same, same_track_of_oracle=float(M[ri, ci].sum()) / len(A),
               big_track_name_mismatches=bad, names_ok=not bad,
               velocity_within_2=float((vel <= 2).mean()) if len(vel) else 1.0, max_velocity_diff=float(vel.max()) if len(vel) else 0.0)
    return out


def drums(ra, rb):
    da = next((t for t in ra['tracks'] if t['is_drum']), None)
    db = next((t for t in rb['tracks'] if t['is_drum']), None)
    out = dict(oracle_router=ra['drum_router'], other_router=rb['drum_router'],
               router_diff=None if ra['drum_router'] is None or rb['drum_router'] is None else abs(ra['drum_router'] - rb['drum_router']),
               same_model=(ra['drum_decode'] or {}).get('model') == (rb['drum_decode'] or {}).get('model'),
               skipped=[isinstance(ra['drum_info'], str), isinstance(rb['drum_info'], str)])
    A = np.array(da['notes']).reshape(-1, 4) if da else np.zeros((0, 4))
    B = np.array(db['notes']).reshape(-1, 4) if db else np.zeros((0, 4))
    fa, fb = np.round(A[:, 0] / FRAME).astype(np.int64), np.round(B[:, 0] / FRAME).astype(np.int64)
    p = match(fa.astype(float), A[:, 2].astype(np.int64), fb.astype(float), B[:, 2].astype(np.int64), tol=0.0)
    out.update(oracle_hits=len(A), other_hits=len(B), matched=len(p), f1=f1(len(p), len(A), len(B)))
    if len(p):
        v = np.abs(A[p[:, 0], 3] - B[p[:, 1], 3])
        out.update(velocity_within_2=float((v <= 2).mean()), max_velocity_diff=float(v.max()))
    return out


def compare(dir_a, dir_b):
    (A, ra), (B, rb) = load_capture(dir_a), load_capture(dir_b)
    out = dict(oracle=str(dir_a), other=str(dir_b), seconds=ra['seconds'],
               tuning_cents=[ra['tuning']['estimate_cents'], rb['tuning']['estimate_cents']],
               retuned=[bool(ra['tuning']['shift_cents']), bool(rb['tuning']['shift_cents'])])
    heads = ['onset', 'frame', 'offset', 'onset2', 'frame2', 'offset2']
    out['maps'] = tensor_gap(A, B, [f'maps.{h}' for h in heads])
    out['sub_maps'] = tensor_gap(A, B, [f'sub.maps.{h}' for h in heads])
    out['taps'] = tensor_gap(A, B, ['sub.tap.hidden', 'sub.tap.stem'])
    out['drum_probs'] = tensor_gap(A, B, [f'drums.{k}' for k in ('kit', 'kit_vel', 'drum', 'timpani', 'vel', 'router')])
    out['transcriber_notes'] = transcriber_notes(ra, rb)
    out['grouping'] = grouping(ra, rb)
    out['drums'] = drums(ra, rb)
    return out


def worst(result):
    """Worst case over all fixtures for each gate: the upstream spread a GPU port may not exceed."""
    def col(f, default=None):
        v = [x for x in (f(d) for d in result.values()) if x is not None]
        return v
    mx = lambda v: max(v) if v else None
    mn = lambda v: min(v) if v else None
    return dict(
        map_max_abs=mx(col(lambda d: max(v.get('max_abs', 0) for v in d['maps'].values()))),
        sub_map_max_abs=mx(col(lambda d: max(v.get('max_abs', 0) for v in d['sub_maps'].values()))),
        tap_max_abs=mx(col(lambda d: max(v.get('max_abs', 0) for v in d['taps'].values()))),
        note_f1_min=mn(col(lambda d: d['transcriber_notes']['f1'])),
        median_onset_err_ms_max=mx(col(lambda d: d['transcriber_notes'].get('median_onset_err_ms'))),
        offsets_within_20ms_min=mn(col(lambda d: d['transcriber_notes'].get('offsets_within_20ms'))),
        unmatched_oracle_onset_prob_max=mx(col(lambda d: d['transcriber_notes']['max_unmatched_oracle_onset_prob'])),
        pitched_velocity_within_2_min=mn(col(lambda d: d['grouping'].get('velocity_within_2'))),
        same_track_min=mn(col(lambda d: d['grouping'].get('same_track'))),
        fixtures_failing_name_check=[k for k, d in result.items() if not d['grouping']['names_ok']],
        drum_router_diff_max=mx(col(lambda d: d['drums']['router_diff'])),
        drum_hit_f1_min=mn(col(lambda d: d['drums']['f1'] if not any(d['drums']['skipped']) else None)),
        drum_velocity_within_2_min=mn(col(lambda d: d['drums'].get('velocity_within_2'))),
        drum_model_differs=[k for k, d in result.items() if not d['drums']['same_model']])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--captures')
    ap.add_argument('--a', default='cpu')
    ap.add_argument('--b', default='cuda')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.captures:
        root = Path(a.captures)
        result = {d.name: compare(d / a.a, d / a.b) for d in sorted(root.iterdir()) if (d / a.a / 'result.json').exists()
                  and (d / a.b / 'result.json').exists()}
    else:
        result = compare(*a.dirs)
    if a.captures:
        result = {'_worst': worst(result), **result}
    if a.out:
        write_json(a.out, result)
    else:
        import json
        print(json.dumps(result, indent=1))

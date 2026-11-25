"""Check a set of captures: hashes, finite values, shapes, grouping coverage, and the weights and fixtures they were made from.

Usage: python verify.py --captures <root> --weights <dir> --fixtures <dir>
Exits non-zero on the first failing capture."""
import argparse
from pathlib import Path
import sys

import numpy as np

from common import digest, load_capture, load_tensors
import json

ap = argparse.ArgumentParser()
ap.add_argument('--captures', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--fixtures', required=True)
a = ap.parse_args()
bad = []


def check(cond, msg):
    if not cond:
        bad.append(msg)


wj = json.loads((Path(a.weights) / 'weights.json').read_text())
for repo, files in wj.items():
    for f in ('config.json', 'model.safetensors'):
        check(digest(Path(a.weights) / repo / f) == files[f]['sha256'], f'weights {repo}/{f}')
fj = json.loads((Path(a.fixtures) / 'fixtures.json').read_text())
for name, e in fj.items():
    check(digest(Path(a.fixtures) / f'{name}.wav') == e['sha256'], f'fixture {name}')

n_checked = 0
for d in sorted(Path(a.captures).glob('*/*')):
    if not (d / 'result.json').exists():
        continue
    tag = f'{d.parent.name}/{d.name}'
    entries = json.loads((d / 'tensors.json').read_text())
    for k, e in entries.items():
        check(digest(d / e['file']) == e['sha256'], f'{tag}: hash {k}')
    T, r = load_capture(d)
    for k, v in T.items():
        if v.dtype.kind == 'f':
            check(np.isfinite(np.asarray(v)).all(), f'{tag}: non-finite {k}')
    heads = ('onset', 'frame', 'offset', 'onset2', 'frame2', 'offset2')
    frames = {T[f'maps.{h}'].shape[0] for h in heads} | {T[f'sub.maps.{h}'].shape[0] for h in heads}
    check(len(frames) == 1, f'{tag}: map lengths differ {frames}')
    check(r['upstream_commit'] == '51ade0f09f2272f4674166026582185d26baaeaf', f'{tag}: commit')
    check(r['repairer'] == ('on' if d.name.endswith('-repair') else 'bypassed'), f'{tag}: repairer')
    if r['repairer'] == 'on':
        n = len(r['repair']['input_notes']['onset'])
        check(T['repair.ev'].shape == (n, 769) and T['repair.hidden'].shape[1:] == (88, 368), f'{tag}: repair shapes')
        check(len(r['repair']['output']['onset']) == r['info']['notes']['transcribed'] - r['info']['notes']['dropped'] + r['info']['notes']['inserted'], f'{tag}: repair counts')
    if 'grouper.meta' in T:
        n = T['grouper.meta'].shape[0]
        check(T['grouper.feat'].shape[0] == n == T['grouper.fsc'].shape[0], f'{tag}: grouper rows')
        check(n == 0 or T['grouper.probs'].shape[0] == n, f'{tag}: grouper probs rows')   # predict() does not run without notes
        ids = sorted(i for _, g in r['groups'] for i in g)
        # upstream leaves a note in no group when the song is too short for the decode (clip-3s: 4 of 10); a note is never in two
        check(len(ids) == len(set(ids)) and all(0 <= i < n for i in ids), f'{tag}: a note is in two groups or out of range')
        pitched = sum(len(t['notes']) for t in r['tracks'] if not t['is_drum'])
        check(pitched == r['info']['notes']['grouped'], f'{tag}: grouped count')
    else:
        check(not r['groups'], f'{tag}: groups without notes')
    if r['drum_hits'] is not None:
        dt = [t for t in r['tracks'] if t['is_drum']]
        check(len(dt[0]['notes'] if dt else []) == len(r['drum_hits']), f'{tag}: drum hits vs track')
    check(all(t['notes'] for t in r['tracks']), f'{tag}: empty track')
    n_checked += 1
print(f'{n_checked} captures checked')
if bad:
    print('FAIL', *bad, sep='\n  ')
    sys.exit(1)
print('PASS')

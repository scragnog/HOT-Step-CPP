"""Build the fixture WAVs from songs our own engine rendered (no third-party audio).

Usage: python fixtures.py --songs <dir with t1..t6 *0.wav> --out <fixtures dir>
The six songs are copied as they are. Five more are derived from them:
  silence-10s   10 s of digital silence (44.1 kHz)
  clip-3s       3 s cut from the piano song, starting at 20 s
  long-260s     piano + jazz-trio + folk songs joined end to end, cut to 260 s (a song over 250 s: the host-buffer path)
  src-32k       the funk-drums song resampled to 32 kHz (the drum model must skip: it needs a source of >= 40 kHz)
  detune-30c    the piano song played 30 cents fast (the retune path; capture.py records the tuning estimate)
All files are 16-bit PCM WAV. fixtures.json records sizes and hashes."""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from common import digest, write_json

ap = argparse.ArgumentParser()
ap.add_argument('--songs', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

songs = {}
for p in sorted(Path(a.songs).glob('t[1-6]-*0.wav')):
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    songs[p.name.rsplit('0.wav', 1)[0]] = (x.mean(1), sr)
if len(songs) != 6:
    raise SystemExit(f'expected six songs, found {sorted(songs)}')
names = sorted(songs)
by = {n.split('-', 1)[0]: n for n in names}

fixtures = {n: songs[n] for n in names}
piano, sr = songs[by['t3']]
fixtures['silence-10s'] = (np.zeros(10 * sr, np.float32), sr)
fixtures['clip-3s'] = (piano[20 * sr:23 * sr].copy(), sr)
long = np.concatenate([songs[by[k]][0] for k in ('t3', 't4', 't5')])[:260 * sr]
fixtures['long-260s'] = (long, sr)
funk, fsr = songs[by['t2']]
fixtures['src-32k'] = (soxr.resample(funk, fsr, 32000, quality='HQ').astype(np.float32), 32000)
r = 2.0 ** (30.0 / 1200.0)                       # played r times faster: pitch x r, 30 cents sharp
fixtures['detune-30c'] = (soxr.resample(piano, sr, sr / r, quality='HQ').astype(np.float32), sr)

report = {}
for name, (x, rate) in fixtures.items():
    path = out / f'{name}.wav'
    sf.write(path, np.clip(x, -1, 1), rate, subtype='PCM_16')
    report[name] = dict(seconds=round(len(x) / rate, 3), sample_rate=rate, sha256=digest(path))
write_json(out / 'fixtures.json', report)
for k, v in report.items():
    print(k, v['seconds'], v['sample_rate'])

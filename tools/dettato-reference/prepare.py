"""Copy the four upstream model repos unmodified into one DETTATO_WEIGHTS directory and record their sha256.

Usage: python prepare.py --out <weights dir> [--cache <huggingface hub dir>]
Downloads nothing itself: it reads the snapshots huggingface_hub already holds (run `hf download kobimusic/dettato` etc. first,
or let the first upstream run fetch them). Files are copied byte for byte, so the hashes equal upstream's."""
import argparse
from pathlib import Path
import shutil

from common import REPOS, digest, write_json

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--cache', default=str(Path.home() / '.cache/huggingface/hub'))
a = ap.parse_args()
out = Path(a.out)
manifest = {}
for stage, name in REPOS.items():
    snaps = sorted((Path(a.cache) / f'models--kobimusic--{name}' / 'snapshots').iterdir())
    if len(snaps) != 1:
        raise SystemExit(f'{name}: expected one cached snapshot, found {len(snaps)}')
    manifest[name] = {'revision': snaps[0].name}
    for f in ('config.json', 'model.safetensors'):
        dst = out / name / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snaps[0] / f, dst)
        manifest[name][f] = {'sha256': digest(dst), 'bytes': dst.stat().st_size}
write_json(out / 'weights.json', manifest)
print(Path(out / 'weights.json').read_text())

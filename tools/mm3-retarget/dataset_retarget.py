# tools/mm3-retarget/dataset_retarget.py — turn a dataset's over-length tracks into a trainable derived manifest
#
# Runs retarget.py over every track in a dataset that exceeds MM3's 6:00 cap, then writes a DERIVED dataset.json
# beside the edits. The derived manifest is the whole integration: `ace-train mm3-codes` resolves audio through each
# sample's `audio_path`, and `ace-train mm3-lm-train` reads `lyrics` from the same file, so pointing both stages at
# the derived manifest picks up the edited audio and the edited lyrics with NO engine change.
#
# `filename` is deliberately left alone — the trainer resolves the MM3 caption as <captions_dir>/<stem>.mm3.txt off
# that field, and the caption still describes the song.
#
# Tracks the tool refuses (no clean repeat to skip) are copied through untouched, so --drop-over-frames still drops
# them and the run is no worse off than today.
#
# Usage:
#   py -3.13 tools/mm3-retarget/dataset_retarget.py <dataset.json> [--out <dir>] [--target 360] [retarget args...]
#     --dry     list what would be edited and stop
#   Anything not recognised here is passed through to retarget.py (--protect, --max-cost, --cpu, --no-vocal, ...).
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
if not argv:
    print(__doc__)
    sys.exit(2)
DS = argv[0]
DRY = '--dry' in argv
TARGET = float(argv[argv.index('--target') + 1]) if '--target' in argv else 360.0
OUT = argv[argv.index('--out') + 1] if '--out' in argv else \
    os.path.join(os.path.dirname(os.path.abspath(DS)), '_mm3-edits')
EATEN = ('--out', '--target')          # consumed here; everything else belongs to retarget.py
passthru = [a for i, a in enumerate(argv[1:], start=1)
            if a != '--dry' and a not in EATEN and argv[i - 1] not in EATEN]

ds = json.load(open(DS, encoding='utf-8'))
samples = ds['samples']
over = [s for s in samples if float(s.get('duration') or 0) > TARGET]
print(f'{len(samples)} samples, {len(over)} over {TARGET:.0f}s')
for s in over:
    print(f"  {float(s['duration'])/60:5.2f} min  {s['filename']}")
if DRY or not over:
    sys.exit(0)

os.makedirs(OUT, exist_ok=True)
missing = [s for s in over if not os.path.exists(s['audio_path'])]
if missing:
    print(f'{len(missing)} source file(s) missing; aborting rather than half-building a manifest')
    for s in missing:
        print('  ' + s['audio_path'])
    sys.exit(1)

cmd = [sys.executable, os.path.join(HERE, 'retarget.py')] + [s['audio_path'] for s in over] + \
      ['--target', str(TARGET)] + passthru
env = dict(os.environ, MM3_RETARGET_OUT=OUT)
print('\n' + ' '.join(cmd[:3]) + f' ... ({len(over)} files)\n')
rc = subprocess.run(cmd, env=env).returncode
if rc != 0:
    print(f'retarget.py exited {rc}')
    sys.exit(rc)

report = {r['file']: r for r in json.load(open(os.path.join(OUT, 'report.json'), encoding='utf-8'))}
edited = skipped = 0
out_samples = []
for s in samples:
    r = report.get(s['audio_path'])
    base = os.path.splitext(os.path.basename(s['audio_path']))[0]
    wav = os.path.join(OUT, base + '.edit.wav')
    if not r or r.get('status') != 'ok' or not os.path.exists(wav):
        if float(s.get('duration') or 0) > TARGET:
            skipped += 1
            why = 'no candidate' if not r else r.get('status', 'not rendered')
            print(f"  KEPT AS-IS {s['filename']}: {why} (still dropped by --drop-over-frames)")
        out_samples.append(s)
        continue
    n = dict(s)
    n['audio_path'] = os.path.abspath(wav)
    n['duration'] = r['edited']
    txt = os.path.join(OUT, base + '.edit.txt')
    if os.path.exists(txt):
        raw = open(txt, encoding='utf-8').read().splitlines()
        for i, line in enumerate(raw):
            if line.strip().lower().startswith('lyrics:'):
                body = '\n'.join(raw[i + 1:]).strip('\n')
                n['lyrics'] = body
                # formatted_lyrics is what the dataset builder derived from the same text; leaving the old copy in
                # place would keep the removed lines visible to anything that reads it instead of `lyrics`.
                if s.get('formatted_lyrics'):
                    n['formatted_lyrics'] = body
                break
    n['retarget'] = dict(source=s['audio_path'], removed=r['best']['removal'],
                         at=[r['best']['t_a'], r['best']['t_b']], cost=r['best']['context_cost'])
    out_samples.append(n)
    edited += 1

derived = dict(ds)
derived['samples'] = out_samples
derived['metadata'] = dict(ds.get('metadata') or {}, retargeted_from=os.path.abspath(DS), retargeted_edits=edited)
dst = os.path.join(OUT, 'dataset.json')
json.dump(derived, open(dst, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f'\n{edited} track(s) edited, {skipped} left for --drop-over-frames')
print(f'derived manifest: {dst}')
print('\nnext:  ace-train mm3-codes    --dataset "%s" ...' % dst)
print('       ace-train mm3-lm-train --manifest "%s" --captions <original dataset dir> ...' % dst)

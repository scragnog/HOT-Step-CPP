"""Build a shareable receipt without private source paths or audio."""
import argparse
import json
from pathlib import Path
from common import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    exp = args.experiment
    read = lambda path: json.loads(path.read_text())
    clips = read(exp / 'clips-v2/clips.json')
    mapping = [('library-vocal', 'v2-vocal'), ('library-vocal-silent-stretch', 'v2-silent'),
               ('library-instrumental', 'v2-instrumental'), ('library-vocal-stem', 'v2-vocal-stem')]
    receipt = dict(schema_version=1, status='ready-for-review',
        artifact_root='_experiments/lead-sheet-reference',
        provenance=read(exp / 'weights/provenance.json'),
        tolerance_file='tolerances.json', inventory_file='inventory.json', captures=[])
    for alias, directory in mapping:
        clip = next(c for c in clips if c['alias'] == alias)
        capture = exp / 'captures' / directory
        run = read(capture / 'manifest.json')
        assert run['input_sha256'] == clip['sha256']
        verification = read(capture / 'verification.json')
        receipt['captures'].append(dict(alias=alias,
            input_sha256=clip['sha256'], source_sha256=clip['source_sha256'],
            start_seconds=clip['start_seconds'], duration_seconds=clip['duration_seconds'],
            capture_directory='captures/' + directory, status=run['status'],
            bpm=run['librosa_bpm'], tuning_bins=run.get('tuning_bins'),
            timings_seconds=run['timings_seconds'], wall_seconds=read(capture/'run.json')['elapsed_seconds'],
            skips=run['skips'], verification=verification,
            files=[dict(path=str(f.relative_to(exp)).replace('\\', '/'),
                        bytes=f.stat().st_size, sha256=digest(f))
                   for f in sorted(capture.rglob('*')) if f.is_file()]))
    receipt['baseline'] = read(exp / 'captures/v2-vocal/baseline-check.json')
    receipt['initial_candidate'] = dict(directory='captures/library-vocal',
        status='rejected-zero-bpm', note='Preserved, excluded from accepted fixture set')
    receipt['limitations'] = [
        'Instrumental source label and vocal-stem provenance are fixture metadata; human listening not performed.',
        'Vocal stem yielded zero program-100 notes, so upstream rejected rolls/MIDI; model outputs retained.',
        'Seconds variant preserves event timings, not a dense grid with an invented seconds-to-step mapping.',
        'HOTSTEP-968 had no clip list; these hashed clips are the proposed shared comparison inputs.',
        'tools/* ignore rule prevents ordinary commit until the owning agent allows this directory.'
    ]
    receipt['codegraph'] = ['query ace-midi-validate: found engine/tools/ace-midi-validate.py',
                            'context reference tensor dumps audio transcription: index busy; file-read fallback']
    write_json(Path(__file__).parent / 'manifest.json', receipt)
    print('Manifest written:', len(receipt['captures']), 'clips,',
          sum(c['verification']['tensors'] for c in receipt['captures']), 'verified tensors')


if __name__ == '__main__':
    main()

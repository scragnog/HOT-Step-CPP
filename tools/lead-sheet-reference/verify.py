"""Verify dump integrity, capture completeness, averaging and event coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from common import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture', type=Path)
    args = p.parse_args()
    root = args.capture
    manifest = json.loads((root / 'manifest.json').read_text())
    entries = json.loads((root / 'tensors.json').read_text())
    arrays = {}
    for name, entry in entries.items():
        path = root / entry['file']
        assert digest(path) == entry['sha256'], name
        assert path.stat().st_size == entry['bytes'], name
        array = np.fromfile(path, '<f4').reshape(entry['shape'])
        assert np.isfinite(array).all(), name
        arrays[name] = array
    for name in ('waveform-padded', 'linear-spectrogram', 'pre-encoder', 'encoder',
                 'pre-decoder', 'prefill.logits', 'cached-step.logits'):
        assert 'mt3.b0.' + name in arrays, name
    assert arrays['mt3.b0.waveform-padded'].shape[-1] == 32767
    assert arrays['mt3.b0.prefill.logits'].shape[1] == 13
    assert arrays['mt3.b0.cached-step.logits'].shape[1] == 13
    if manifest['librosa_bpm'] > 0:
        assert 'tuning_bins' in manifest
        for fold in range(5):
            assert f'chord.fold{fold}.cnn' in arrays
            assert f'chord.fold{fold}.lstm' in arrays
            np.testing.assert_array_equal(arrays[f'chord.fold{fold}.crop-18-270'][0],
                                          arrays['chord.cqt-magnitude'][:, 18:270])
        for head in range(6):
            mean = np.mean([arrays[f'chord.fold{fold}.head{head}'] for fold in range(5)], axis=0)
            np.testing.assert_array_equal(mean, arrays[f'chord.mean.head{head}'])
            np.testing.assert_allclose(mean.sum(-1), 1, atol=1e-6, rtol=0)
    notes = json.loads((root / 'merged-notes-seconds.json').read_text())
    vocals = [n for n in notes if n['program'] == 100 and not n['is_drum']]
    assert vocals == json.loads((root / 'vocal-program-100-seconds.json').read_text())
    chords_path = root / 'chords-seconds.json'
    chords = json.loads(chords_path.read_text()) if chords_path.exists() else []
    coverage = dict(vocal_notes=len(vocals), drum_notes=sum(n['is_drum'] for n in notes),
        repeated_vocal_pitches={str(p): n for p, n in Counter(x['pitch'] for x in vocals).items() if n > 1},
        inversion_labels=sorted({c[0] for c in chords if '/' in c[0]}),
        chunk_crossing_notes=sum(int(n['onset'] / (32767/16000)) != int(n['offset'] / (32767/16000)) for n in notes),
        non_120_estimated_bpm=manifest['librosa_bpm'] not in (0, 120),
        inferred_labels_are_not_human_annotations=True)
    if manifest['status'] == 'captured':
        for name, width in [('pianoroll', 256), ('chord', 12), ('drum_pianoroll', 128)]:
            array = arrays['roll.' + name]
            assert array.shape[1] == width
            assert np.isin(array, [0, 1]).all()
        import mido
        for name in ('melody', 'chord', 'drums'):
            mido.MidiFile(root / 'midi' / (name + '.mid'))
    receipt = dict(integrity='PASS', tensors=len(entries), status=manifest['status'], coverage=coverage)
    write_json(root / 'verification.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

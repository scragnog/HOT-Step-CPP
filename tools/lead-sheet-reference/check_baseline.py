"""Compare captured output to an uninstrumented upstream run on CPU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from common import Dumps, load_upstream, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    args = p.parse_args()
    torch = load_upstream(args.upstream)
    import numpy as np
    from mulacover._symbolic_transcription.melody import MelodyTranscriber
    from mulacover._symbolic_transcription.harmony import ChordTranscriber
    from mulacover.symbolic import SymbolicCondition
    started = time.perf_counter()
    read = lambda name: json.loads((args.capture / name).read_text())
    notes = MelodyTranscriber(args.weights / 'yourmt3/last.ckpt', 'cpu').transcribe(args.audio)
    assert [asdict(n) for n in notes] == read('merged-notes-seconds.json')
    bpm = read('manifest.json')['librosa_bpm']
    chords = ChordTranscriber(args.weights / 'chord', 'cpu').transcribe(args.audio, bpm)
    assert [list(c) for c in chords] == read('chords-rounded-beats.json')
    condition = SymbolicCondition.from_transcription(notes, chords, bpm)
    entries = read('tensors.json')
    for name, tensor in condition.to_tensors().items():
        entry = entries['roll.' + name]
        expected = np.fromfile(args.capture / entry['file'], '<f4').reshape(entry['shape'])
        np.testing.assert_array_equal(tensor.numpy(), expected)
    # A second valid condition explicitly omits drums, as the cover API permits.
    no_drums = SymbolicCondition(melody=condition.melody, chords=condition.chords, bpm=bpm)
    dumps = Dumps(args.capture / 'without-drums')
    for name, tensor in no_drums.to_tensors().items():
        dumps.put('roll.' + name, tensor)
    no_drums.save_midi(args.capture / 'without-drums' / 'midi')
    write_json(args.capture / 'without-drums' / 'triplets-sixteenths.json',
               {k: getattr(no_drums, k).tolist() for k in ('melody', 'chords', 'drums')})
    result = dict(status='PASS', comparison='Exact merged notes, rounded chords and all four roll/mask arrays',
                  elapsed_seconds=time.perf_counter()-started,
                  no_drums='Conditioning variant, not a source-separated audio claim')
    write_json(args.capture / 'baseline-check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

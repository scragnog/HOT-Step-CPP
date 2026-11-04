"""Prepare anonymous library excerpts; preserve private provenance separately."""
import argparse
import json
from pathlib import Path
import sqlite3
from common import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--instrumental-source', type=Path)
    args = p.parse_args()
    import soundfile as sf
    connection = sqlite3.connect(args.library.joinpath('hotstep.db').as_uri() + '?mode=ro', uri=True)
    rows = connection.execute('SELECT id, audio_url, bpm, generation_params, lyrics FROM songs ORDER BY created_at DESC').fetchall()
    selected = []
    for instrumental in (False, True):
        if instrumental and args.instrumental_source:
            path = args.instrumental_source
            info = sf.info(path)
            audio, sr = sf.read(path, frames=min(info.frames, 16 * info.samplerate), dtype='float32', always_2d=True)
            output = args.out / 'library-instrumental.wav'
            sf.write(output, audio, sr, subtype='FLOAT')
            selected.append(dict(alias='library-instrumental', sha256=digest(output),
                source_sha256=digest(path), source_path=str(path.resolve()), start_seconds=0,
                duration_seconds=len(audio)/sr, coverage='Existing instrumental test fixture; human content annotation pending'))
            continue
        for song_id, url, bpm, raw, lyrics in rows:
            params = json.loads(raw or '{}')
            if bool(params.get('instrumental', False)) != instrumental:
                continue
            if not instrumental and len((lyrics or '').strip()) < 100:
                continue
            path = args.library / 'audio' / Path(url).name
            if not path.is_file():
                continue
            info = sf.info(path)
            if info.duration < 20:
                continue
            start = min(30, max(0, info.duration / 3))
            alias = 'library-instrumental' if instrumental else 'library-vocal'
            audio, sr = sf.read(path, start=int(start * info.samplerate), frames=16 * info.samplerate, dtype='float32', always_2d=True)
            args.out.mkdir(parents=True, exist_ok=True)
            output = args.out / (alias + '.wav')
            sf.write(output, audio, sr, subtype='FLOAT')
            selected.append(dict(alias=alias, sha256=digest(output), source_id=song_id,
                source_sha256=digest(path), start_seconds=start, duration_seconds=len(audio)/sr,
                library_bpm=bpm, instrumental_request=instrumental,
                coverage='Metadata-based candidate; note repetition, inversions and actual vocal/drum presence require verification'))
            if not instrumental:
                # Keep the original as well; this derivative exercises a known silent stretch.
                audio[5*sr:7*sr] = 0
                silent = args.out / 'library-vocal-silent-stretch.wav'
                sf.write(silent, audio, sr, subtype='FLOAT')
                selected.append(dict(selected[-1], alias='library-vocal-silent-stretch',
                    sha256=digest(silent), edit='Samples from 5.0 through 7.0 seconds set to zero'))
            break
    write_json(args.out / 'clips.json', selected)
    print(json.dumps(selected, indent=2))
    if len(selected) != 3:
        raise RuntimeError('Could not find both library candidates')


if __name__ == '__main__':
    main()

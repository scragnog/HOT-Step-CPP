"""Download pinned reference weights; run from the isolated upstream directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hf = '5e66c1ea173a8186e0d20432b841d3180cc015b5'
    gh = '481f4ce703f8822b99f4037e9104ba1760e21ea3'
    name = 'amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt'
    sources = [('yourmt3/last.ckpt', 'huggingface-space', hf,
                f'https://huggingface.co/spaces/mimbres/YourMT3/resolve/{hf}/{name}')]
    for fold in range(5):
        name = f'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict'
        sources.append((f'chord/{name}', 'github', gh,
                        f'https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/{gh}/cache_data/{name}'))
    manifest = {'upstream_commit': 'e27fa7fa0da7c9817879cc6210cc5693f9327946', 'weights': []}
    for relative, host, revision, url in sources:
        path = args.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download into a temporary file so an interrupted transfer is never reused.
        partial = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(path)
        entry = dict(path=relative, host=host, revision=revision, url=url,
                     bytes=path.stat().st_size, sha256=sha256(path))
        manifest['weights'].append(entry)
        (args.out / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()

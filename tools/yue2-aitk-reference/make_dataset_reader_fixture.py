"""CPU-only valid and malformed immutable-cache cases for the native reader."""
import copy
import json
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
latent = struct.pack('<128f', *[(i % 13 - 6) * .03125 for i in range(128)])
(out / 'latent.bin').write_bytes(latent)
h = 14695981039346656037
for value in latent:
    h = ((h ^ value) * 1099511628211) & ((1 << 64) - 1)
item = dict(id='tiny', frames=2, latent_file='latent.bin', latent_fnv1a64=f'{h:016x}',
            semantic_tokens=[12, 42], prefix_full_ids=[1, 2], prefix_off_ids=[1, 2], abc_ids=[])
base = dict(schema_version=1, recipe_version='aitk-yue2-2026-09-16', cot='full',
            base_sha256='a' * 64, source_manifest_sha256='b' * 64, items=[item])
cases = []
def case(name, obj, accept=False):
    (out / f'{name}.json').write_text(json.dumps(obj), encoding='utf-8')
    cases.append(dict(name=name, mode='accept' if accept else 'reject'))
case('valid', base, True)
for key, value in [('schema_version', 2), ('cot', 'off'), ('items', []), ('base_sha256', 'invalid')]:
    obj = copy.deepcopy(base); obj[key] = value; case(key, obj)
for key, value in [('latent_file', '../latent.bin'), ('id', 'bad\0name'),
                   ('frames', 3), ('semantic_tokens', [12, 32768]),
                   ('prefix_full_ids', []), ('prefix_off_ids', [184704]),
                   ('latent_fnv1a64', '0' * 16), ('abc_ids', [1] * 24576)]:
    obj = copy.deepcopy(base); obj['items'][0][key] = value; case(key, obj)
obj = copy.deepcopy(base); obj['items'] *= 2; case('duplicate_ids', obj)
obj = copy.deepcopy(base); obj['extra'] = 'bad\0value'; case('extra_nul', obj)
obj = copy.deepcopy(base); obj['extra'] = 0
for _ in range(66): obj['extra'] = [obj['extra']]
case('depth', obj)
(out / 'duplicate_keys.json').write_text(json.dumps(base).replace('"frames": 2', '"frames": 2, "frames": 2'), encoding='utf-8')
cases.append(dict(name='duplicate_keys', mode='reject'))
bad_latent = struct.pack('<128f', float('nan'), *([0.] * 127))
(out / 'nonfinite.bin').write_bytes(bad_latent)
h = 14695981039346656037
for value in bad_latent: h = ((h ^ value) * 1099511628211) & ((1 << 64) - 1)
obj = copy.deepcopy(base); obj['items'][0].update(latent_file='nonfinite.bin', latent_fnv1a64=f'{h:016x}')
case('nonfinite', obj)
(out / 'cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
print(f'{len(cases)} dataset reader cases prepared')

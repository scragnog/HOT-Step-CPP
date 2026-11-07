// mm3Ladder.test.ts — which album presets follow "Use this checkpoint", and
// how saved MM3 previews are listed back after a reload.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { mm3PresetFollows } from './lyricStudioExport.js';
import { listMm3RunPreviews } from './mm3Preview.js';

const ds = { slug: 'albumx', lyricsSetId: 7 };
const ref = (run: string, step: number) => `${run}\\ckpt-${step}\\adapter_model.safetensors`;

test('presets follow the chosen checkpoint, and move again when another is picked', () => {
  const r = 'albumx-2026-11-07_11-21-20';
  // The direct preset follows whatever it held.
  assert.equal(mm3PresetFollows({ lyrics_set_id: 7, mm3_adapter_path: '' }, ds, ref(r, 200)), true);
  assert.equal(mm3PresetFollows({ lyrics_set_id: 7, mm3_adapter_path: 'other-2026-01-01_00-00-00/x.safetensors' }, ds, ref(r, 200)), true);
  // Another step of the same run, or another run of this dataset, moves too.
  assert.equal(mm3PresetFollows({ lyrics_set_id: 9, mm3_adapter_path: ref(r, 600) }, ds, ref(r, 200)), true);
  assert.equal(mm3PresetFollows({ lyrics_set_id: 9, mm3_adapter_path: ref('albumx-2026-10-01_00-00-00', 300) }, ds, ref(r, 200)), true);
  // Already there (either slash, any case): idempotent.
  assert.equal(mm3PresetFollows({ lyrics_set_id: 7, mm3_adapter_path: ref(r, 200).replace(/\\/g, '/').toUpperCase() }, ds, ref(r, 200)), false);
  // Another dataset's adapter, including a slug that only shares a prefix, is left alone.
  assert.equal(mm3PresetFollows({ lyrics_set_id: 9, mm3_adapter_path: ref('albumx_live-2026-10-01_00-00-00', 300) }, ds, ref(r, 200)), false);
  assert.equal(mm3PresetFollows({ lyrics_set_id: 9, mm3_adapter_path: '' }, ds, ref(r, 200)), false);
});

test('saved previews list from their sidecar, or from the file name before sidecars existed', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-ladder-'));
  try {
    fs.writeFileSync(path.join(dir, 'step-000200-artist.wav'), 'x');
    fs.writeFileSync(path.join(dir, 'step-000100-artist.wav'), 'xx');
    fs.writeFileSync(path.join(dir, 'step-000100-artist-s77.wav'), 'xxx');
    fs.writeFileSync(path.join(dir, 'step-000100-artist-s77.wav.json'), JSON.stringify({ seed: 77, seconds: 30, caption: 'cap', ms: 9000 }));
    fs.writeFileSync(path.join(dir, 'base-control.wav'), 'x');
    fs.writeFileSync(path.join(dir, 'notes.txt'), 'ignored');
    const list = listMm3RunPreviews(dir, { seconds: 40, seed: 424242 });
    // Same-step takes order by render time, which ties here; compare by seed.
    assert.deepEqual(list.map(p => [p.step, p.base, p.kind, p.seed, p.seconds]).sort((a, b) => Number(a[0]) - Number(b[0]) || Number(a[3]) - Number(b[3])), [
      [0, true, 'control', 424242, 40],
      [100, false, 'artist', 77, 30],
      [100, false, 'artist', 424242, 40],
      [200, false, 'artist', 424242, 40],
    ]);
    assert.deepEqual(list.map(p => p.step), [0, 100, 100, 200]);
    assert.equal(list.find(p => p.seed === 77)?.caption, 'cap');
    assert.equal(listMm3RunPreviews(path.join(dir, 'missing'), { seconds: 40, seed: 1 }).length, 0);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

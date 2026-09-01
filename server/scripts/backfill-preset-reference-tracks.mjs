#!/usr/bin/env node
// backfill-preset-reference-tracks.mjs — give older album presets a reference track
//
// The Lyric Studio export only started hanging one of the dataset's own tracks
// on the album preset (the timbre / mastering reference) on 2026-09-01. Albums
// exported by earlier training runs have the DiT and planner adapters wired up
// and that field empty, which is exactly the case this script fixes.
//
// It changes NOTHING else: the preset is read back and every other column is
// round-tripped, because PUT .../preset is a full overwrite. A preset that
// already names a reference file that still exists is left alone — a hand-
// picked reference outranks ours.
//
// The pick itself comes from the server (GET .../lyric-studio returns
// `referenceTrack`), so this and a live export can never disagree.
//
// Usage — dry run first, it is the default:
//   node server/scripts/backfill-preset-reference-tracks.mjs <dataset-name>...
//   node server/scripts/backfill-preset-reference-tracks.mjs --apply <dataset-name>...
//   node server/scripts/backfill-preset-reference-tracks.mjs --apply --from <name> --to <name>
//
// --from/--to select an inclusive range of the datasets list in its own order.
// The app must be running (it owns the SQLite connection).

import fs from 'fs';

const BASE = process.env.HOTSTEP_URL || 'http://localhost:3001';

const argv = process.argv.slice(2);
const apply = argv.includes('--apply');
const flagValue = (name) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : null;
};
const from = flagValue('--from');
const to = flagValue('--to');
const names = argv.filter((a, i) =>
  !a.startsWith('--') && argv[i - 1] !== '--from' && argv[i - 1] !== '--to');

async function api(path, init) {
  const res = await fetch(`${BASE}${path}`, init);
  const body = await res.text();
  if (!res.ok) throw new Error(`${init?.method ?? 'GET'} ${path} → ${res.status} ${body.slice(0, 300)}`);
  return body ? JSON.parse(body) : null;
}

const listed = await api('/api/training/datasets');
const all = Array.isArray(listed) ? listed : (listed.datasets ?? []);

let targets;
if (from) {
  const start = all.findIndex(d => d.name === from || d.slug === from);
  if (start < 0) throw new Error(`--from ${from}: no such dataset`);
  let end = all.length - 1;
  if (to) {
    end = all.findIndex(d => d.name === to || d.slug === to);
    if (end < 0) throw new Error(`--to ${to}: no such dataset`);
  }
  targets = all.slice(start, end + 1);
} else {
  targets = names.map(n => {
    const hit = all.find(d => d.name === n || d.slug === n);
    if (!hit) throw new Error(`no dataset named ${n}`);
    return hit;
  });
}

if (!targets.length) {
  console.error('Nothing to do — pass dataset names, or --from <name> [--to <name>].');
  process.exit(1);
}

console.log(`${apply ? 'APPLYING to' : 'DRY RUN over'} ${targets.length} dataset(s)\n`);

const counts = { set: 0, kept: 0, noRef: 0, noSet: 0, failed: 0 };

for (const ds of targets) {
  const label = ds.name.padEnd(32);
  try {
    const preview = await api(`/api/training/datasets/${encodeURIComponent(ds.id)}/lyric-studio`);
    const lyricsSetId = ds.lyricsSetId || preview.existingLyricsSetId || 0;
    if (!lyricsSetId) { counts.noSet++; console.log(`${label} SKIP  no Lyric Studio album linked`); continue; }
    if (!preview.referenceTrack) { counts.noRef++; console.log(`${label} SKIP  no usable audio in the dataset`); continue; }

    const { preset } = await api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`);
    const existing = preset?.reference_track_path ?? '';
    if (existing && fs.existsSync(existing)) {
      counts.kept++;
      console.log(`${label} KEEP  ${existing.split(/[\/]/).pop()}`);
      continue;
    }

    if (apply) {
      // Full overwrite — every other column has to go back in untouched.
      await api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          adapter_path: preset?.adapter_path ?? undefined,
          adapter_scale: preset?.adapter_scale ?? undefined,
          adapter_group_scales: preset?.adapter_group_scales ?? undefined,
          reference_track_path: preview.referenceTrack,
          audio_cover_strength: preset?.audio_cover_strength ?? undefined,
          lm_adapter_path: preset?.lm_adapter_path ?? undefined,
          lm_adapter_scale: preset?.lm_adapter_scale ?? undefined,
        }),
      });
    }
    counts.set++;
    console.log(`${label} ${apply ? 'SET  ' : 'WOULD'} ${preview.referenceTrack}`);
  } catch (err) {
    counts.failed++;
    console.log(`${label} FAIL  ${err.message}`);
  }
}

console.log(`\n${apply ? 'set' : 'would set'}: ${counts.set} | already had one: ${counts.kept} `
  + `| no album: ${counts.noSet} | no audio: ${counts.noRef} | failed: ${counts.failed}`);
if (!apply && counts.set) console.log('Re-run with --apply to write these.');
process.exit(counts.failed ? 1 : 0);

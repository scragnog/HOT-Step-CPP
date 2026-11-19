// lyrics-drift.ts — pool the CTC-versus-Whisper audit across the tracks on disk.
//
//   npm run lyrics:drift           totals, per section occurrence, per track
//   npm run lyrics:drift -- --all  also name the files it could not use
//
// `runSongAlignment` measures how far the CTC path strays from the words
// Whisper heard and writes it into the `.lyrics.json` as
// `diagnostics.anchorDrift` on a full-track run. One track's number decides
// nothing: the question the engine corridor (D2-C) asks is whether drift
// concentrates where a corridor would help — a chorus that comes back — or
// spreads evenly, in which case a corridor would be tuning by anecdote. This
// reads every sidecar it can and puts those rows side by side.
//
// Files are skipped rather than guessed at. A range run's drift is about a crop
// and its line indices are its own, so the pipeline does not record one; a file
// aligned without the CTC stage has no audit at all; and an audit written before
// the section breakdown can speak to a track but not to a repeat.

import fs from 'node:fs';
import path from 'node:path';
import { config } from '../src/config.js';
import { summariseDrift, type DriftRecord } from '../src/services/lyricsAlignment.js';
import type { LyricsJson } from '../src/services/lyricsReconcile.js';

const all = process.argv.includes('--all');

/** Every sidecar in the audio directory that parsed, whether it has an audit or not. */
function readSidecars(): Array<{ name: string; sidecar: Partial<LyricsJson> }> {
  const dir = config.data.audioDir;
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(file => file.endsWith('.lyrics.json'))
    .sort()
    .flatMap(file => {
      try {
        const sidecar = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf-8')) as Partial<LyricsJson>;
        return [{ name: file.replace(/\.lyrics\.json$/, ''), sidecar }];
      } catch {
        // A half-written sidecar is not evidence; it is skipped like any file
        // that cannot answer.
        return [];
      }
    });
}

function report(): void {
  const records: DriftRecord[] = [];
  const skipped: Array<{ name: string; reason: string }> = [];

  for (const { name, sidecar } of readSidecars()) {
    const diagnostics = sidecar.diagnostics;
    const drift = diagnostics?.anchorDrift;
    if (diagnostics?.range) {
      skipped.push({ name, reason: 'range run (a crop, not a track)' });
    } else if (!drift) {
      skipped.push({
        name,
        reason: `no audit — ${diagnostics?.fallback || 'the CTC and Whisper stages did not both run'}`,
      });
    } else if (!drift.sections) {
      skipped.push({ name, reason: 'older audit, no section breakdown' });
    } else {
      records.push({ track: name, drift });
    }
  }

  if (records.length === 0) {
    console.log('No track has a full-track anchor audit yet.');
    console.log('Run a whole-track alignment with both the CTC aligner and Whisper, then apply it:');
    console.log('the audit is written into the sidecar with the timings.');
    if (skipped.length) {
      console.log(`${skipped.length} sidecar${skipped.length === 1 ? ' is' : 's are'} not usable.`);
    }
    for (const entry of all ? skipped : skipped.slice(0, 5)) {
      console.log(`  ${entry.name}: ${entry.reason}`);
    }
    if (!all && skipped.length > 5) console.log(`  ... ${skipped.length - 5} more (--all)`);
    return;
  }

  const summary = summariseDrift(records);
  const pct = summary.words ? `${((summary.outliers / summary.words) * 100).toFixed(1)}%` : 'n/a';
  console.log(`Tracks            ${summary.tracks}`);
  console.log(`Anchor words      ${summary.words}`);
  console.log(`Outliers          ${summary.outliers} (${pct} of pairs)`);
  console.log(`Median of medians ${summary.medianMs} ms`);
  console.log(`Worst track P90   ${summary.worstP90Ms} ms`);

  console.log('');
  console.log('Section             #  tracks   words    median  outliers');
  for (const section of summary.sections) {
    const label = (section.section || '(unlabelled)').slice(0, 18).padEnd(18);
    const median = `${section.medianMs} ms`.padStart(8);
    console.log(
      `${label} ${String(section.occurrence).padStart(2)}  ${String(section.tracks).padStart(6)}  `
      + `${String(section.words).padStart(6)}  ${median}  ${String(section.outliers).padStart(8)}`,
    );
  }

  console.log('');
  for (const { track, drift } of records) {
    console.log(`${track}: median ${drift.medianMs} ms, P90 ${drift.p90Ms} ms, `
      + `${drift.outliers} outliers of ${drift.anchors} anchor words`);
    for (const worst of drift.worst.slice(0, 3)) {
      const where = worst.section ? ` (${worst.section})` : '';
      console.log(`  line ${worst.line}${where} word ${worst.word}: ${worst.driftMs} ms — `
        + `ctc ${worst.ctcSec}s, heard ${worst.anchorSec}s`);
    }
  }

  if (skipped.length) {
    console.log('');
    console.log(`${skipped.length} sidecar${skipped.length === 1 ? '' : 's'} not counted`
      + `${all ? '' : ' (--all names them)'}.`);
    if (all) for (const entry of skipped) console.log(`  ${entry.name}: ${entry.reason}`);
  }

  console.log('');
  console.log('These rows are the D2-C evidence: a corridor is worth building when the repeat');
  console.log('rows (occurrence 2 and up) carry the outliers, not when they look like the');
  console.log('first-occurrence rows and the drift is spread across the track.');
}

try {
  report();
} catch (err: any) {
  console.error(err?.message || err);
  process.exitCode = 1;
}

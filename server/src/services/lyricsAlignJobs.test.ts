import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  buildPolishPreview, findRunning, refusesOverwrite, writeTimingsFile, type AlignJobView,
} from './lyricsAlignJobs.js';
import type { LyricsLine, LyricsWord } from './lyricsReconcile.js';

function word(text: string, start: number, end: number): LyricsWord {
  return { word: text, start, end, confidence: 1, source: 'matched' };
}

function line(words: LyricsWord[]): LyricsLine {
  return {
    start: words[0].start,
    end: words[words.length - 1].end,
    text: words.map(w => w.word).join(' '),
    words,
  };
}

/** A sidecar as the aligner left it, with the diagnostics a polish run reads. */
function sidecar(lines: LyricsLine[]) {
  return {
    version: 1,
    method: 'mms-fa',
    whisperModel: 'ggml-large-v3-turbo.bin',
    vocalsIsolated: false,
    lines,
    diagnostics: { dtw: true },
  };
}

test('hand-placed timings refuse a machine run unless the caller forces it', () => {
  assert.equal(refusesOverwrite({ editedByHand: true }, undefined), true);
  assert.equal(refusesOverwrite({ editedByHand: true }, false), true);
  assert.equal(refusesOverwrite({ editedByHand: true }, true), false);
  assert.equal(refusesOverwrite({ editedByHand: false }, undefined), false);
  assert.equal(refusesOverwrite(null, undefined), false);
  assert.equal(refusesOverwrite(undefined, true), false);
});

test('the per-track lock still sees a long run after a short one repointed latest', () => {
  const view = (jobId: string, status: AlignJobView['status']): AlignJobView => ({
    jobId, audioUrl: 'x.wav', status, stage: status, startedAt: 1,
  });
  const jobs = new Map([
    ['full', { key: '/audio/x.lyrics.json', view: view('full', 'running') }],
    ['polish', { key: '/audio/x.lyrics.json', view: view('polish', 'succeeded') }],
  ]);
  // `latest` points at the polish run; the lock is about the full one.
  assert.equal(findRunning(jobs, '/audio/x.lyrics.json')?.jobId, 'full');
  assert.equal(findRunning(jobs, '/audio/other.lyrics.json'), null);

  // A run that ended, however it ended, releases the track.
  jobs.set('full', { key: '/audio/x.lyrics.json', view: view('full', 'cancelled') });
  assert.equal(findRunning(jobs, '/audio/x.lyrics.json'), null);
});

test('the one writer bumps the revision, clears the hand mark and rebuilds the .lrc', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lyrics-write-'));
  try {
    const jsonPath = path.join(dir, 'x.lyrics.json');
    const lrcPath = path.join(dir, 'x.lrc');
    const written = writeTimingsFile({
      data: { version: 1, revision: 4, editedByHand: true },
      lines: [line([word('one', 1, 1.5), word('two', 1.5, 2)])],
      jsonPath,
      lrcPath,
      title: 'Title',
      artist: 'Artist',
      expectedRevision: 4,
    });
    const onDisk = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

    assert.deepEqual(written, { revision: 5, lines: 1, words: 2 });
    assert.equal(onDisk.revision, 5, 'an open editor’s next save is refused');
    assert.equal(onDisk.editedByHand, false, 'these times are not hand-placed');
    assert.equal(onDisk.lines[0].words[1].end, 2);
    const lrc = fs.readFileSync(lrcPath, 'utf-8');
    assert.match(lrc, /00:01/, 'the player reads this file, so it matches the sidecar');
    assert.match(lrc, /one two/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('the writer keeps the hand mark when the lines still contain hand placements', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lyrics-write-'));
  try {
    const jsonPath = path.join(dir, 'x.lyrics.json');
    const carried = line([word('keep', 1, 2)]);
    carried.checked = true;
    writeTimingsFile({
      data: { version: 1, revision: 1, editedByHand: false },
      lines: [carried, line([word('machine', 3, 3.5)])],
      jsonPath,
      lrcPath: path.join(dir, 'x.lrc'),
      title: '',
      artist: '',
      expectedRevision: 1,
    });
    const onDisk = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    assert.equal(onDisk.editedByHand, true, 'the next run still has to ask');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a polish run keeps the track’s anchor audit instead of erasing it', () => {
  // The audit describes the CTC pass against Whisper's anchors, and a polish run
  // has neither. It is carried over so a repair pass does not throw away the
  // track's only evidence for or against an engine corridor.
  const audit = {
    anchors: 2, words: 2, medianMs: 40, p90Ms: 90, outliers: 0,
    worst: [{ line: 0, word: 1, driftMs: 90, anchorSec: 1.4, ctcSec: 1.49, section: 'Chorus' }],
    sections: [{ section: 'Chorus', occurrence: 1, words: 2, medianMs: 40, p90Ms: 90, outliers: 0 }],
  };
  const existing = {
    ...sidecar([line([word('one', 1.0, 1.6), word('two', 1.2, 1.5)])]),
    diagnostics: { dtw: true, anchorDrift: audit },
  };
  const preview = buildPolishPreview(existing, []);

  assert.deepEqual(preview.diagnostics.anchorDrift, audit);
  assert.deepEqual(preview.data.diagnostics.anchorDrift, audit,
    'the file Apply writes still carries the audit');
});

test('a polish preview holds the repaired lines, never the originals', () => {
  // Two words where the second begins before the first ends: the monotonic
  // repair has to move it. A preview built with `polishLines({ dryRun: true })`
  // would list that repair in `repairs[]` and then hand back the untouched
  // lines, so Apply would report work and write none of it.
  const original = [line([word('one', 1.0, 1.6), word('two', 1.2, 1.5)])];
  const preview = buildPolishPreview(sidecar(original), []);

  assert.ok(preview.result.repairs && preview.result.repairs.length > 0,
    'the overlapping defect must produce at least one repair');
  assert.equal(preview.result.pending, true, 'a polish run is a preview');
  assert.equal(preview.result.dryRun, true);
  assert.deepEqual(preview.diagnostics.repairs, preview.result.repairs,
    'the diagnostics and the job result describe the same repairs');

  for (const repair of preview.result.repairs!) {
    const applied = preview.lines[repair.line].words[repair.word];
    assert.equal(applied.start, repair.to.start, `word ${repair.line}:${repair.word} start`);
    assert.equal(applied.end, repair.to.end, `word ${repair.line}:${repair.word} end`);
  }

  // The file that would be written is the repaired one, and not the array the
  // caller passed in: a preview that aliased it would let a later edit mutate
  // a diff already on screen.
  assert.notDeepEqual(preview.data.lines, original);
  assert.deepEqual(preview.data.lines, preview.lines);
  assert.equal(preview.data.editedByHand, false);
  assert.equal(original[0].words[1].start, 1.2, 'the caller’s sidecar is untouched');
});

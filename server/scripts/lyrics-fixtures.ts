// lyrics-fixtures.ts — save hand-checked lyric timings, then score runs on them.
//
//   npm run lyrics:fixtures -- save "<audioUrl>"
//   npm run lyrics:fixtures -- score "<audioUrl>"
//   npm run lyrics:fixtures -- list
//
// `save` copies the hand-checked lines out of the track's current
// `.lyrics.json` — the ones ticked in the lyric editor because a person placed
// them by ear — into the fixture store. A line the machine timed and nobody
// checked is not ground truth, so it is left out. `score` compares the sidecar
// as it is now (after a machine run) against that fixture and prints the numbers
// the scope document asks for: median and P90 onset error, the share inside
// 100 ms and 250 ms, monotonic violations, and the second-chorus offset.
//
// The fixture lives in the data directory, not the repo: it is made of a user's
// audio and their own judgement, and it must survive re-alignment of the sidecar.

import fs from 'node:fs';
import path from 'node:path';
import { config } from '../src/config.js';
import { sidecarPath } from '../src/services/lrcExport.js';
import { handCheckedLines, scoreFixture, type LyricsFixture } from '../src/services/lyricsFixture.js';
import type { LyricsLine } from '../src/services/lyricsReconcile.js';

const ROOT = path.join(config.data.dir, 'fixtures', 'lyrics');

function slugOf(audioUrl: string): string {
  const name = audioUrl.split('?')[0].split('/').pop() || 'track';
  return name.replace(/\.[a-z0-9]+$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '-');
}

function fixturePath(audioUrl: string): string {
  return path.join(ROOT, `${slugOf(audioUrl)}.json`);
}

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function sidecarOf(audioUrl: string): { lines?: unknown[] } | null {
  return readJson<{ lines?: unknown[] }>(sidecarPath(config.data.audioDir, audioUrl, '.lyrics.json'));
}

function save(audioUrl: string): void {
  const sidecar = sidecarOf(audioUrl);
  if (!sidecar || !Array.isArray(sidecar.lines) || sidecar.lines.length === 0) {
    throw new Error(`No word timings for ${audioUrl}. Open it in the lyric editor first.`);
  }
  const checked = handCheckedLines(sidecar.lines as LyricsLine[]);
  if (checked.length === 0) {
    throw new Error(
      `No hand-checked lines in ${audioUrl}. In the lyric editor, tick a line's check box `
      + 'for every line you placed by ear; a fixture holds only those, so a machine guess '
      + 'never becomes the reference.',
    );
  }
  fs.mkdirSync(ROOT, { recursive: true });
  const fixture: LyricsFixture = {
    version: 1,
    audioUrl,
    createdAt: new Date().toISOString(),
    lines: checked,
  };
  fs.writeFileSync(fixturePath(audioUrl), JSON.stringify(fixture, null, 2));
  console.log(
    `Saved fixture ${path.basename(fixturePath(audioUrl))}: ${checked.length} checked `
    + `of ${sidecar.lines.length} lines`,
  );
}

function score(audioUrl: string): void {
  const fixture = readJson<LyricsFixture>(fixturePath(audioUrl));
  if (!fixture) throw new Error(`No fixture for ${audioUrl}. Save one after placing the words by ear.`);
  const sidecar = sidecarOf(audioUrl);
  if (!sidecar || !Array.isArray(sidecar.lines)) throw new Error(`No timings to score for ${audioUrl}.`);

  const result = scoreFixture(fixture, sidecar.lines as LyricsFixture['lines']);
  const pct = (share: number) => `${(share * 100).toFixed(1)}%`;
  console.log(`Fixture  ${path.basename(fixturePath(audioUrl))} (saved ${fixture.createdAt})`);
  console.log(`Words    ${result.words} (matched ${result.matched}, missing ${result.missing})`);
  console.log(`Onset    median ${result.medianMs} ms, P90 ${result.p90Ms} ms, max ${result.maxMs} ms`);
  console.log(`Within   100 ms ${pct(result.within100)}, 250 ms ${pct(result.within250)}`);
  console.log(`Monotonic violations ${result.monotonicViolations}`);
  console.log(`Second chorus offset ${result.secondChorusOffsetMs == null ? 'n/a' : `${result.secondChorusOffsetMs} ms`}`);
  if (result.sections.length) {
    console.log('Sections');
    for (const section of result.sections) {
      console.log(`  ${section.section} #${section.occurrence}: ${section.words} words, median ${section.medianMs} ms`);
    }
  }
}

function list(): void {
  if (!fs.existsSync(ROOT)) {
    console.log('No fixtures yet.');
    return;
  }
  for (const file of fs.readdirSync(ROOT).filter(name => name.endsWith('.json')).sort()) {
    const fixture = readJson<LyricsFixture>(path.join(ROOT, file));
    const words = fixture?.lines.reduce((n, line) => n + (line.words?.length ?? 0), 0) ?? 0;
    console.log(`${file}: ${words} words, ${fixture?.createdAt ?? 'unknown date'}`);
  }
}

const [command, audioUrl] = process.argv.slice(2);
try {
  if (command === 'save' && audioUrl) save(audioUrl);
  else if (command === 'score' && audioUrl) score(audioUrl);
  else if (command === 'list') list();
  else {
    console.log('Usage: npm run lyrics:fixtures -- save|score "<audioUrl>"   (or: list)');
    process.exitCode = 2;
  }
} catch (err: any) {
  console.error(err?.message || err);
  process.exitCode = 1;
}

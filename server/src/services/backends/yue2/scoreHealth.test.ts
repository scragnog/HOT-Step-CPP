import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyYue2Score } from './scoreHealth.js';

const HEADER = 'X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n';

function song(bars: number, vocalEvery: number, sections: string[]): string {
  const lines: string[] = [HEADER];
  let bar = 0;
  const per = Math.round(bars / sections.length);
  for (const section of sections) {
    lines.push(`% ${section}`, 'V: Vocal');
    const chunk: string[] = [];
    for (let i = 0; i < per; i++, bar++) chunk.push(bar % vocalEvery === 0 ? '"C"C4D4E4F4' : 'z16');
    lines.push(chunk.join('|') + '|', 'V: Ins', `Z${per}|`);
  }
  return lines.join('\n');
}

test('a normal verse/chorus sheet is healthy', () => {
  const h = classifyYue2Score(song(120, 2, ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro']));
  assert.equal(h.verdict, 'healthy');
  assert.equal(h.bars, 120);
  assert.ok(Math.abs(h.vocalShare - 0.5) < 0.05);
  assert.equal(h.estSeconds, 240);
});

test('the same sheet is long, not runaway, when the render hit its cap', () => {
  assert.equal(classifyYue2Score(song(120, 2, ['verse', 'chorus']), 'limit_hit').verdict, 'long');
});

test('hundreds of vocal-less bars are runaway', () => {
  const h = classifyYue2Score(song(200, 1000, ['intro', 'verse']));
  assert.equal(h.verdict, 'runaway');
  assert.ok(h.longestSilentRun >= 64);
});

test('an intro>verse loop of section markers is runaway', () => {
  const loop = Array.from({ length: 12 }, (_, i) => (i % 2 ? 'verse' : 'intro'));
  assert.equal(classifyYue2Score(song(96, 2, loop)).verdict, 'runaway');
});

test('Z-rests count as whole bars and chord symbols are ignored', () => {
  const abc = `${HEADER}% verse\nV: Vocal\n"F#"z16|Z4|"B"c4c4A2c2-"B"c2d2-|\n`;
  const h = classifyYue2Score(abc);
  assert.equal(h.bars, 6);
  assert.equal(h.vocalBars, 1);
});

test('no vocal bars is unknown', () => {
  assert.equal(classifyYue2Score('').verdict, 'unknown');
});

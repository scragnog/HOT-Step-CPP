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

// Legibility: what a plan the verdict passed can still get wrong.
function sheet(vocalBars: string[], insBars: string[], sections = ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro']): string {
  const lines: string[] = [HEADER];
  const per = Math.ceil(vocalBars.length / sections.length);
  sections.forEach((section, s) => {
    lines.push(`% ${section}`, 'V: Vocal', vocalBars.slice(s * per, (s + 1) * per).join('|') + '|',
      'V: Ins', insBars.slice(s * per, (s + 1) * per).join('|') + '|');
  });
  return lines.join('\n');
}
const CHORDS = ['"C"', '"F"', '"G"', '"Am"'];
// A tune that never plays the same 4-bar phrase twice: pitch walks with the bar.
const tune = (n: number) => Array.from({ length: n }, (_, i) => `${CHORDS[i % 4]}${'CDEFGABc'[i % 8]}4${'CDEFGABc'[(i * 3) % 8]}4${'CDEFGABc'[(i * 5 + 1) % 8]}8`);

test('a varied sheet raises no legibility flag', () => {
  const h = classifyYue2Score(sheet(tune(96), tune(96)), undefined, '[Intro]\n[Verse 1]\nla\n[Chorus]\nla\n[Verse 2]\nla\n[Chorus]\nla\n[Outro]');
  assert.deepEqual(h.legibility.flags, []);
  assert.equal(h.legibility.lyricSections, 6);
  assert.ok(h.legibility.insLoop.bars < 8);
});

test('a two-bar riff played for the whole sheet is a loop, one chord and one pitch are flagged', () => {
  const ins = Array.from({ length: 96 }, (_, i) => (i % 2 ? '"D#m"d24-d4c4' : '"D#m"d8D4F4d4A4F4d4-'));
  const vocal = Array.from({ length: 96 }, (_, i) => (i % 8 ? '"D#m"z32' : '"D#m"d32'));
  const h = classifyYue2Score(sheet(vocal, ins));
  assert.equal(h.legibility.insLoop.period, 2);
  assert.equal(h.legibility.insLoop.bars, 96);
  assert.match(h.legibility.flags[0], /Ins voice repeats a 2-bar riff for 96 bars/);
  assert.ok(h.legibility.flags.some(f => /1 pitch/.test(f)));
  assert.ok(h.legibility.flags.some(f => /1 chord/.test(f)));
});

test('a plan with fewer sections than the lyric is flagged', () => {
  const h = classifyYue2Score(sheet(tune(48), tune(48), ['verse', 'chorus']), undefined, '[Verse 1]\nla\n[Chorus]\nla\n[Verse 2]\nla\n[Bridge]\nla\n[Chorus]\nla');
  assert.match(h.legibility.flags[0], /2 section\(s\) planned for 5 lyric section tags/);
});

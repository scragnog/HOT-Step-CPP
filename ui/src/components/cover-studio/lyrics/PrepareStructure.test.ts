import assert from 'node:assert/strict';
import test from 'node:test';
import { mergeScoreSections } from '@/utils/abc/abcSectionMove';
import { barTimes, clock, createPlaybackOwner, mainMeter, sectionSpans, shortBars, spanName, startPlayer, stopPlayer, type AbcPlayer } from '@/components/cover-studio/build/prepareView';

const HEADER = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=120',
  'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C#m'];
// The verse / pre-chorus / chorus excerpt with its 3/4 and 1/4 bars, as the transcriber wrote it.
const REPORT = [...HEADER,
  '% verse', 'V: Vocal', '"C#m"z4c4c4c4^B4c4d4c4-|"C#m"c4z8c4^B4c4d4A4-|', 'V: Ins', 'Z2|',
  'V: Vocal', 'M:3/4', '"A"A4z12e4e4|', 'V: Ins', 'M:3/4', 'Z|',
  '% pre-chorus', 'V: Vocal', 'M:4/4', '"G#"z4A4G4F8E4D4C4|', 'V: Ins', 'M:4/4', 'Z|',
  '% chorus', 'V: Vocal', 'M:1/4', '"C#m"z4c4|', 'V: Ins', 'M:1/4', 'Z|',
  'V: Vocal', 'M:4/4', '"C#m"c4c4c4z4B4B4c4c4-|"C#m"c4z8c2c2c4d4e4e4-|', 'V: Ins', 'M:4/4', 'Z2|',
  '% chorus', 'V: Vocal', '"A"e2d2c4z8e4e4f4f2e2|', 'V: Ins', 'Z|',
  'V: Vocal', 'M:3/4', '"G#"c4d6c2d8z4|', 'V: Ins', 'M:3/4', 'Z|',
].join('\n');
const IDS = ['s1', 's2', 's3', 's4'];

test('sections: ids, labels, occurrence among repeated labels and bar ranges', () => {
  const spans = sectionSpans(REPORT, IDS);
  assert.ok(Array.isArray(spans));
  assert.deepEqual(spans.map(s => [s.id, s.label, s.occurrence, s.repeats, s.start, s.end]),
    [['s1', 'verse', 1, 1, 1, 3], ['s2', 'pre-chorus', 1, 1, 4, 4], ['s3', 'chorus', 1, 2, 5, 7], ['s4', 'chorus', 2, 2, 8, 9]]);
  assert.equal(spanName(spans[0]), 'verse');
  assert.equal(spanName(spans[3]), 'chorus (2/2)');
});

test('sections: a marker count that differs from the ids is refused, not guessed', () => {
  assert.match(sectionSpans(REPORT, ['s1']) as string, /above the K: line/);
  assert.equal(typeof sectionSpans('X:1\nM:4/4\nQ:1/4=120\n', []), 'string');
});

test('bar clock: score time from M: and Q:, a 3/4 bar is shorter', () => {
  const times = barTimes(REPORT)!;
  assert.deepEqual(times.start.slice(0, 4), [0, 2, 4, 5.5]);
  assert.equal(times.end[2], 5.5);
  assert.equal(times.start.length, 9);
  assert.equal(barTimes('X:1\nK:C\nC4|'), null, 'no meter and tempo');
});

test('merging sections does not move the clock', () => {
  const merged = mergeScoreSections(REPORT, 0, 'first') as { abc: string };
  assert.deepEqual(barTimes(merged.abc), barTimes(REPORT));
});

test('short bars: interior bars shorter than the usual one, never the first or last', () => {
  const short = shortBars(REPORT);
  assert.ok(short.includes(3) && short.includes(5), `3/4 and 1/4 bars flagged: ${short}`);
  assert.ok(!short.includes(1) && !short.includes(9), 'the first and last bars are protected');
  assert.deepEqual(shortBars('X:1\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nV:Vocal\nCDEF|GABc|cBAG|\n'), []);
});

test('clock formats minutes and seconds', () => {
  assert.equal(clock(0), '0:00');
  assert.equal(clock(65.9), '1:05');
});

// abcjs's controller as it behaves at runtime: play() toggles isStarted and the timer, pause() (the embedded
// player's own pause button) stops the timer but leaves isStarted set.
const fakePlayer = () => {
  const log: string[] = [];
  const player = {
    isStarted: false, timer: { isRunning: false }, log,
    async play() { player.isStarted = !player.isStarted; player.timer.isRunning = player.isStarted; log.push(player.isStarted ? 'play' : 'pause'); },
    seek(position: number) { log.push(`seek ${position}`); },
    pauseFromControls() { player.timer.isRunning = false; },
  };
  return player satisfies AbcPlayer & Record<string, unknown>;
};
const hear = async (player: ReturnType<typeof fakePlayer>, at: number) => { await startPlayer(player); player.seek(at, 'seconds'); };

test('Hear score, Stop, Hear score sounds again (the player is toggled off, not just paused)', async () => {
  const player = fakePlayer();
  await hear(player, 4);
  assert.equal(player.timer.isRunning, true);
  await stopPlayer(player);
  assert.equal(player.timer.isRunning, false);
  assert.equal(player.isStarted, false);
  await hear(player, 4);
  assert.equal(player.timer.isRunning, true, 'the second Hear score is audible');
  assert.deepEqual(player.log, ['play', 'seek 4', 'pause', 'play', 'seek 4']);
});

test('a player paused from its own controls is resumed by the next Hear score, and a sounding one is not toggled', async () => {
  const player = fakePlayer();
  await hear(player, 0);
  player.pauseFromControls();
  assert.equal(player.isStarted, true);
  await hear(player, 8);
  assert.equal(player.timer.isRunning, true);
  const before = player.log.length;
  await hear(player, 12);
  assert.deepEqual(player.log.slice(before), ['seek 12'], 'seeking a sounding player does not toggle it');
  await stopPlayer(fakePlayer());
});

test('one source at a time, in both directions, however the score was started', () => {
  const calls: string[] = [];
  const owner = createPlaybackOwner({ recording: () => calls.push('pause recording'), score: () => calls.push('stop score') });
  owner.start('score');            // the embedded player's play button
  owner.start('recording');        // then Hear recording
  assert.deepEqual(calls, ['stop score']);
  owner.start('score');            // the embedded player starts during the recording
  assert.deepEqual(calls, ['stop score', 'pause recording']);
  owner.ended('score');
  owner.start('score');
  assert.deepEqual(calls, ['stop score', 'pause recording'], 'nothing else to silence once the score gave up its claim');
  owner.ended('recording');        // a stale end from the other source changes nothing
  assert.equal(owner.current, 'score');
});

test('main meter: the meter most bars are in, so one 2/4 bar does not make a 2/4 score', () => {
  assert.equal(mainMeter(REPORT), '4/4');
  const twoFour = [...HEADER.map(l => l === 'M:4/4' ? 'M:2/4' : l), '% verse', 'V: Vocal', 'c8c8|c8c8|c16|', 'V: Ins', 'Z3|'].join('\n');
  assert.equal(mainMeter(twoFour), '2/4');
  assert.equal(mainMeter(''), null);
});

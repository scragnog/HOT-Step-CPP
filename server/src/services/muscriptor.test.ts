import test from 'node:test';
import assert from 'node:assert/strict';
import { partialMidiFromEvents, readGrid, alignToAudio, exportArgs, speculativeArgs, modelDir } from './muscriptor.js';
import { parseMidiFile } from './midiParser.js';

test('speculativeArgs drafts with small only for large, when on and small is downloaded', () => {
  assert.deepEqual(speculativeArgs('large', true, true), ['--draft-model', modelDir('small'), '--speculative-k', '4']);
  assert.deepEqual(speculativeArgs('large', false, true), []);
  assert.deepEqual(speculativeArgs('large', true, false), []);
  assert.deepEqual(speculativeArgs('medium', true, true), []);
  assert.deepEqual(speculativeArgs('small', true, true), []);
});

test('partialMidiFromEvents keeps completed notes and drops the open one', () => {
  const mid = partialMidiFromEvents([
    { type: 'progress', completed: 0, total: 2 },
    { type: 'note_start', index: 0, pitch: 60, time: 0.5, instrument: 'acoustic_piano' },
    { type: 'note_start', index: 1, pitch: 36, time: 1, instrument: 'drums' },
    { type: 'note_end', index: 1, time: 1.01 },
    { type: 'note_end', index: 0, time: 1.5 },
    { type: 'note_start', index: 2, pitch: 40, time: 2, instrument: 'electric_bass' },
  ]);
  assert.ok(mid);
  const parsed = parseMidiFile(mid);
  assert.equal(parsed.noteCount, 2);
  const piano = parsed.notes.find(n => n.pitch === 60)!;
  assert.equal(piano.channel, 0);
  assert.ok(Math.abs(piano.start - 0.5) < 0.002 && Math.abs(piano.duration - 1) < 0.002);
  assert.equal(parsed.notes.find(n => n.pitch === 36)!.channel, 9);
});

test('partialMidiFromEvents writes the event velocity, 100 without one', () => {
  const mid = partialMidiFromEvents([
    { type: 'note_start', index: 0, pitch: 60, time: 0, instrument: 'voice', velocity: 37 },
    { type: 'note_end', index: 0, time: 1 },
    { type: 'note_start', index: 1, pitch: 62, time: 2, instrument: 'voice' },
    { type: 'note_end', index: 1, time: 3 },
  ])!;
  assert.deepEqual(parseMidiFile(mid).notes.map(n => n.velocity), [37, 100]);
});

test('partialMidiFromEvents is null when no note completed', () => {
  assert.equal(partialMidiFromEvents([{ type: 'note_start', index: 0, pitch: 60, time: 0, instrument: 'voice' }]), null);
});

test('partialMidiFromEvents pads a zero-length note instead of merging it into the next', () => {
  const mid = partialMidiFromEvents([
    { type: 'note_start', index: 0, pitch: 60, time: 0, instrument: 'acoustic_piano' },
    { type: 'note_end', index: 0, time: 0 },
    { type: 'note_start', index: 1, pitch: 60, time: 1, instrument: 'acoustic_piano' },
    { type: 'note_end', index: 1, time: 2 },
  ])!;
  const notes = parseMidiFile(mid).notes.map(n => [n.start, n.duration]);
  assert.equal(notes.length, 2);
  assert.ok(Math.abs(notes[0][0]) < 0.002 && Math.abs(notes[0][1] - 0.01) < 0.002);
  assert.ok(Math.abs(notes[1][0] - 1) < 0.002 && Math.abs(notes[1][1] - 1) < 0.002);
});

test('readGrid + alignToAudio move MIDI times back onto the audio', () => {
  const grid = readGrid(JSON.stringify({ grid: {
    bpm: 100, beats_per_bar: 4, steady: true, onset_delay: 0.01, subdivision: 2,
    audio_shift_s: -2.2, beats: [0.25, 0.85], bars: [0.25],
  } }))!;
  assert.equal(grid.beatsPerBar, 4);
  const out = alignToAudio({ durationSec: 9, notes: [{ start: 2.45, duration: 1 }] }, grid);
  assert.ok(Math.abs(out.notes[0].start - 0.25) < 1e-9);
  assert.ok(Math.abs(out.durationSec - 1.25) < 1e-9);
  assert.equal(readGrid('{"tracked_beats":[]}'), null);
});

test('exportArgs maps options to ace-midi flags and drops bad values', () => {
  assert.deepEqual(exportArgs({ bpm: 98, tempoMode: 'fixed', beatScale: 2, quantize: true, grid: 4, strength: 0.5 }),
    ['--bpm', '98', '--fixed-tempo', '--beat-scale', '2', '--quantize', '--grid', '4', '--quantize-strength', '0.5']);
  assert.deepEqual(exportArgs({ bpm: 5000, beatScale: 3, grid: 4, beatsPerBar: 2.5 }), []);
  assert.deepEqual(exportArgs({ fillGaps: 0.5, trimLeadingBars: true }), ['--fill-gaps', '0.5', '--trim-leading-bars']);
});
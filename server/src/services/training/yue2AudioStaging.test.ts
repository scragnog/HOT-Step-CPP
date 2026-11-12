import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { scanAudioFiles } from './datasetScan.js';
import { selectYue2Audio, stageYue2Audio } from './yue2AudioStaging.js';

function fixture(t: test.TestContext) {
  const root = fs.mkdtempSync(path.resolve('.yue2-staging-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'source');
  const output = path.join(root, 'output');
  fs.mkdirSync(source);
  const write = (name: string, content = name) => {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
    return file;
  };
  const samples = (recursive: boolean) => scanAudioFiles(source, recursive).map(f => ({
    audioPath: f.absPath, relPath: f.relPath, fileMissing: false, excluded: f.relPath === 'excluded.wav',
  }));
  return { root, source, output, write, samples };
}

test('stages exactly included dataset files, honoring recursive selection', t => {
  const f = fixture(t);
  f.write('root.wav');
  f.write('excluded.wav');
  f.write('nested/deep.flac');
  f.write('unsupported.opus');
  // Original route handed the engine this flat scan: excluded in, nested out.
  assert.deepEqual(scanAudioFiles(f.source, false).map(s => s.relPath), ['excluded.wav', 'root.wav', 'unsupported.opus']);
  const samples = [...f.samples(true), {
    audioPath: path.join(f.source, 'missing.wav'), relPath: 'missing.wav', fileMissing: true, excluded: false,
  }];
  const selected = selectYue2Audio(samples);
  assert.equal(selected.unsupported, 1);
  assert.equal(selected.needFfmpeg, 1);
  const dir = stageYue2Audio(samples, f.output);
  assert.deepEqual(fs.readdirSync(dir).sort(), ['deep.flac', 'root.wav']);
  stageYue2Audio(f.samples(false), f.output);
  assert.deepEqual(fs.readdirSync(dir), ['root.wav']);
  assert.equal(fs.readFileSync(path.join(f.source, 'excluded.wav'), 'utf8'), 'excluded.wav');
});

test('flattens duplicate names and pairs all engine sidecars, preserving unique basenames', t => {
  const f = fixture(t);
  for (const name of ['one/a.wav', 'two/a.wav', 'one__a.wav', 'unique.mp3']) f.write(name);
  for (const prefix of ['one/a', 'two/a', 'one__a']) {
    for (const suffix of ['.txt', '.yue2.txt', '.mm3.txt']) f.write(prefix + suffix);
  }
  const dir = stageYue2Audio(f.samples(true), f.output);
  const audio = fs.readdirSync(dir).filter(n => n.endsWith('.wav') || n.endsWith('.mp3'));
  assert.equal(audio.length, 4);
  assert.ok(audio.includes('unique.mp3'));
  assert.ok(audio.includes('one__a.wav'));
  assert.ok(audio.includes('two__a.wav'));
  for (const name of audio.filter(n => n.endsWith('.wav'))) {
    const original = fs.readFileSync(path.join(dir, name), 'utf8').replace(/\.wav$/, '');
    for (const suffix of ['.txt', '.yue2.txt', '.mm3.txt']) {
      assert.equal(fs.readFileSync(path.join(dir, name.replace(/\.wav$/, suffix)), 'utf8'), original + suffix);
    }
  }
});

test('copy fallback preserves bytes and mtime, and repeat prepare drops excluded audio and sidecars', t => {
  const f = fixture(t);
  const file = f.write('track.wav', 'audio bytes');
  f.write('track.txt', 'lyrics: words');
  f.write('keep.wav');
  const when = new Date('2026-01-01T00:00:00Z');
  fs.utimesSync(file, when, when);
  let copies = 0;
  const dir = stageYue2Audio(f.samples(true), f.output, () => { copies++; throw new Error('EXDEV'); });
  assert.equal(copies, 3);
  assert.equal(fs.readFileSync(path.join(dir, 'track.wav'), 'utf8'), 'audio bytes');
  assert.equal(fs.statSync(path.join(dir, 'track.wav')).mtimeMs, when.getTime());
  stageYue2Audio(f.samples(true).map(s => ({ ...s, excluded: s.relPath === 'track.wav' })), f.output);
  assert.deepEqual(fs.readdirSync(dir), ['keep.wav']);
  assert.ok(fs.existsSync(file));
});

test('refuses staging from its own persistent files before removing them', t => {
  const f = fixture(t);
  f.write('track.wav');
  const dir = stageYue2Audio(f.samples(true), f.output);
  assert.throws(() => stageYue2Audio([{
    audioPath: path.join(dir, 'track.wav'), relPath: 'track.wav', excluded: false, fileMissing: false,
  }], f.output), /own source audio/);
  assert.ok(fs.existsSync(path.join(dir, 'track.wav')));
});

test('reserves every sidecar destination across distinct audio stems', t => {
  const f = fixture(t);
  f.write('track.wav');
  f.write('track.yue2.wav');
  f.write('track.yue2.txt', 'planner caption or second track metadata');
  const dir = stageYue2Audio(f.samples(true), f.output);
  const audio = fs.readdirSync(dir).filter(n => n.endsWith('.wav'));
  assert.equal(audio.length, 2);
  for (const name of audio) {
    const original = fs.readFileSync(path.join(dir, name), 'utf8');
    const suffix = original === 'track.wav' ? '.yue2.txt' : '.txt';
    assert.equal(fs.readFileSync(path.join(dir, name.replace(/\.wav$/, suffix)), 'utf8'),
      'planner caption or second track metadata');
  }
});

test('rebuilds staging under a non-ASCII output without retaining old files', t => {
  const f = fixture(t);
  f.write('Song \u2014 Caf\u00e9.wav', 'original audio');
  f.write('keep.wav');
  const output = path.join(f.output, 'Run \u2014 Caf\u00e9');
  const dir = stageYue2Audio(f.samples(true), output);
  stageYue2Audio(f.samples(true).filter(s => s.relPath === 'keep.wav'), output);
  assert.deepEqual(fs.readdirSync(dir), ['keep.wav']);
  assert.equal(fs.readFileSync(path.join(f.source, 'Song \u2014 Caf\u00e9.wav'), 'utf8'), 'original audio');
});

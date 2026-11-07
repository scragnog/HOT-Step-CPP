import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test, { after } from 'node:test';
import { stageYue2Audio, yue2SeparationAudioDir } from './yue2AudioStaging.js';

// Runner imports recover persisted jobs and pipelines. Keep these tests away
// from the app's training state before loading any of those modules.
const trainingRoot = fs.mkdtempSync(path.resolve('.yue2-staging-callers-'));
process.env.TRAINING_DIR = path.join(trainingRoot, 'training');
const { yue2CaptionsStale } = await import('./yue2AitkPrepareRunner.js');
const { yue2StemSources, yue2VocalPath } = await import('./yue2Stems.js');
after(() => fs.rmSync(trainingRoot, { recursive: true, force: true }));

function fixture(t: test.TestContext, copy = false) {
  const root = fs.mkdtempSync(path.join(trainingRoot, 'case-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'source');
  const output = path.join(root, 'output');
  fs.mkdirSync(source);
  fs.mkdirSync(output);
  const write = (name: string, text = name) => {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, text);
    return file;
  };
  const stage = (names: string[]) => stageYue2Audio(names.map(relPath => ({
    relPath, audioPath: path.join(source, relPath), excluded: false, fileMissing: false,
  })), output, copy ? () => { throw new Error('EXDEV'); } : fs.linkSync);
  const manifestPath = path.join(output, 'yue2_preprocess.json');
  const manifest = (dir: string, caption = '') => {
    const sources = fs.readdirSync(dir).filter(n => n.endsWith('.wav')).map(name => ({
      name, source: path.join(dir, name), caption,
    }));
    fs.writeFileSync(manifestPath, JSON.stringify({ audio_dir: dir, caption_mode: 'yue2', sources }));
    return sources;
  };
  return { source, output, write, stage, manifest, manifestPath };
}

test('caption staleness sees newly created original planner sidecars after staging', t => {
  const f = fixture(t);
  f.write('nested/track.wav');
  const dir = f.stage(['nested/track.wav']);
  f.manifest(dir);
  assert.equal(yue2CaptionsStale(f.manifestPath), false);
  f.write('nested/track.yue2.txt', 'new planner caption');
  assert.equal(yue2CaptionsStale(f.manifestPath), true);
  assert.equal(fs.readFileSync(path.join(dir, 'track.yue2.txt'), 'utf8'), 'new planner caption');
  f.manifest(dir, 'new planner caption');
  assert.equal(yue2CaptionsStale(f.manifestPath), false);
});

test('caption staleness refreshes copies and replaced hardlinks without changing originals', t => {
  for (const copy of [false, true]) {
    const f = fixture(t, copy);
    f.write('one/track.wav');
    f.write('two/track.wav');
    const original = f.write('one/track.yue2.txt', 'old caption');
    const dir = f.stage(['one/track.wav', 'two/track.wav']);
    const sources = f.manifest(dir, 'old caption');
    // A sidecar writer may atomically replace the inode the staging link used.
    fs.unlinkSync(original);
    f.write('one/track.yue2.txt', 'updated caption');
    assert.equal(yue2CaptionsStale(f.manifestPath), true);
    const staged = sources.find(s => s.name === 'one__track.wav')!;
    const sidecar = staged.source.replace(/\.wav$/, '.yue2.txt');
    assert.equal(fs.readFileSync(sidecar, 'utf8'), 'updated caption');
    assert.equal(fs.readFileSync(original, 'utf8'), 'updated caption');
    // Copy fallback also needs refresh after an in-place edit.
    fs.writeFileSync(original, 'edited again');
    yue2CaptionsStale(f.manifestPath);
    assert.equal(fs.readFileSync(sidecar, 'utf8'), 'edited again');
    fs.unlinkSync(original);
    yue2CaptionsStale(f.manifestPath);
    assert.equal(fs.existsSync(sidecar), false);
  }
});

test('caption refresh replaces copied sidecars whose names hold non-ASCII characters', t => {
  const f = fixture(t, true);
  f.write('Song — Café.wav');
  f.write('Song — Café.txt', 'old lyrics');
  const dir = f.stage(['Song — Café.wav']);
  f.manifest(dir);
  f.write('Song — Café.txt', 'new lyrics');
  yue2CaptionsStale(f.manifestPath);
  assert.equal(fs.readFileSync(path.join(dir, 'Song — Café.txt'), 'utf8'), 'new lyrics');
});

test('separation reads nested and renamed staged sources with the exact aligner stem names', t => {
  const f = fixture(t);
  for (const name of ['one/track.wav', 'two/track.wav', 'unique.wav']) f.write(name);
  f.write('excluded.wav');
  const dir = f.stage(['one/track.wav', 'two/track.wav', 'unique.wav']);
  const manifestSources = f.manifest(dir);
  const audioDir = yue2SeparationAudioDir(f.manifestPath, f.source);
  assert.equal(audioDir, dir);
  const separationSources = yue2StemSources(audioDir);
  assert.deepEqual(separationSources.map(s => path.basename(s)).sort(), manifestSources.map(s => s.name).sort());
  for (const source of manifestSources) {
    assert.equal(yue2VocalPath('stems', source.source), path.join('stems', path.parse(source.name).name, 'vocals.wav'));
  }
  fs.writeFileSync(f.manifestPath, JSON.stringify({ sources: [] }));
  assert.equal(yue2SeparationAudioDir(f.manifestPath, f.source), f.source);
  fs.writeFileSync(f.manifestPath, JSON.stringify({ audio_dir: path.join(dir, 'moved-away'), sources: [] }));
  assert.equal(yue2SeparationAudioDir(f.manifestPath, f.source), f.source);
});

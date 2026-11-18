import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { replaceSelectedCaptions as replace, CaptionReplaceError } from './captionReplace.js';

function fixture(t: { after(fn: () => void): void }) {
  const root = fs.mkdtempSync(path.resolve('.test-caption-replace-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const samples = ['one', 'two'].map(sampleId => ({ sampleId, audioPath: path.join(root, `${sampleId}.wav`), sidecarPath: path.join(root, `${sampleId}.txt`) }));
  const write = (file: string, text: string) => fs.writeFileSync(path.join(root, file), text, 'utf8');
  const read = (file: string) => fs.readFileSync(path.join(root, file), 'utf8');
  return { root, samples, write, read };
}
const input = { sampleIds: ['one'], find: 'guitar', replacement: 'cello' };

test('preview is read-only; apply changes only selected captions, preserving Unicode, lyrics and CRLF', t => {
  const f = fixture(t);
  const original = '\uFEFFcaption: café guitar, guitar\r\n  温かい guitar\r\ngenre: guitar\r\nbpm: 90\r\nlyrics:\r\nguitar 歌詞\r\ncaption: guitar\r\n';
  f.write('one.txt', original);
  f.write('one.yue2.txt', 'café guitar / guitar 日本語\n');
  f.write('one.mm3.txt', 'guitar');
  f.write('two.txt', 'caption: guitar\n');
  f.write('dataset.json', '{"caption":"guitar"}');
  const preview = replace(f.root, f.samples, input);
  assert.equal(preview.tracks, 1);
  assert.deepEqual(preview.files.map(c => c.file), ['one.txt', 'one.yue2.txt']);
  assert.equal(f.read('one.txt'), original);
  replace(f.root, f.samples, { ...input, previewToken: preview.previewToken });
  assert.equal(f.read('one.txt'), original.replace('café guitar, guitar', 'café cello, cello').replace('温かい guitar', '温かい cello'));
  assert.equal(f.read('one.yue2.txt'), 'café cello / cello 日本語\n');
  assert.equal(f.read('two.txt'), 'caption: guitar\n');
  assert.equal(f.read('one.mm3.txt'), 'guitar');
  assert.equal(f.read('dataset.json'), '{"caption":"guitar"}');
  assert.equal(fs.readdirSync(f.root).some(p => p.endsWith('.tmp')), false);
});

test('literal case-sensitive replacement supports punctuation, dollar signs, deletion and duplicate selection', t => {
  const f = fixture(t);
  f.write('one.txt', 'caption: a.* $& a.* A.*\nlyrics:\na.*\n');
  const request = { sampleIds: ['one', 'one'], find: 'a.*', replacement: '$&' };
  const preview = replace(f.root, f.samples, request);
  assert.equal(preview.files.length, 1);
  replace(f.root, f.samples, { ...request, previewToken: preview.previewToken });
  assert.equal(f.read('one.txt'), 'caption: $& $& $& A.*\nlyrics:\na.*\n');
  const deletion = { ...input, find: '$&', replacement: '' };
  replace(f.root, f.samples, { ...deletion, previewToken: replace(f.root, f.samples, deletion).previewToken });
  assert.equal(f.read('one.txt'), 'caption:    A.*\nlyrics:\na.*\n');
});

test('rejects stale previews before writing any file', t => {
  const f = fixture(t);
  f.write('one.txt', 'caption: guitar\n');
  f.write('one.yue2.txt', 'guitar');
  const preview = replace(f.root, f.samples, input);
  f.write('one.yue2.txt', 'guitar edited');
  assert.throws(() => replace(f.root, f.samples, { ...input, previewToken: preview.previewToken }), /changed since preview/);
  assert.equal(f.read('one.txt'), 'caption: guitar\n');
});

test('preview binds selection, find, replacement and newly appearing files', t => {
  const f = fixture(t);
  f.write('one.txt', 'caption: guitar');
  f.write('two.txt', 'caption: guitar');
  const previewToken = replace(f.root, f.samples, input).previewToken;
  for (const changed of [{ replacement: 'bass' }, { find: 'cello' }, { sampleIds: ['two'] }]) {
    assert.throws(() => replace(f.root, f.samples, { ...input, ...changed, previewToken }), /changed since preview/);
  }
  f.write('one.yue2.txt', 'guitar');
  assert.throws(() => replace(f.root, f.samples, { ...input, previewToken }), /changed since preview/);
});

test('missing captions and lyrics-only files have no matches', t => {
  const f = fixture(t);
  f.write('one.txt', 'guitar in plain lyrics\n');
  assert.equal(replace(f.root, f.samples, input).files.length, 0);
  assert.equal(replace(f.root, f.samples, { ...input, sampleIds: ['two'] }).files.length, 0);
});

test('invalid requests, unknown and missing tracks are rejected', t => {
  const f = fixture(t);
  for (const invalid of [{ find: '' }, { replacement: 'caption\nlyrics: bad' }, { sampleIds: [] }, { sampleIds: [3] }, { previewToken: 3 }]) {
    assert.throws(() => replace(f.root, f.samples, { ...input, ...invalid } as any), CaptionReplaceError);
  }
  assert.throws(() => replace(f.root, f.samples, { ...input, sampleIds: ['unknown'] }), /missing/);
  assert.throws(() => replace(f.root, f.samples.map(s => ({ ...s, fileMissing: true })), input), /missing/);
});

test('linked caption files and paths outside the dataset are rejected', t => {
  const f = fixture(t);
  f.write('two.txt', 'caption: guitar');
  fs.linkSync(path.join(f.root, 'two.txt'), path.join(f.root, 'one.txt'));
  assert.throws(() => replace(f.root, f.samples, input), /Linked/);
  assert.equal(f.read('two.txt'), 'caption: guitar');
  assert.throws(() => replace(f.root, [{ ...f.samples[0], sidecarPath: path.join(f.root, '..', 'outside.txt') }], input), /outside/);
});

test('rename failure preserves original file and removes the temporary file', t => {
  const f = fixture(t);
  f.write('one.txt', 'caption: guitar');
  const previewToken = replace(f.root, f.samples, input).previewToken;
  const rename = t.mock.method(fs, 'renameSync', () => { throw new Error('rename failed'); });
  assert.throws(() => replace(f.root, f.samples, { ...input, previewToken }), /rename failed/);
  rename.mock.restore();
  assert.equal(f.read('one.txt'), 'caption: guitar');
  assert.equal(fs.readdirSync(f.root).some(p => p.endsWith('.tmp')), false);
});

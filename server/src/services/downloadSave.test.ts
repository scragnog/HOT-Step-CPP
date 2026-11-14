import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { parse as dotenvParse } from 'dotenv';
import { formatEnvValue, parseEnvValue } from '../config.js';
import { revealCommand, safeFilename, saveIntoDir, validateDownloadDir } from './downloadSave.js';

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'dlsave-'));
const source = (dir: string) => { const f = path.join(dir, 'src.bin'); fs.writeFileSync(f, 'audio'); return f; };

test('safeFilename drops directories, absolute prefixes and traversal', () => {
  assert.equal(safeFilename('..\\..\\evil.flac'), 'evil.flac');
  assert.equal(safeFilename('../../evil.flac'), 'evil.flac');
  assert.equal(safeFilename('C:\\Windows\\evil.flac'), 'evil.flac');
  assert.equal(safeFilename('/etc/passwd'), 'passwd');
  assert.equal(safeFilename('..'), 'download');
  assert.equal(safeFilename(''), 'download');
});

test('safeFilename strips reserved characters and device names', () => {
  assert.equal(safeFilename('a<b>:"c|d?e*.wav'), 'abcde.wav');
  assert.equal(safeFilename('trailing. .'), 'trailing');
  assert.equal(safeFilename('CON.wav'), '_CON.wav');
  assert.equal(safeFilename('lpt1.flac'), '_lpt1.flac');
  assert.equal(safeFilename('console.wav'), 'console.wav');
});

test('saveIntoDir keeps traversal names inside the folder', () => {
  const root = tmp(), out = path.join(root, 'out');
  const saved = saveIntoDir(out, '..\\..\\evil.flac', source(root));
  assert.equal(path.dirname(saved), fs.realpathSync(out));
  assert.equal(path.basename(saved), 'evil.flac');
  assert.ok(!fs.existsSync(path.join(root, 'evil.flac')));
});

test('saveIntoDir never overwrites: renames to (2), (3)', () => {
  const root = tmp(), out = path.join(root, 'out'), src = source(root);
  const a = saveIntoDir(out, 'song.flac', src);
  fs.writeFileSync(a, 'first');
  const b = saveIntoDir(out, 'song.flac', src);
  const c = saveIntoDir(out, 'song.flac', src);
  assert.equal(path.basename(b), 'song (2).flac');
  assert.equal(path.basename(c), 'song (3).flac');
  assert.equal(fs.readFileSync(a, 'utf8'), 'first');
});

test('saveIntoDir does not follow a junction/symlink planted in the folder', () => {
  const root = tmp(), out = path.join(root, 'out'), outside = path.join(root, 'outside');
  fs.mkdirSync(out); fs.mkdirSync(outside);
  const link = path.join(out, 'song.flac');
  fs.symlinkSync(outside, link, 'junction'); // junction on Windows, plain symlink elsewhere
  const saved = saveIntoDir(out, 'song.flac', source(root));
  assert.equal(path.basename(saved), 'song (2).flac');
  assert.deepEqual(fs.readdirSync(outside), []);
});

test('saveIntoDir does not write through a dangling file symlink', () => {
  const root = tmp(), out = path.join(root, 'out'), target = path.join(root, 'target.flac');
  fs.mkdirSync(out);
  try { fs.symlinkSync(target, path.join(out, 'song.flac'), 'file'); } catch { return; } // no symlink privilege
  const saved = saveIntoDir(out, 'song.flac', source(root));
  assert.equal(path.basename(saved), 'song (2).flac');
  assert.ok(!fs.existsSync(target));
});

test('validateDownloadDir rejects relative paths and creates missing ones', () => {
  assert.throws(() => validateDownloadDir('relative/dir'), /absolute/);
  const dir = path.join(tmp(), 'a', 'b');
  validateDownloadDir(dir);
  assert.ok(fs.statSync(dir).isDirectory());
});

test('validateDownloadDir probe does not truncate a planted symlink', () => {
  const root = tmp(), out = path.join(root, 'out'), target = path.join(root, 'target.txt');
  fs.mkdirSync(out); fs.writeFileSync(target, 'data');
  try { fs.symlinkSync(target, path.join(out, `.hotstep-write-test-${process.pid}`), 'file'); } catch { return; }
  validateDownloadDir(out);
  assert.equal(fs.readFileSync(target, 'utf8'), 'data');
});

test('env values round-trip through the .env writer and dotenv', () => {
  for (const v of ['D:/Music #1', 'D:\\new\\tab #1', '/mnt/a "b" c', "it's #x", ' lead', 'plain/path']) {
    assert.equal(dotenvParse(`K=${formatEnvValue(v)}`).K, v);
    assert.equal(parseEnvValue(formatEnvValue(v)), v);
  }
  assert.throws(() => formatEnvValue('a#`b'), /backtick/);
});

test('revealCommand selects a file inside the folder and ignores anything else', () => {
  const dir = tmp();
  const real = fs.realpathSync(dir);
  const inside = path.join(real, 'song.flac');
  fs.writeFileSync(inside, 'x');
  const outside = source(tmp());
  assert.deepEqual(revealCommand(dir, inside, 'win32'), { cmd: 'explorer.exe', args: ['/select,', inside] });
  assert.deepEqual(revealCommand(dir, inside, 'darwin'), { cmd: 'open', args: ['-R', inside] });
  assert.deepEqual(revealCommand(dir, outside, 'win32'), { cmd: 'explorer.exe', args: [real] });
  assert.deepEqual(revealCommand(dir, path.join(real, 'missing.flac'), 'win32'), { cmd: 'explorer.exe', args: [real] });
  assert.deepEqual(revealCommand(dir, 'song.flac', 'win32'), { cmd: 'explorer.exe', args: [real] });
  assert.deepEqual(revealCommand(dir, undefined, 'linux'), { cmd: 'xdg-open', args: [real] });
});

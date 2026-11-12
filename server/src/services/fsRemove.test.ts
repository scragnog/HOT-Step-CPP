import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { removeSync } from './fsRemove.js';

function fixture(t: test.TestContext): string {
  const root = fs.mkdtempSync(path.resolve('.fs-remove-test-'));
  t.after(() => removeSync(root, { recursive: true, force: true }));
  return root;
}

test('deletes non-ASCII files and folders, including ASCII entries under them', t => {
  const root = fixture(t);
  const file = path.join(root, 'Song \u2014 Caf\u00e9.wav');
  const dir = path.join(root, 'Folder \u2014 Caf\u00e9');
  fs.writeFileSync(file, 'audio');
  fs.mkdirSync(path.join(dir, 'nested'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'nested', 'ascii.txt'), 'text');
  fs.writeFileSync(path.join(dir, 'Song \u2014 Caf\u00e9.txt'), 'text');
  removeSync(file, { force: true });
  removeSync(dir, { recursive: true, force: true });
  assert.equal(fs.existsSync(file), false);
  assert.equal(fs.existsSync(dir), false);
});

test('retains missing-path and nonrecursive-directory errors', t => {
  const root = fixture(t);
  const missing = path.join(root, 'missing');
  assert.throws(() => removeSync(missing), { code: 'ENOENT' });
  removeSync(missing, { force: true });
  assert.throws(() => removeSync(root));
  assert.ok(fs.existsSync(root));
});

test('fallback removes a junction or symlink without deleting its target', t => {
  const root = fixture(t);
  const target = path.join(root, 'target');
  const link = path.join(root, 'Link \u2014 Caf\u00e9');
  fs.mkdirSync(target);
  fs.writeFileSync(path.join(target, 'keep.txt'), 'keep');
  fs.symlinkSync(target, link, process.platform === 'win32' ? 'junction' : 'dir');
  const noop = t.mock.method(fs, 'rmSync', () => {});
  try { removeSync(link, { recursive: true, force: true }); }
  finally { noop.mock.restore(); }
  assert.equal(fs.existsSync(link), false);
  assert.equal(fs.readFileSync(path.join(target, 'keep.txt'), 'utf8'), 'keep');
});

test('reports the path if fallback deletion fails', t => {
  const root = fixture(t);
  const file = path.join(root, 'locked.txt');
  fs.writeFileSync(file, 'keep');
  const noop = t.mock.method(fs, 'rmSync', () => {});
  const fail = t.mock.method(fs, 'unlinkSync', () => { throw new Error('access denied'); });
  try { assert.throws(() => removeSync(file, { force: true }), err =>
    err instanceof Error && err.message.includes(file) && err.message.includes('access denied')); }
  finally { noop.mock.restore(); fail.mock.restore(); }
});

test('throws if fallback also silently leaves the path behind', t => {
  const root = fixture(t);
  const file = path.join(root, 'retained.txt');
  fs.writeFileSync(file, 'keep');
  const rm = t.mock.method(fs, 'rmSync', () => {});
  const unlink = t.mock.method(fs, 'unlinkSync', () => {});
  try { assert.throws(() => removeSync(file, { force: true }), err =>
    err instanceof Error && err.message.includes(file) && err.message.includes('still exists')); }
  finally { rm.mock.restore(); unlink.mock.restore(); }
});

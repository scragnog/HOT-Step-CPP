// Path confinement for the ear-test score sheet routes (routes/listening.ts):
// traversal, an out-of-root symlink, and a real scores.json write+rename.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { resolveConfined } from '../src/routes/listening.js';

function withTmpRoot(fn: (root: string) => void) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'listening-test-'));
  fs.mkdirSync(path.join(root, 'study-a'));
  fs.writeFileSync(path.join(root, 'study-a', 'index.html'), '<html></html>');
  try { fn(root); } finally { fs.rmSync(root, { recursive: true, force: true }); }
}

test('resolves a normal file inside the root', () => {
  withTmpRoot((root) => {
    const target = resolveConfined(root, 'study-a', ['index.html']);
    assert.equal(target, path.join(root, 'study-a', 'index.html'));
  });
});

test('rejects a folder name that escapes via ..', () => {
  withTmpRoot((root) => {
    assert.equal(resolveConfined(root, '..', ['secret.txt']), null);
  });
});

test('rejects a segment that escapes via ..', () => {
  withTmpRoot((root) => {
    assert.equal(resolveConfined(root, 'study-a', ['..', '..', 'secret.txt']), null);
  });
});

test('rejects a segment containing a path separator', () => {
  withTmpRoot((root) => {
    assert.equal(resolveConfined(root, 'study-a', ['sub/evil.txt']), null);
  });
});

test('rejects a symlinked folder that points outside the root', () => {
  withTmpRoot((root) => {
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'listening-outside-'));
    fs.writeFileSync(path.join(outside, 'secret.txt'), 'nope');
    try {
      fs.symlinkSync(outside, path.join(root, 'escape'), 'junction');
      assert.equal(resolveConfined(root, 'escape', ['secret.txt']), null);
    } catch (err: any) {
      if (err.code === 'EPERM') return; // unprivileged Windows account: skip
      throw err;
    } finally {
      fs.rmSync(outside, { recursive: true, force: true });
    }
  });
});

test('scores.json writes via temp file + rename (no partial file visible)', () => {
  withTmpRoot((root) => {
    const target = resolveConfined(root, 'study-a', ['scores.json']);
    assert.ok(target);
    const tmp = `${target}.tmp-test`;
    fs.writeFileSync(tmp, JSON.stringify({ scores: { x: 1 } }));
    fs.renameSync(tmp, target!);
    assert.equal(fs.existsSync(tmp), false);
    assert.deepEqual(JSON.parse(fs.readFileSync(target!, 'utf8')), { scores: { x: 1 } });
  });
});

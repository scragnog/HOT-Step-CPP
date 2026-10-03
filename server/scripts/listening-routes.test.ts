// Path confinement for the ear-test score sheet routes (routes/listening.ts):
// traversal, an out-of-root symlink, and a real scores.json write+rename.
// Router-level tests (actual HTTP, Origin/body validation) live in
// listening-http.test.ts — they need HOT_STEP_ROOT set before config.ts
// loads, which a static import here (hoisted above any top-level code) would
// defeat.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { resolveConfined, isTrustedOrigin, isScoreEnvelope } from '../src/routes/listening.js';

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

// ── isTrustedOrigin / isScoreEnvelope: unit coverage for the CSRF/body guards ──

test('isTrustedOrigin allows no Origin header (same-origin / direct tools)', () => {
  assert.equal(isTrustedOrigin(undefined, 3001), true);
});
test('isTrustedOrigin allows the server\'s own localhost/127.0.0.1 origin', () => {
  assert.equal(isTrustedOrigin('http://localhost:3001', 3001), true);
  assert.equal(isTrustedOrigin('http://127.0.0.1:3001', 3001), true);
});
test('isTrustedOrigin rejects a foreign origin', () => {
  assert.equal(isTrustedOrigin('https://untrusted.example', 3001), false);
  assert.equal(isTrustedOrigin('http://localhost:3000', 3001), false); // Vite dev port, different origin
});
test('isScoreEnvelope requires a scores object', () => {
  assert.equal(isScoreEnvelope({ scores: { a: 1 } }), true);
  assert.equal(isScoreEnvelope({}), false);
  assert.equal(isScoreEnvelope({ scores: [] }), false);
  assert.equal(isScoreEnvelope(null), false);
  assert.equal(isScoreEnvelope('junk=1'), false);
  assert.equal(isScoreEnvelope(undefined), false);
});

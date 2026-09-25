// Tier 0 — static checks. Needs no running app.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { MIN, REPO, sh, walk, tail } from './lib.mjs';

const SERVER = path.join(REPO, 'server');
const UI = path.join(REPO, 'ui');

test('server type-checks (tsc --noEmit)', { timeout: 10 * MIN }, () => {
  const r = sh('npx', ['tsc', '--noEmit'], { cwd: SERVER });
  assert.equal(r.status, 0, tail(r.out));
});

test('ui type-checks (tsc --noEmit -p tsconfig.app.json)', { timeout: 10 * MIN }, () => {
  // A bare `tsc --noEmit` in ui/ checks nothing: the root tsconfig has no inputs.
  const r = sh('npx', ['tsc', '--noEmit', '-p', 'tsconfig.app.json'], { cwd: UI });
  assert.equal(r.status, 0, tail(r.out));
});

test('engine fork hooks and ggml patches intact (verify-hooks.ps1)', { timeout: 2 * MIN }, (t) => {
  if (process.platform !== 'win32') return t.skip('verify-hooks.ps1 is a Windows script');
  const r = sh('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(REPO, 'engine', 'verify-hooks.ps1')]);
  assert.equal(r.status, 0, tail(r.out));
});

test('server unit tests (src/**/*.test.ts via tsx --test)', { timeout: 15 * MIN }, (t) => {
  const files = walk(path.join(SERVER, 'src'), /\.test\.ts$/).map((f) => path.relative(SERVER, f));
  assert.ok(files.length > 0, 'no unit tests found under server/src');
  const r = sh('npx', ['tsx', '--test', ...files], { cwd: SERVER });
  assert.equal(r.status, 0, tail(r.out, 80));
  const ran = Number(r.out.match(/tests (\d+)/)?.[1] ?? 0);
  const failed = Number(r.out.match(/fail (\d+)/)?.[1] ?? 0);
  t.diagnostic(`${files.length} files, ${ran} tests, ${failed} failed`);
  assert.ok(ran > 0, `tsx --test reported no tests:\n${tail(r.out, 20)}`);
  assert.equal(failed, 0);
});

test('release prerequisites: weights on Hugging Face, data files packaged', { timeout: 10 * MIN }, () => {
  const args = [path.join(SERVER, 'scripts', 'check-release-prereqs.mjs')];
  if (process.env.GATE_OFFLINE === '1') args.push('--offline');
  const r = sh(process.execPath, args);
  assert.equal(r.status, 0, tail(r.out, 60));
});

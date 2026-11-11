// An open VST editor window holds an update, whether or not the request named a uid.
// Stub children only: no vst-host process starts. Fixtures live under the OS temp folder.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

test('every GUI child blocks maintenance.begin() until it exits, with or without a uid', async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-vst-gui-'));
  const d = { updaterDir: path.join(base, 'updater'), installRoot: path.join(base, 'install'), dataRoot: path.join(base, 'data') };
  for (const p of Object.values(d)) fs.mkdirSync(p, { recursive: true });
  process.env.HOT_STEP_ROOT = d.installRoot;
  process.env.HOT_STEP_DATA_ROOT = d.dataRoot;
  process.env.HOT_STEP_PROFILE_DIR = path.join(base, 'profile');

  const cp = createRequire(import.meta.url)('node:child_process');
  const original = cp.spawn;
  const children: EventEmitter[] = [];
  cp.spawn = () => {
    const child = Object.assign(new EventEmitter(), { pid: 12345, exitCode: null, killed: false, stdout: null, stderr: null, unref() {} });
    children.push(child);
    return child;
  };
  syncBuiltinESMExports();
  try {
    const { config } = await import('../config.js');
    Object.defineProperty(config.vst, 'exe', { get: () => process.execPath });
    const { default: router } = await import('./vst.js');
    const { maintenance, releaseLease } = await import('../services/updates/maintenance.js');
    const gui = (router.stack.find((l: any) => l.route?.path === '/gui') as any).route.stack[0].handle;
    const begin = () => maintenance.begin({ ...d, preflight: () => [], blockers: () => [] });
    const refused = () => assert.rejects(begin(), (e: any) => e.status === 409 && e.blockers.some((s: string) => s.includes('VST plugin window')));

    for (const body of [{ pluginPath: 'fixture.vst3' }, { pluginPath: 'fixture.vst3', uid: 'chain-1' }]) {
      let reply: any;
      gui({ body }, { json(v: any) { reply = v; } });
      assert.equal(reply.ok, true, JSON.stringify(body));
      await refused();
      children.at(-1)!.emit('exit', 0, null);
      const taken = await begin();
      releaseLease(d, taken.lease.id);
      maintenance.open();
    }
    assert.equal(children.length, 2);
  } finally {
    cp.spawn = original;
    syncBuiltinESMExports();
    try { fs.rmSync(base, { recursive: true, force: true }); } catch { /* a handle may linger on Windows; temp folder */ }
  }
});

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import express from 'express';
import { execFileSync } from 'child_process';
import { RUNTIME } from '../dataroot/bootstrap.js';
import { initLayout, dataLayout, readPointer, writePointer } from '../dataroot/paths.js';
import { getDataLocation, getBusyBlockers, registerBusyCheck, previewOperation, scheduleOperation, cancelPending } from '../dataroot/pointer.js';
import { createDataFolderRouter } from './settingsDataFolder.js';

test('Settings previews, refuses busy work, persists before restart and separates all four operations', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-settings-'));
  const original = { ...RUNTIME };
  const appRoot = path.join(dir, 'app');
  const root = path.join(dir, 'data');
  const pointer = path.join(dir, 'profile', 'data-location.json');
  const existing = path.join(dir, 'existing');
  const oldApp = path.join(dir, 'previous');
  fs.mkdirSync(appRoot);
  fs.mkdirSync(path.join(oldApp, 'data'), { recursive: true });
  fs.writeFileSync(path.join(oldApp, 'data', 'audio.wav'), 'old audio');
  initLayout(dataLayout(root));
  initLayout(dataLayout(existing));
  writePointer(pointer, { version: 1, root, pending: null });
  Object.assign(RUNTIME, { mode: 'release', appRoot, dataRoot: root, pointerPath: pointer, override: null });
  let blockers: string[] = [];
  registerBusyCheck('settings-test', () => blockers);
  let restartCount = 0;
  let failRestart = false;
  const app = express();
  app.use(express.json());
  app.use('/data-folder', createDataFolderRouter({
    location: getDataLocation, blockers: getBusyBlockers,
    preview: (kind, target) => previewOperation(kind, target, null),
    schedule: (kind, target) => scheduleOperation(kind, target, null),
    cancel: cancelPending,
    restart: () => {
      assert.ok(readPointer(pointer)?.pending, 'pending must be durable before restart');
      if (failRestart) throw new Error('marker unavailable');
      fs.writeFileSync(path.join(dir, 'profile', 'restart-requested'), 'restart');
      restartCount++;
    },
  }));
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.once('listening', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}/data-folder`;
  const post = async (suffix: string, body: unknown) => {
    const response = await fetch(base + suffix, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return { status: response.status, body: await response.json() };
  };
  try {
    assert.equal((await post('/preview', { kind: 'migrate', target: existing })).status, 400);
    assert.equal((await post('/schedule', { kind: 'move', target: './relative' })).status, 400);
    for (const kind of ['move', 'repoint', 'create', 'import'] as const) {
      const target = kind === 'repoint' ? existing : kind === 'import' ? oldApp : path.join(dir, kind);
      const input = { kind, target };
      const preview = await post('/preview', input);
      assert.equal(preview.body.ok, true, preview.body.errors?.join('; '));
      assert.equal(readPointer(pointer)?.pending, null, 'preview must not write');
      for (const source of ['generation', 'training', 'audio/workflow queue', 'download']) {
        blockers = [source];
        const refused = await post('/schedule', input);
        assert.equal(refused.status, 409);
        assert.deepEqual(refused.body.blockers, [source]);
        assert.equal(readPointer(pointer)?.pending, null);
      }
      blockers = [];
      const scheduled = await post('/schedule', input);
      assert.equal(scheduled.status, 200, scheduled.body.error);
      const saved = readPointer(pointer)!.pending!;
      assert.equal(saved.kind, kind);
      assert.equal(saved.destination, kind === 'import' ? root : target);
      if (kind === 'import') assert.equal(saved.source, oldApp);
      if (kind === 'repoint' || kind === 'create') assert.equal(saved.mappings.length, 0);
      assert.equal(readPointer(pointer)?.root, root, 'active root cannot hot-apply');
      assert.ok(fs.existsSync(root));
      assert.ok(fs.existsSync(path.join(oldApp, 'data', 'audio.wav')));
      assert.equal((await post('/schedule', input)).status, 409, 'second change must refuse');
      assert.equal((await post('/cancel', {})).body.cancelled, true);
    }
    assert.equal(restartCount, 4);
    failRestart = true;
    const failed = await post('/schedule', { kind: 'repoint', target: existing });
    assert.equal(failed.status, 503);
    assert.equal(readPointer(pointer)?.pending?.kind, 'repoint', 'failed restart retains scheduled change');
    const location = await (await fetch(base)).json();
    assert.equal(location.pending.destination, existing);
    await post('/cancel', {});
    RUNTIME.override = root;
    assert.equal((await post('/schedule', { kind: 'repoint', target: existing })).status, 400);
    RUNTIME.mode = 'dev'; RUNTIME.appRoot = null; RUNTIME.pointerPath = null;
    assert.equal((await post('/schedule', { kind: 'move', target: existing })).status, 400);
  } finally {
    blockers = [];
    Object.assign(RUNTIME, original);
    await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('release restart writes the profile marker before scheduling graceful shutdown', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-restart-'));
  const appRoot = path.join(dir, 'app');
  const root = path.join(dir, 'data');
  const profile = path.join(dir, 'profile');
  fs.mkdirSync(appRoot);
  initLayout(dataLayout(root));
  writePointer(path.join(profile, 'data-location.json'), { version: 1, root, pending: null });
  const moduleUrl = new URL('./shutdown.ts', import.meta.url).href;
  const script = `
    const { requestWorkerRestart } = await import(${JSON.stringify(moduleUrl)});
    const fs = await import('node:fs');
    const path = await import('node:path');
    const assert = await import('node:assert/strict');
    let scheduled = false;
    globalThis.setTimeout = (fn, ms) => {
      assert.equal(ms, 300);
      assert.ok(fs.existsSync(path.join(process.env.HOT_STEP_PROFILE_DIR, 'restart-requested')));
      scheduled = true;
      return 0;
    };
    requestWorkerRestart();
    assert.ok(scheduled);
    assert.equal(fs.existsSync(path.join(process.env.HOT_STEP_ROOT, '.restart-requested')), false);
    process.exit(0);
  `;
  try {
    execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', script], {
      env: { ...process.env, HOT_STEP_ROOT: appRoot, HOT_STEP_PROFILE_DIR: profile, HOT_STEP_DATA_ROOT: '' },
      stdio: 'pipe',
    });
  } finally {
    // The fresh files can still be held briefly (indexer or AV) under the full
    // suite. Async rm: on Node 24 Windows, rmSync ignores maxRetries for EPERM.
    await fs.promises.rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  }
});

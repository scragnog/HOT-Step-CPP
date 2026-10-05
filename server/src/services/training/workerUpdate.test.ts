import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { PROJECT_ROOT } from '../../config.js';
import { changedSteps, currentCommit, getUpdate, ggmlPointerChanged, runUpdatePlan, startUpdate, updateGgml } from './workerUpdate.js';
import { workerStatus } from './trainingWorkers.js';

const git = (...args: string[]) => execFileSync('git', args, { cwd: PROJECT_ROOT, encoding: 'utf8' }).trim();
async function mockWorker(handler: (req: http.IncomingMessage, res: http.ServerResponse) => void) {
  const server = http.createServer(handler);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  return { server, url: `http://127.0.0.1:${(server.address() as { port: number }).port}` };
}

test('mocked worker status classifies current, behind and diverged commits', async () => {
  const head = currentCommit();
  const parent = git('rev-parse', 'HEAD~1');
  let remote = head;
  const { server, url } = await mockWorker((_req, res) => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ commit: remote, idle: true, version: 'test', aceServer: { status: 'ok', version: 'engine-test' } }));
  });
  try {
    const worker = { name: 'mock', url };
    assert.deepEqual((({ relation, behind }) => ({ relation, behind }))(await workerStatus(worker)), { relation: 'current', behind: 0 });
    remote = parent;
    const behind = await workerStatus(worker);
    assert.equal(behind.relation, 'behind');
    assert.ok((behind.behind ?? 0) >= 1);
    remote = '0'.repeat(40);
    assert.equal((await workerStatus(worker)).relation, 'diverged');
  } finally { server.close(); }
});

function operations(record: string[], options: { advance?: boolean; busy?: boolean; hooksFail?: boolean } = {}) {
  const step = (name: string) => async () => { record.push(name); };
  return {
    advance: () => options.advance !== false,
    idle: () => { if (options.busy) throw new Error('Worker has an active or queued job'); record.push('idle'); },
    reset: step('reset'), recoverGgml: async () => { record.push('ggml recovery'); if (options.hooksFail) throw new Error('Engine hook verification failed'); },
    serverInstall: step('server ci'), uiInstall: step('ui ci'),
    uiBuild: step('ui build'), engineBuild: step('engine build'),
    restart: () => { record.push('restart marker'); },
  };
}

test('queued or running work and diverged history stop before reset', async () => {
  for (const busy of [true, false]) {
    const order: string[] = [];
    await assert.rejects(runUpdatePlan(['engine/src/a.cpp'], false, operations(order, { busy, advance: busy })));
    assert.ok(!order.includes('reset'));
    assert.ok(!order.includes('restart marker'));
  }
});

test('lockfile and engine changes select steps; restart marker is last', async () => {
  assert.deepEqual(changedSteps(['server/package-lock.json', 'ui/package-lock.json', 'engine/src/a.cpp']), { serverInstall: true, uiInstall: true, engineBuild: true });
  const withoutEngine: string[] = [];
  await runUpdatePlan(['ui/src/App.tsx'], false, operations(withoutEngine));
  assert.deepEqual(withoutEngine, ['idle', 'reset', 'ui build', 'restart marker']);
  const all: string[] = [];
  await runUpdatePlan(['server/package-lock.json', 'ui/package-lock.json', 'engine/src/a.cpp'], false, operations(all));
  assert.deepEqual(all, ['idle', 'reset', 'server ci', 'ui ci', 'ui build', 'engine build', 'restart marker']);
});

test('ggml pointer change recovers after reset; hook failure stops before engine build', async () => {
  assert.equal(ggmlPointerChanged('base', 'target', rev => rev === 'base:engine/ggml' ? 'old' : 'new'), true);
  assert.equal(ggmlPointerChanged('base', 'target', () => 'same'), false);
  const changed: string[] = [];
  await runUpdatePlan(['engine/ggml'], true, operations(changed));
  assert.deepEqual(changed, ['idle', 'reset', 'ggml recovery', 'ui build', 'engine build', 'restart marker']);
  const failed: string[] = [];
  await assert.rejects(runUpdatePlan(['engine/ggml'], true, operations(failed, { hooksFail: true })), /hook verification failed/);
  assert.deepEqual(failed, ['idle', 'reset', 'ggml recovery']);
});

function forkFixture(missingObject = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-fork-update-'));
  const source = path.join(root, 'source');
  const fork = path.join(root, 'fork');
  const worker = path.join(root, 'worker');
  const run = (cwd: string, ...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: 'pipe' }).trim();
  const commit = (cwd: string) => run(cwd, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture');
  fs.mkdirSync(source); fs.mkdirSync(worker);
  run(source, 'init');
  fs.writeFileSync(path.join(source, 'tracked.txt'), 'original');
  run(source, 'add', 'tracked.txt'); commit(source);
  run(root, 'clone', source, fork);
  run(worker, 'init');
  run(worker, '-c', 'protocol.file.allow=always', 'submodule', 'add', source, 'engine/ggml');
  commit(worker);
  const oldPointer = run(worker, 'rev-parse', 'HEAD:engine/ggml');
  const kernelFiles = [
    'src/ggml-cuda/convrot8.cu', 'src/ggml-cuda/convrot8.cuh',
    'src/ggml-cuda/fattn-train.cu', 'src/ggml-cuda/fattn-train.cuh',
    'src/ggml-vulkan/vulkan-shaders/fa_train_test.comp',
  ];
  for (const file of kernelFiles) {
    const full = path.join(fork, ...file.split('/'));
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, `committed ${file}`);
  }
  run(fork, 'add', '.'); commit(fork);
  const forkPointer = run(fork, 'rev-parse', 'HEAD');
  run(worker, 'config', '-f', '.gitmodules', 'submodule.engine/ggml.url', fork.replaceAll('\\', '/'));
  run(worker, 'add', '.gitmodules');
  run(worker, 'update-index', '--add', '--cacheinfo', `160000,${missingObject ? 'a'.repeat(40) : forkPointer},engine/ggml`);
  commit(worker);
  return { root, worker, source, fork, oldPointer, forkPointer, kernelFiles, run };
}

test('pointer-changing fork update syncs the URL and retains committed kernels from a dirty submodule', async () => {
  const fixture = forkFixture();
  const { root, worker, source, fork, oldPointer, forkPointer, kernelFiles, run } = fixture;
  const previousProtocol = process.env.GIT_ALLOW_PROTOCOL;
  process.env.GIT_ALLOW_PROTOCOL = 'file';
  try {
    assert.equal(ggmlPointerChanged(`${run(worker, 'rev-parse', 'HEAD^')}`, `${run(worker, 'rev-parse', 'HEAD')}`, rev => run(worker, 'rev-parse', rev)), true);
    assert.equal(run(path.join(worker, 'engine/ggml'), 'rev-parse', 'HEAD'), oldPointer);
    assert.equal(run(worker, 'config', '--get', 'submodule.engine/ggml.url').replaceAll('\\', '/'), source.replaceAll('\\', '/'));
    fs.writeFileSync(path.join(worker, 'engine/ggml/tracked.txt'), 'dirty worker edit');
    const events: string[] = [];
    try {
      await updateGgml(worker, line => events.push(line), async () => { events.push('hooks'); });
    } catch (error) {
      assert.fail(`${error}; git output: ${events.join(' | ')}`);
    }
    assert.equal(run(worker, 'config', '--get', 'submodule.engine/ggml.url').replaceAll('\\', '/'), fork.replaceAll('\\', '/'));
    assert.equal(run(path.join(worker, 'engine/ggml'), 'rev-parse', 'HEAD'), forkPointer);
    assert.equal(run(path.join(worker, 'engine/ggml'), 'status', '--porcelain'), '');
    assert.equal(fs.readFileSync(path.join(worker, 'engine/ggml/tracked.txt'), 'utf8'), 'original');
    for (const file of kernelFiles) assert.equal(fs.readFileSync(path.join(worker, 'engine/ggml', ...file.split('/')), 'utf8'), `committed ${file}`);
    assert.equal(events.at(-1), 'hooks');
  } finally {
    if (previousProtocol === undefined) delete process.env.GIT_ALLOW_PROTOCOL;
    else process.env.GIT_ALLOW_PROTOCOL = previousProtocol;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('missing fork object fails before hook verification', async () => {
  const { root, worker } = forkFixture(true);
  const previousProtocol = process.env.GIT_ALLOW_PROTOCOL;
  process.env.GIT_ALLOW_PROTOCOL = 'file';
  try {
    let verified = false;
    await assert.rejects(updateGgml(worker, () => {}, async () => { verified = true; }), /git exited/);
    assert.equal(verified, false);
  } finally {
    if (previousProtocol === undefined) delete process.env.GIT_ALLOW_PROTOCOL;
    else process.env.GIT_ALLOW_PROTOCOL = previousProtocol;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('hook failure after a fork checkout stops the update', async () => {
  const { root, worker } = forkFixture();
  const previousProtocol = process.env.GIT_ALLOW_PROTOCOL;
  process.env.GIT_ALLOW_PROTOCOL = 'file';
  try {
    fs.writeFileSync(path.join(worker, 'engine/verify-hooks.ps1'), "Write-Output '[FAIL] fixture hook'\n");
    await assert.rejects(updateGgml(worker, () => {}), /hook verification failed/);
  } finally {
    if (previousProtocol === undefined) delete process.env.GIT_ALLOW_PROTOCOL;
    else process.env.GIT_ALLOW_PROTOCOL = previousProtocol;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('non-recursive reset preserves tracked ggml changes', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-submodule-reset-'));
  const source = path.join(root, 'source');
  const parent = path.join(root, 'parent');
  const run = (cwd: string, ...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: 'pipe' }).trim();
  const commit = (cwd: string) => run(cwd, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture');
  try {
    fs.mkdirSync(source); fs.mkdirSync(parent);
    run(source, 'init');
    fs.writeFileSync(path.join(source, 'tracked.txt'), 'base');
    run(source, 'add', 'tracked.txt'); commit(source);
    run(parent, 'init');
    run(parent, '-c', 'protocol.file.allow=always', 'submodule', 'add', source, 'engine/ggml');
    commit(parent);
    run(parent, 'config', 'submodule.recurse', 'true');
    const tracked = path.join(parent, 'engine', 'ggml', 'tracked.txt');
    fs.writeFileSync(tracked, 'worker patch');
    run(parent, '-c', 'submodule.recurse=false', 'reset', '--hard', 'HEAD');
    assert.equal(fs.readFileSync(tracked, 'utf8'), 'worker patch');
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('mocked worker with an active job refuses update before bundle creation', async () => {
  const { server, url } = await mockWorker((_req, res) => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ commit: git('rev-parse', 'HEAD~1'), idle: false }));
  });
  try {
    const name = `busy-${Date.now()}`;
    startUpdate(name, url, '');
    for (let n = 0; n < 50 && getUpdate(name)?.status === 'preparing'; n++) await new Promise(resolve => setTimeout(resolve, 20));
    const update = getUpdate(name);
    assert.equal(update?.status, 'failed');
    assert.match(update?.error ?? '', /active or queued/);
  } finally { server.close(); }
});

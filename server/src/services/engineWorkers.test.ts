// Multi-GPU engine workers (GH #13): the empty opt-in must launch exactly the
// engine it always did, and with two workers every engine call, log line,
// crash epoch and restore must stay on the worker it was bound to. Two fake
// HTTP engines stand in for two cards; no GPU is touched.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import type { AddressInfo } from 'node:net';

process.env.ACESTEPCPP_HOST = '127.0.0.1';
process.env.ACESTEPCPP_EXE = process.execPath;   // only needs to exist
process.env.ACESTEPCPP_WORKER_GPUS = '';

function fakeChild(pid: number): any {
  const c: any = new EventEmitter();
  c.stdout = new EventEmitter();
  c.stderr = new EventEmitter();
  c.pid = pid;
  c.exitCode = null;
  c.signalCode = null;
  c.kill = () => true;
  return c;
}

/** A fake ace-server: answers /health, and /job?id=7 with its own name. */
function fakeEngine(name: string, hits: string[]) {
  return http.createServer((req, res) => {
    hits.push(`${name} ${req.url}`);
    res.setHeader('content-type', 'application/json');
    if (req.url === '/health') return res.end('{"status":"ok"}');
    if (req.url?.startsWith('/job?id=7')) return res.end(JSON.stringify({ status: 'running', engine: name }));
    res.end('{}');
  });
}

async function listen(srv: http.Server, port = 0): Promise<number> {
  await new Promise<void>((resolve, reject) => {
    srv.once('error', reject);
    srv.listen(port, '127.0.0.1', () => { srv.off('error', reject); resolve(); });
  });
  return (srv.address() as AddressInfo).port;
}

/** Two fake engines on adjacent ports, the layout resolveWorkers() hands out. */
async function twoEngines(hits: string[]) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const a = fakeEngine('A', hits);
    const b = fakeEngine('B', hits);
    const pa = await listen(a);
    try {
      await listen(b, pa + 1);
      return { a, b, pa };
    } catch {
      a.close();
    }
  }
  throw new Error('no adjacent ports free');
}

const GPUS = [
  { index: 0, uuid: 'GPU-aaaa', name: 'Card A', memoryMB: 24576 },
  { index: 1, uuid: 'GPU-bbbb', name: 'Card B', memoryMB: 16384 },
];

test('resolveWorkers: empty, unknown and duplicate opt-ins', async () => {
  const { resolveWorkers, parseWorkerGpus } = await import('./engineWorkers.js');
  assert.deepEqual(parseWorkerGpus(' GPU-aaaa, gpu-AAAA ,0, MIG-x,'), ['GPU-aaaa', 'MIG-x']);

  const empty = resolveWorkers('', GPUS, 8085);
  assert.equal(empty.workers.length, 1);
  assert.equal(empty.workers[0].gpuUuid, null, 'empty opt-in keeps the Settings/Auto pick');
  assert.equal(empty.log, null);

  const unknown = resolveWorkers('GPU-zzzz', GPUS, 8085);
  assert.equal(unknown.workers.length, 1);
  assert.equal(unknown.workers[0].gpuUuid, null, 'no opted-in card present: legacy single engine');

  const none = resolveWorkers('GPU-aaaa,GPU-bbbb', [], 8085);
  assert.equal(none.workers[0].gpuUuid, null, 'non-CUDA machine: legacy single engine');

  const two = resolveWorkers('GPU-bbbb,GPU-aaaa,GPU-cccc', GPUS, 8085, [8086]);
  assert.deepEqual(two.workers.map(w => [w.id, w.gpuUuid, w.logFile]),
    [['w0', 'GPU-bbbb', 'ace_engine.log'], ['w1', 'GPU-aaaa', 'ace_engine.w1.log']]);
  assert.equal(two.workers[1].port, 8087, 'secondary ports skip reserved ones');
  assert.match(two.log!, /1 opted-in GPU\(s\) not present/);
});

test('empty opt-in preserves legacy launch', async () => {
  const hits: string[] = [];
  const srv = fakeEngine('A', hits);
  const { config } = await import('../config.js');
  const proc = await import('./aceEngineProcess.js');
  const workers = await import('./engineWorkers.js');
  const logs = await import('../routes/logs.js');
  const { aceClient } = await import('./aceClient.js');
  const { buildGpuEnv } = await import('./gpuDevices.js');
  const fakeGpus = GPUS;

  const spawned: Array<{ args: string[]; env: NodeJS.ProcessEnv }> = [];
  const kids: any[] = [];
  proc.testHooks.spawn = ((_exe: string, args: string[], opts: any) => {
    spawned.push({ args, env: opts.env });
    const c = fakeChild(5000 + kids.length); kids.push(c); return c;
  }) as any;

  const port = await listen(srv);
  const savedPort = config.aceServer.port;
  try {
    for (const [label, cuda, portOverride] of [
      ['auto', '', null], ['manual', 'GPU-bbbb', null], ['port override', '', port],
    ] as const) {
      config.aceServer.cudaVisibleDevices = cuda;
      config.aceServer.port = portOverride ?? savedPort;
      workers.setEngineWorkersForTest(null);
      const table = workers.engineWorkers();
      assert.equal(table.length, 1, `${label}: one worker`);
      const w0 = table[0];
      assert.equal(w0.id, 'w0');
      assert.equal(w0.gpuUuid, null, `${label}: Settings pick, not a pinned UUID`);

      proc.startAceServer();
      const { args, env } = spawned[spawned.length - 1];
      // The legacy spawn: same leading args, configured port, same GPU env.
      assert.deepEqual(args.slice(0, 6), ['--models', config.aceServer.models, '--host', '127.0.0.1',
        '--port', String(config.aceServer.port)], `${label}: args`);
      const legacyEnv = buildGpuEnv(process.env, undefined, fakeGpus).env;
      const legacyFromProc = buildGpuEnv(process.env).env;
      assert.equal(env.CUDA_VISIBLE_DEVICES, legacyFromProc.CUDA_VISIBLE_DEVICES, `${label}: CUDA_VISIBLE_DEVICES`);
      assert.equal(env.CUDA_DEVICE_ORDER, legacyFromProc.CUDA_DEVICE_ORDER, `${label}: CUDA_DEVICE_ORDER`);
      if (cuda) assert.equal(legacyEnv.CUDA_VISIBLE_DEVICES, 'GPU-bbbb', 'manual pick passes through');

      // Primary URL: the configured port, no scope needed.
      assert.equal(config.aceServer.url, `http://127.0.0.1:${config.aceServer.port}`, `${label}: url`);
      assert.equal(w0.url, config.aceServer.url);

      // Log routing: engine lines land untagged, in ace_engine.log.
      const seen: any[] = [];
      const off = logs.subscribeLines(l => { if (l.source === 'engine') seen.push(l); });
      kids[kids.length - 1].stdout.emit('data', Buffer.from(`hello from ${label}\n`));
      off();
      assert.equal(seen.length, 1);
      assert.equal(seen[0].text, `hello from ${label}`);
      assert.equal('worker' in seen[0], false, `${label}: primary lines carry no worker tag`);
      assert.equal(w0.logFile, 'ace_engine.log');

      kids[kids.length - 1].exitCode = 0;   // fake pid: never let stop reach taskkill
      await proc.stopAceServer('test', 100, { suspend: false });
    }

    // With the port override in place the bare client reaches that engine.
    config.aceServer.port = port;
    await aceClient.health();
    assert.ok(hits.includes('A /health'), 'bare client talks to the configured port');
  } finally {
    config.aceServer.port = savedPort;
    config.aceServer.cudaVisibleDevices = '';
    workers.setEngineWorkersForTest(null);
    for (const k of kids) k.exitCode = 0;
    srv.close();
  }
});

test('two workers: identical engine job ids, logs, selection restore, crash and restart', async () => {
  const hits: string[] = [];
  const { a, b, pa } = await twoEngines(hits);
  const { config } = await import('../config.js');
  const proc = await import('./aceEngineProcess.js');
  const workers = await import('./engineWorkers.js');
  const logs = await import('../routes/logs.js');
  const { aceClient } = await import('./aceClient.js');
  const state = await import('../engineState.js');

  const kids: any[] = [];
  proc.testHooks.spawn = ((_exe: string, _args: string[], opts: any) => {
    const c = fakeChild(6000 + kids.length); c.env = opts.env; c.args = _args; kids.push(c); return c;
  }) as any;
  proc.testHooks.restartHealthTimeoutMs = 2000;
  proc.testHooks.listGpus = () => GPUS;
  proc.testHooks.lateReadyPollMs = 50;

  const savedPort = config.aceServer.port;
  config.aceServer.port = pa;
  const { workers: table } = workers.resolveWorkers('GPU-aaaa,GPU-bbbb', GPUS, pa, []);
  workers.setEngineWorkersForTest(table);
  const [w0, w1] = table;
  assert.equal(w1.port, pa + 1);

  // A selection the server persists and replays into every restarted engine.
  const restores: string[] = [];
  proc.onEngineRestarted(async () => {
    await fetch(`${config.aceServer.url}/models/select?role=lm`);
    restores.push(workers.currentWorker().id);
  });

  try {
    // Secondary boot: spawned pinned to its own card and port, restore hook in its scope.
    const up = await proc.startEngineWorkers();
    assert.deepEqual(up, [true]);
    const k1 = kids[0];
    assert.equal(k1.env.CUDA_VISIBLE_DEVICES, 'GPU-bbbb', 'w1 pinned by UUID');
    assert.equal(k1.args[k1.args.indexOf('--port') + 1], String(pa + 1));
    assert.equal(w1.ready, true);
    assert.equal(w1.pid, 6000);
    assert.deepEqual(restores, ['w1']);
    assert.ok(hits.includes('B /models/select?role=lm'), 'w1 restore reached engine B');
    assert.ok(!hits.includes('A /models/select?role=lm'), 'and not engine A');

    // Identical engine job id on both workers: each scope reads its own.
    const [onA, onB] = await Promise.all([
      workers.runWithWorker(w0, async () => { await new Promise(r => setTimeout(r, 20)); return aceClient.pollJob('7'); }),
      workers.runWithWorker(w1, () => aceClient.pollJob('7')),
    ]);
    assert.equal((onA as any).engine, 'A');
    assert.equal((onB as any).engine, 'B');
    assert.equal(((await aceClient.pollJob('7')) as any).engine, 'A', 'outside any scope: the primary');
    assert.equal(workers.workerForJob({ workerId: 'w1' }), w1);
    assert.equal(workers.workerForJob({}), w0);

    // Switching workers mid-job is refused.
    await assert.rejects(workers.runWithWorker(w0, () => workers.runWithWorker(w1, () => 1)), /cannot switch/);

    // Log isolation: a subscriber sees engine lines of its own worker only.
    const fromW0: string[] = [];
    const fromW1: string[] = [];
    const offs = [
      logs.subscribeLines(l => { if (l.source === 'engine') fromW0.push(l.text); }),
      await workers.runWithWorker(w1, () => logs.subscribeLines(l => { if (l.source === 'engine') fromW1.push(l.text); })),
    ];
    k1.stdout.emit('data', Buffer.from('[LM-Phase1] Step 3 on B\n[Load] DiT backend: CUDA0\n'));
    logs.pushLog('[LM-Phase1] Step 9 on A', 'engine');
    logs.pushLog('server line', 'server');
    offs.forEach(off => off());
    assert.deepEqual(fromW0, ['[LM-Phase1] Step 9 on A']);
    assert.deepEqual(fromW1, ['[LM-Phase1] Step 3 on B', '[Load] DiT backend: CUDA0']);
    assert.equal(w1.device, 'CUDA0', 'w1 records its own device');
    assert.notEqual(state.engineDevice, 'CUDA0', 'without touching the primary\'s');

    // Idle-only restart: refused while a call is bound to the worker.
    await workers.runWithWorker(w1, async () => {
      await assert.rejects(proc.restartEngineWorker('w1'), /busy/);
    });

    // w0 is the live primary: bootstrap's setEngineReady is what it reports.
    state.setEngineReady(true, 'Ready');
    assert.equal(w0.ready, true);
    assert.equal(w0.status, 'Ready');
    assert.deepEqual(workers.engineWorkerStatus()[0].ready, true);

    // Crash of w1: its epoch moves, the primary's does not, the primary stays ready.
    const w0Seq = w0.crashSeq;
    const seqInW1 = await workers.runWithWorker(w1, () => proc.engineCrashSeq());
    k1.exitCode = 1;
    k1.emit('exit', 1, null);
    assert.equal(w1.crashSeq, seqInW1 + 1);
    assert.equal(await workers.runWithWorker(w1, () => proc.engineCrashSeq()), seqInW1 + 1);
    assert.equal(proc.engineCrashSeq(), w0Seq, 'primary epoch unchanged');
    assert.equal(state.engineReady, true, 'primary readiness untouched');
    assert.equal(w1.pid, null);
    assert.equal(w1.ready, false, 'a crashed worker is down at once');

    // Restart w1 while idle: the respawn timer is superseded, selection restored again.
    const before = hits.filter(h => h === 'B /models/select?role=lm').length;
    assert.equal(await proc.restartEngineWorker('w1'), true);
    assert.equal(kids.length, 2, 'one respawn, no duplicate from the crash timer');
    assert.deepEqual(restores, ['w1', 'w1']);
    assert.equal(hits.filter(h => h === 'B /models/select?role=lm').length, before + 1);
    await new Promise(r => setTimeout(r, 3300));   // past the crash timer's 3 s
    assert.equal(kids.length, 2, 'stale crash timer never fires');
    assert.deepEqual(workers.engineWorkerStatus().map(s => [s.id, s.ready, s.gpuUuid]),
      [['w0', true, 'GPU-aaaa'], ['w1', true, 'GPU-bbbb']]);

    // Automatic recovery: the crash timer respawns, probes /health and restores
    // in w1's scope, keeping the crash limiter's count.
    const crash = async () => {
      const k = kids[kids.length - 1];
      k.exitCode = 1;
      k.emit('exit', 1, null);
      assert.equal(w1.ready, false, 'down from the crash until /health answers');
    };
    const recovered = async (n: number) => {
      const until = Date.now() + 5000;
      while (!(kids.length === n && w1.ready && restores.length === n) && Date.now() < until) await new Promise(r => setTimeout(r, 50));
      assert.equal(kids.length, n, 'respawned once');
      assert.equal(w1.ready, true, 'ready again after /health');
      assert.equal(restores.length, n, 'selection restored after recovery');
    };
    const seqBefore = w1.crashSeq;
    await crash();
    await recovered(3);
    assert.deepEqual(restores, ['w1', 'w1', 'w1'], 'selection restored after the automatic respawn');
    assert.equal(w1.pid, 6002);
    await crash();
    await recovered(4);
    await crash();   // third crash inside the window: the limiter gives up
    await new Promise(r => setTimeout(r, 3300));
    assert.equal(kids.length, 4, 'limiter count survived the automatic respawns');
    assert.equal(w1.crashSeq, seqBefore + 3);
    assert.match(w1.status, /crashed 3 times/);
    assert.equal(w0.ready, true, 'primary untouched throughout');
  } finally {
    for (const k of kids) k.exitCode = 0;   // fake pid: never let stop reach taskkill
    await proc.stopEngineWorkers('test', 100, { suspend: false });
    config.aceServer.port = savedPort;
    workers.setEngineWorkersForTest(null);
    a.close(); b.close();
  }
});

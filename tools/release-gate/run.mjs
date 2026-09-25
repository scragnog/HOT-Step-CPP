#!/usr/bin/env node
// tools/release-gate/run.mjs — pre-release go/no-go for HOT-Step.
//
//   node tools/release-gate/run.mjs                      attach to the app already running on :3001
//   node tools/release-gate/run.mjs --zip release/out/HOT-Step-CPP-vX.Y.Z-win-x64-cuda.zip
//   node tools/release-gate/run.mjs --dir <extracted release folder>
//   node tools/release-gate/run.mjs --dev                spawn the source tree on :3199 with a fresh data dir
//   node tools/release-gate/run.mjs --tiers 0-3          run a subset (ranges and lists: 0-2,5)
//   node tools/release-gate/run.mjs --skip 6             everything but training
//   node tools/release-gate/run.mjs --update-goldens     rewrite the tier 7 references
//
// Exit 0 = GO. Exit 1 = NO-GO. Exit 2 = could not run (app busy, boot failed).
// Each tier is a node:test file in this folder; run.mjs only orders them,
// stops after a failed gate tier, restores the backend, stages the renders for
// the ear test and writes logs/release-gate/<stamp>/summary.{md,json}.
import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const MIN = 60_000;

const TIERS = [
  { n: 0, file: 't0-static.test.mjs', name: 'Static checks', gate: true, app: false },
  { n: 1, file: 't1-boot.test.mjs', name: 'Boot and read-only API', gate: true },
  { n: 2, file: 't2-ace-gen.test.mjs', name: 'ACE generation modes', gate: true },
  { n: 3, file: 't3-plugins-adapters.test.mjs', name: 'Plugins and adapters change the output' },
  { n: 4, file: 't4-backends.test.mjs', name: 'MM3 and YuE2 backends' },
  { n: 5, file: 't5-audio-tools.test.mjs', name: 'Audio tools' },
  { n: 6, file: 't6-training.test.mjs', name: 'Training, a few steps of each trainer' },
  { n: 7, file: 't7-golden.test.mjs', name: 'Golden fingerprints', warnOnly: true },
];

const HELP = fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1, 12).map((l) => l.replace(/^\/\/ ?/, '')).join('\n');

function parseArgs(argv) {
  const o = {
    zip: null, dir: null, dev: false, url: 'http://localhost:3001',
    port: 3199, enginePort: 18085,
    models: path.join(REPO, 'models'), adapters: path.join(REPO, 'adapters'),
    tiers: TIERS.map((t) => t.n), skip: [], updateGoldens: false, strictGolden: false,
    offline: false, keep: false, listen: true,
  };
  const tierList = (s) => s.split(',').flatMap((part) => {
    const m = part.match(/^(\d+)-(\d+)$/);
    if (m) return Array.from({ length: +m[2] - +m[1] + 1 }, (_, i) => +m[1] + i);
    return [Number(part)];
  });
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--zip') o.zip = path.resolve(next());
    else if (a === '--dir') o.dir = path.resolve(next());
    else if (a === '--dev') o.dev = true;
    else if (a === '--url') o.url = next();
    else if (a === '--port') o.port = Number(next());
    else if (a === '--engine-port') o.enginePort = Number(next());
    else if (a === '--models') o.models = path.resolve(next());
    else if (a === '--adapters') o.adapters = path.resolve(next());
    else if (a === '--tiers') o.tiers = tierList(next());
    else if (a === '--skip') o.skip = tierList(next());
    else if (a === '--update-goldens') o.updateGoldens = true;
    else if (a === '--strict-golden') o.strictGolden = true;
    else if (a === '--offline') o.offline = true;
    else if (a === '--keep') o.keep = true;
    else if (a === '--no-listen') o.listen = false;
    else if (a === '--help' || a === '-h') { console.log(HELP); process.exit(0); }
    else { console.error(`unknown argument ${a}\n${HELP}`); process.exit(2); }
  }
  o.tiers = o.tiers.filter((n) => !o.skip.includes(n));
  return o;
}

const stamp = () => {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
};
const log = (msg) => console.log(`[gate ${new Date().toTimeString().slice(0, 8)}] ${msg}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tail = (file, n = 30) => (fs.existsSync(file) ? fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).slice(-n).join('\n') : '');

async function getJson(url, timeoutMs = 5000) {
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  return res.json();
}

// ── Booting the app under test ───────────────────────────────────────────────
function extractZip(zip, into) {
  fs.mkdirSync(into, { recursive: true });
  let r = spawnSync('tar', ['-xf', zip, '-C', into], { encoding: 'utf8', windowsHide: true });
  if (r.status !== 0 && process.platform === 'win32') {
    r = spawnSync('powershell', ['-NoProfile', '-Command', `Expand-Archive -LiteralPath '${zip}' -DestinationPath '${into}' -Force`], { encoding: 'utf8', windowsHide: true });
  }
  if (r.status !== 0) throw new Error(`could not extract ${zip}: ${r.stderr || r.stdout}`);
  const entries = fs.readdirSync(into);
  return entries.length === 1 && fs.statSync(path.join(into, entries[0])).isDirectory() ? path.join(into, entries[0]) : into;
}

function bootApp(opts, runDir) {
  let root = null;
  let cmd;
  let args;
  let cwd;
  const env = {
    ...process.env,
    SERVER_PORT: String(opts.port),
    ACESTEPCPP_PORT: String(opts.enginePort),
    ACESTEPCPP_MODELS: opts.models,
    ACESTEPCPP_ADAPTERS: opts.adapters,
  };
  if (opts.zip) root = extractZip(opts.zip, path.join(runDir, 'app'));
  else if (opts.dir) root = opts.dir;
  if (root) {
    const serverMjs = path.join(root, 'server', 'server.mjs');
    if (!fs.existsSync(serverMjs)) throw new Error(`not a release build: ${serverMjs} is missing`);
    cmd = ['runtime/node.exe', 'runtime/bin/node', 'runtime/node'].map((p) => path.join(root, p)).find(fs.existsSync) ?? process.execPath;
    args = [serverMjs];
    cwd = root;
    env.HOT_STEP_ROOT = root;
  } else {
    root = REPO;
    cmd = 'npx';
    args = ['tsx', 'src/index.ts'];
    cwd = path.join(REPO, 'server');
    env.DATA_DIR = path.join(runDir, 'data');
  }
  const out = fs.createWriteStream(path.join(runDir, 'app.log'));
  const child = spawn(cmd, args, { cwd, env, shell: cmd === 'npx', stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  child.stdout.pipe(out, { end: false });
  child.stderr.pipe(out, { end: false });
  log(`spawned ${cmd} ${args.join(' ')} (pid ${child.pid}) with HOT_STEP_ROOT=${env.HOT_STEP_ROOT ?? '(dev)'}`);
  return { child, root, url: `http://localhost:${opts.port}` };
}

async function waitReady(url, child, runDir, timeoutMs = 20 * MIN) {
  const started = Date.now();
  let last = '';
  for (;;) {
    if (child && child.exitCode !== null) throw new Error(`app exited with code ${child.exitCode}\n${tail(path.join(runDir, 'app.log'))}`);
    try {
      const h = await getJson(`${url}/api/health`);
      if (h?.engine?.ready) return h;
      last = h?.engine?.bootStatus ?? h?.status ?? '';
    } catch (err) { last = err.message; }
    if (Date.now() - started > timeoutMs) throw new Error(`engine never became ready: ${last}`);
    if ((Date.now() - started) % 30_000 < 3000) log(`waiting for engine: ${last}`);
    await sleep(3000);
  }
}

async function shutdownApp(app) {
  if (!app?.child) return;
  try { await fetch(`${app.url}/api/shutdown`, { method: 'POST', signal: AbortSignal.timeout(10_000) }); } catch { /* it may close the socket first */ }
  for (let i = 0; i < 30 && app.child.exitCode === null; i++) await sleep(1000);
  if (app.child.exitCode === null) {
    log('app did not stop on request; killing the process tree');
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(app.child.pid), '/T', '/F'], { windowsHide: true });
    else app.child.kill('SIGKILL');
  }
}

// ── Tiers ────────────────────────────────────────────────────────────────────
function parseTap(file) {
  const text = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  const num = (k) => Number((text.match(new RegExp(`^# ${k} (\\d+)`, 'm')) ?? [])[1] ?? 0);
  const failed = [...text.matchAll(/^\s*not ok \d+ - (.+?)\s*$/gm)].map((m) => m[1]).filter((n) => !/\.test\.mjs$/.test(n));
  return { pass: num('pass'), fail: num('fail'), skipped: num('skipped'), failed };
}

function runTier(tier, env, runDir) {
  const tap = path.join(runDir, `t${tier.n}.tap`);
  const logPath = path.join(runDir, `t${tier.n}.log`);
  const args = [
    '--test',
    '--test-reporter=spec', '--test-reporter-destination=stdout',
    '--test-reporter=tap', `--test-reporter-destination=${tap}`,
    path.join(HERE, tier.file),
  ];
  return new Promise((resolve) => {
    const started = Date.now();
    console.log(`\n=== Tier ${tier.n}: ${tier.name} ===`);
    const child = spawn(process.execPath, args, { env, cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const out = fs.createWriteStream(logPath);
    const tee = (d) => { process.stdout.write(d); out.write(d); };
    child.stdout.on('data', tee);
    child.stderr.on('data', tee);
    child.on('close', (code) => {
      out.end();
      resolve({ ...tier, code, seconds: Math.round((Date.now() - started) / 1000), ...parseTap(tap) });
    });
  });
}

// ── Ear test hand-off ────────────────────────────────────────────────────────
function stageListening(runStamp, renders) {
  const dir = path.join(REPO, '_experiments', '_LISTENING', `${runStamp}-release-gate`);
  fs.mkdirSync(path.join(dir, 'renders'), { recursive: true });
  const tracks = renders.map((r, i) => {
    const file = `renders/${String(i + 1).padStart(2, '0')}-${path.basename(r.file)}`;
    fs.copyFileSync(r.file, path.join(dir, file));
    const tier = r.label.match(/^t(\d)/)?.[1] ?? '?';
    return { id: r.label, round: 1, group: `Tier ${tier}`, order: i + 1, label: r.label, file };
  });
  const study = {
    id: `release-gate-${runStamp}`,
    title: `Release gate ${runStamp}`,
    intro: 'Every render here completed mechanically. Your ear decides whether it is music. Score 1-5.',
    criteria: [
      { key: 'music', name: 'Music', desc: 'sounds like music, not noise, silence or a loop', series: 'quality' },
      { key: 'audio', name: 'Audio', desc: 'clean: no hiss, phasing, crackle or clipping', series: 'quality' },
      { key: 'coherence', name: 'Coherence', desc: 'holds together and ends properly', series: 'quality' },
      { key: 'prompt', name: 'Prompt', desc: 'matches the caption and lyrics it was given', series: 'adherence' },
    ],
    series: { adherence: { name: 'Adherence', color: 'var(--accent)' }, quality: { name: 'Quality', color: 'var(--warn)' } },
    roundNames: { 1: 'Release gate renders' },
    tracks,
  };
  fs.writeFileSync(path.join(dir, 'study.json'), JSON.stringify(study, null, 2));
  const maker = path.join(REPO, '.claude', 'skills', 'ear-test-scoresheet', 'make-scoresheet.mjs');
  if (fs.existsSync(maker)) spawnSync(process.execPath, [maker, path.join(dir, 'study.json')], { stdio: 'inherit' });
  else fs.writeFileSync(path.join(dir, 'README.txt'), 'Score sheet generator not on this checkout; play the files in renders/ in order.\n');
  return dir;
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const runStamp = stamp();
  const runDir = path.join(REPO, 'logs', 'release-gate', runStamp);
  fs.mkdirSync(runDir, { recursive: true });
  fs.writeFileSync(path.join(runDir, 'run.json'), JSON.stringify({ startedAt: new Date().toISOString(), opts }, null, 2));
  const selected = TIERS.filter((t) => opts.tiers.includes(t.n));
  const needsApp = selected.some((t) => t.app !== false);
  const mode = opts.zip ? 'zip' : opts.dir ? 'dir' : opts.dev ? 'dev' : 'attach';
  log(`run ${runStamp}: tiers ${selected.map((t) => t.n).join(',')}, mode ${mode}`);

  let app = null;
  let url = opts.url;
  let appRoot = REPO;
  const results = [];
  let stateBackend = null;
  const stateFile = path.join(runDir, 'state.json');
  const readState = () => { try { return JSON.parse(fs.readFileSync(stateFile, 'utf8')); } catch { return {}; } };

  process.on('SIGINT', async () => { log('interrupted'); await shutdownApp(app); process.exit(2); });

  try {
    if (needsApp) {
      if (mode !== 'attach') {
        app = bootApp(opts, runDir);
        url = app.url;
        appRoot = app.root;
      }
      log(`waiting for ${url} (engine ready)`);
      await waitReady(url, app?.child ?? null, runDir);
      const q = await getJson(`${url}/api/generate/queue`);
      if (q.running || q.pending) throw new Error(`generation queue is busy (running=${q.running}, pending=${q.pending}); the gate needs an idle app`);
      const { jobs = [] } = await getJson(`${url}/api/training/jobs`);
      const live = jobs.find((j) => j.status === 'running' || j.status === 'queued');
      if (live) throw new Error(`training job ${live.kind} is ${live.status}; the gate needs an idle app`);
      stateBackend = (await getJson(`${url}/api/backends`)).activeId;
      log(`app idle; active backend ${stateBackend}`);
    }

    const env = {
      ...process.env,
      GATE_URL: url,
      GATE_RUN_DIR: runDir,
      GATE_APP_ROOT: appRoot,
      GATE_OFFLINE: opts.offline ? '1' : '',
      GATE_UPDATE_GOLDENS: opts.updateGoldens ? '1' : '',
    };

    let stopped = false;
    for (const tier of selected) {
      if (stopped) { results.push({ ...tier, code: null, note: 'not run: an earlier gate tier failed' }); continue; }
      const r = await runTier(tier, env, runDir);
      results.push(r);
      if (r.code !== 0 && tier.gate) { stopped = true; log(`tier ${tier.n} is a gate and failed; later tiers are not run`); }
    }

    if (needsApp && stateBackend) {
      try {
        const now = (await getJson(`${url}/api/backends`)).activeId;
        if (now !== stateBackend) {
          await fetch(`${url}/api/backends/active`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: stateBackend }) });
          log(`restored active backend ${stateBackend}`);
        }
      } catch (err) { log(`could not restore the backend: ${err.message}`); }
    }
  } catch (err) {
    log(`could not run: ${err.message}`);
    if (app) await shutdownApp(app);
    fs.writeFileSync(path.join(runDir, 'summary.md'), `# Release gate ${runStamp}: NOT RUN\n\n${err.message}\n`);
    process.exit(2);
  }

  // ── Verdict ────────────────────────────────────────────────────────────────
  const blocking = results.filter((r) => (r.code === null) || (r.code !== 0 && !(r.warnOnly && !opts.strictGolden)));
  const warnings = results.filter((r) => r.code !== null && r.code !== 0 && r.warnOnly && !opts.strictGolden);
  const verdict = blocking.length ? 'NO-GO' : 'GO';

  const renders = (readState().renders ?? []).filter((r) => fs.existsSync(r.file));
  let listening = null;
  if (opts.listen && renders.length) {
    try { listening = stageListening(runStamp, renders); } catch (err) { log(`could not stage the ear test: ${err.message}`); }
  }

  const rows = results.map((r) => {
    const status = r.code === null ? 'not run' : r.code === 0 ? 'pass' : r.warnOnly && !opts.strictGolden ? 'WARN' : 'FAIL';
    return `| ${r.n} | ${r.name} | ${status} | ${r.pass ?? ''} | ${r.fail ?? ''} | ${r.skipped ?? ''} | ${r.seconds ?? ''} |`;
  });
  const failedTests = results.flatMap((r) => (r.failed ?? []).map((n) => `- tier ${r.n}: ${n}`));
  const md = [
    `# Release gate ${runStamp}: ${verdict}`,
    '',
    `App: ${url} (${mode})${app ? `, root ${appRoot}` : ''}`,
    '',
    '| Tier | Name | Result | Pass | Fail | Skip | Seconds |',
    '|---|---|---|---|---|---|---|',
    ...rows,
    '',
    failedTests.length ? `Failed tests:\n${failedTests.join('\n')}` : 'No failed tests.',
    warnings.length ? `\nWarnings (not blocking): tier ${warnings.map((w) => w.n).join(', ')}` : '',
    listening ? `\nEar test: ${listening}\\index.html (${renders.length} renders)` : '',
    `\nLogs: ${runDir}`,
    '',
  ].join('\n');
  fs.writeFileSync(path.join(runDir, 'summary.md'), md);
  fs.writeFileSync(path.join(runDir, 'summary.json'), JSON.stringify({ stamp: runStamp, verdict, mode, url, results, listening }, null, 2));
  console.log(`\n${md}`);

  if (app && !opts.keep) await shutdownApp(app);
  else if (app) log(`app left running at ${url} (--keep)`);
  process.exit(verdict === 'GO' ? 0 : 1);
}

main().catch((err) => { console.error(err); process.exit(2); });

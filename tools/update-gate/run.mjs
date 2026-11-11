import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import http from 'node:http';
import { spawn, spawnSync, execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { writeStubs } from '../../release/smoke-artifact.mjs';
import { chromium, browserOptions } from './browser.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture');
const evidence = path.join(here, 'evidence');
const archiveDir = path.join(root, 'source/release/out');
const baseTag = 'v91.0.0', nextTag = 'v91.0.1';
const port = 13948, enginePort = 18948, url = `http://127.0.0.1:${port}`;
const require = createRequire(import.meta.url);
const Database = require(path.join(root, 'source/server/node_modules/better-sqlite3'));
const pause = ms => new Promise(r => setTimeout(r, ms));
const write = (f, value) => fs.writeFileSync(f, JSON.stringify(value, null, 2) + '\n');
const read = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const results = [];
fs.mkdirSync(evidence, { recursive: true });
function ps(script) {
  return execFileSync('pwsh', ['-NoProfile', '-Command', script], { encoding: 'utf8', timeout: 60_000, windowsHide: true });
}
function psQuote(s) { return "'" + s.replaceAll("'", "''") + "'"; }
async function until(fn, ms = 90_000) {
  const deadline = Date.now() + ms;
  let last;
  while (Date.now() < deadline) { try { const v = await fn(); if (v) return v; } catch (e) { last = e; } await pause(250); }
  throw new Error(`Timed out after ${ms}ms: ${last?.message || 'condition stayed false'}`);
}
async function api(route, body) {
  const r = await fetch(url + route, { method: body === undefined ? 'GET' : 'POST', headers: { 'content-type': 'application/json', origin: url }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(15_000) });
  return { status: r.status, body: await r.json() };
}
const health = () => api('/api/health').then(r => r.status === 200 && r.body);
function snapshotRows(data) {
  const db = new Database(path.join(data, 'library/hotstep.db'), { readonly: true });
  try {
    return Object.fromEntries(db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all().map(({ name }) => [name, db.prepare(`SELECT * FROM "${name.replaceAll('"', '""')}"`).all().map(r => JSON.stringify(r)).sort()]));
  } finally { db.close(); }
}
function seed(data) {
  const db = new Database(path.join(data, 'library/hotstep.db'));
  try {
    const user = db.prepare('SELECT id FROM users LIMIT 1').get();
    assert.ok(user);
    db.prepare("INSERT OR REPLACE INTO songs(id,user_id,title) VALUES ('update-gate-song',?,'preserve this library song')").run(user.id);
    db.prepare("INSERT OR REPLACE INTO settings(key,value) VALUES ('update-gate','original')").run();
  } finally { db.close(); }
  fs.appendFileSync(path.join(data, 'settings/.env'), '\nUPDATE_GATE_SETTING=original\n');
}
function compareRows(after, before) {
  // Instance UUID/PID/heartbeat necessarily change across a restart. Retain those
  // rows in evidence, but compare every persistent library/settings table exactly.
  const content = ({ server_instances, ...rest }) => rest;
  assert.deepEqual(content(after), content(before));
}
async function stopTree(child, name) {
  if (!child || child.exitCode !== null) return;
  const all = JSON.parse(ps('Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress'));
  const tree = all.filter(p => p.ProcessId === child.pid);
  for (let i = 0; i < tree.length; i++) tree.push(...all.filter(p => p.ParentProcessId === tree[i].ProcessId && !tree.some(t => t.ProcessId === p.ProcessId)));
  const daemon = JSON.parse(execFileSync(process.env.MULTICA_CLI_PATH, ['daemon', 'status', '--output', 'json'], { encoding: 'utf8', timeout: 15_000, windowsHide: true }));
  assert.ok(!tree.some(p => p.ProcessId === daemon.pid), 'refuse to kill daemon');
  write(path.join(evidence, `${name}-killed-tree.json`), tree);
  if (tree.length) {
    const killed = spawnSync('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { encoding: 'utf8', timeout: 20_000, windowsHide: true });
    fs.writeFileSync(path.join(evidence, `${name}-kill.txt`), killed.stdout + killed.stderr);
    assert.equal(killed.status, 0);
  }
  await until(() => child.exitCode !== null || child.signalCode !== null, 10_000);
}
const assetName = `HOT-Step-CPP-${nextTag}-win-x64-cpu.zip`;
function releaseFeed() {
  return {
    tag_name: nextTag, name: 'Updater acceptance candidate', draft: false, prerelease: false,
    published_at: '2026-11-11T00:00:00Z', html_url: `https://github.com/scragnog/HOT-Step-CPP/releases/tag/${nextTag}`,
    assets: [assetName, assetName + '.sha256'].map((name, i) => ({ id: i + 1, name, size: fs.statSync(path.join(archiveDir, name)).size, digest: 'sha256:' + crypto.createHash('sha256').update(fs.readFileSync(path.join(archiveDir, name))).digest('hex'), browser_download_url: `https://github.com/scragnog/HOT-Step-CPP/releases/download/${nextTag}/${name}` })),
  };
}
async function fixture(name, fault = '') {
  const dir = path.join(root, name), install = path.join(dir, 'install'), data = path.join(dir, 'data'), profile = path.join(dir, 'profile');
  assert.ok(!fs.existsSync(dir), `Refusing to overwrite ${dir}`);
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(data, { recursive: true });
  fs.mkdirSync(profile, { recursive: true });
  writeStubs(path.join(data, 'models'));
  const cachedRuntime = path.join(root, 'clean-ready/data/tools/runtime');
  if (fs.existsSync(cachedRuntime)) fs.cpSync(cachedRuntime, path.join(data, 'tools/runtime'), { recursive: true });
  ps(`Expand-Archive -LiteralPath ${psQuote(path.join(archiveDir, `HOT-Step-CPP-${baseTag}-win-x64-cpu.zip`))} -DestinationPath ${psQuote(install)}`);
  const payload = read(path.join(install, 'current.json')).payload;
  const info = read(path.join(install, 'app', payload, 'build-info.json'));
  const cfgFile = path.join(evidence, `${name}-config.json`);
  const cfg = { fault, archiveDir, data, sha: info.sha, release: releaseFeed(), trace: path.join(evidence, `${name}-trace.jsonl`), marker: path.join(evidence, `${name}-marker.json`) };
  write(cfgFile, cfg);
  if (name !== 'clean-install') {
    // A previous successful check with an expired manual throttle avoids a minute per crash case.
    const empty = { etag: null, data: null, okAt: null, attemptAt: null, error: null, retryUntil: null };
    fs.mkdirSync(path.join(data, 'settings'), { recursive: true });
    write(path.join(data, 'settings/updates.json'), { version: 1, channel: 'release', master: empty, release: { ...empty, okAt: Date.now(), attemptAt: Date.now() - 120_000, data: { tag: nextTag, name: cfg.release.name, url: cfg.release.html_url, publishedAt: cfg.release.published_at } } });
  }
  const env = { ...process.env, UPDATE_GATE_CONFIG: cfgFile, NODE_OPTIONS: `--import=${pathToFileURL(path.join(here, 'preload.mjs')).href}`, HOT_STEP_PROFILE_DIR: profile, HOT_STEP_DATA_ROOT: data, SERVER_HOST: '127.0.0.1', SERVER_PORT: String(port), ACESTEPCPP_HOST: '127.0.0.1', ACESTEPCPP_PORT: String(enginePort), ACESTEPCPP_WARM_ON_STARTUP: '0', TRAINING_WORKERS: '', CUDA_VISIBLE_DEVICES: '-1' };
  // No machine-specific path/API settings are inherited into the portable fixture.
  for (const k of Object.keys(env)) if (/^(ACESTEPCPP_MODELS|ACESTEPCPP_ADAPTERS|ACESTEPCPP_EXE|DATA_DIR|DOWNLOAD_DIR|TENSORRT_LIBS|HOT_STEP_ROOT|HOT_STEP_INSTALL_ROOT|HOT_STEP_UPDATER_DIR|HOT_STEP_MAINTENANCE_TOKEN)$/.test(k)) delete env[k];
  let serial = 0;
  function start() {
    const log = fs.openSync(path.join(evidence, `${name}-launcher-${++serial}.log`), 'w');
    const child = spawn('cmd.exe', ['/d', '/s', '/c', `"${path.join(install, 'HOT-Step.bat')}"`], { cwd: install, env, windowsVerbatimArguments: true, windowsHide: true, stdio: ['ignore', log, log] });
    fs.closeSync(log);
    child.on('error', e => { throw e; });
    return child;
  }
  return { name, dir, install, data, profile, payload, cfgFile, cfg, env, start };
}
async function run(name, action) {
  if (process.argv[2] && !process.argv[2].split(',').includes(name) && process.argv[2] !== 'all') return;
  console.log(`START ${name}`);
  const at = Date.now();
  try { const details = await action(); results.push({ name, verdict: 'PASS', seconds: (Date.now() - at) / 1000, ...details }); console.log(`PASS ${name}`); }
  catch (e) { results.push({ name, verdict: 'FAIL', seconds: (Date.now() - at) / 1000, error: e.stack, details: e.details }); console.error(`FAIL ${name}: ${e.stack}`); }
  write(path.join(evidence, `${name}-result.json`), results.at(-1));
}

await run('clean', async () => {
  const f = await fixture('clean-confirm');
  let child, second, browser;
  try {
    browser = await chromium.launch(browserOptions);
    const page = await browser.newPage();
    child = f.start();
    await until(health);
    fs.writeFileSync(path.join(evidence, 'clean-listeners.txt'), ps(`Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in ${port},${enginePort} } | Format-Table -AutoSize | Out-String`));
    await page.goto(url + '/settings');
    await page.getByRole('button', { name: /^Updates/ }).click();
    await page.locator('.settings-updates').getByText('Installed build', { exact: true }).waitFor();
    seed(f.data);
    const before = snapshotRows(f.data);
    const envBefore = fs.readFileSync(path.join(f.data, 'settings/.env'), 'utf8');
    write(path.join(evidence, 'clean-before-rows.json'), before);
    second = f.start();
    await until(() => fs.readFileSync(path.join(evidence, `${f.name}-launcher-2.log`), 'utf8').includes('already running'), 15_000);
    await stopTree(second, 'second-launcher');
    // Startup checking consumes the same one-minute throttle as Check now.
    // It does not discover a downloadable candidate; the next manual check does.
    await page.locator('.settings-updates').getByRole('button', { name: /Check now/ }).click();
    await page.locator('.settings-updates').getByRole('button', { name: /Download update/ }).waitFor({ timeout: 20_000 });
    write(path.join(evidence, 'clean-offered.json'), (await api('/api/updates/status')).body);
    await page.locator('.settings-updates').getByRole('button', { name: /Download update/ }).click();
    const status = await until(async () => {
      const s = (await api('/api/updates/status')).body;
      if (s.prepare.phase === 'failed') throw new Error(s.prepare.error);
      return s.prepare.phase === 'ready' && s;
    }, 120_000);
    write(path.join(evidence, 'clean-ready.json'), status);
    assert.equal(status.prepare.offered, false);
    await page.locator('.settings-updates').getByRole('button', { name: 'Restart and install', exact: true }).waitFor();
    // Added and modified payload files must be refused while the server stays up.
    const added = path.join(f.install, 'app', f.payload, 'user-added.txt');
    fs.writeFileSync(added, 'fixture hand edit');
    const refusal = await api('/api/updates/apply', { candidateId: status.prepare.candidate.id });
    write(path.join(evidence, 'hand-added-refusal.json'), refusal);
    assert.equal(refusal.status, 400);
    assert.match(JSON.stringify(refusal), /user-added.txt/);
    fs.unlinkSync(added);
    const versionFile = path.join(f.install, 'app', f.payload, 'VERSION');
    const original = fs.readFileSync(versionFile);
    fs.appendFileSync(versionFile, '\nhand edit');
    const edited = await api('/api/updates/apply', { candidateId: status.prepare.candidate.id });
    fs.writeFileSync(versionFile, original);
    write(path.join(evidence, 'hand-edited-refusal.json'), edited);
    assert.equal(edited.status, 400);
    assert.match(JSON.stringify(edited), /VERSION/);
    await page.locator('.settings-updates').getByRole('button', { name: 'Restart and install', exact: true }).click();
    await page.getByRole('button', { name: 'Restart and install', exact: true }).last().click();
    const result = await until(() => fs.existsSync(path.join(f.install, 'updater/last-result.json')) && read(path.join(f.install, 'updater/last-result.json')), 180_000);
    write(path.join(evidence, 'clean-last-result.json'), result);
    assert.equal(result.outcome, 'installed');
    await until(health);
    const after = snapshotRows(f.data);
    write(path.join(evidence, 'clean-after-rows.json'), after);
    compareRows(after, before);
    assert.equal(fs.readFileSync(path.join(f.data, 'settings/.env'), 'utf8'), envBefore);
    const final = (await api('/api/updates/status')).body;
    write(path.join(evidence, 'clean-final-status.json'), final);
    assert.equal(final.installed.tag, nextTag);
    assert.equal(final.install.lastResult.outcome, 'installed');
    const engine = await fetch(`http://127.0.0.1:${enginePort}/health`).then(r => r.json());
    write(path.join(evidence, 'clean-engine-health.json'), engine);
    await pause(12_000);
    return { result, engine, browserPages: browser.contexts().flatMap(c => c.pages()).length, health: await health(), secondLauncher: 'refused', handEdits: 'added and modified refused' };
  } finally {
    await stopTree(second, 'second-launcher-final');
    await stopTree(child, 'clean');
    await browser?.close();
  }
});
async function prepare() {
  const checked = await api('/api/updates/check', { channel: 'release' });
  assert.ok(checked.body.prepare.candidate, JSON.stringify(checked));
  const id = checked.body.prepare.candidate.id;
  assert.equal((await api('/api/updates/prepare', { candidateId: id })).status, 202);
  return until(async () => {
    const s = (await api('/api/updates/status')).body;
    if (s.prepare.phase === 'failed') throw new Error(s.prepare.error);
    return s.prepare.phase === 'ready' && s;
  }, 120_000);
}
for (const fault of ['failed-health', 'no-models', 'before-pointer', 'after-pointer', 'mid-restore', 'after-commit']) {
  await run(fault, async () => {
    const f = await fixture(fault + '-run', fault);
    let child, browser;
    try {
      // Keep a real tab connected so the unmodified browser helper does not open the desktop browser.
      browser = await chromium.launch(browserOptions);
      const page = await browser.newPage();
      child = f.start();
      await until(health);
      await page.goto(url + '/settings');
      await page.getByRole('button', { name: /^Updates/ }).click();
      await page.locator('.settings-updates').getByText('Installed build', { exact: true }).waitFor();
      seed(f.data);
      const before = snapshotRows(f.data);
      const envBefore = fs.readFileSync(path.join(f.data, 'settings/.env'), 'utf8');
      write(path.join(evidence, `${fault}-before-rows.json`), before);
      const staged = await prepare();
      const applied = await api('/api/updates/apply', { candidateId: staged.prepare.candidate.id });
      write(path.join(evidence, `${fault}-apply.json`), applied);
      assert.equal(applied.status, 202, JSON.stringify(applied));
      if (!['failed-health', 'no-models'].includes(fault)) {
        await until(() => fs.existsSync(f.cfg.marker), fault === 'mid-restore' ? 180_000 : 90_000);
        const tx = read(path.join(f.install, 'updater/transaction.json'));
        write(path.join(evidence, `${fault}-interrupted-tx.json`), tx);
        write(path.join(evidence, `${fault}-interrupted-pointer.json`), read(path.join(f.install, 'current.json')));
        if (fault === 'after-commit') {
          assert.equal(tx.state, 'committed');
          const db = new Database(path.join(f.data, 'library/hotstep.db'));
          db.prepare("UPDATE settings SET value='committed value must survive' WHERE key='update-gate'").run();
          db.close();
          before.settings = snapshotRows(f.data).settings;
        }
        await stopTree(child, fault + '-interruption');
        // Remove injection for the real next HOT-Step.bat start.
        f.cfg.fault = '';
        write(f.cfgFile, f.cfg);
        child = f.start();
      }
      const resultFile = path.join(f.install, 'updater/last-result.json');
      const result = await until(() => fs.existsSync(resultFile) && read(resultFile), 180_000);
      write(path.join(evidence, `${fault}-last-result.json`), result);
      assert.equal(result.outcome, fault === 'after-commit' ? 'installed' : 'rolled-back');
      await until(health);
      const after = snapshotRows(f.data);
      write(path.join(evidence, `${fault}-after-rows.json`), after);
      compareRows(after, before);
      assert.equal(fs.readFileSync(path.join(f.data, 'settings/.env'), 'utf8'), envBefore);
      const final = (await api('/api/updates/status')).body;
      write(path.join(evidence, `${fault}-final-status.json`), final);
      assert.equal(final.installed.tag, fault === 'after-commit' ? nextTag : baseTag);
      return { result, tablesCompared: Object.keys(after).length, settingsMatch: true, health: await health() };
    } finally { await stopTree(child, fault + '-final'); await browser?.close(); }
  });
}
await run('busy-lock', async () => {
  const f = await fixture('busy-lock');
  let child, lock, browser, downloadServer;
  try {
    browser = await chromium.launch(browserOptions);
    const page = await browser.newPage();
    child = f.start();
    await until(health);
    await page.goto(url + '/settings');
    await page.getByRole('button', { name: /^Updates/ }).click();
    await page.locator('.settings-updates').getByText('Installed build', { exact: true }).waitFor();
    const staged = await prepare();
    const registry = (await api('/api/model-manager/registry')).body;
    const file = registry.files.find(f => f.id === 'lm-1.7b-q80');
    assert.ok(file);
    downloadServer = http.createServer((req, res) => {
      res.writeHead(200, { 'content-length': String(file.sizeBytes) });
      const timer = setInterval(() => res.write(Buffer.alloc(16_384)), 100);
      res.on('close', () => clearInterval(timer));
    });
    await new Promise(resolve => downloadServer.listen(0, '127.0.0.1', resolve));
    f.cfg.busyUrl = `https://huggingface.co/${file.repo}/resolve/main/${file.repoPath || file.filename}`;
    f.cfg.busyLocalUrl = `http://127.0.0.1:${downloadServer.address().port}/model`;
    write(f.cfgFile, f.cfg);
    const started = await api('/api/model-manager/download', { fileId: file.id });
    assert.equal(started.status, 200);
    async function downloadJob() {
      const response = await fetch(url + '/api/model-manager/downloads', { signal: AbortSignal.timeout(5000) });
      const reader = response.body.getReader();
      try {
        let text = '';
        while (!text.includes('\n\n')) text += Buffer.from((await reader.read()).value).toString();
        return JSON.parse(text.match(/^data: (.+)$/m)[1]).jobs.find(j => j.jobId === started.body.jobId);
      } finally { await reader.cancel(); }
    }
    const before = await until(async () => { const j = await downloadJob(); return j?.bytesDownloaded > 0 && j; });
    const refused = await api('/api/updates/apply', { candidateId: staged.prepare.candidate.id });
    assert.equal(refused.status, 409, JSON.stringify(refused));
    assert.ok(refused.body.blockers.some(b => b.includes(file.filename)));
    const after = await until(async () => { const j = await downloadJob(); return j.status === 'downloading' && j.bytesDownloaded > before.bytesDownloaded && j; });
    write(path.join(evidence, 'busy-refusal.json'), { refused, before, after });
    assert.equal((await api(`/api/model-manager/download/${started.body.jobId}/cancel`, {})).body.ok, true);
    await until(async () => (await downloadJob()).status === 'cancelled');
    // A read-sharing handle allows validation, but refuses rename/delete of the old file.
    const lockedFile = path.join(f.install, 'app', f.payload, 'README.txt');
    const lockMarker = path.join(evidence, 'old-payload-lock-ready');
    const lockScript = `$f=[IO.File]::Open(${psQuote(lockedFile)},[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read); [IO.File]::WriteAllText(${psQuote(lockMarker)},'locked'); try { Start-Sleep -Seconds 180 } finally { $f.Dispose() }`;
    lock = spawn('pwsh', ['-NoProfile', '-Command', lockScript], { windowsHide: true, stdio: 'ignore' });
    await until(() => fs.existsSync(lockMarker), 10_000);
    const applied = await api('/api/updates/apply', { candidateId: staged.prepare.candidate.id });
    assert.equal(applied.status, 202, JSON.stringify(applied));
    const receipt = await until(() => fs.existsSync(path.join(f.install, 'updater/last-result.json')) && read(path.join(f.install, 'updater/last-result.json')));
    write(path.join(evidence, 'locked-last-result.json'), receipt);
    assert.equal(receipt.outcome, 'installed');
    assert.equal(lock.exitCode, null, 'old file remained locked throughout activation');
    assert.ok(fs.existsSync(lockedFile));
    assert.equal((await api('/api/updates/status')).body.installed.tag, nextTag);
    return { refused, progressBefore: before.bytesDownloaded, progressAfter: after.bytesDownloaded, receipt, oldFileStillLocked: true };
  } finally {
    await stopTree(lock, 'old-file-lock');
    await stopTree(child, 'busy-lock-final');
    downloadServer?.closeAllConnections();
    await new Promise(resolve => downloadServer ? downloadServer.close(resolve) : resolve());
    await browser?.close();
  }
});

await run('protocol-zero', async () => {
  const f = await fixture('protocol-zero');
  const flat = path.join(f.dir, 'flat');
  ps(`Expand-Archive -LiteralPath ${psQuote(path.join(archiveDir, 'HOT-Step-CPP-v91.0.2-win-x64-cpu.zip'))} -DestinationPath ${psQuote(flat)}`);
  assert.equal(read(path.join(flat, 'build-info.json')).updaterProtocol, 0);
  assert.ok(!fs.existsSync(path.join(flat, 'current.json')));
  assert.ok(!fs.existsSync(path.join(flat, 'app')));
  const log = fs.openSync(path.join(evidence, 'protocol-zero-server.log'), 'w');
  // The flat bat hardcodes its browser URL to 3001. Run its packaged server command
  // directly with the fixture's ports; no launcher/IPC is appropriate to protocol 0.
  const child = spawn(path.join(flat, 'runtime/node.exe'), [path.join(flat, 'server/server.mjs')], { cwd: flat, env: { ...f.env, HOT_STEP_ROOT: flat }, windowsHide: true, stdio: ['ignore', log, log] });
  fs.closeSync(log);
  try {
    await until(health);
    const refused = await api('/api/updates/apply', { candidateId: '00000000-0000-4000-8000-000000000000' });
    write(path.join(evidence, 'protocol-zero-refusal.json'), refused);
    assert.equal(refused.status, 409);
    assert.match(refused.body.error, /protocol 0/);
    return { flatLayout: true, refused };
  } finally { await stopTree(child, 'protocol-zero-final'); }
});
await run('recovery-no-server', async () => {
  const f = await fixture('recovery-verified', 'before-pointer');
  let child, browser;
  try {
    browser = await chromium.launch(browserOptions);
    const page = await browser.newPage();
    page.on('framenavigated', frame => { if (frame === page.mainFrame()) fs.appendFileSync(path.join(evidence, 'recovery-navigation.log'), frame.url() + '\n'); });
    child = f.start();
    await until(health);
    await page.goto(url + '/settings');
    await page.getByRole('button', { name: /^Updates/ }).click();
    await page.locator('.settings-updates').getByText('Installed build', { exact: true }).waitFor();
    seed(f.data);
    await prepare();
    write(path.join(f.install, 'updater/last-result.json'), { format: 1, outcome: 'installed', from: f.payload, to: f.payload.replace(baseTag, nextTag), at: '2026-01-01T00:00:00.000Z', message: 'Stale fixture receipt' });
    await page.reload();
    await page.getByRole('button', { name: /^Updates/ }).click();
    await page.locator('.settings-updates').getByRole('button', { name: 'Restart and install', exact: true }).click();
    await page.getByRole('button', { name: 'Restart and install', exact: true }).last().click();
    await page.locator('.settings-update-overlay').waitFor();
    await until(() => fs.existsSync(f.cfg.marker));
    const overlayPresent = await page.locator('.settings-update-overlay').count();
    fs.writeFileSync(path.join(evidence, 'recovery-before-kill.txt'), await page.locator('body').innerText());
    await page.screenshot({ path: path.join(evidence, 'recovery-before-kill.png') });
    await stopTree(child, 'recovery-no-server-interruption');
    const tx = read(path.join(f.install, 'updater/transaction.json'));
    const manifest = read(path.join(tx.snapshot.dir, 'manifest.json'));
    const entry = manifest.entries.find(e => e.existed && e.kind === 'db');
    assert.ok(entry);
    // Damage the disposable snapshot to exercise the real no-server recovery path.
    fs.appendFileSync(path.join(tx.snapshot.dir, entry.name), 'injected damage');
    f.cfg.fault = '';
    write(f.cfgFile, f.cfg);
    child = f.start();
    const result = await until(() => {
      const resultFile = path.join(f.install, 'updater/last-result.json');
      if (!fs.existsSync(resultFile)) return false;
      const r = read(resultFile);
      return r.at !== '2026-01-01T00:00:00.000Z' && r;
    });
    assert.equal(result.outcome, 'recovery-required');
    write(path.join(evidence, 'recovery-no-server-last-result.json'), result);
    await until(() => fs.readFileSync(path.join(evidence, `${f.name}-launcher-2.log`), 'utf8').includes('Recovery required'));
    if (!overlayPresent) {
      const images = await page.locator('img').evaluateAll(imgs => imgs.map(i => ({ alt: i.alt })));
      const e = new Error('Reconnect overlay unmounted on server disconnect; offline splash replaces the five-minute launcher guidance');
      e.details = { result, images, overlayCount: await page.locator('.settings-update-overlay').count(), evidence: 'recovery-before-kill.png; recovery-verified-launcher-2.log' };
      throw e;
    }
    await page.getByText(/launcher window/).waitFor({ timeout: 310_000 });
    const text = await page.locator('.settings-update-overlay').innerText();
    fs.writeFileSync(path.join(evidence, 'recovery-no-server-overlay.txt'), text);
    await page.locator('.settings-update-overlay').screenshot({ path: path.join(evidence, 'recovery-no-server.png') });
    let listening = true;
    try { await health(); } catch { listening = false; }
    assert.equal(listening, false);
    return { result, text, timeout: 'Real five-minute wait; launcher refused damaged snapshot and started no server' };
  } finally { await stopTree(child, 'recovery-no-server-final'); await browser?.close(); }
});
console.log(JSON.stringify(results, null, 2));
process.exitCode = results.some(r => r.verdict === 'FAIL') ? 1 : 0;

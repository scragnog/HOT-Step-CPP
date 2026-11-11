import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture';
const kit = path.join(root, 'owner-kit-1ced828d');
const source = '1ced828d7273b01c9269b8f74f5334c6fc55845c';
assert.ok(!fs.existsSync(kit) || fs.readdirSync(kit, { recursive: true, withFileTypes: true }).every(e => e.isDirectory()), 'refuse to overwrite a populated owner kit');
const write = (name, value) => fs.writeFileSync(path.join(kit, name), JSON.stringify(value, null, 2) + '\n');
const quote = s => "'" + s.replaceAll("'", "''") + "'";
for (const name of ['assets', 'profile', 'data/settings', 'logs']) fs.mkdirSync(path.join(kit, name), { recursive: true });
const hashes = {};
for (const version of ['94.0.0', '94.0.1']) {
  const name = `HOT-Step-CPP-v${version}-win-x64-cuda13.1.zip`;
  // The local packager uses "cuda" for its filename; published updater assets
  // use the toolkit-specific name from build-info. The ZIP bytes stay intact.
  const localName = `HOT-Step-CPP-v${version}-win-x64-cuda.zip`;
  fs.copyFileSync(path.join(root, 'source/release/out', localName), path.join(kit, 'assets', name));
  const digest = crypto.createHash('sha256').update(fs.readFileSync(path.join(kit, 'assets', name))).digest('hex');
  assert.ok(fs.readFileSync(path.join(root, 'source/release/out', localName + '.sha256'), 'utf8').toLowerCase().includes(digest));
  fs.writeFileSync(path.join(kit, 'assets', name + '.sha256'), `${digest}  ${name}\n`);
  hashes[name] = digest;
}
execFileSync('pwsh', ['-NoProfile', '-Command', `Expand-Archive -LiteralPath ${quote(path.join(kit, 'assets/HOT-Step-CPP-v94.0.0-win-x64-cuda13.1.zip'))} -DestinationPath ${quote(path.join(kit, 'install'))}`], { timeout: 120_000, windowsHide: true });
const pointer = JSON.parse(fs.readFileSync(path.join(kit, 'install/current.json'), 'utf8'));
const info = JSON.parse(fs.readFileSync(path.join(kit, 'install/app', pointer.payload, 'build-info.json'), 'utf8'));
assert.equal(info.sha, source);
assert.equal(info.updaterProtocol, 1);
assert.equal(info.variant, 'cuda13.1');
assert.ok(fs.existsSync(path.join(kit, 'install/app', pointer.payload, 'engine/ggml-cuda.dll')));
const candidate = 'HOT-Step-CPP-v94.0.1-win-x64-cuda13.1.zip';
const release = { tag_name: 'v94.0.1', name: 'Owner portable update check', draft: false, prerelease: false, published_at: '2026-11-11T00:00:00Z', html_url: 'https://github.com/scragnog/HOT-Step-CPP/releases/tag/v94.0.1', assets: [candidate, candidate + '.sha256'].map((name, i) => ({ id: i + 1, name, size: fs.statSync(path.join(kit, 'assets', name)).size, digest: 'sha256:' + crypto.createHash('sha256').update(fs.readFileSync(path.join(kit, 'assets', name))).digest('hex'), browser_download_url: `https://github.com/scragnog/HOT-Step-CPP/releases/download/v94.0.1/${name}` })) };
write('feed.json', { sha: source, release });
fs.copyFileSync(path.join(here, 'owner-feed.mjs'), path.join(kit, 'feed.mjs'));
const empty = { etag: null, data: null, okAt: null, attemptAt: null, error: null, retryUntil: null };
write('data/settings/updates.json', { version: 1, channel: 'release', master: empty, release: { ...empty, okAt: Date.now(), attemptAt: Date.now() - 120_000, data: { tag: release.tag_name, name: release.name, url: release.html_url, publishedAt: release.published_at } } });
fs.writeFileSync(path.join(kit, 'preflight.mjs'), `import net from 'node:net';
for (const port of [13960, 18960]) {
  const server = net.createServer();
  await new Promise((resolve, reject) => server.once('error', reject).listen(port, '127.0.0.1', resolve));
  await new Promise(resolve => server.close(resolve));
}
console.log('Owner kit ports are free. Open http://localhost:13960/settings');
`);
const script = ['@echo off', 'setlocal', 'cd /d "%~dp0"', 'set "NODE_OPTIONS="', '"%~dp0install\\runtime\\node.exe" "%~dp0preflight.mjs"', 'if errorlevel 1 (', '  echo Owner kit ports are busy. Close the other kit window first.', '  pause', '  exit /b 1', ')',
  ...['ACESTEPCPP_EXE','ACESTEPCPP_MODELS','ACESTEPCPP_ADAPTERS','DATA_DIR','DOWNLOAD_DIR','TENSORRT_LIBS','HOT_STEP_ROOT','HOT_STEP_INSTALL_ROOT','HOT_STEP_UPDATER_DIR','HOT_STEP_MAINTENANCE_TOKEN','CUDA_VISIBLE_DEVICES'].map(k => `set "${k}="`),
  'set "HOT_STEP_PROFILE_DIR=%~dp0profile"', 'set "HOT_STEP_DATA_ROOT=%~dp0data"', 'set "SERVER_HOST=127.0.0.1"', 'set "SERVER_PORT=13960"', 'set "ACESTEPCPP_HOST=127.0.0.1"', 'set "ACESTEPCPP_PORT=18960"', 'set "ACESTEPCPP_WARM_ON_STARTUP=0"', 'set "TRAINING_WORKERS="',
  // --import takes a URL: a bare D:\ path is read as scheme "d:" and Node exits before the launcher runs.
  // ponytail: no percent-encoding, so the kit path must have no spaces; the fixture root has none.
  'set "KIT_URL=%~dp0"', 'set "NODE_OPTIONS=--import=file:///%KIT_URL:\\=/%feed.mjs"',
  // OWNER_KIT_CHECK=1 (verify-owner-kit.mjs) runs everything above, then asks the preloaded feed for its tag instead of starting the app.
  'if "%OWNER_KIT_CHECK%"=="1" goto check', 'call "%~dp0install\\HOT-Step.bat"', 'exit /b %ERRORLEVEL%', ':check',
  `"%~dp0install\\runtime\\node.exe" -e "fetch('https://api.github.com/repos/scragnog/HOT-Step-CPP/releases/latest').then(r=>r.json()).then(j=>process.exit(j.tag_name==='${release.tag_name}'?0:1))"`, 'exit /b %ERRORLEVEL%'];
fs.writeFileSync(path.join(kit, 'Start-owner-check.cmd'), script.join('\r\n') + '\r\n');
fs.writeFileSync(path.join(kit, 'README.txt'), `Owner portable-update check — not started by the agent

Both packages come from ${source}; fixture-only versions v94.0.0 -> v94.0.1,
updater protocol 1, CUDA 13.1. Existing native binaries were packaged with
-SkipEngine; this is the first CUDA runtime check. No weights or fake models
are supplied. The first start may download CUDA runtime files into data/.

1. When the GPU is available, double-click Start-owner-check.cmd. Keep its window.
2. Open http://localhost:13960/settings if the browser does not open there.
   If Model Manager opens for the empty models folder, close it first.
3. Open Updates. Confirm v94.0.0. Click Check now, then Download update.
   If Check now is throttled by startup, wait one minute and click it again.
4. Click Restart and install, then confirm. Keep the original tab open.
5. Expect Update installed for v94.0.1, then click Reload yourself.
6. Confirm v94.0.1, one browser tab, and no engine failure. An empty-model
   notice is expected until models are installed; no generation is required.
7. Report the outcome and any unexpected message. Keep this kit for inspection.

Only this pair's release/feed URLs are served from assets/ by feed.mjs; the
normal updater still checks the archive and hashes. These are local fixture
tags, not public releases; the GitHub release-notes link will not exist.
Always use Start-owner-check.cmd so the feed seam and isolation stay active.

Server 13960; engine 18960. profile/ and data/ belong only to this kit.
App code is in install/. Candidate ZIP and checksums are in assets/.
Result: install/updater/last-result.json. Session logs: data/logs/.
Stop with Ctrl+C in the launcher window and let it exit. Close the browser tab.
Do not run another copy on these ports. No live checkout or real data is used.
`);
const proof = { source, kit, variant: 'cuda13.1', protocol: 1, versions: ['v94.0.0', 'v94.0.1'], ports: { server: 13960, engine: 18960 }, hashes, started: false, native: 'Reused CUDA 13.1 binaries; no native engine started during packaging. Packager smoke-boot uses an absent engine path.' };
write('kit.json', proof);
fs.writeFileSync(path.join(here, 'evidence/round3/owner-kit.json'), JSON.stringify(proof, null, 2));
console.log(JSON.stringify(proof, null, 2));

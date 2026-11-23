// Production bundles only. No app server, engine, dev port or real data is used.
// Run: node scripts/check-pwa.mjs [--playwright <module path>]
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';

const ui = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const req = createRequire(import.meta.url);
const arg = process.argv.indexOf('--playwright');
const workerRef = process.argv.indexOf('--worker-ref');
const sheetRef = process.argv.indexOf('--sheet-ref');
const { chromium } = req(arg >= 0 ? resolve(process.argv[arg + 1]) : 'playwright');
await mkdir(join(ui, 'node_modules/.cache'), { recursive: true });
const temp = await mkdtemp(join(ui, 'node_modules/.cache/pwa-fixture-'));
let browser;
let server;
let active;
const counts = new Map();
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.webmanifest': 'application/manifest+json', '.png': 'image/png', '.webp': 'image/webp' };
try {
  for (const version of ['a', 'b']) {
    const publicDir = join(temp, `public-${version}`);
    await cp(join(ui, 'public'), publicDir, { recursive: true });
    if (workerRef >= 0) await writeFile(join(publicDir, 'sw.js'), execFileSync('git', ['show', `${process.argv[workerRef + 1]}:ui/public/sw.js`], { cwd: ui }));
    const manifest = JSON.parse(await readFile(join(publicDir, 'manifest.webmanifest'), 'utf8'));
    manifest.name = `PWA fixture ${version}`;
    await writeFile(join(publicDir, 'manifest.webmanifest'), JSON.stringify(manifest));
    // Valid images with different bytes, no source assets are changed.
    if (version === 'b') await cp(join(publicDir, 'icon-512.png'), join(publicDir, 'icon-192.png'));
    console.log(`Building production fixture ${version}...`);
    await build({ root: ui, publicDir, logLevel: 'error',
      plugins: sheetRef < 0 ? [] : [{
        name: 'original-sheet-repro', enforce: 'pre',
        load(id) {
          if (id.replaceAll('\\', '/').endsWith('/src/components/shared/overlays/BottomSheet.tsx')) {
            return execFileSync('git', ['show', `${process.argv[sheetRef + 1]}:ui/src/components/shared/BottomSheet.tsx`], { cwd: ui, encoding: 'utf8' });
          }
        },
      }], build: {
      outDir: join(temp, version), emptyOutDir: true,
      rollupOptions: { input: { app: join(ui, 'index.html'), sheet: join(ui, 'scripts/pwa-fixture.html') } },
    } });
  }
  active = join(temp, 'a');
  const workerA = await readFile(join(active, 'sw.js'), 'utf8');
  const workerB = await readFile(join(temp, 'b/sw.js'), 'utf8');
  assert.ok(!workerA.includes('__HOTSTEP_'), 'worker template must be compiled');
  assert.ok(workerA !== workerB, 'metadata-only update must change worker');
  server = createServer(async (request, response) => {
    const url = new URL(request.url, 'http://localhost');
    counts.set(url.pathname, (counts.get(url.pathname) || 0) + 1);
    response.setHeader('Cache-Control', 'no-store');
    // Offline testing intentionally disconnects the app; suppress its automatic
    // health-recovery reload while this fixture controls navigation itself.
    if (url.pathname === '/api/health') { response.writeHead(503).end(); return; }
    if (/^\/(api|audio|references)\//.test(url.pathname)) {
      response.setHeader('Content-Type', 'application/json');
      let data = {};
      if (url.pathname === '/api/auth/auto') data = { token: 'fixture', user: { id: 'fixture', username: 'Fixture' } };
      else if (url.pathname.includes('/songs')) data = { songs: [], total: 0 };
      else if (url.pathname.includes('/queue')) data = { items: [], jobs: [], status: 'idle' };
      response.end(JSON.stringify(data));
      return;
    }
    const relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
    if (relative.includes('..')) { response.writeHead(400).end(); return; }
    try {
      let file = join(active, relative);
      let body;
      try { body = await readFile(file); } catch {
        if (extname(relative)) { response.writeHead(404).end(); return; }
        file = join(active, 'index.html'); body = await readFile(file);
      }
      response.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream');
      response.end(body);
    } catch { response.writeHead(500).end(); }
  });
  await new Promise((ok, fail) => { server.once('error', fail); server.listen(0, '127.0.0.1', ok); });
  const base = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  await context.addInitScript(() => {
    window.openedSongs = [];
    window.addEventListener('hotstep:open-song', e => window.openedSongs.push(e.detail.songId));
    const add = ServiceWorkerContainer.prototype.addEventListener;
    ServiceWorkerContainer.prototype.addEventListener = function(type, ...args) {
      if (type === 'message') window.notificationListenerMounted = true;
      return add.call(this, type, ...args);
    };
  });
  const page = await context.newPage();
  page.setDefaultTimeout(20000);
  await page.goto(base);
  await page.waitForFunction(() => !!navigator.serviceWorker.controller, undefined, { timeout: 30000 });
  const cacheName = await page.evaluate(async () => (await caches.keys()).find(k => k.startsWith('hotstep-shell-')));
  assert.ok(cacheName);
  await page.evaluate(async () => {
    await caches.open('another-app-cache');
    await caches.open('hotstep-shell-obsolete');
  });
  for (const path of ['/api/fixture', '/audio/fixture.wav', '/references/fixture', '/assets/range-test']) {
    await page.evaluate(async path => { await fetch(path, { headers: { Range: 'bytes=0-10' } }); }, path);
    assert.ok(counts.get(path), `${path} reached network`);
  }
  const cached = await page.evaluate(async () => {
    const keys = await caches.keys();
    return (await Promise.all(keys.map(async key => (await (await caches.open(key)).keys()).map(r => new URL(r.url).pathname)))).flat();
  });
  assert.ok(!cached.some(p => /fixture|range-test/.test(p) && !p.endsWith('pwa-fixture.html')), 'API/media/range must not enter shell cache');
  const assets = cached.filter(p => p.startsWith('/assets/'));
  assert.ok(assets.length > 0, 'built assets must be precached before offline use');
  await context.setOffline(true);
  await page.goto(`${base}/offline-route`);
  assert.ok(await page.locator('#root').count());
  for (const asset of assets) assert.equal(await page.evaluate(async p => (await fetch(p)).status, asset), 200);
  await context.setOffline(false);
  console.log('PASS production registration, offline shell/assets, API/audio/range bypass');
  await page.goto(`${base}/scripts/pwa-fixture.html`);

  // Revalidate fixed URLs immediately, even before a worker update is requested.
  active = join(temp, 'b');
  assert.equal(await page.evaluate(async () => (await (await fetch('/manifest.webmanifest')).json()).name), 'PWA fixture b');
  const icon = Buffer.from(await page.evaluate(async () => Array.from(new Uint8Array(await (await fetch('/icon-192.png')).arrayBuffer()))));
  assert.deepEqual(icon, await readFile(join(active, 'icon-192.png')));
  await page.evaluate(async () => Promise.race([
    (await navigator.serviceWorker.getRegistration()).update(),
    new Promise((_, reject) => setTimeout(() => reject(new Error('worker update timeout')), 20000)),
  ]));
  // Wait on the worker lifecycle itself before reading the finished cleanup.
  await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.getRegistration();
    const worker = registration.installing || registration.waiting;
    if (worker && worker.state !== 'activated') await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`activation timeout: ${worker.state}`)), 20000);
      worker.addEventListener('statechange', () => {
        if (worker.state === 'activated') { clearTimeout(timer); resolve(); }
        if (worker.state === 'redundant') { clearTimeout(timer); reject(new Error('updated worker became redundant')); }
      });
    });
  });
  const keys = await page.evaluate(() => caches.keys());
  assert.ok(!keys.includes(cacheName), `old shell retired: ${keys}`);
  assert.ok(keys.includes('another-app-cache'));
  assert.ok(!keys.includes('hotstep-shell-obsolete'));
  assert.equal(keys.filter(k => k.startsWith('hotstep-shell-')).length, 1);
  console.log('PASS metadata/icon revalidation, second build activation, app-only cache cleanup');

  // Exercise the actual worker handler and mounted app listener. Synthetic events
  // cannot grant OS focus/openWindow permission; replace only those two methods.
  const worker = context.serviceWorkers()[0];
  async function click(songId, cold = false) {
    return worker.evaluate(async ({ songId, cold }) => {
      const match = self.clients.matchAll.bind(self.clients);
      const open = self.clients.openWindow.bind(self.clients);
      let opened;
      self.clients.matchAll = async options => cold ? [] : (await match(options)).map(c => { c.focus = async () => c; return c; });
      self.clients.openWindow = async url => { opened = url; return null; };
      try {
        const event = new Event('notificationclick');
        let done;
        Object.defineProperties(event, {
          notification: { value: { data: { songId }, close() {} } },
          waitUntil: { value: promise => { done = promise; } },
        });
        self.dispatchEvent(event);
        await done;
        return opened;
      } finally { self.clients.matchAll = match; self.clients.openWindow = open; }
    }, { songId, cold });
  }
  // Reload online so the full app has its notification listener mounted.
  await page.goto(base);
  await page.waitForFunction(() => window.notificationListenerMounted === true);
  await click('existing-song');
  await page.waitForFunction(() => window.openedSongs.includes('existing-song'));
  const coldUrl = await click('cold-song', true);
  assert.equal(new URL(coldUrl).searchParams.get('open-song'), 'cold-song');
  const coldPage = await context.newPage();
  await coldPage.goto(coldUrl);
  await coldPage.waitForFunction(() => window.openedSongs.includes('cold-song'));
  assert.equal(new URL(coldPage.url()).searchParams.has('open-song'), false);
  console.log('PASS notificationclick existing-client handshake and cold-start song delivery (synthetic OS boundary)');

  const sheet = await context.newPage();
  await sheet.goto(`${base}/scripts/pwa-fixture.html`);
  await sheet.locator('#opener').click();
  await sheet.waitForFunction(() => document.activeElement?.id === 'sheet-first');
  await sheet.locator('#sheet-first').click();
  await sheet.waitForFunction(() => document.activeElement?.id === 'child-first');
  await sheet.keyboard.press('Tab');
  assert.equal(await sheet.evaluate(() => document.activeElement.id), 'child-last');
  await sheet.keyboard.press('Escape');
  assert.equal(await sheet.getByRole('dialog', { name: 'Child', exact: true }).count(), 0);
  assert.equal(await sheet.getByRole('dialog', { name: 'Parent', exact: true }).count(), 1);
  await sheet.keyboard.press('Escape');
  assert.equal(await sheet.getByRole('dialog', { name: 'Parent', exact: true }).count(), 0);
  assert.equal(await sheet.evaluate(() => document.activeElement.id), 'opener');
  console.log('PASS sheet child layering, parent trap suspension, Escape ownership and focus return');
} finally {
  await browser?.close();
  if (server) { server.closeAllConnections(); await new Promise(ok => server.close(ok)); }
  await rm(temp, { recursive: true, force: true });
}

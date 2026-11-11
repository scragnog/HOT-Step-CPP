import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const kit = path.join(process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture', 'owner-kit-1ced828d');
const cfg = JSON.parse(fs.readFileSync(path.join(kit, 'feed.json'), 'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(kit, 'kit.json'), 'utf8'));
const quote = s => "'" + s.replaceAll("'", "''") + "'";
const infos = [];
for (const [name, expectedHash] of Object.entries(manifest.hashes)) {
  const zip = path.join(kit, 'assets', name);
  const info = JSON.parse(execFileSync('pwsh', ['-NoProfile', '-Command', `Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead(${quote(zip)})
try {
  $entry = @($zip.Entries | Where-Object { $_.FullName -match '^app/[^/]+/build-info.json$' })
  if ($entry.Count -ne 1) { throw 'Expected one payload build-info' }
  $reader = [IO.StreamReader]::new($entry[0].Open())
  try { $reader.ReadToEnd() } finally { $reader.Dispose() }
} finally { $zip.Dispose() }`], { encoding: 'utf8', windowsHide: true, timeout: 30_000 }));
  assert.equal(info.sha, manifest.source);
  assert.equal(info.updaterProtocol, 1);
  assert.equal(info.variant, 'cuda13.1');
  assert.ok(name.includes(info.tag));
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(zip)).digest('hex'), expectedHash);
  infos.push(info);
}
for (const name of ['feed.mjs', 'preflight.mjs']) execFileSync(process.execPath, ['--check', path.join(kit, name)], { windowsHide: true, timeout: 10_000 });
// Exercise only the feed seam; never launch the kit/server/engine. Unknown URLs
// go to a local fake fetch, so validation cannot contact a remote service.
globalThis.fetch = async input => Response.json({ fallback: String(input) });
await import(pathToFileURL(path.join(kit, 'feed.mjs')).href);
assert.deepEqual(await (await fetch('https://api.github.com/repos/scragnog/HOT-Step-CPP/releases/latest')).json(), cfg.release);
assert.deepEqual(await (await fetch(`https://api.github.com/repos/scragnog/HOT-Step-CPP/commits/${cfg.release.tag_name}`)).json(), { sha: manifest.source });
for (const asset of cfg.release.assets) {
  const response = await fetch(asset.browser_download_url);
  const hash = crypto.createHash('sha256');
  let bytes = 0;
  for await (const chunk of response.body) { hash.update(chunk); bytes += chunk.length; }
  assert.equal(bytes, asset.size);
  assert.equal('sha256:' + hash.digest('hex'), asset.digest);
}
assert.deepEqual(await (await fetch('https://example.invalid/unrelated')).json(), { fallback: 'https://example.invalid/unrelated' });
assert.equal(fs.existsSync(path.join(kit, 'data/library')), false);
assert.equal(fs.existsSync(path.join(kit, 'install/updater/last-result.json')), false);
assert.deepEqual(fs.readdirSync(path.join(kit, 'profile')), []);
const result = { verdict: 'PASS', kit, infos, checks: ['both archive identities/hashes', 'feed/commit/asset/checksum responses', 'unrelated URL passthrough', 'feed/preflight syntax', 'no database, receipt or populated profile'], started: false };
fs.writeFileSync(path.join(here, 'evidence/round3/owner-kit-validation.json'), JSON.stringify(result, null, 2));
console.log('PASS owner kit: identities, hashes and feed seam verified; kit not started');

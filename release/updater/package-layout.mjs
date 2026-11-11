// package-layout.mjs — the last step before a Windows package is zipped.
// release/build-release.ps1 and the build-windows job of release.yml both run
// it, so the fixture tests cover what CI ships.
//
//   node release/updater/package-layout.mjs <staging>
//
// build-info.json updaterProtocol 0 (what releases ship today): the flat
// package is left exactly as it is. Protocol 1 or higher: the assembled
// package moves into app/<id>/ and the stable root is added:
//   HOT-Step.bat, launcher.mjs, runtime/node.exe (the Node that runs the
//   launcher), current.json, README.txt, app/<id>/
// app/<id>/HOT-Step.bat is removed (only the launcher starts a payload), and
// app/<id>/payload-manifest.json records the SHA-256 of every shipped file so
// an update can refuse a payload someone has edited or added files to.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { payloadIdFor } from './launcher.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const MANIFEST_FILE = 'payload-manifest.json';
export const ROOT_ENTRIES = ['HOT-Step.bat', 'launcher.mjs', 'runtime', 'current.json', 'README.txt', 'app'];

function* files(dir, rel = '') {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) yield* files(path.join(dir, e.name), r);
    else if (e.isFile()) yield r;
    else throw new Error(`package-layout: ${r} is not a plain file`);
  }
}

/** sha256 of every file under `dir` except the manifest itself, keyed by '/'-separated path. */
export function hashTree(dir) {
  const out = {};
  for (const rel of files(dir)) {
    if (rel === MANIFEST_FILE) continue;
    out[rel] = crypto.createHash('sha256').update(fs.readFileSync(path.join(dir, ...rel.split('/')))).digest('hex');
  }
  return out;
}

export function packageLayout(staging) {
  const entries = fs.readdirSync(staging);
  if (entries.includes('app') || entries.includes('current.json')) throw new Error('package-layout: staging is already laid out');
  const info = JSON.parse(fs.readFileSync(path.join(staging, 'build-info.json'), 'utf8'));
  if (!(info.updaterProtocol >= 1)) return { layout: 'flat', id: null };
  const id = payloadIdFor(info);
  const tmp = path.join(staging, `.payload-${process.pid}`);
  fs.mkdirSync(tmp);
  for (const e of entries) fs.renameSync(path.join(staging, e), path.join(tmp, e));
  fs.mkdirSync(path.join(staging, 'app'));
  const payload = path.join(staging, 'app', id);
  fs.renameSync(tmp, payload);
  fs.rmSync(path.join(payload, 'HOT-Step.bat'), { force: true });
  fs.writeFileSync(path.join(payload, MANIFEST_FILE), JSON.stringify({ format: 1, files: hashTree(payload) }) + '\n');

  fs.copyFileSync(path.join(here, 'HOT-Step.bat'), path.join(staging, 'HOT-Step.bat'));
  fs.copyFileSync(path.join(here, 'launcher.mjs'), path.join(staging, 'launcher.mjs'));
  fs.copyFileSync(path.join(here, '..', 'README.txt'), path.join(staging, 'README.txt'));
  fs.mkdirSync(path.join(staging, 'runtime'));
  fs.copyFileSync(path.join(payload, 'runtime', 'node.exe'), path.join(staging, 'runtime', 'node.exe'));
  fs.writeFileSync(path.join(staging, 'current.json'), JSON.stringify({ format: 1, payload: id }) + '\n');
  return { layout: 'versioned', id };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const staging = process.argv[2];
  if (!staging || !fs.existsSync(staging)) throw new Error(`package-layout: staging folder not found: ${staging}`);
  const r = packageLayout(staging);
  console.log(r.id ? `package-layout: versioned, payload app/${r.id}` : 'package-layout: protocol 0, flat package unchanged');
}

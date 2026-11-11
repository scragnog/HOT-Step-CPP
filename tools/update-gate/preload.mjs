// Process-local fault injection. No package source or payload manifest is changed.
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import childProcess from 'node:child_process';
import https from 'node:https';
import http from 'node:http';
import { Readable } from 'node:stream';
import { createRequire, syncBuiltinESMExports } from 'node:module';

const cfgPath = process.env.UPDATE_GATE_CONFIG;
if (!cfgPath) throw new Error('UPDATE_GATE_CONFIG is required');
const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
const append = message => fs.appendFileSync(cfg.trace, JSON.stringify({ at: new Date().toISOString(), pid: process.pid, entry: path.basename(process.argv[1] || ''), payload: process.env.HOT_STEP_ROOT, message }) + '\n');
const emit = net.Server.prototype.emit;
net.Server.prototype.emit = function (event, ...args) {
  if (event === 'error') append(`net error ${args[0]?.stack}`);
  if (event === 'request' && args[0].url === '/api/maintenance/status') {
    const response = args[1], end = response.end;
    response.end = function (chunk, ...rest) {
      try { append({ attestation: true, body: { engine: JSON.parse(String(chunk)).engine } }); } catch {}
      return end.call(this, chunk, ...rest);
    };
  }
  return emit.call(this, event, ...args);
};
const nativeFetch = globalThis.fetch;
const httpsGet = https.get;
https.get = function (url, options, callback) {
  const live = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
  if (live.busyUrl && String(url) === live.busyUrl) {
    append('controlled model download');
    return http.get(live.busyLocalUrl, options, callback);
  }
  return httpsGet.apply(this, arguments);
};
globalThis.fetch = async (input, init) => {
  const url = String(input?.url || input);
  const api = 'https://api.github.com/repos/scragnog/HOT-Step-CPP';
  if (url === `${api}/releases/latest`) {
    append('feed latest');
    return Response.json(cfg.release);
  }
  if (url === `${api}/commits/${cfg.release.tag_name}`) {
    append('feed commit');
    return Response.json({ sha: cfg.sha });
  }
  const asset = cfg.release.assets.find(a => a.browser_download_url === url);
  if (asset) {
    append(`asset ${asset.name}`);
    return new Response(Readable.toWeb(fs.createReadStream(path.join(cfg.archiveDir, asset.name))), { headers: { 'content-length': String(asset.size) } });
  }
  const response = await nativeFetch(input, init);
  if (url.includes('/api/maintenance/status') || url.endsWith('/health')) {
    try { append({ url, status: response.status, body: await response.clone().json() }); } catch {}
  }
  return response;
};

const entry = path.basename(process.argv[1] || '');
const nativeSpawn = childProcess.spawn;
childProcess.spawn = function (exe, args, options) {
  if (String(exe).endsWith('ace-server.exe')) append({ engineSpawn: String(exe) });
  return nativeSpawn.apply(this, arguments);
};
if (entry === 'open-browser-if-needed.mjs') {
  const log = console.log;
  console.log = (...args) => { append(args.join(' ')); log(...args); };
  const spawn = childProcess.spawn;
  childProcess.spawn = function (exe, args, options) {
    if (exe === 'explorer.exe') append(`browser-open ${args.join(' ')}`);
    return spawn.apply(this, arguments);
  };
}
const fault = cfg.fault;
const freeze = point => {
  append(`frozen ${point}`);
  fs.writeFileSync(cfg.marker, JSON.stringify({ point, pid: process.pid }));
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 300_000);
  throw new Error(`Fault controller did not kill process at ${point}`);
};
const rename = fs.renameSync;
fs.renameSync = function (from, to) {
  const target = String(to);
  let value;
  if (entry === 'launcher.mjs' && ['current.json', 'transaction.json'].includes(path.basename(target))) {
    value = JSON.parse(fs.readFileSync(from, 'utf8'));
  }
  const nextPointer = path.basename(target) === 'current.json' && value?.payload?.startsWith(cfg.release.tag_name + '-');
  if (nextPointer && fault === 'before-pointer') freeze(fault);
  const result = rename.apply(this, arguments);
  if (nextPointer && fault === 'after-pointer') freeze(fault);
  if (entry === 'launcher.mjs' && path.basename(target) === 'transaction.json' && value?.state === 'committed' && ['after-commit', 'committed-missing'].includes(fault)) freeze(fault);
  if (entry === 'updater-tool.mjs' && String(from).endsWith('.restore-tmp') && fault === 'mid-restore') freeze(fault);
  return result;
};
syncBuiltinESMExports();

const candidate = process.env.HOT_STEP_ROOT?.includes(cfg.release.tag_name + '-');
if (entry === 'server.mjs' && (fault === 'both-no-models' || (candidate && fault === 'no-models'))) {
  process.env.ACESTEPCPP_MODELS = path.join(cfg.data, 'empty-models');
  fs.mkdirSync(process.env.ACESTEPCPP_MODELS, { recursive: true });
  append(`injected ${fault}`);
}
if (entry === 'server.mjs' && candidate && fault === 'committed-engine-absent') {
  process.env.ACESTEPCPP_EXE = path.join(cfg.data, 'deliberately-absent-engine.exe');
  append('injected committed engine absent');
}
if (entry === 'server.mjs' && candidate && ['failed-health', 'mid-restore'].includes(fault)) {
  // An absent engine gives the real server a permanently failed attestation.
  process.env.ACESTEPCPP_EXE = path.join(cfg.data, 'deliberately-absent-engine.exe');
  const require = createRequire(path.join(process.env.HOT_STEP_ROOT, 'server', 'server.mjs'));
  const Database = require('better-sqlite3');
  const db = new Database(path.join(cfg.data, 'library', 'hotstep.db'));
  db.prepare("UPDATE songs SET title = 'candidate must be rolled back' WHERE id = 'update-gate-song'").run();
  db.prepare("UPDATE settings SET value = 'candidate' WHERE key = 'update-gate'").run();
  db.close();
  fs.appendFileSync(path.join(cfg.data, 'settings', '.env'), '\nUPDATE_GATE_CANDIDATE=must-be-rolled-back\n');
  append(`candidate changed song/settings/env; injected ${fault}`);
}
append('preload installed');

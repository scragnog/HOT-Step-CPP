// Windows archive gate. See smoke-artifact.md for usage and retained evidence.
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { checkPackageBuildInfo } from './check-layout.mjs';
import { readPointer } from './updater/launcher.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const contained = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
};

export function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer().once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(error => error ? reject(error) : resolve(port));
    });
  });
}

// All spawned commands have a deadline and a finally block that reaps their tree.
// On Windows even node's direct child can have children (smoke-boot -> server).
export async function withProcess(exe, args, options, timeoutMs, logFile, action) {
  const log = fs.openSync(logFile, 'a');
  const child = spawn(exe, args, { ...options, windowsHide: true, stdio: ['ignore', log, log] });
  let outcome;
  const closed = new Promise(resolve => {
    child.once('error', error => { outcome = { error }; resolve(); });
    child.once('close', (code, signal) => { outcome = { code, signal }; resolve(); });
  });
  let timer;
  let interrupt;
  const interrupted = new Promise((_, reject) => {
    interrupt = () => reject(new Error(`Interrupted while running ${exe}`));
    process.once('SIGINT', interrupt);
    process.once('SIGTERM', interrupt);
  });
  try {
    return await Promise.race([
      action({ child, closed, outcome: () => outcome }),
      interrupted,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Timed out after ${timeoutMs}ms: ${exe}`)), timeoutMs); }),
    ]);
  } finally {
    clearTimeout(timer);
    process.removeListener('SIGINT', interrupt);
    process.removeListener('SIGTERM', interrupt);
    let reapTimer;
    try {
      if (!outcome && child.pid) {
        if (process.platform === 'win32') {
          await new Promise((resolve, reject) => {
            const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
            const killTimer = setTimeout(() => { killer.kill(); reject(new Error(`taskkill timed out for PID ${child.pid}`)); }, 10000);
            killer.once('error', error => { clearTimeout(killTimer); reject(error); });
            killer.once('close', () => { clearTimeout(killTimer); resolve(); });
          });
        } else {
          child.kill('SIGKILL');
        }
      }
      // close must be observed; a successful signal alone is not cleanup proof.
      await Promise.race([closed, new Promise((_, reject) => { reapTimer = setTimeout(() => reject(new Error(`PID ${child.pid} was not reaped`)), 5000); })]);
    } finally {
      clearTimeout(reapTimer);
      fs.closeSync(log);
    }
  }
}

export async function runCommand(exe, args, options, timeoutMs, logFile) {
  return withProcess(exe, args, options, timeoutMs, logFile, async state => {
    await state.closed;
    const result = state.outcome();
    if (result.error) throw result.error;
    if (result.code !== 0) throw new Error(`${path.basename(exe)} exited ${result.code} (signal ${result.signal}); see ${logFile}`);
  });
}

export function writeStubs(directory) {
  fs.mkdirSync(directory, { recursive: true });
  const u32 = value => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b; };
  const u64 = value => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; };
  const string = value => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);
  for (const arch of ['acestep-lm', 'acestep-dit', 'acestep-text-enc', 'acestep-vae']) {
    fs.writeFileSync(path.join(directory, `stub-${arch}.gguf`), Buffer.concat([
      Buffer.from('GGUF'), u32(3), u64(0), u64(1), string('general.architecture'), u32(8), string(arch),
    ]));
  }
}

export async function smokeEngine(exe, root, models, env, logFile, timeoutMs = 60000, prefixArgs = []) {
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  await withProcess(exe, [...prefixArgs, '--models', models, '--host', '127.0.0.1', '--port', String(port)],
    { cwd: root, env }, timeoutMs, logFile, async state => {
      const assertAlive = () => {
        const result = state.outcome();
        if (result) throw new Error(`Engine exited before smoke completed: ${JSON.stringify(result)}; see ${logFile}`);
      };
      const get = async endpoint => {
        const response = await fetch(`${base}/${endpoint}`, { signal: AbortSignal.timeout(2000) });
        if (!response.ok) throw new Error(`GET /${endpoint}: HTTP ${response.status}`);
        // Drain the body under the same request deadline.
        return response.text();
      };
      for (;;) {
        assertAlive();
        try { await get('health'); break; } catch { /* bounded by the process deadline */ }
        await delay(100);
      }
      for (const endpoint of ['health', 'props', 'plugins', 'props']) {
        assertAlive();
        const body = await get(endpoint);
        fs.appendFileSync(logFile, `\nGET /${endpoint}: ${body}\n`);
      }
      await delay(2000);
      assertAlive();
    });
}

// CUDA runtime DLLs are downloaded, not shipped. Same contract as the app:
// engine/.cuda-version picks the registry's runtime pack (server/src/index.ts
// getRequiredRuntimeIds lists the same ids), each file comes from Hugging Face
// at repo/repoPath, and the folder goes on PATH (config.ts RUNTIME_LIB_DIR).
export function cudaRuntimeFiles(registry, major) {
  const pack = registry.packs.find(p => p.id === (major <= 12 ? 'cuda12-runtime' : 'cuda-runtime'));
  if (!pack) throw new Error(`Registry has no CUDA ${major} runtime pack`);
  return pack.fileIds.map(id => {
    const file = registry.files.find(f => f.id === id);
    if (!file?.repo || !file.filename || !file.sizeBytes) throw new Error(`Registry runtime entry ${id} is incomplete`);
    return file;
  });
}

// actions/cache key: changes only when a runtime entry itself changes.
export function cudaRuntimeKey(registry, major) {
  const entries = cudaRuntimeFiles(registry, major).map(f => [f.id, f.filename, f.sizeBytes, f.repo, f.repoPath || f.filename]);
  return `cuda${major}-runtime-${crypto.createHash('sha256').update(JSON.stringify(entries)).digest('hex').slice(0, 16)}`;
}

export async function provisionCudaRuntime(registry, major, dir, timeoutMs = 900000) {
  fs.mkdirSync(dir, { recursive: true });
  const started = Date.now();
  const stats = { key: cudaRuntimeKey(registry, major), dir, bytes: 0, downloadedBytes: 0, downloaded: [] };
  for (const file of cudaRuntimeFiles(registry, major)) {
    const target = path.join(dir, file.filename);
    stats.bytes += file.sizeBytes;
    if (fs.statSync(target, { throwIfNoEntry: false })?.size === file.sizeBytes) continue;
    const url = `https://huggingface.co/${file.repo}/resolve/main/${file.repoPath || file.filename}`;
    const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
    await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(`${target}.part`));
    const size = fs.statSync(`${target}.part`).size;
    if (size !== file.sizeBytes) throw new Error(`${file.filename}: got ${size} bytes, registry says ${file.sizeBytes}`);
    fs.renameSync(`${target}.part`, target);
    stats.downloadedBytes += size;
    stats.downloaded.push(file.filename);
  }
  stats.seconds = (Date.now() - started) / 1000;
  return stats;
}

// Loader DLLs that only a GPU driver installs. A backend that cannot load
// because one of these is absent has an identified cause: no driver.
const DRIVER_DLLS = { 'nvcuda.dll': 'NVIDIA driver', 'vulkan-1.dll': 'Vulkan loader' };
const GPU_BACKENDS = { cuda: { reg: 'CUDA', file: 'ggml-cuda.dll' }, vulkan: { reg: 'Vulkan', file: 'ggml-vulkan.dll' } };

// Policy for the ace-server --backend-check report: returns [notes, failures].
export function assertBackends(report, variant) {
  const notes = [];
  const failures = [];
  const cpu = report.registered?.find(r => r.name === 'CPU');
  if (!report.cpu?.initialised || !cpu?.devices?.length) failures.push('CPU backend did not initialise');
  else notes.push(`CPU: ${report.cpu.description.trim()} via ${path.basename(report.cpu.module || '?')}`);
  if (!report.selected) failures.push('no backend selected');
  for (const f of report.files || []) {
    if (!f.load_error) continue;
    // Only a driver DLL that is not there at all (126, not on the search path)
    // proves "no driver"; a present one failing with 127/193/1114 is a defect.
    const driver = f.missing?.code === 126 && f.missing.exists === false && DRIVER_DLLS[f.missing.dll?.toLowerCase()];
    if (driver && f.file === GPU_BACKENDS[variant]?.file) notes.push(`${f.file}: ${driver} absent (${f.missing.chain})`);
    else failures.push(`${f.file} failed to load: ${f.load_error}${f.missing ? `; missing ${f.missing.chain} (${f.missing.error})` : '; no missing import found'}`);
  }
  const gpu = GPU_BACKENDS[variant];
  if (gpu) {
    const reg = report.registered?.find(r => r.name === gpu.reg);
    const file = report.files?.find(f => f.file === gpu.file);
    if (reg) {
      const reason = reg.devices.length ? ''
        : `; ${report.log?.find(l => l.toLowerCase().includes(variant) && !l.startsWith('load_backend')) || 'no reason logged'}`;
      notes.push(`${gpu.reg}: registered, ${reg.devices.length} device(s)${reason}`);
    } else if (!file) {
      failures.push(`package DLL ${gpu.file} is missing`);
    } else if (!file.load_error) {
      if (variant === 'vulkan' && file.vk_create_instance === -9) notes.push('Vulkan: no installable client driver (VK_ERROR_INCOMPATIBLE_DRIVER)');
      else failures.push(`${gpu.file} loaded but did not register: ${JSON.stringify(file)}`);
    }
  }
  return [notes, failures];
}

export async function checkBackends(exe, root, env, logFile, variant, timeoutMs = 60000) {
  await runCommand(exe, ['--backend-check'], { cwd: root, env }, timeoutMs, logFile);
  const line = fs.readFileSync(logFile, 'utf8').split(/\r?\n/).reverse().find(l => l.startsWith('{"registered"'));
  if (!line) throw new Error(`No --backend-check report in ${logFile}`);
  const [notes, failures] = assertBackends(JSON.parse(line), variant);
  if (failures.length) throw new Error(`Backend check (${variant}): ${failures.join('; ')}`);
  return notes;
}

// The gate must be able to fail on this artifact: take away a package DLL and,
// for CUDA, a provisioned runtime DLL, and require each check to fail.
async function negativeChecks(root, env, work, variant, runtimeDir, timeoutMs) {
  const exe = path.join(root, 'engine/ace-server.exe');
  const expectFail = async (name, runEnv) => {
    const log = path.join(work, `backend-negative-${name}.log`);
    try { await checkBackends(exe, root, runEnv, log, variant, timeoutMs); }
    catch (error) {
      // Only the engine or the report may fail it; a harness error is not proof.
      if (!/^(Backend check|ace-server\.exe exited|No --backend-check report|Timed out)/.test(error.message)) throw error;
      return `${name}: failed as required (${error.message.split('\n')[0]})`;
    }
    throw new Error(`Negative check ${name} passed; the backend gate cannot see that loss`);
  };
  const results = [];
  // CPU-only: every ggml-cpu*.dll, since a dynamic-backend build falls back
  // to the next CPU variant (the static CI build has just ggml-cpu.dll).
  const engineDir = path.join(root, 'engine');
  const removed = GPU_BACKENDS[variant] ? [GPU_BACKENDS[variant].file]
    : fs.readdirSync(engineDir).filter(f => /^ggml-cpu.*\.dll$/i.test(f));
  if (!removed.length) throw new Error('No CPU backend DLL to remove for the negative check');
  for (const f of removed) fs.renameSync(path.join(engineDir, f), path.join(engineDir, `${f}.removed`));
  try { results.push(await expectFail(`no-${removed.length > 1 ? 'ggml-cpu-all' : removed[0]}`, env)); }
  finally { for (const f of removed) fs.renameSync(path.join(engineDir, `${f}.removed`), path.join(engineDir, f)); }
  if (runtimeDir) {
    // A partial copy, so the shared runtime cache is never modified.
    const partial = path.join(work, 'runtime-partial');
    fs.mkdirSync(partial);
    const dropped = fs.readdirSync(runtimeDir).filter(f => f.toLowerCase().endsWith('.dll'))
      .sort((a, b) => fs.statSync(path.join(runtimeDir, b)).size - fs.statSync(path.join(runtimeDir, a)).size)[0];
    for (const f of fs.readdirSync(runtimeDir).filter(f => f !== dropped && f.toLowerCase().endsWith('.dll'))) {
      fs.copyFileSync(path.join(runtimeDir, f), path.join(partial, f));
    }
    results.push(await expectFail(`no-${dropped}`, { ...env, PATH: env.PATH.replace(runtimeDir, partial) }));
  }
  return results;
}

/** Resolve and check an extracted archive before starting anything from it. */
export function artifactPayload(root) {
  const payload = fs.existsSync(path.join(root, 'current.json'))
    ? path.join(root, 'app', readPointer(root)) : root;
  for (const file of ['runtime/node.exe', 'server/server.mjs', 'engine/ace-server.exe']) {
    if (!fs.statSync(path.join(payload, file), { throwIfNoEntry: false })?.isFile()) throw new Error(`Archive is missing ${file}`);
  }
  const info = JSON.parse(fs.readFileSync(path.join(payload, 'build-info.json'), 'utf8'));
  const errors = checkPackageBuildInfo(root, { os: 'windows', variant: info.variant });
  if (errors.length) throw new Error(`Archive package check: ${errors.join('; ')}`);
  return payload;
}

export async function smokeArtifact(zipArg, { outputParent = os.tmpdir(), timeoutMs = 60000, runtimeDir } = {}) {
  if (process.platform !== 'win32') throw new Error('This gate requires Windows and the packaged Windows runtime');
  if (!Number.isFinite(timeoutMs) || timeoutMs < 100 || timeoutMs > 600000) throw new Error('timeoutMs must be 100..600000');
  const zip = path.resolve(zipArg);
  if (!fs.statSync(zip).isFile()) throw new Error(`Not an archive: ${zip}`);
  const parent = fs.realpathSync(outputParent);
  const checkout = fs.realpathSync(path.join(here, '..'));
  if (contained(checkout, parent)) throw new Error('Extraction directory must be outside the checkout');
  const work = fs.mkdtempSync(path.join(parent, 'hotstep-artifact-'));
  const root = path.join(work, 'HOT Step smoke');
  const evidence = { zip, root, status: 'running' };
  const report = () => fs.writeFileSync(path.join(work, 'result.json'), JSON.stringify(evidence, null, 2));
  console.log(`smoke-artifact: evidence ${work}`);
  report();
  try {
    await runCommand('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      '$ErrorActionPreference = "Stop"; Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory($env:HOTSTEP_SMOKE_ZIP, $env:HOTSTEP_SMOKE_DEST)'],
    { env: { ...process.env, HOTSTEP_SMOKE_ZIP: zip, HOTSTEP_SMOKE_DEST: root } }, 120000, path.join(work, 'extract.log'));
    const payload = artifactPayload(root);
    evidence.payload = payload;
    const temp = path.join(work, 'temp');
    fs.mkdirSync(temp);
    const env = { ...process.env, TEMP: temp, TMP: temp,
      HOT_STEP_ROOT: payload, HOT_STEP_DATA_ROOT: path.join(work, 'data'), HOT_STEP_PROFILE_DIR: path.join(work, 'profile'),
      // Do not let a developer's DLL search paths mask an incomplete package.
      PATH: `${path.join(process.env.SystemRoot || 'C:\\Windows', 'System32')};${process.env.SystemRoot || 'C:\\Windows'}`,
    };
    await runCommand(path.join(payload, 'runtime/node.exe'), [path.join(here, 'smoke-boot.mjs'), path.join(payload, 'server/server.mjs'), String(timeoutMs / 1000)],
      { cwd: payload, env }, timeoutMs + 10000, path.join(work, 'server.log'));
    evidence.server = 'pass';
    const models = path.join(work, 'stub-models');
    writeStubs(models);
    // Local packages may omit the coarse markers; build identity is required
    // and the package check has already matched any markers that are present.
    const info = JSON.parse(fs.readFileSync(path.join(payload, 'build-info.json'), 'utf8'));
    const variant = info.variant.startsWith('cuda') ? 'cuda' : info.variant;
    let backendEnv = env;
    if (variant === 'cuda') {
      const major = Number(info.variant.match(/^cuda(\d+)/)[1]);
      const registry = JSON.parse(fs.readFileSync(path.join(payload, 'server/data/model-registry.json'), 'utf8'));
      runtimeDir = path.resolve(runtimeDir || path.join(work, 'data', 'tools', 'runtime'));
      evidence.runtime = await provisionCudaRuntime(registry, major, runtimeDir);
      backendEnv = { ...env, PATH: `${runtimeDir};${env.PATH}` };
    } else {
      runtimeDir = undefined;
    }
    await smokeEngine(path.join(payload, 'engine/ace-server.exe'), payload, models, backendEnv, path.join(work, 'engine.log'), timeoutMs);
    evidence.engine = 'pass';
    evidence.backends = await checkBackends(path.join(payload, 'engine/ace-server.exe'), payload, backendEnv, path.join(work, 'backend.log'), variant, timeoutMs);
    evidence.negative = await negativeChecks(payload, backendEnv, work, variant, runtimeDir, timeoutMs);
    evidence.status = 'pass';
    console.log(`smoke-artifact: PASS - extracted portable server, engine endpoints and ${variant} backends`);
    for (const line of [...evidence.backends, ...evidence.negative]) console.log(`  ${line}`);
    return work;
  } catch (error) {
    evidence.status = 'fail';
    evidence.error = error.message;
    throw error;
  } finally {
    report(); // Keep the extraction, isolated state and logs on both outcomes.
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const checkoutRegistry = () => JSON.parse(fs.readFileSync(path.join(here, '../server/src/data/model-registry.json'), 'utf8'));
  try {
    if (args[0] === 'payload-root' && args.length === 2) {
      console.log(artifactPayload(path.resolve(args[1])));
    } else if (args[0] === 'runtime-key' && args.length === 2) {
      console.log(cudaRuntimeKey(checkoutRegistry(), Number(args[1])));
    } else if (args[0] === 'runtime-fetch' && args.length === 3) {
      console.log(JSON.stringify(await provisionCudaRuntime(checkoutRegistry(), Number(args[1]), path.resolve(args[2]))));
    } else if (args.length >= 1 && args.length <= 4 && !args[0].startsWith('runtime-')) {
      const [zip, parent, seconds = '60', runtimeDir] = args;
      await smokeArtifact(zip, { outputParent: parent || undefined, timeoutMs: Number(seconds) * 1000, runtimeDir });
    } else {
      console.error('Usage: node release/smoke-artifact.mjs <windows.zip> [output-parent] [timeout-seconds] [cuda-runtime-dir]\n'
        + '       node release/smoke-artifact.mjs payload-root <extracted-root>\n'
        + '       node release/smoke-artifact.mjs runtime-key <cuda-major>\n'
        + '       node release/smoke-artifact.mjs runtime-fetch <cuda-major> <dir>');
      process.exitCode = 2;
    }
  } catch (error) { console.error(`smoke-artifact: FAIL - ${error.message}`); process.exitCode = 1; }
}

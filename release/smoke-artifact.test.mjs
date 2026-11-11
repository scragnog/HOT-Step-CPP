import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { artifactPayload, assertBackends, cudaRuntimeFiles, cudaRuntimeKey, runCommand, smokeArtifact, smokeEngine, writeStubs } from './smoke-artifact.mjs';
import { buildInfo, UPDATER_PROTOCOL } from './build-info.mjs';
import { packageLayout } from './updater/package-layout.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(here, 'out');
fs.mkdirSync(out, { recursive: true });

function fixture(t) {
  const root = fs.mkdtempSync(path.join(out, 'artifact-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

function alive(pid) {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

for (const protocol of [0, 1]) {
  test(`archive payload: protocol ${protocol}, package validation and missing binary`, t => {
    const root = fixture(t);
    const info = { format: 1, tag: 'v1.2.3', sha: 'a'.repeat(40), engineSha: 'b'.repeat(40),
      channel: 'release', os: 'windows', arch: 'x64', variant: 'cpu', updaterProtocol: protocol };
    for (const file of ['runtime/node.exe', 'server/server.mjs', 'server/updater-tool.mjs', 'engine/ace-server.exe', 'HOT-Step.bat']) {
      fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
      fs.writeFileSync(path.join(root, file), 'fixture');
    }
    fs.writeFileSync(path.join(root, 'build-info.json'), JSON.stringify(info));
    fs.writeFileSync(path.join(root, 'VERSION'), '1.2.3');
    const { id } = packageLayout(root);
    const payload = id ? path.join(root, 'app', id) : root;
    assert.equal(artifactPayload(root), payload);
    fs.writeFileSync(path.join(payload, 'VERSION'), '9.9.9');
    assert.throws(() => artifactPayload(root), /disagrees with VERSION/);
    fs.writeFileSync(path.join(payload, 'VERSION'), '1.2.3');
    if (id) {
      fs.writeFileSync(path.join(root, 'unexpected.txt'), 'bad root');
      assert.throws(() => artifactPayload(root), /root: unexpected/);
      fs.unlinkSync(path.join(root, 'unexpected.txt'));
      fs.writeFileSync(path.join(root, 'current.json'), JSON.stringify({ format: 1, payload: '../escape' }));
      assert.throws(() => artifactPayload(root), /not a valid version pointer/);
      fs.writeFileSync(path.join(root, 'current.json'), JSON.stringify({ format: 1, payload: id }));
    }
    fs.unlinkSync(path.join(payload, 'engine/ace-server.exe'));
    assert.throws(() => artifactPayload(root), /missing engine\/ace-server.exe/);
  });
}

test('protocol default is Windows-only and explicit protocol wins on every platform', () => {
  for (const [os, arch, variant] of [['windows', 'x64', 'cpu'], ['linux', 'x64', 'cpu'], ['macos', 'arm64', 'metal']]) {
    const opts = { os, arch, variant };
    assert.equal(buildInfo(opts).updaterProtocol, os === 'windows' ? UPDATER_PROTOCOL : 0);
    for (const protocol of [0, 1]) assert.equal(buildInfo({ ...opts, protocol }).updaterProtocol, protocol);
  }
});

// Real child processes exercise HTTP errors, crashes, hangs and cleanup without
// a GPU, native compiler or release archive. Full archive runs are separate.
const engineFixture = `
import fs from 'node:fs';
import http from 'node:http';
const [mode, pidFile, requestsFile, ...args] = process.argv.slice(2);
fs.writeFileSync(pidFile, String(process.pid));
if (mode === 'exit') process.exit(13);
if (mode === 'hang') { setInterval(() => {}, 1000); }
else {
  const port = Number(args[args.indexOf('--port') + 1]);
  http.createServer((req, res) => {
    fs.appendFileSync(requestsFile, req.url + '\\n');
    if (mode === 'bad-props' && req.url === '/props') res.writeHead(500);
    if (mode === 'stalled-props' && req.url === '/props') return;
    res.end('{}');
  }).listen(port, '127.0.0.1');
}
`;

for (const mode of ['pass', 'exit', 'hang', 'bad-props', 'stalled-props']) {
  test(`engine ${mode}: result and child cleanup`, { timeout: 15000 }, async t => {
    const root = fixture(t);
    const script = path.join(root, 'engine.mjs');
    const pidFile = path.join(root, 'pid');
    const requests = path.join(root, 'requests');
    const models = path.join(root, 'models');
    writeStubs(models);
    fs.writeFileSync(script, engineFixture);
    const run = smokeEngine(process.execPath, root, models, process.env, path.join(root, 'engine.log'),
      mode === 'hang' ? 1000 : 7000, [script, mode, pidFile, requests]);
    if (mode === 'pass') {
      await run;
      assert.deepEqual(fs.readFileSync(requests, 'utf8').trim().split('\n'), ['/health', '/health', '/props', '/plugins', '/props']);
    } else {
      await assert.rejects(run, mode === 'exit' ? /Engine exited/ : mode === 'hang' ? /Timed out/ : mode === 'bad-props' ? /HTTP 500/ : /abort|timeout/i);
    }
    assert.equal(alive(Number(fs.readFileSync(pidFile, 'utf8'))), false, 'engine process must be reaped');
    assert.ok(fs.existsSync(path.join(root, 'engine.log')), 'retain failure logs');
  });
}

test('failed command preserves exit code and output', { timeout: 15000 }, async t => {
  const root = fixture(t);
  const log = path.join(root, 'failure.log');
  await assert.rejects(runCommand(process.execPath, ['-e', 'console.error("bundle boot failed"); process.exit(13)'], {}, 3000, log), /exited 13/);
  assert.match(fs.readFileSync(log, 'utf8'), /bundle boot failed/);
});

test('Windows timeout kills the child and grandchild', { skip: process.platform !== 'win32', timeout: 15000 }, async t => {
  const root = fixture(t);
  const pidFile = path.join(root, 'pids');
  const script = `const fs = require('node:fs'); const { spawn } = require('node:child_process');
    const nested = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
    fs.writeFileSync(process.argv[1], JSON.stringify([process.pid, nested.pid])); setInterval(() => {}, 1000);`;
  await assert.rejects(runCommand(process.execPath, ['-e', script, pidFile], {}, 1500, path.join(root, 'tree.log')), /Timed out/);
  for (const pid of JSON.parse(fs.readFileSync(pidFile, 'utf8'))) assert.equal(alive(pid), false, `PID ${pid} survived timeout`);
});

test('archive rejects checkout extraction and missing packaged runtime', { skip: process.platform !== 'win32', timeout: 20000 }, async t => {
  const root = fixture(t);
  const source = path.join(root, 'source');
  fs.mkdirSync(source);
  fs.writeFileSync(path.join(source, 'README.txt'), 'incomplete release fixture');
  const zip = path.join(root, 'broken.zip');
  await runCommand('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
    '$ErrorActionPreference = "Stop"; Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::CreateFromDirectory($env:SMOKE_SOURCE, $env:SMOKE_ZIP)'],
  { env: { ...process.env, SMOKE_SOURCE: source, SMOKE_ZIP: zip } }, 10000, path.join(root, 'zip.log'));
  await assert.rejects(smokeArtifact(zip, { outputParent: root }), /outside the checkout/);
  // The gate chooses its own unique external directory; locate it from its log.
  let evidence;
  const original = console.log;
  console.log = value => { if (value.startsWith('smoke-artifact: evidence ')) evidence = value.slice('smoke-artifact: evidence '.length); };
  try {
    await assert.rejects(smokeArtifact(zip), /missing runtime\/node.exe/);
    const result = JSON.parse(fs.readFileSync(path.join(evidence, 'result.json'), 'utf8'));
    assert.equal(path.basename(result.root), 'HOT Step smoke');
    assert.ok(fs.existsSync(path.join(evidence, 'HOT Step smoke', 'README.txt')));
    assert.equal(result.status, 'fail');
  } finally {
    console.log = original;
    if (evidence) fs.rmSync(evidence, { recursive: true, force: true });
  }
});

// --backend-check reports as ace-server prints them, reduced to what the policy reads.
const cpuReg = { name: 'CPU', module: 'C:\e\ggml-cpu.dll', devices: [{ name: 'CPU', type: 'cpu' }] };
const report = (extra = {}) => ({
  registered: [cpuReg], selected: 'CPU', cpu: { initialised: true, description: 'Test CPU ', module: cpuReg.module },
  files: [{ file: 'ggml-cpu.dll', registered: true }], log: [], ...extra,
});
const cudaFile = extra => ({ files: [{ file: 'ggml-cpu.dll', registered: true }, { file: 'ggml-cuda.dll', registered: false, ...extra }] });
const missing = (dll, chain = dll, code = 126, exists = false) => ({ load_error: '126: The specified module could not be found.', load_error_code: 126,
  missing: { dll, chain, error: `${code}: loader error`, code, exists } });

const policy = [
  ['cpu passes', 'cpu', report(), null],
  ['cpu not initialised', 'cpu', report({ cpu: { initialised: false } }), /CPU backend did not initialise/],
  ['nothing selected', 'cpu', report({ selected: '' }), /no backend selected/],
  ['cuda registered with no device', 'cuda', report({ registered: [{ name: 'CUDA', devices: [] }, cpuReg], files: [{ file: 'ggml-cuda.dll', registered: true }],
    log: ['ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version'] }), null],
  ['cuda package DLL missing', 'cuda', report(), /package DLL ggml-cuda.dll is missing/],
  ['cuda runtime DLL missing', 'cuda', report(cudaFile(missing('cublasLt64_13.dll', 'cublas64_13.dll -> cublasLt64_13.dll'))), /missing cublas64_13.dll -> cublasLt64_13.dll/],
  ['cuda driver absent', 'cuda', report(cudaFile(missing('nvcuda.dll'))), null],
  ['bare error 126', 'cuda', report(cudaFile({ load_error: '126: The specified module could not be found.' })), /no missing import found/],
  ['ABI: missing export', 'cuda', report(cudaFile({ load_error: '127: The specified procedure could not be found.' })), /127/],
  ['loads but never registers', 'cuda', report(cudaFile({ init_ok: false })), /loaded but did not register/],
  ['vulkan without a driver', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, init_ok: false, vk_create_instance: -9 }] }), null],
  ['vulkan loader absent', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, ...missing('vulkan-1.dll') }] }), null],
  ['present vulkan-1.dll missing an export', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, ...missing('vulkan-1.dll', 'vulkan-1.dll', 127, true) }] }), /vulkan-1.dll \(127/],
  ['present vulkan-1.dll is a bad image', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, ...missing('vulkan-1.dll', 'vulkan-1.dll', 193, true) }] }), /vulkan-1.dll \(193/],
  ['present vulkan-1.dll init fails', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, ...missing('vulkan-1.dll', 'vulkan-1.dll', 1114, true) }] }), /vulkan-1.dll \(1114/],
  ['present nvcuda.dll with 126 below it', 'cuda', report(cudaFile(missing('nvcuda.dll', 'nvcuda.dll', 126, true))), /nvcuda.dll \(126/],
  ['report without absence proof', 'cuda', report(cudaFile({ load_error: '126', missing: { dll: 'nvcuda.dll', chain: 'nvcuda.dll', error: '126' } })), /failed to load/],
  ['vulkan init fails with a driver', 'vulkan', report({ files: [{ file: 'ggml-vulkan.dll', registered: false, init_ok: false, vk_create_instance: 0 }] }), /did not register/],
  ['driver DLL excuses only the expected GPU', 'cpu', report({ files: [{ file: 'ggml-cuda.dll', registered: false, ...missing('nvcuda.dll') }] }), /ggml-cuda.dll failed to load/],
  ['CPU variant DLL fails to load', 'vulkan', report({ registered: [{ name: 'Vulkan', devices: [] }, cpuReg],
    files: [{ file: 'ggml-vulkan.dll', registered: true }, { file: 'ggml-cpu-haswell.dll', registered: false, ...missing('VCOMP140.DLL') }] }), /ggml-cpu-haswell.dll failed to load/],
];
for (const [name, variant, input, failure] of policy) {
  test(`backend policy: ${name}`, () => {
    const [, failures] = assertBackends(input, variant);
    if (failure) assert.match(failures.join('; '), failure);
    else assert.deepEqual(failures, []);
  });
}

test('CUDA runtime comes from the registry pack and keys on its entries', () => {
  const registry = JSON.parse(fs.readFileSync(path.join(here, '../server/src/data/model-registry.json'), 'utf8'));
  assert.deepEqual(cudaRuntimeFiles(registry, 13).map(f => f.filename).sort(), ['cublas64_13.dll', 'cublasLt64_13.dll', 'cudart64_13.dll']);
  assert.deepEqual(cudaRuntimeFiles(registry, 12).map(f => f.filename).sort(), ['cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll']);
  const key = cudaRuntimeKey(registry, 13);
  const edited = structuredClone(registry);
  edited.files.find(f => f.id === 'cuda-rt-cudart').sizeBytes += 1;
  assert.notEqual(cudaRuntimeKey(edited, 13), key);
  edited.files.push({ id: 'unrelated', filename: 'x.gguf' });
  assert.equal(cudaRuntimeKey(edited, 12), cudaRuntimeKey(registry, 12));
});

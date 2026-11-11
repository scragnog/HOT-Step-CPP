// Source checks for every release assembler, including local packaging.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BUILD_INFO_FILE, buildInfoErrors } from './build-info.mjs';
import { payloadIdFor } from './updater/launcher.mjs';
import { MANIFEST_FILE, ROOT_ENTRIES } from './updater/package-layout.mjs';

export function checkReleaseLayout(root) {
  const errors = [];
  const read = name => fs.readFileSync(path.join(root, name), 'utf8');
  const wf = read('.github/workflows/release.yml');
  const assemblers = [
    ...['windows', 'linux', 'macos'].map(platform => {
      const match = wf.match(new RegExp(`^  build-${platform}:[\\s\\S]*?(?=^  [a-z][\\w-]*:|$(?![\\s\\S]))`, 'm'));
      return [`release.yml ${platform}`, match?.[0] ?? ''];
    }),
    ['release/build-release.ps1', read('release/build-release.ps1')],
    ['package-release.sh', read('package-release.sh')],
  ];
  for (const [name, src] of assemblers) {
    if (/windows|ps1/.test(name) && !/updater[\\/]package-layout\.mjs/.test(src)) errors.push(`${name}: does not run release/updater/package-layout.mjs before zipping`);
    if (!src.includes('release/.env.example') && !src.includes('$ReleaseDir ".env.example"')) errors.push(`${name}: missing release env template`);
    for (const file of ['profile-path.mjs', 'README.txt', 'HOT-Step.']) {
      if (!src.includes(file)) errors.push(`${name}: missing ${file}`);
    }
    if (!/release[\\/]build-info\.mjs|"build-info\.mjs"/.test(src) || !src.includes('--package')) errors.push(`${name}: does not write and check build-info.json`);
    if (!/server[\\/]src[\\/]data[\\/](?:\*|\.)/.test(src)) errors.push(`${name}: resources must copy wholesale into server/data`);
    if (!/(?:\$dataOut|SRVOUT\/data|STAGING_DIR\}\/server\/data)/.test(src)) errors.push(`${name}: missing app-owned server/data target`);
    if (/(?:mkdir[^\n]*|New-Item[^\n]*)(?:\$(?:DIST|dist)|\$\{STAGING_DIR\})\/(?:models|adapters)/.test(src)) errors.push(`${name}: creates mutable folders inside app`);
    const binaryList = src.match(/\$engineFiles\s*=\s*@\(([\s\S]*?)\)/)?.[1]
      ?? [...src.matchAll(/for bin in ([^;]+); do/g)].map(m => m[1]).find(s => s.includes('neural-codec') && !s.includes('engine/build/')) ?? '';
    for (const binary of ['ace-server', 'ace-train', 'ace-midi', 'ace-caption', 'mastering', 'mp3-codec', 'neural-codec', 'vst-host', 'quantize']) {
      if (!new RegExp(`(?:^|["\\s])${binary}(?:\\.exe)?(?:$|["\\s,])`).test(binaryList)) errors.push(`${name}: missing engine binary ${binary}`);
    }
  }
  const template = read('release/.env.example');
  for (const line of template.split(/\r?\n/)) {
    if (/^(DATA_DIR|ACESTEPCPP_MODELS|ACESTEPCPP_ADAPTERS|TRAINING_DIR|LYRICS_EXPORT_DIR|MUSCRIPTOR_MODELS_DIR|WHISPER_EXE|WHISPER_MODELS_DIR|TENSORRT_LIBS)=.+/.test(line)) errors.push(`release template pins a managed path: ${line.split('=')[0]}`);
  }
  for (const launcher of ['release/HOT-Step.bat', 'release/HOT-Step.sh']) {
    const src = read(launcher);
    if (!src.includes('profile-path.mjs') || !src.includes('RESTART_MARKER')) errors.push(`${launcher}: restart must use the profile helper`);
    if (src.includes('.restart-requested') || /mkdir[^\n]*models/.test(src)) errors.push(`${launcher}: writes state inside app`);
    if (/powershell|pwsh/i.test(src)) errors.push(`${launcher}: spawns PowerShell, which behaviour-based antivirus quarantines`);
    if (/HOT_STEP_DATA_ROOT|HOT_STEP_PROFILE_DIR/.test(src)) errors.push(`${launcher}: pins the data root or profile, so two unpacked releases would stop sharing one library`);
  }
  const rootBat = read('release/updater/HOT-Step.bat');
  if (!rootBat.includes('runtime\\node.exe') || !rootBat.includes('launcher.mjs')) errors.push('release/updater/HOT-Step.bat: must run launcher.mjs on the root runtime');
  if (/powershell|pwsh/i.test(rootBat)) errors.push('release/updater/HOT-Step.bat: spawns PowerShell, which behaviour-based antivirus quarantines');
  return errors;
}

const BACKEND_LIB = /ggml-(cuda|vulkan|hip)\.(?:dll|so|dylib)$/;

/** Checks an assembled package root against the asset it is meant to be: the
 *  flat layout (protocol 0) or the versioned one package-layout.mjs makes. */
export function checkPackageBuildInfo(root, opts) {
  if (!fs.existsSync(path.join(root, 'current.json'))) return checkPayload(root, opts, 0);
  const errors = [];
  let id = null;
  try { id = JSON.parse(fs.readFileSync(path.join(root, 'current.json'), 'utf8')).payload; } catch { errors.push('current.json: unreadable'); }
  for (const e of fs.readdirSync(root)) if (!ROOT_ENTRIES.includes(e)) errors.push(`root: unexpected ${e}`);
  for (const f of ['HOT-Step.bat', 'launcher.mjs', 'runtime/node.exe', 'README.txt']) if (!fs.existsSync(path.join(root, f))) errors.push(`root: missing ${f}`);
  const apps = fs.existsSync(path.join(root, 'app')) ? fs.readdirSync(path.join(root, 'app')) : [];
  if (apps.length !== 1 || apps[0] !== id) errors.push(`app/: must hold exactly the payload current.json names (${id})`);
  if (errors.length) return errors;
  const payload = path.join(root, 'app', id);
  errors.push(...checkPayload(payload, opts, 1));
  if (errors.length) return errors;
  const info = JSON.parse(fs.readFileSync(path.join(payload, BUILD_INFO_FILE), 'utf8'));
  if (payloadIdFor(info) !== id) errors.push(`app/${id}: name does not match build-info.json (${payloadIdFor(info)})`);
  if (fs.existsSync(path.join(payload, 'HOT-Step.bat'))) errors.push(`app/${id}: still has the flat HOT-Step.bat`);
  for (const f of ['server/updater-tool.mjs', 'runtime/node.exe', MANIFEST_FILE]) if (!fs.existsSync(path.join(payload, f))) errors.push(`app/${id}: missing ${f}`);
  return errors;
}

function checkPayload(root, { os, variant }, minProtocol) {
  let info;
  try { info = JSON.parse(fs.readFileSync(path.join(root, BUILD_INFO_FILE), 'utf8')); }
  catch (e) { return [`${BUILD_INFO_FILE}: ${e.code === 'ENOENT' ? 'missing' : 'unreadable'}`]; }
  const errors = buildInfoErrors(info).map(e => `${BUILD_INFO_FILE}: ${e}`);
  if (info?.os !== os || info?.variant !== variant) errors.push(`${BUILD_INFO_FILE}: is ${info?.os} ${info?.variant}, expected ${os} ${variant}`);
  if (!errors.length && minProtocol && info.updaterProtocol < minProtocol) errors.push(`${BUILD_INFO_FILE}: the versioned layout needs updaterProtocol ${minProtocol} or higher`);
  if (errors.length) return errors;
  let version = '';
  try { version = fs.readFileSync(path.join(root, 'VERSION'), 'utf8').trim().replace(/^v/, ''); } catch { errors.push('VERSION: missing'); }
  if (info.tag && version !== info.tag.slice(1)) errors.push(`${BUILD_INFO_FILE}: tag ${info.tag} disagrees with VERSION ${version}`);
  const engine = path.join(root, 'engine');
  const libs = fs.existsSync(engine) ? fs.readdirSync(engine).map(f => f.match(BACKEND_LIB)?.[1]).filter(Boolean) : [];
  const backend = variant.startsWith('cuda') ? 'cuda' : variant === 'rocm' ? 'hip' : variant === 'vulkan' ? 'vulkan' : null;
  if (backend && !libs.includes(backend)) errors.push(`engine: ${variant} package has no ggml-${backend} library`);
  for (const lib of libs) if (lib !== backend) errors.push(`engine: ${variant} package ships a ggml-${lib} library`);
  // The coarse markers are optional (local packagers skip them) but must agree when present.
  const marker = f => { try { return fs.readFileSync(path.join(engine, f), 'utf8').trim(); } catch { return null; } };
  const coarse = backend === 'cuda' ? 'cuda' : variant;
  if (![null, coarse].includes(marker('.variant'))) errors.push(`engine/.variant: expected ${coarse}, found ${marker('.variant')}`);
  const major = backend === 'cuda' ? variant.match(/^cuda(\d+)/)[1] : null;
  if (![null, major].includes(marker('.cuda-version'))) errors.push(`engine/.cuda-version: expected ${major ?? 'none'}, found ${marker('.cuda-version')}`);
  return errors;
}

// node release/check-layout.mjs --package <root> --os <os> --variant <variant>
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const opts = {};
  for (let i = 2; i < process.argv.length; i += 2) opts[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
  const errors = opts.package ? checkPackageBuildInfo(opts.package, opts)
    : checkReleaseLayout(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
  for (const e of errors) console.error(`check-layout: ${e}`);
  if (errors.length) process.exit(1);
  console.log(`check-layout: ${opts.package ? `${opts.package} ${opts.variant}` : 'assemblers'} OK`);
}

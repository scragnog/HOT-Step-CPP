// Writes build-info.json, the exact identity of a portable package, beside
// VERSION in the package root. Every release assembler calls this one script.
//
//   node release/build-info.mjs <packageRoot> --os windows|linux|macos --arch x64|arm64 --variant <asset variant> [--tag v1.2.3] [--protocol 1]
//
// --protocol overrides UPDATER_PROTOCOL for one build (a protocol-1 fixture for
// updater tests); release/updater/package-layout.mjs then builds the versioned layout.
//
// The tag is kept only when it names the commit being packaged; anything else
// (a preflight version, a local build off a tag) packages as channel "master"
// with tag null.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const BUILD_INFO_FILE = 'build-info.json';
export const BUILD_INFO_FORMAT = 1;
/** What release builds ship. Stays 0 until the updater's acceptance run passes. */
export const UPDATER_PROTOCOL = 0;
/** Highest protocol release/updater/launcher.mjs speaks (LAUNCHER_PROTOCOL). */
export const MAX_UPDATER_PROTOCOL = 1;
/** Exact asset variants per os/arch, matching the release.yml matrices. */
export const VARIANTS = {
  'windows/x64': ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'cpu'],
  'linux/x64': ['cuda13.1', 'cuda12.8', 'cuda12-volta', 'vulkan', 'rocm', 'cpu'],
  'macos/arm64': ['metal'],
};

/** Contract violations in a parsed build-info.json; mirrored by server/src/config.ts parseBuildInfo. */
export function buildInfoErrors(info) {
  if (!info || typeof info !== 'object') return ['not an object'];
  const bad = [];
  const want = (field, ok) => { if (!ok) bad.push(`bad ${field} ${JSON.stringify(info[field])}`); };
  want('format', info.format === BUILD_INFO_FORMAT);
  want('tag', info.tag === null || (typeof info.tag === 'string' && /^v\S+$/.test(info.tag)));
  want('sha', typeof info.sha === 'string' && /^[0-9a-f]{40}$/.test(info.sha));
  want('engineSha', typeof info.engineSha === 'string' && /^[0-9a-f]{40}$/.test(info.engineSha));
  want('channel', info.channel === (info.tag ? 'release' : 'master'));
  want('variant', VARIANTS[`${info.os}/${info.arch}`]?.includes(info.variant) === true);
  want('updaterProtocol', Number.isInteger(info.updaterProtocol) && info.updaterProtocol >= 0 && info.updaterProtocol <= MAX_UPDATER_PROTOCOL);
  return bad;
}

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const git = (...args) => execFileSync('git', args, { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();

export function buildInfo({ os, arch, variant, tag, protocol }) {
  if (!VARIANTS[`${os}/${arch}`]?.includes(variant)) throw new Error(`build-info: no ${variant} asset for ${os}/${arch}`);
  const updaterProtocol = protocol === undefined ? (os === 'windows' ? UPDATER_PROTOCOL : 0) : Number(protocol);
  if (!Number.isInteger(updaterProtocol) || updaterProtocol < 0 || updaterProtocol > MAX_UPDATER_PROTOCOL) throw new Error(`build-info: unsupported updater protocol ${protocol}`);
  const sha = git('rev-parse', 'HEAD');
  let tagSha = '';
  if (tag) { try { tagSha = git('rev-parse', '--verify', '--quiet', `refs/tags/${tag}^{commit}`); } catch { /* no such tag */ } }
  const exactTag = tag && tagSha === sha ? tag : null;
  if (tag && !exactTag) console.warn(`build-info: ${tag} does not name HEAD; packaging as master`);
  if (git('status', '--porcelain', '--untracked-files=no')) console.warn('build-info: tracked files differ from HEAD; sha names HEAD only');
  return {
    format: BUILD_INFO_FORMAT,
    tag: exactTag,
    sha,
    channel: exactTag ? 'release' : 'master',
    os,
    arch,
    variant,
    // Tree id of engine/ at HEAD: changes with any engine source or ggml pin change.
    engineSha: git('rev-parse', 'HEAD:engine'),
    updaterProtocol,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [root, ...rest] = process.argv.slice(2);
  const opts = {};
  for (let i = 0; i < rest.length; i += 2) opts[rest[i].replace(/^--/, '')] = rest[i + 1];
  if (!root || !fs.existsSync(root)) throw new Error(`build-info: package root not found: ${root}`);
  const info = buildInfo(opts);
  fs.writeFileSync(path.join(root, BUILD_INFO_FILE), JSON.stringify(info, null, 2) + '\n');
  console.log(`build-info: ${info.os}/${info.arch} ${info.variant} ${info.tag ?? 'untagged'} ${info.sha.slice(0, 12)} protocol ${info.updaterProtocol}`);
}

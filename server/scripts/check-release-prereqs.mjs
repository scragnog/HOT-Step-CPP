#!/usr/bin/env node
//
// check-release-prereqs.mjs — does a packaged build have everything a user needs?
//
// A feature can pass every local test because THIS machine happens to hold a
// file that was never part of the distribution. Nothing else catches that: the
// paths are resolved at runtime, so tsc is happy, the build is green, and the
// feature is dead for everyone who downloads it. It has happened twice —
// MM3 training gated on two GGUFs that only existed on the dev box (#137), and
// the MM3 caption corpus, which is committed but was not copied into the
// release archives (#139).
//
// This script checks the two ways that goes wrong:
//
//   1. WEIGHTS — every file in model-registry.json must actually exist in the
//      Hugging Face repo it names, at the size the registry claims.
//   2. DATA — every file the server reads from its data directory at runtime
//      must be copied into the package by .github/workflows/release.yml.
//
// Run before pushing a release tag:
//     node server/scripts/check-release-prereqs.mjs
//
// Exit 0 = everything a user needs is reachable. Exit 1 = do not ship.
//
// Network: one HF API call per distinct repo, unauthenticated. Pass
// --offline to skip part 1 (part 2 needs no network).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const DATA_DIR = path.join(REPO_ROOT, 'server', 'src', 'data');
const REGISTRY = path.join(DATA_DIR, 'model-registry.json');
const WORKFLOW = path.join(REPO_ROOT, '.github', 'workflows', 'release.yml');

const OFFLINE = process.argv.includes('--offline');

const problems = [];
const notes = [];

// ── 1. Registry files exist on Hugging Face ─────────────────────────────────

/** Every (repo, path, expected size) the registry promises a user can download,
 *  including the companion files (LICENSE and friends) that ride along. */
function registryDownloads() {
  const reg = JSON.parse(fs.readFileSync(REGISTRY, 'utf-8'));
  const out = [];
  for (const f of reg.files) {
    out.push({ id: f.id, repo: f.repo, repoPath: f.repoPath || f.filename, size: f.sizeBytes });
    for (const c of f.companions || []) {
      out.push({ id: `${f.id} (companion)`, repo: f.repo, repoPath: c.repoPath || c.filename, size: null });
    }
  }
  return out;
}

/** Every file entry carries the fields the Model Manager reads without a
 *  guard. `mm3-rec7-f16` shipped without a `tags` array (a62054f9) and
 *  ModelRow's `file.tags.includes(...)` blanked the whole MiniMax-Music3 tab
 *  (#137). The UI is guarded now too, but the catalogue is the contract. */
function checkFileSchema() {
  const reg = JSON.parse(fs.readFileSync(REGISTRY, 'utf-8'));
  const seen = new Set();
  for (const f of reg.files) {
    const where = `file "${f.id ?? '?'}"`;
    if (typeof f.id !== 'string' || !f.id) problems.push('a file entry has no id');
    else if (seen.has(f.id)) problems.push(`${where}: duplicate id`);
    seen.add(f.id);
    if (typeof f.filename !== 'string' || !f.filename) problems.push(`${where}: no filename`);
    if (typeof f.repo !== 'string' || !f.repo) problems.push(`${where}: no repo`);
    if (typeof f.sizeBytes !== 'number' || !(f.sizeBytes > 0)) problems.push(`${where}: sizeBytes missing or not positive`);
    if (!Array.isArray(f.tags)) problems.push(`${where}: no tags array (ModelRow reads it)`);
    if (f.companions !== undefined && !Array.isArray(f.companions)) problems.push(`${where}: companions is not an array`);
  }
}

/** Pack ids must resolve to real file entries, or the pack renders as a card
 *  that can never reach "installed" — how #120 started. */
function checkPackIds() {
  const reg = JSON.parse(fs.readFileSync(REGISTRY, 'utf-8'));
  const ids = new Set(reg.files.map(f => f.id));
  for (const pack of reg.packs) {
    if (!Array.isArray(pack.fileIds)) {
      problems.push(`pack "${pack.id}" has no fileIds array`);
      continue;
    }
    for (const id of pack.fileIds) {
      if (!ids.has(id)) problems.push(`pack "${pack.id}" lists unknown file id "${id}"`);
    }
  }
}

async function repoBlobs(repo) {
  const url = `https://huggingface.co/api/models/${repo}?blobs=true`;
  const res = await fetch(url);
  if (!res.ok) return { error: `${res.status} ${res.statusText}` };
  const json = await res.json();
  const map = new Map();
  for (const s of json.siblings || []) map.set(s.rfilename, s.size ?? null);
  return { map, gated: json.gated, private: json.private };
}

async function checkHuggingFace() {
  const downloads = registryDownloads();
  const byRepo = new Map();
  for (const d of downloads) {
    if (!byRepo.has(d.repo)) byRepo.set(d.repo, []);
    byRepo.get(d.repo).push(d);
  }

  for (const [repo, wanted] of byRepo) {
    const info = await repoBlobs(repo);
    if (info.error) {
      problems.push(`repo ${repo} is not reachable (${info.error}) — ${wanted.length} catalogue entries point at it`);
      continue;
    }
    if (info.private) problems.push(`repo ${repo} is PRIVATE — nobody but you can download from it`);
    if (info.gated) notes.push(`repo ${repo} is gated; users need an HF token`);

    for (const d of wanted) {
      if (!info.map.has(d.repoPath)) {
        problems.push(`${d.id}: ${repo}/${d.repoPath} does not exist on Hugging Face`);
        continue;
      }
      const actual = info.map.get(d.repoPath);
      if (d.size != null && actual != null && actual !== d.size) {
        problems.push(`${d.id}: size mismatch — registry says ${d.size}, HF has ${actual}`);
      }
    }
    console.log(`  ${repo}: ${wanted.length} entries checked`);
  }
}

// ── 2. Runtime data files are packaged ──────────────────────────────────────
//
// Portable mode (a release archive) reads these from server/data/. They are
// committed under server/src/data/, and get there only because release.yml
// copies them one by one — so a new data file is invisible to users until
// someone remembers to add a copy line, in all three places it appears
// (Windows, Linux, macOS).

function checkPackagedData() {
  if (!fs.existsSync(WORKFLOW)) {
    notes.push('release.yml not found — skipped the packaged-data check');
    return;
  }
  const wf = fs.readFileSync(WORKFLOW, 'utf-8');

  // The workflow should copy the directory wholesale, once per platform, so a
  // new data file needs no workflow edit at all. Count those first; only fall
  // back to per-file matching if someone has gone back to naming files.
  const wholeDir = (wf.match(/server\/src\/data\/(\*|\.)(?![\w.-])/g) || []).length;
  if (wholeDir >= 3) {
    console.log(`  server/src/data/ copied wholesale (${wholeDir} platforms) — ${fs.readdirSync(DATA_DIR).length} files covered`);
    return;
  }
  if (wholeDir > 0) {
    problems.push(`release.yml copies server/src/data wholesale in only ${wholeDir} of 3 platform jobs`);
  }
  for (const name of fs.readdirSync(DATA_DIR)) {
    const copies = wf.split(`server/src/data/${name}`).length - 1;
    if (copies === 0) {
      problems.push(`server/src/data/${name} is never copied into a release package — the feature that reads it is dead in every download`);
    } else if (copies + wholeDir < 3) {
      problems.push(`server/src/data/${name} is copied ${copies}x in release.yml; expected 3 (Windows, Linux, macOS)`);
    } else {
      console.log(`  ${name}: packaged (${copies} copy steps)`);
    }
  }
}

// ── 3. Engine binaries the server spawns are packaged ───────────────────────
//
// Third instance of the same fault: v1.2.1 shipped without ace-train and every
// Training Studio job 503'd; every release through v1.3 shipped without
// ace-caption, so the MOSS captioner could never appear no matter how many GB
// of MOSS weights a user downloaded. The engine builds sixteen executables and
// the release packages nine of them, by hand, in three separate places.
//
// The rule enforced here: if server/src names a CMake executable target, that
// binary must be in the package. Dev-only tools are listed below so a mention
// in a comment cannot fail the check — anything else new defaults to "must
// ship", which is the safe direction.

const CMAKE = path.join(REPO_ROOT, 'engine', 'CMakeLists.txt');

/** Tools that exist for development and are deliberately not distributed. */
const DEV_ONLY = new Set([
  'sa3-ggml-test', 'moss-ggml-test', 'bs-roformer-test', 'mdx23c-test',
  'ace-synth', 'ace-lm', 'ace-understand',   // CLI equivalents of what ace-server does in-process
]);

function checkEngineBinaries() {
  if (!fs.existsSync(CMAKE) || !fs.existsSync(WORKFLOW)) {
    notes.push('CMakeLists.txt or release.yml not found — skipped the binary check');
    return;
  }
  const cmake = fs.readFileSync(CMAKE, 'utf-8');
  const wf = fs.readFileSync(WORKFLOW, 'utf-8');
  const targets = [...cmake.matchAll(/^\s*add_executable\(([A-Za-z0-9_-]+)/gm)].map(m => m[1]);

  // Server sources, flattened once — the reference test is a word match.
  const srcDir = path.join(REPO_ROOT, 'server', 'src');
  let src = '';
  (function walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith('.ts')) src += fs.readFileSync(p, 'utf-8');
    }
  })(srcDir);

  for (const t of targets) {
    if (DEV_ONLY.has(t)) continue;
    const referenced = new RegExp(`\\b${t.replace(/[-]/g, '\\-')}\\b`).test(src);
    if (!referenced) {
      notes.push(`engine target "${t}" is not referenced by the server — not checked`);
      continue;
    }
    // Packaged if it appears in release.yml outside of prose. Both the
    // PowerShell array and the shell for-loop name it bare or with .exe.
    const packaged = new RegExp(`["\\s]${t}(\\.exe)?["\\s,]`).test(wf);
    if (!packaged) {
      problems.push(`engine binary "${t}" is spawned by the server but never packaged by release.yml — the feature that spawns it is dead in every download`);
    } else {
      console.log(`  ${t}: packaged`);
    }
  }
}

// ── Run ─────────────────────────────────────────────────────────────────────

console.log('Registry pack ids');
checkFileSchema();
checkPackIds();

console.log('\nEngine binaries');
checkEngineBinaries();

console.log('\nPackaged runtime data');
checkPackagedData();

if (OFFLINE) {
  notes.push('--offline: did not check Hugging Face');
} else {
  console.log('\nHugging Face availability');
  await checkHuggingFace();
}

console.log('');
for (const n of notes) console.log(`note: ${n}`);
if (problems.length === 0) {
  console.log('OK — everything the app resolves at runtime is reachable by a user.');
  process.exit(0);
}
console.log(`\n${problems.length} problem(s) — DO NOT SHIP:`);
for (const p of problems) console.log(`  - ${p}`);
process.exit(1);

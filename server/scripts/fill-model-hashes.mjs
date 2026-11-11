#!/usr/bin/env node
// Read Hugging Face LFS metadata only; never download model contents.
// Preview: node server/scripts/fill-model-hashes.mjs
// Apply:   node server/scripts/fill-model-hashes.mjs --write
// HF_TOKEN optionally supplies access to gated repositories.
import fs from 'node:fs';

const registryPath = new URL('../src/data/model-registry.json', import.meta.url);
const original = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
const metadata = new Map();
const errors = new Map();
for (const repo of new Set(original.files.map(f => f.repo).filter(Boolean))) {
  try {
    let response;
    for (let attempt = 0; attempt < 3; attempt++) {
      response = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`, {
        headers: process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {},
        signal: AbortSignal.timeout(30_000),
      });
      if (response.status !== 429 && response.status < 500) break;
      await response.body?.cancel();
      if (attempt < 2) await new Promise(resolve => setTimeout(resolve, (attempt + 1) * 2000));
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.siblings)) throw new Error('API response has no file inventory');
    metadata.set(repo, new Map(data.siblings.map(f => [f.rfilename, f.lfs?.sha256])));
    console.log(`Read ${repo}`);
  } catch (error) {
    errors.set(repo, error.message);
  }
}

// Re-read immediately before writing so concurrent registry additions and
// existing hashes survive. A new repo will be reported for the next run.
const text = fs.readFileSync(registryPath, 'utf8');
const registry = JSON.parse(text);
let added = 0;
let conflicts = 0;
const missing = [];
for (const file of registry.files) {
  const repoPath = file.repoPath || file.filename;
  const inventory = metadata.get(file.repo);
  const hash = inventory?.get(repoPath);
  if (typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)) {
    if (!file.sha256) { file.sha256 = hash; added++; }
    else if (file.sha256 !== hash) {
      console.error(`CONFLICT ${file.id}: retained ${file.sha256}; HF ${hash}`);
      conflicts++;
    }
  }
  if (!file.sha256) {
    const reason = !file.repo ? 'no repo'
      : errors.has(file.repo) ? `repository unavailable: ${errors.get(file.repo)}`
      : !inventory ? 'repo added during this run; run again'
      : !inventory.has(repoPath) ? `path absent from HF: ${repoPath}`
      : 'not an LFS file; HF provides no content sha256';
    missing.push(`${file.id}: ${reason}`);
  }
}
if (process.argv.includes('--write') && added) {
  const newline = text.includes('\r\n') ? '\r\n' : '\n';
  fs.writeFileSync(registryPath, (JSON.stringify(registry, null, 2) + '\n').replaceAll('\n', newline));
}
console.log(`${process.argv.includes('--write') ? 'Added' : 'Would add'} ${added} hashes; ${conflicts} conflicts; ${missing.length} without a hash.`);
for (const line of missing) console.log(`MISSING ${line}`);
if (conflicts || errors.size) process.exitCode = 1;

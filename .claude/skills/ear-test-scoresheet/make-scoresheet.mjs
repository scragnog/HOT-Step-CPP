#!/usr/bin/env node
// Writes the ear-test score sheet (index.html) for a study.
//
//   node .claude/skills/ear-test-scoresheet/make-scoresheet.mjs <study.json>
//
// study.json sits in the study's folder; index.html is written next to it.
// Track `file` paths are relative to that folder (e.g. "round1/rbf/03-nar400.wav").
// Re-run after adding tracks or rounds: scores live in the browser and in
// scores.json, keyed by track id, so they survive a regenerated page.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const studyPath = path.resolve(process.argv[2] ?? '');
if (!process.argv[2] || !fs.existsSync(studyPath)) {
  console.error('usage: node make-scoresheet.mjs <study.json>');
  process.exit(1);
}
const study = JSON.parse(fs.readFileSync(studyPath, 'utf8'));
for (const k of ['id', 'title', 'criteria', 'series', 'tracks']) {
  if (!study[k]) throw new Error(`study.json is missing "${k}"`);
}
const ids = new Set();
for (const t of study.tracks) {
  if (!t.id || ids.has(t.id)) throw new Error(`track id missing or repeated: ${t.id}`);
  ids.add(t.id);
  // A relative URL the browser resolves against index.html's own folder.
  t.url = t.file.split(/[\\/]/).map(encodeURIComponent).join('/');
}
const here = path.dirname(fileURLToPath(import.meta.url));
const template = fs.readFileSync(path.join(here, 'template.html'), 'utf8');
// "</" inside the inlined JSON would end the <script> block early.
const json = JSON.stringify(study).replace(/<\//g, '<\\/');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const html = template.replace('__TITLE__', esc(study.title)).replace('__STUDY__', () => json);
const out = path.join(path.dirname(studyPath), 'index.html');
fs.writeFileSync(out, html);
const missing = study.tracks.filter(t => !fs.existsSync(path.join(path.dirname(studyPath), t.file))).length;
console.log(`wrote ${out}: ${study.tracks.length} tracks, ${missing} not rendered yet`);

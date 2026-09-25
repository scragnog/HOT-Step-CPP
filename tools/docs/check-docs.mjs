#!/usr/bin/env node
// check-docs.mjs — fail when the docs have drifted from the code.
//
//   node tools/docs/check-docs.mjs
//
// Checks:
//   1. generated tables are current (delegates to build-docs.mjs --check)
//   2. every UI studio folder has a user page (STUDIO_PAGES below)
//   3. every skill folder is listed in .claude/skills/README.md
//   4. every relative link and image in a tracked .md resolves
//   5. FEATURES.md links to every studio page
// Exit 1 on any failure. Node builtins only.

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const rel = (p) => path.relative(ROOT, p).replaceAll('\\', '/');
const exists = (p) => fs.existsSync(path.join(ROOT, p));
const fail = [];
const warn = [];

// UI component folder -> the user page that documents it. Shared/plumbing folders map to the
// page that covers them. Add a row when you add a studio; the check fails until you do.
const STUDIO_PAGES = {
  'insta-gen': 'docs/user/studios/insta-gen.md',
  create: 'docs/user/studios/create.md',
  library: 'docs/user/studios/library.md',
  player: 'docs/user/studios/library.md',
  playlist: 'docs/user/studios/library.md',
  details: 'docs/user/studios/library.md',
  'lyric-studio': 'docs/user/studios/lyric-studio.md',
  'cover-studio': 'docs/user/studios/cover-studio.md',
  'repaint-studio': 'docs/user/studios/repaint-studio.md',
  'stem-studio': 'docs/user/studios/stem-studio.md',
  'stem-builder': 'docs/user/studios/stem-builder.md',
  'song-builder': 'docs/user/studios/song-builder.md',
  storm: 'docs/user/studios/storm.md',
  'midi-studio': 'docs/user/studios/midi-studio.md',
  'training-studio': 'docs/user/studios/training-studio.md',
  settings: 'docs/user/studios/settings.md',
  terminal: 'docs/user/studios/settings.md',
  'model-manager': 'docs/user/studios/model-manager.md',
  assistant: 'docs/user/studios/assistant.md',
  'global-bar': 'docs/user/generation.md',
  sidebar: null, // navigation only
  shared: null,
};

// 1. generated tables
const gen = spawnSync(process.execPath, [path.join(ROOT, 'tools/docs/build-docs.mjs'), '--check'], { encoding: 'utf8' });
if (gen.status !== 0) fail.push((gen.stderr || gen.stdout).trim());

// 2. studio pages
for (const dir of fs.readdirSync(path.join(ROOT, 'ui/src/components'), { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name)) {
  if (!(dir in STUDIO_PAGES)) fail.push(`ui/src/components/${dir} has no entry in STUDIO_PAGES (tools/docs/check-docs.mjs): write its docs/user page and add the row`);
  else if (STUDIO_PAGES[dir] && !exists(STUDIO_PAGES[dir])) fail.push(`ui/src/components/${dir} maps to ${STUDIO_PAGES[dir]}, which does not exist`);
}

// 3. skills index
const skillsReadme = fs.readFileSync(path.join(ROOT, '.claude/skills/README.md'), 'utf8');
for (const s of fs.readdirSync(path.join(ROOT, '.claude/skills'), { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name)) {
  if (!skillsReadme.includes(`${s}/SKILL.md`)) fail.push(`.claude/skills/${s} is not listed in .claude/skills/README.md`);
}

// 4. links
const tracked = execFileSync('git', ['ls-files', '*.md', '**/*.md'], { cwd: ROOT, encoding: 'utf8' }).split(/\r?\n/).filter((f) => f && !f.startsWith('node_modules/'));
const untrackedNew = execFileSync('git', ['ls-files', '--others', '--exclude-standard', '*.md', '**/*.md'], { cwd: ROOT, encoding: 'utf8' }).split(/\r?\n/).filter(Boolean);
const linkRe = /!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
for (const file of [...tracked, ...untrackedNew]) {
  const raw = fs.readFileSync(path.join(ROOT, file), 'utf8');
  const text = raw.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '');
  for (const m of text.matchAll(linkRe)) {
    let target = m[1];
    if (/^(https?:|mailto:|#|data:)/.test(target)) continue;
    target = decodeURIComponent(target.split('#')[0]);
    if (!target) continue;
    const resolved = path.resolve(path.dirname(path.join(ROOT, file)), target);
    if (fs.existsSync(resolved)) continue;
    // A missing screenshot that the page already flags with <!-- screenshot: needed --> is a
    // known gap, reported but not fatal, so CI does not block on images only a human can take.
    if (m[0].startsWith('!') && raw.includes('screenshot: needed')) warn.push(`${file}: screenshot missing ${m[1]}`);
    else fail.push(`${file}: broken link ${m[1]}`);
  }
}

// 5. FEATURES.md reaches every studio page
if (exists('FEATURES.md')) {
  const features = fs.readFileSync(path.join(ROOT, 'FEATURES.md'), 'utf8');
  for (const page of new Set(Object.values(STUDIO_PAGES).filter(Boolean))) {
    if (!features.includes(page.replace(/^docs\//, ''))) fail.push(`FEATURES.md does not link to ${page}`);
  }
}

if (warn.length) console.log(`[check-docs] ${warn.length} warning(s):\n- ${warn.join('\n- ')}`);
if (fail.length) {
  console.error(`[check-docs] ${fail.length} problem(s):\n- ${fail.join('\n- ')}`);
  process.exit(1);
}
console.log('[check-docs] ok');

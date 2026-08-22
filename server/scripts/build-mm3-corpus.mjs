// build-mm3-corpus.mjs — compile MiniMax's Structured Caption reference corpus
// into the single JSON file the deterministic caption composer reads at runtime.
//
// Source of truth is the vendored upstream library:
//   .claude/skills/mm3-captioning/upstream/references/genre-router.md
//   .claude/skills/mm3-captioning/upstream/references/index-<family>.md   (18)
//   .claude/skills/mm3-captioning/upstream/templates/<slug>_NNNN.txt      (1000)
//
// `.claude/` is absent from a portable release, so the compiled corpus is
// committed to server/src/data/. Re-run this after any upstream refresh:
//   node server/scripts/build-mm3-corpus.mjs
//
// Output shape (see services/lireek/mm3Corpus.ts for the consuming types):
//   { builtFrom, families: string[], cards: [{ id, family, style, secondary,
//     bpm, key, scale, gender, sections, fields: {<12 labels>} }] }

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const UPSTREAM = path.join(REPO, '.claude', 'skills', 'mm3-captioning', 'upstream');
const OUT = path.join(REPO, 'server', 'src', 'data', 'mm3-corpus.json');

/** The 12 content labels every reference caption carries, in emission order. */
const LABELS = [
  'Basic Attributes',
  'Global Emotional Progression',
  'Application Scenarios & Imagery',
  'Sonics & Production Profile',
  'Vocal Gender & Timbre',
  'Vocal Style',
  'Harmony/Backing Vocals',
  'Vocal FX',
  'Primary',
  'Secondary',
  'Groove & Foundation Progression',
  'Embellishments, Textures & Spatial FX',
];

/** Section words that appear in the corpus prose, normalised to singular. */
const SECTION_RE =
  /\b(intro|verses?|pre-?chorus(?:es)?|post-?chorus|choruse?s?|bridge|breakdown|drop|outro|hook|refrain|coda|interlude|solo|finale)\b/gi;

function canonSection(w) {
  const s = w.toLowerCase().replace(/[- ]/g, '-');
  if (s.startsWith('pre')) return 'pre-chorus';
  if (s.startsWith('post')) return 'post-chorus';
  if (s.startsWith('choru')) return 'chorus';
  if (s.startsWith('vers')) return 'verse';
  return s.replace(/e?s$/, '');
}

function parseTemplate(txt) {
  const fields = {};
  for (const line of txt.split('\n')) {
    const m = line.match(/^([A-Z][^:]{2,70}):\s*(.*)$/);
    if (m && LABELS.includes(m[1]) && m[2].trim()) fields[m[1]] = m[2].trim();
  }
  return fields;
}

/** Which of the four vocal-gender shapes this card uses. */
function detectGender(vocalField) {
  const v = vocalField || '';
  const hasF = /\bfemale\b/i.test(v);
  // strip "female" first so its trailing "male" doesn't count as a male singer
  const hasM = /\bmale\b/i.test(v.replace(/female/gi, ''));
  if (hasF && hasM) return 'duet';
  if (hasF) return 'female';
  if (hasM) return 'male';
  return 'unknown';
}

// ── read the 18 family index cards ──────────────────────────────────────────
const refDir = path.join(UPSTREAM, 'references');
const tplDir = path.join(UPSTREAM, 'templates');
if (!fs.existsSync(refDir) || !fs.existsSync(tplDir)) {
  console.error(`upstream corpus not found under ${UPSTREAM}`);
  process.exit(1);
}

const cards = [];
const families = [];
let missingTemplate = 0;
let incomplete = 0;

for (const file of fs.readdirSync(refDir).filter((f) => f.startsWith('index-'))) {
  const family = file.replace(/^index-|\.md$/g, '');
  families.push(family);
  for (const line of fs.readFileSync(path.join(refDir, file), 'utf8').split('\n')) {
    if (!line.startsWith('| `')) continue;
    const col = line.split('|').map((s) => s.trim()).filter((_, i) => i > 0);
    if (col.length < 8) continue;

    const id = col[0].replace(/`/g, '');
    const tplPath = path.join(tplDir, `${id}.txt`);
    if (!fs.existsSync(tplPath)) { missingTemplate++; continue; }

    const fields = parseTemplate(fs.readFileSync(tplPath, 'utf8'));
    // Two upstream templates omit "Primary:". Keep them — every other column is
    // usable and the composer selects per-field, so only that one column is lost.
    if (LABELS.some((l) => !fields[l])) incomplete++;

    const tk = col[3] || '';
    const ba = fields['Basic Attributes'] || '';
    const keyM = ba.match(/key is ([A-G][#b]?),? and scale is (major|minor)/i);

    // Per-field section vocabulary — the composer rejects any field mentioning
    // a section the requested arrangement does not have.
    const sections = {};
    for (const [label, text] of Object.entries(fields)) {
      sections[label] = [...new Set((text.match(SECTION_RE) || []).map(canonSection))];
    }

    cards.push({
      id,
      family,
      style: col[1],
      secondary: col[2] === '—' ? '' : col[2],
      bpm: parseInt((tk.match(/(\d+)\s*BPM/i) || [])[1] || '0', 10) || 0,
      key: keyM ? keyM[1] : '',
      scale: keyM ? keyM[2].toLowerCase() : '',
      gender: detectGender(fields['Vocal Gender & Timbre']),
      sections,
      fields,
    });
  }
}

families.sort();
cards.sort((a, b) => a.id.localeCompare(b.id));

const out = { builtFrom: 'mm3-captioning/upstream', labels: LABELS, families, cards };
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(out));

const byGender = cards.reduce((a, c) => ((a[c.gender] = (a[c.gender] || 0) + 1), a), {});
console.log(`mm3-corpus.json written: ${cards.length} cards, ${families.length} families`);
console.log(`  gender pools: ${Object.entries(byGender).map(([k, v]) => `${k}=${v}`).join(' ')}`);
if (missingTemplate) console.log(`  ${missingTemplate} index rows had no template file (skipped)`);
if (incomplete) console.log(`  ${incomplete} templates are missing at least one label (kept, that column unused)`);
console.log(`  ${(fs.statSync(OUT).size / 1048576).toFixed(1)} MB -> ${path.relative(REPO, OUT)}`);

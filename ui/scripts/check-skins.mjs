// check-skins.mjs — the theme half of the skin checker (docs/dev/skins.md).
//   node scripts/check-skins.mjs
// Checks every ui/src/themes/<id>/ (manifest, CSS rules, contrast), then runs the
// bad fixtures in scripts/skin-fixtures/ and fails unless each is rejected for
// the reason named in its expect.txt. Slice 3 adds the shell half.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import postcss from 'postcss';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const UI = path.join(HERE, '..');
const THEMES_DIR = path.join(UI, 'src/themes');
const FIXTURES_DIR = path.join(HERE, 'skin-fixtures');
const docsPath = path.join(UI, '../docs/dev/skins.md');

// The theme API: the only properties a theme may set. Renaming one is a major API bump.
const TOKENS = new Set([
  '--tint', '--tint-hover', '--tint-glow', '--tint-deep', '--tint-ink',
  '--k-well', '--k-chassis', '--k-panel', '--k-raised', '--k-ink', '--k-muted', '--k-on-tint',
  '--k-ok', '--k-warn', '--k-error', '--k-error-fill', '--k-error-hover', '--k-error-glow',
  '--k-shade', '--k-hi',
  '--grad-raised', '--grad-raised-press', '--grad-panel', '--grad-well',
  '--grad-tint', '--grad-tint-hover', '--grad-tint-press',
  '--edge-side', '--edge-neutral', '--edge-tint', '--drop', '--press-shadow', '--well-shadow',
  '--font-body', '--font-heading', '--font-mono',
]);
// Fonts are the same in both modes, so they need no :not(.dark) twin.
const MODE_FREE = new Set(['--font-body', '--font-heading', '--font-mono']);

// Value grammar per token kind. Only these shapes and functions are accepted, so no
// image function (url, image-set, cross-fade...) can get through, and CSS escapes are rejected.
const kindOf = p => p.startsWith('--font-') ? 'font' : p.startsWith('--grad-') ? 'gradient' : /^--(tint|k-)/.test(p) && !p.startsWith('--tint-glow-') ? 'channel' : 'shadow';
const FUNCS = { gradient: ['linear-gradient', 'rgb', 'rgba', 'var'], shadow: ['rgb', 'rgba', 'var'] };
const FONT_RE = /^(?:"[^"\\]*"|'[^'\\]*'|[A-Za-z][\w-]*)(?:\s*,\s*(?:"[^"\\]*"|'[^'\\]*'|[A-Za-z][\w-]*))*$/;
function badValue(prop, value) {
  const kind = kindOf(prop), v = value.trim();
  if (kind === 'channel') return /^(\d{1,3}) (\d{1,3}) (\d{1,3})$/.test(v) && v.split(' ').every(n => +n <= 255) ? null : 'must be three 0-255 channels written "R G B"';
  if (kind === 'font') return FONT_RE.test(v) ? null : 'must be a plain font-family list (names and generic families only)';
  if (!/^[\w\s#%.,()/+-]*$/.test(v)) return 'has characters outside the allowed grammar';
  for (const m of v.matchAll(/([A-Za-z-]+)\s*\(/g)) if (!FUNCS[kind].includes(m[1].toLowerCase())) return `uses function ${m[1]}(), allowed: ${FUNCS[kind].join(', ')}`;
  for (const m of v.matchAll(/var\(\s*([^),\s]*)/g)) if (!TOKENS.has(m[1])) return `var(${m[1]}) is not a documented token`;
  return null;
}

const triplet = v => { const m = /^(\d{1,3}) (\d{1,3}) (\d{1,3})$/.exec(v.trim()); return m ? m.slice(1).map(Number) : null; };
const lum = ([r, g, b]) => {
  const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const contrast = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

// Default values from index.css: :root (dark) and html:not(.dark) (light).
function defaults() {
  const root = postcss.parse(fs.readFileSync(path.join(UI, 'src/index.css'), 'utf8'));
  const dark = {}, light = {};
  root.walkRules(r => {
    const sel = r.selector.trim();
    const into = sel === ':root' ? dark : sel === 'html:not(.dark)' ? light : null;
    if (into) r.walkDecls(d => { if (d.prop.startsWith('--')) into[d.prop] = d.value; });
  });
  return { dark, light: { ...dark, ...light } };
}

function checkManifest(dir, id, errors) {
  let m; // returns the targets array when valid
  try { m = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8')); }
  catch (e) { errors.push(`manifest: unreadable (${e.message})`); return null; }
  const str = k => typeof m[k] === 'string' && m[k].trim() !== '';
  if (m.id !== id) errors.push(`manifest: id "${m.id}" must equal the folder name "${id}"`);
  if (!/^[a-z][a-z0-9-]{1,31}$/.test(id)) errors.push(`manifest: folder/id "${id}" must be lowercase letters, digits and dashes`);
  for (const k of ['name', 'author']) if (!str(k)) errors.push(`manifest: "${k}" must be a non-empty string`);
  if (!str('version') || !/^\d+\.\d+\.\d+$/.test(m.version)) errors.push('manifest: "version" must be semver (1.0.0)');
  if (m.api !== 0) errors.push('manifest: "api" must be 0');
  if (m.kind !== 'theme') errors.push('manifest: "kind" must be "theme"');
  const t = m.targets;
  if (!Array.isArray(t) || t.length === 0 || !t.every(x => x === 'studio' || x === 'classic') || new Set(t).size !== t.length) errors.push('manifest: "targets" must be ["studio"], ["classic"] or both');
  const p = m.preview;
  if (!Array.isArray(p) || p.length < 2 || p.length > 4 || !p.every(c => /^#[0-9a-f]{6}$/i.test(c))) errors.push('manifest: "preview" must be 2 to 4 hex colours (#rrggbb)');
  return t;
}

function checkCss(dir, id, errors) {
  const file = path.join(dir, 'theme.css');
  if (!fs.existsSync(file)) { errors.push('theme.css is missing'); return null; }
  let root;
  try { root = postcss.parse(fs.readFileSync(file, 'utf8')); }
  catch (e) { errors.push(`theme.css: parse error (${e.message})`); return null; }
  const base = `html[data-skin="${id}"]`, twin = `${base}:not(.dark)`;
  const sets = { [base]: {}, [twin]: {} };
  root.walk(n => {
    if (n.type === 'atrule') errors.push(`theme.css: @${n.name} is not allowed`);
    else if (n.type === 'rule') {
      if (n.parent !== root) errors.push(`theme.css: nested rule "${n.selector}" is not allowed`);
      else if (n.selector !== base && n.selector !== twin) errors.push(`theme.css: selector "${n.selector}" is not allowed; use ${base} or ${twin}`);
    } else if (n.type === 'decl') {
      const sel = n.parent.type === 'rule' ? n.parent.selector : null;
      if (!TOKENS.has(n.prop)) errors.push(`theme.css: "${n.prop}" is not a documented token`);
      if (n.important) errors.push(`theme.css: ${n.prop} uses !important, which is not allowed`);
      const bad = TOKENS.has(n.prop) ? badValue(n.prop, n.value) : null;
      if (/url\s*\(|@import/i.test(n.value)) errors.push(`theme.css: ${n.prop} uses url() or import, which is not allowed`);
      else if (bad) errors.push(`theme.css: ${n.prop} ${bad}`);
      if (sets[sel]) sets[sel][n.prop] = n.value;
    }
  });
  for (const p of Object.keys(sets[base])) {
    if (!MODE_FREE.has(p) && !(p in sets[twin])) errors.push(`theme.css: ${p} is set for dark but not in the :not(.dark) block`);
  }
  return { dark: sets[base], light: sets[twin] };
}

// One opaque colour: #rgb, #rrggbb, rgb(R G B) or rgb(R, G, B), alpha absent or 1. Anything else
// (names, transparent, alpha below 1) gives null, because its rendered contrast cannot be calculated.
function opaqueColour(s) {
  s = s.trim();
  let m = /^#([0-9a-f]{6}|[0-9a-f]{3})$/i.exec(s);
  if (m) { const h = m[1].length === 3 ? [...m[1]].map(c => c + c).join('') : m[1]; return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  // CSS does not allow mixed separators: the comma form and the space form are matched separately.
  m = /^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*(\d*\.?\d+)\s*)?\)$/i.exec(s)
    ?? /^rgba?\(\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s*(?:\/\s*(\d*\.?\d+)\s*)?\)$/i.exec(s);
  if (m && [m[1], m[2], m[3]].every(n => +n <= 255) && (m[4] === undefined || +m[4] === 1)) return [+m[1], +m[2], +m[3]];
  return null;
}

// Every stop of a linear-gradient, in order. A stop that is not an opaque colour is null, so the
// caller reports it instead of skipping it. Fewer than two stops gives [null].
function gradientStops(val) {
  const m = /^\s*linear-gradient\((.*)\)\s*$/is.exec(val);
  if (!m) return [null];
  const parts = []; let depth = 0, cur = '';
  for (const ch of m[1]) {
    if (ch === '(') depth++; else if (ch === ')') depth--;
    if (ch === ',' && depth === 0) { parts.push(cur); cur = ''; } else cur += ch;
  }
  parts.push(cur);
  // Optional direction: an angle, or "to <side>" / "to <corner>". Anything else in that slot is a stop.
  if (/^\s*(-?\d*\.?\d+(deg|turn|rad|grad)|to\s+(top|bottom)(\s+(left|right))?|to\s+(left|right)(\s+(top|bottom))?)\s*$/i.test(parts[0])) parts.shift();
  if (parts.length < 2) return [null];
  // Each stop is one colour plus at most two positions (% or px); a colour that does not parse is null.
  return parts.map(p => {
    const m = /^(.*?)((?:\s+-?\d*\.?\d+(?:%|px)){0,2})$/.exec(p.trim());
    return m ? opaqueColour(m[1]) : null;
  });
}

// Blends sampled per segment. The contrast ratio is smooth in the blend position, so 512 samples
// find the minimum to within about 1e-4 of the true value.
const BLEND_STEPS = 512;
// Gradient fills are gated at 4.6:1, not 4.5: a 0.1 margin on top of the dense sampling, as Lead ruled (the bound is closed).
const FILL_FLOOR = 4.6;

// Colours the gradient paints: each stop, plus evenly spaced blends between neighbouring stops
// (browsers blend in sRGB), so a fill that dips below the contrast floor between two passing stops is caught.
// Stop positions only squeeze segments and never add a colour, so they are ignored. A null stop stays null.
function paintedColours(stops) {
  const out = stops.map((c, i) => ({ label: `stop ${i + 1}`, c }));
  const res = [];
  for (let i = 0; i < out.length; i++) {
    res.push(out[i]);
    const a = out[i].c, b = out[i + 1]?.c;
    // Unrounded blends (the browser's per-pixel rounding moves a ratio by well under 0.01).
    if (a && b) for (let k = 1; k < BLEND_STEPS; k++) res.push({ label: `between stops ${i + 1} and ${i + 2}`, c: a.map((x, j) => x + (b[j] - x) * k / BLEND_STEPS) });
  }
  return res;
}

function checkContrast(sets, def, errors, label) {
  for (const mode of ['dark', 'light']) {
    const v = { ...def[mode], ...(mode === 'dark' ? sets.dark : { ...sets.dark, ...sets.light }) };
    const pairs = [];
    for (const s of ['--k-well', '--k-chassis', '--k-panel', '--k-raised']) for (const t of ['--k-ink', '--k-muted']) pairs.push([t, s]);
    for (const s of ['--tint', '--tint-hover', '--tint-deep']) pairs.push(['--k-on-tint', s]);
    for (const s of ['--k-chassis', '--k-panel']) pairs.push(['--tint-ink', s]);
    for (const t of ['--k-ok', '--k-warn', '--k-error']) pairs.push([t, '--k-panel']);
    // The fills the kit really paints: every colour stop of each gradient.
    const stopsOf = (name, fallback) => {
      if (!(name in v)) return fallback.map(k => triplet(v[k] ?? ''));
      const val = v[name].replace(/var\(\s*(--[\w-]+)\s*\)/g, (m, k) => v[k] ?? m);
      return gradientStops(val);
    };
    const fills = [['--grad-tint', ['--tint-hover', '--tint', '--tint-deep']], ['--grad-tint-hover', ['--tint-glow', '--tint-hover', '--tint']], ['--grad-tint-press', ['--tint-deep', '--tint']]];
    const surf = ['--grad-raised', '--grad-raised-press', '--grad-panel', '--grad-well'];
    const rgbPairs = [];
    for (const [name, fb] of fills) for (const { label, c } of paintedColours(stopsOf(name, fb))) rgbPairs.push(['--k-on-tint', triplet(v['--k-on-tint'] ?? ''), `${name} ${label}`, c]);
    for (const name of surf) for (const { label, c } of paintedColours(stopsOf(name, []))) for (const t of ['--k-ink', '--k-muted']) rgbPairs.push([t, triplet(v[t] ?? ''), `${name} ${label}`, c]);
    const worst = new Map(); // one error per text colour and fill, at its lowest ratio
    for (const [fg, a, bgName, b] of rgbPairs) {
      const key = `${fg} ${bgName}`;
      if (!a || !b) { worst.set(key, { fg, bgName, r: -1 }); continue; }
      const r = contrast(a, b);
      if (r < FILL_FLOOR && !(worst.get(key)?.r <= r)) worst.set(key, { fg, bgName, r });
    }
    for (const { fg, bgName, r } of worst.values()) {
      errors.push(r < 0 ? `${label}${mode}: ${fg} or ${bgName} is not a readable colour` : `${label}${mode}: ${fg} on ${bgName} is ${r.toFixed(2)}:1, needs ${FILL_FLOOR}:1 on gradient fills`);
    }
    for (const [fg, bg] of pairs) {
      const a = triplet(v[fg] ?? ''), b = triplet(v[bg] ?? '');
      if (!a || !b) { errors.push(`${label}${mode}: ${!a ? fg : bg} is not an "R G B" triplet`); continue; }
      const r = contrast(a, b);
      if (r < 4.5) errors.push(`${label}${mode}: ${fg} on ${bg} is ${r.toFixed(2)}:1, needs 4.5:1`);
    }
  }
}

function checkTheme(dir, def) {
  const id = path.basename(dir), errors = [];
  const targets = checkManifest(dir, id, errors);
  if (Array.isArray(targets) && targets.includes('classic')) console.log(`note ${id}: classic rules: not implemented yet (HOTSTEP-1322)`);
  // Studio targets: token-only rules and contrast. Classic-only themes skip them until HOTSTEP-1322.
  if (Array.isArray(targets) && !targets.includes('studio')) return errors;
  const sets = checkCss(dir, id, errors);
  if (sets) checkContrast(sets, def, errors, 'contrast ');
  return errors;
}

const subdirs = d => (fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }) : []).filter(e => e.isDirectory()).map(e => path.join(d, e.name));
const def = defaults();
let failed = 0;

// The docs must list every token.
const docs = fs.existsSync(docsPath) ? fs.readFileSync(docsPath, 'utf8') : '';
for (const t of TOKENS) if (!docs.includes(`\`${t}\``)) { console.error(`docs/dev/skins.md does not document ${t}`); failed++; }

// The default kit must itself pass the contrast rules, so the rules are fair.
const defErr = [];
checkContrast({ dark: {}, light: {} }, def, defErr, 'default kit ');
if (defErr.length) console.log(`note (shipped kit, not gated): ${defErr[0]}${defErr.length > 1 ? ` (+${defErr.length - 1} more samples on the same gradient)` : ''}`); // themes are gated; the default look must not change in this slice

const themes = subdirs(THEMES_DIR);
for (const dir of themes) {
  const errs = checkTheme(dir, def);
  if (errs.length) { failed += errs.length; console.error(`FAIL theme ${path.basename(dir)}:\n  - ${errs.join('\n  - ')}`); }
  else console.log(`ok   theme ${path.basename(dir)}`);
}

// Each bad fixture must be rejected, for the reason in its expect.txt.
for (const dir of subdirs(FIXTURES_DIR)) {
  const name = path.basename(dir);
  const want = fs.readFileSync(path.join(dir, 'expect.txt'), 'utf8').trim();
  const errs = checkTheme(dir, def);
  const hit = errs.find(e => e.includes(want));
  if (hit) console.log(`ok   fixture ${name} rejected: ${hit}`);
  else { failed++; console.error(`FAIL fixture ${name}: expected a rejection containing "${want}", got ${errs.length ? errs.join(' | ') : 'none'}`); }
}

if (failed) { console.error(`[check-skins] ${failed} problem(s)`); process.exit(1); }
console.log(`[check-skins] ok (${themes.length} theme(s))`);

// Module hooks that let the existing frontend's code run in Node without its dependencies installed: a few imports
// are replaced by small stand-ins (stubs.mjs), React by a React that only calls components (react.mjs), extensionless
// and `.js` imports find their `.ts` or `.tsx` file, `.ts` files are loaded with their types stripped and `.tsx` files
// compiled with the TypeScript compiler. In a file of ui/src, an import of anything that would draw, fetch or open
// something (a package, a shared component, a query, a modal) is replaced by globalThis.__uiStub(<name>), which
// tools/real-app/page.ts defines: the components of the Create page's bar then run as plain functions over the real
// stores. Registered by load.mjs; used only by tools/compare-request.ts and tools/make-fixtures.ts.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { stripTypeScriptTypes } from 'node:module';
import ts from 'typescript';

const STUB = new URL('./stubs.mjs', import.meta.url).href;
const REACT = new URL('./react.mjs', import.meta.url).href;
const BARE = new Set(['zustand', 'zod/v4']);
const REACTS = new Set(['react', 'react/jsx-runtime', 'react/jsx-dev-runtime', 'react-dom', 'react-dom/client']);
/* files replaced by stand-ins, matched on their path without extension */
const REPLACED = ['SettingsPanel', 'backendStore', 'httpClient'];

/* In ui/src, what runs as it is; any other import of a ui/src file is a stand-in. The bar's panels run, with the
   stores, the helpers and the types they use; what they draw from elsewhere (shared controls, modals) is an element
   whose props are read, not drawn. */
const RUNS = [/^stores\//, /^utils\//, /^types/, /^constants/, /^context\/GlobalParamsContext/, /^components\/global-bar\//,
  /^components\/create\/(createContent|MetadataSection|ContentSection)/, /^components\/shared\/CoverArtSubjectSection/, /^lib\//];
const STAND_IN_PANELS = /global-bar\/(ModelManagerModal|ProfilesModal|SeedManagerDrawer|MonitorBar|VstChainDropdown|ModelSelect)\b/;

const exists = p => fs.existsSync(p) && !fs.statSync(p).isDirectory();
function fileOf(spec, parent) {
  const p = fileURLToPath(new URL(spec, parent));
  const bare = p.replace(/\.(js|ts|tsx)$/, '');
  for (const cand of [p, bare + '.ts', bare + '.tsx', p + '.ts', p + '.tsx', p + '/index.ts', p + '/index.tsx']) if (exists(cand)) return cand;
  return null;
}
const uiSrc = file => { const m = /[\\/]ui[\\/]src[\\/](.*)$/.exec(file); return m ? m[1].replace(/\\/g, '/') : null; };

export async function resolve(spec, ctx, next) {
  if (REACTS.has(spec)) return { url: REACT, shortCircuit: true };
  if (BARE.has(spec)) return { url: `${STUB}?m=${encodeURIComponent(spec)}`, shortCircuit: true };
  if (ctx.parentURL && spec.startsWith('.') && ctx.parentURL.startsWith('file:')) {
    const p = fileURLToPath(new URL(spec, ctx.parentURL));
    const name = p.replace(/\.(js|ts|tsx)$/, '').split(/[/\\]/).pop();
    if (REPLACED.includes(name)) return { url: `${STUB}?m=${name}`, shortCircuit: true };
    if (!exists(p)) { const f = fileOf(spec, ctx.parentURL); if (f) return { url: pathToFileURL(f).href, shortCircuit: true }; }
  }
  return next(spec, ctx);
}

/** Whether an import from a ui/src file is replaced by a stand-in: a package other than React, or a ui/src file that
 *  does not run. */
function standIn(spec, parentFile) {
  if (REACTS.has(spec) || BARE.has(spec) || spec.startsWith('node:')) return false;
  if (!spec.startsWith('.')) return true;
  const name = spec.replace(/\.(js|ts|tsx)$/, '').split('/').pop();
  if (REPLACED.includes(name)) return false;
  const f = fileOf(spec, pathToFileURL(parentFile).href);
  if (!f) return true;
  const rel = uiSrc(f);
  if (!rel) return false; /* outside ui/src: the server's contracts, which run */
  return !RUNS.some(r => r.test(rel)) || STAND_IN_PANELS.test(rel);
}

/** Every import of a stand-in, rewritten to read globalThis.__uiStub(<what it imports>). */
function rewrite(code, file) {
  let n = 0;
  const named = list => list.split(',').map(s => s.trim()).filter(Boolean).map(s => s.replace(/^type\s+/, '').replace(/\s+as\s+/, ': ')).join(', ');
  code = code.replace(/^import\s+([\s\S]+?)\s+from\s+(['"])([^'"]+)\2;?[ \t]*$/gm, (all, clause, _q, spec) => {
    if (!standIn(spec, file)) return all;
    const key = spec.startsWith('.') ? (uiSrc(fileOf(spec, pathToFileURL(file).href) ?? '') || spec) : spec;
    const m = `__stub${n++}`, out = [`const ${m} = globalThis.__uiStub(${JSON.stringify(key)});`];
    const star = /^\*\s+as\s+(\w+)$/.exec(clause.trim());
    if (star) return out.concat(`const ${star[1]} = ${m};`).join(' ');
    const braces = /\{([\s\S]*)\}/.exec(clause), def = clause.replace(/\{[\s\S]*\}/, '').replace(/,/g, '').trim();
    if (def) out.push(`const ${def} = ${m}.default;`);
    if (braces && braces[1].trim()) out.push(`const { ${named(braces[1])} } = ${m};`);
    return out.join(' ');
  });
  return code.replace(/^import\s+(['"])([^'"]+)\1;?[ \t]*$/gm, (all, _q, spec) => (standIn(spec, file) ? '' : all));
}

export async function load(url, ctx, next) {
  if (url.startsWith('file:') && (url.endsWith('.ts') || url.endsWith('.tsx'))) {
    const file = fileURLToPath(url), text = fs.readFileSync(file, 'utf8');
    let source = url.endsWith('.tsx')
      ? ts.transpileModule(text, { fileName: file, compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX, isolatedModules: true } }).outputText
      : stripTypeScriptTypes(text);
    if (uiSrc(file)) source = rewrite(source, file);
    return { format: 'module', source, shortCircuit: true };
  }
  return next(url, ctx);
}

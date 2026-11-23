// Module hooks that let the existing frontend's request code run in Node without its dependencies installed:
// a few imports are replaced by small stand-ins (stubs.mjs), extensionless and `.js` imports find their `.ts`
// file, and every `.ts` file is loaded with its types stripped. Registered by load.mjs; used only by
// tools/compare-request.ts.

import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { stripTypeScriptTypes } from 'node:module';

const STUB = new URL('./stubs.mjs', import.meta.url).href;
const BARE = new Set(['zustand', 'zod/v4']);
/* files replaced by stand-ins, matched on their path without extension */
const REPLACED = ['SettingsPanel', 'backendStore', 'httpClient'];

export async function resolve(spec, ctx, next) {
  if (BARE.has(spec)) return { url: `${STUB}?m=${encodeURIComponent(spec)}`, shortCircuit: true };
  if (ctx.parentURL && spec.startsWith('.') && ctx.parentURL.startsWith('file:')) {
    const p = fileURLToPath(new URL(spec, ctx.parentURL));
    const bare = p.replace(/\.(js|ts|tsx)$/, '');
    const name = bare.split(/[/\\]/).pop();
    if (REPLACED.includes(name)) return { url: `${STUB}?m=${name}`, shortCircuit: true };
    if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) {
      for (const cand of [bare + '.ts', p + '.ts', p + '/index.ts']) {
        if (fs.existsSync(cand)) return { url: pathToFileURL(cand).href, shortCircuit: true };
      }
    }
  }
  return next(spec, ctx);
}

export async function load(url, ctx, next) {
  if (url.startsWith('file:') && url.endsWith('.ts')) {
    const source = stripTypeScriptTypes(fs.readFileSync(fileURLToPath(url), 'utf8'));
    return { format: 'module', source, shortCircuit: true };
  }
  return next(url, ctx);
}

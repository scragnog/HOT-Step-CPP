// The slice's matrix, gathered from slice-states/*.mjs: each work package adds a file of its own (its STATES and the
// regions its states compare, SCOPES), and none edits another's. A scope of the same name in two files is joined.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'slice-states');
const mods = await Promise.all(fs.readdirSync(dir).filter(f => f.endsWith('.mjs')).sort().map(f => import(pathToFileURL(path.join(dir, f)).href)));
export const STATES = mods.flatMap(m => m.STATES ?? []);
const ids = new Set(); for (const s of STATES) { if (ids.has(s.id)) throw new Error(`the state ${s.id} is defined twice in slice-states/`); ids.add(s.id); }
export const SCOPES = {};
for (const m of mods) for (const [k, v] of Object.entries(m.SCOPES ?? {})) SCOPES[k] = [...new Set([...(SCOPES[k] ?? []), ...v])];

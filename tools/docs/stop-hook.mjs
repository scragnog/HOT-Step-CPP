#!/usr/bin/env node
// stop-hook.mjs — Claude Code Stop hook. If the working tree has code changes but no doc
// changes, block the stop once with a reminder. Second stop in a row is allowed through
// (stop_hook_active), so this can never loop. Codex has no hooks; the shared rule lives in AGENTS.md.
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

let input = '';
try { input = fs.readFileSync(0, 'utf8'); } catch {}
let payload = {};
try { payload = JSON.parse(input || '{}'); } catch {}
if (payload.stop_hook_active) process.exit(0);

const cwd = payload.cwd || process.cwd();
const git = (args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).split(/\r?\n/).filter(Boolean);
let changed = [];
try { changed = [...new Set([...git(['diff', '--name-only', 'HEAD']), ...git(['ls-files', '--others', '--exclude-standard'])])]; } catch { process.exit(0); }

const code = changed.filter((f) => /^(server\/src|ui\/src|engine\/(src|tools|plugins)|plugins)\//.test(f) && !/\.md$/.test(f));
const docs = changed.filter((f) => /^(docs\/|README\.md|FEATURES\.md|AGENTS\.md|\.claude\/skills\/|server\/src\/data\/assistant-knowledge\.md)/.test(f));
if (code.length && !docs.length) {
  process.stderr.write(
    `Code changed with no documentation change. Changed: ${code.slice(0, 8).join(', ')}${code.length > 8 ? ', ...' : ''}.\n` +
    `Check the ownership table in AGENTS.md ("Documentation is part of the change") and update the owning page, run "node tools/docs/build-docs.mjs" if routes, plugins or the model registry changed, then "node tools/docs/check-docs.mjs". If no user- or developer-visible behaviour changed, say so in one line and stop again.\n`
  );
  process.exit(2);
}
process.exit(0);

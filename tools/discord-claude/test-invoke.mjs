// Diagnostic: exercise the exact spawn/stdin/parse path the bridge uses,
// without Discord. `node test-invoke.mjs "your prompt"` — prints the parsed
// result + session id. Uses sonnet to keep test cost down.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const prompt = process.argv[2] ?? 'Reply with exactly: BRIDGE-OK\nSecond line to prove multi-line stdin works.';

const args = [
  '-p',
  '--model', 'claude-sonnet-5',
  '--output-format', 'json',
  '--allowedTools', 'Read,Grep,Glob',
  '--disallowedTools', '"Read(**/.env),Read(**/.env.*),Read(**/*.pem),Read(**/*token*),Read(**/sessions.json)"',
  '--strict-mcp-config', '--mcp-config', JSON.stringify(path.join(HERE, 'mcp-empty.json')),
  '--max-turns', '3',
];

const child = spawn('claude', args, { cwd: REPO, shell: process.platform === 'win32', windowsHide: true });
let stdout = '', stderr = '';
child.stdout.on('data', d => { stdout += d; });
child.stderr.on('data', d => { stderr += d; });
child.on('close', (code) => {
  try {
    const events = JSON.parse(stdout);
    const done = (Array.isArray(events) ? events : [events]).findLast(e => e?.type === 'result');
    const init = (Array.isArray(events) ? events : [events]).find(e => e?.type === 'system');
    console.log('exit:', code);
    console.log('result:', done?.result);
    console.log('session:', done?.session_id);
    console.log('mcp servers visible:', (init?.mcp_servers ?? []).length);
    console.log('permission denials:', JSON.stringify(done?.permission_denials ?? []));
  } catch (e) {
    console.log('PARSE FAIL:', e.message, '\nstdout head:', stdout.slice(0, 400), '\nstderr:', stderr.slice(0, 400));
  }
});
child.stdin.write(prompt);
child.stdin.end();

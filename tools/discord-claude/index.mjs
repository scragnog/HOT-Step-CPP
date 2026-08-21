// discord-claude bridge — the MM3 working-group thread <-> Claude Code.
//
// Behavior:
//   - Buffers EVERY message in the allowed channels (rolling, per channel) so
//     Claude sees the conversation, not just the ping.
//   - INVOKES Claude only when @mentioned by an ALLOWLISTED user id.
//   - Each invocation runs `claude -p` headless with cwd = the HOT-Step repo,
//     so it inherits CLAUDE.md, memory, skills, and the docs/plans corpus.
//     Tools are restricted to read-only (Read/Grep/Glob) — the bot can cite
//     the scoreboard; it cannot edit files or run commands.
//   - Per-channel session continuity via --resume (session id persisted to
//     sessions.json beside this file).
//   - `!model fable|opus|sonnet|<full-id>` (allowlisted only) switches the
//     runtime default model. `!model` reports it.
//
// Setup: see README.md (bot token, MessageContent intent, invite URL, .env).

import { Client, GatewayIntentBits, Partials } from 'discord.js';
import { spawn } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';
import * as transcript from './transcript.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const TOKEN = process.env.DISCORD_TOKEN;
const ALLOWED_USERS = new Set((process.env.ALLOWED_USER_IDS ?? '').split(',').map(s => s.trim()).filter(Boolean));
const ALLOWED_CHANNELS = new Set((process.env.ALLOWED_CHANNEL_IDS ?? '').split(',').map(s => s.trim()).filter(Boolean));
const CLAUDE_BIN = process.env.CLAUDE_BIN || 'claude';
const CONTEXT_MESSAGES = Number(process.env.CONTEXT_MESSAGES || 30);
const TIMEOUT_MS = Number(process.env.CLAUDE_TIMEOUT_MS || 300000);
const INTERJECT_PORT = Number(process.env.INTERJECT_PORT || 47821);

const MODEL_ALIASES = {
  fable: 'claude-fable-5',
  opus: 'claude-opus-5',
  sonnet: 'claude-sonnet-5',
};
let currentModel = process.env.CLAUDE_MODEL || 'claude-fable-5';

if (!TOKEN || ALLOWED_USERS.size === 0 || ALLOWED_CHANNELS.size === 0) {
  console.error('Missing DISCORD_TOKEN / ALLOWED_USER_IDS / ALLOWED_CHANNEL_IDS — see .env.example');
  process.exit(1);
}

// ── per-channel state ───────────────────────────────────────────────────────

const SESSIONS_FILE = path.join(HERE, 'sessions.json');
let sessions = {};
try { sessions = JSON.parse(fs.readFileSync(SESSIONS_FILE, 'utf-8')); } catch { /* fresh start */ }
const saveSessions = () => fs.writeFileSync(SESSIONS_FILE, JSON.stringify(sessions, null, 1));

const busy = new Set();    // channelId — one invocation at a time per channel

// Every message in an allowed channel is written to logs/discord/ as it lands
// — bots and our own replies included. That log replaces the old in-memory
// buffer, which was empty on restart and lost everything said between pings.
const log = (msg) => {
  try { transcript.appendMany(msg.channelId, [transcript.toRecord(msg)]); }
  catch (e) { console.error('[bridge] transcript write failed:', String(e.message ?? e).slice(0, 200)); }
};

// ── Claude invocation ───────────────────────────────────────────────────────

function runClaude(prompt, channelId) {
  return new Promise((resolve) => {
    // The prompt goes via STDIN, never argv: on Windows the claude.cmd shim
    // forces shell launching, where multi-line/quoted argv WILL mangle. Every
    // remaining arg is shell-safe by construction (no spaces or quotes), and
    // the empty MCP config travels as a file path for the same reason.
    const args = [
      '-p',
      '--model', currentModel,
      '--output-format', 'json',
      '--allowedTools', 'Read,Grep,Glob',
      // Read-only is the capability CEILING; this narrows disclosure within
      // it. Secrets are unreadable even by name — an injected "print the
      // .env" gets a permission denial, not the bot token.
      '--disallowedTools', '"Read(**/.env),Read(**/.env.*),Read(**/*.pem),Read(**/*token*),Read(**/sessions.json),Read(**/logs/discord/**)"',
      // No MCP servers in Discord-facing sessions: the permission layer would
      // deny their tools anyway, but Gmail/home-automation shouldn't even be
      // visible to a session that untrusted text can steer.
      '--strict-mcp-config', '--mcp-config', JSON.stringify(path.join(HERE, 'mcp-empty.json')),
      '--max-turns', '15',
    ];
    if (sessions[channelId]) args.push('--resume', sessions[channelId]);
    const child = spawn(CLAUDE_BIN, args, {
      cwd: REPO,
      shell: process.platform === 'win32', // claude is a .cmd shim on Windows
      timeout: TIMEOUT_MS,
      windowsHide: true,
    });
    let stdout = '', stderr = '';
    child.stdout.on('data', d => { stdout += d; });
    child.stderr.on('data', d => { stderr += d; });
    child.on('error', (e) => resolve({ text: `(bridge error: ${String(e.message ?? e).slice(0, 300)})` }));
    child.on('close', () => {
      try {
        // --output-format json emits an ARRAY of events (verified live);
        // the terminal entry has type "result" with .result/.session_id.
        const parsed = JSON.parse(stdout);
        const events = Array.isArray(parsed) ? parsed : [parsed];
        const done = events.findLast?.(e => e?.type === 'result') ?? events[events.length - 1];
        if (done?.session_id) { sessions[channelId] = done.session_id; saveSessions(); }
        resolve({ text: done?.result ?? '(empty result)' });
      } catch {
        resolve({ text: String(stdout).trim() || `(no output; stderr: ${stderr.slice(0, 200)})` });
      }
    });
    child.stdin.write(prompt);
    child.stdin.end();
  });
}

// Discord hard limit is 2000 chars per message.
async function replyChunked(msg, text) {
  const chunks = [];
  let rest = text.trim() || '(empty reply)';
  while (rest.length > 0) {
    chunks.push(rest.slice(0, 1900));
    rest = rest.slice(1900);
  }
  let target = msg;
  for (const c of chunks.slice(0, 5)) { // cap runaway replies
    target = await target.reply({ content: c, allowedMentions: { repliedUser: false } });
  }
}

// ── Discord wiring ──────────────────────────────────────────────────────────

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent],
  partials: [Partials.Channel],
});

client.on('clientReady', () => console.log(`[bridge] logged in as ${client.user.tag}, model ${currentModel}`));

client.on('messageCreate', async (msg) => {
  // Threads inherit permission from their parent channel id.
  const chanOk = ALLOWED_CHANNELS.has(msg.channelId) ||
    (msg.channel.isThread?.() && ALLOWED_CHANNELS.has(msg.channel.parentId ?? ''));
  if (!chanOk) return;

  // Log BEFORE the bot-skip: the bridge's own replies are part of the
  // conversation, and omitting them is why the ping path could only stay
  // coherent via --resume.
  log(msg);
  if (msg.author.bot) return;

  const fromAllowed = ALLOWED_USERS.has(msg.author.id);

  // !model — allowlisted control of the runtime default.
  if (fromAllowed && msg.content.startsWith('!model')) {
    const arg = msg.content.split(/\s+/)[1];
    if (arg) {
      currentModel = MODEL_ALIASES[arg.toLowerCase()] ?? arg;
      await msg.reply({ content: `model → \`${currentModel}\``, allowedMentions: { repliedUser: false } });
    } else {
      await msg.reply({ content: `model: \`${currentModel}\``, allowedMentions: { repliedUser: false } });
    }
    return;
  }

  if (!msg.mentions.has(client.user)) return;
  if (!fromAllowed) return; // silent — no toy for trolls
  if (busy.has(msg.channelId)) {
    await msg.react('⏳').catch(() => {});
    return;
  }

  busy.add(msg.channelId);
  try {
    await msg.channel.sendTyping().catch(() => {});
    const context = await buildContext(msg.channel);
    // persona.md is re-read per reply so edits apply live. It leads the
    // prompt; the fixed footer below it names the docs entry points and
    // restates the respond-to-the-ping task so a persona edit can't
    // accidentally delete the operating instructions.
    let persona = '';
    try { persona = fs.readFileSync(path.join(HERE, 'persona.md'), 'utf-8').trim(); } catch { /* optional */ }
    const prompt =
      (persona ? persona + '\n\n---\n\n' : '') +
      `Project state, if needed: start with docs/plans/2026-08-20-mm3-training-studio.md and ` +
      `docs/plans/2026-08-18-encoder-training-plan.md (encoder scoreboard). ` +
      `Recent thread messages for context:\n\n${context}\n\n` +
      `The last message mentions you — respond to it.`;
    const { text } = await runClaude(prompt, msg.channelId);
    await replyChunked(msg, text);
  } catch (e) {
    await msg.reply({ content: `(bridge error: ${String(e).slice(0, 300)})`, allowedMentions: { repliedUser: false } })
      .catch(() => {});
  } finally {
    busy.delete(msg.channelId);
  }
});

// ── Local interject endpoint ────────────────────────────────────────────────
//
// POST http://127.0.0.1:<INTERJECT_PORT>/interject  {"note": "...", "channelId": "..."}
// From Scragnog's console only (loopback bind — not reachable off-box). The
// bot fetches FRESH channel history from Discord (not just the live buffer,
// which is empty at startup and misses offline gaps), composes one in-context
// message under the persona, and posts it as a normal channel message — no
// ping involved. `note` optionally steers the topic; `channelId` defaults to
// the first allowed channel.

// Single source of context for BOTH entry points. We hit the Discord API
// first so anything said while the bridge was down gets folded in — the log
// self-heals on the next invocation — then read the window back off disk.
async function buildContext(channel) {
  try {
    const batch = await channel.messages.fetch({ limit: 100 });
    if (transcript.appendMany(channel.id, [...batch.values()].map(transcript.toRecord))) {
      transcript.dedupeSort(channel.id);
    }
  } catch { /* offline or missing perms — fall through to what is already on disk */ }
  return transcript.format(transcript.readChannel(channel.id).slice(-CONTEXT_MESSAGES));
}

async function interject(channelId, note) {
  const channel = await client.channels.fetch(channelId);
  if (!channel?.isTextBased?.()) throw new Error(`channel ${channelId} is not text-based`);
  if (busy.has(channelId)) throw new Error('busy — an invocation is already running for that channel');
  busy.add(channelId);
  try {
    await channel.sendTyping().catch(() => {});
    const context = await buildContext(channel);
    let persona = '';
    try { persona = fs.readFileSync(path.join(HERE, 'persona.md'), 'utf-8').trim(); } catch { /* optional */ }
    const prompt =
      (persona ? persona + '\n\n---\n\n' : '') +
      `Project state, if needed: start with docs/plans/2026-08-20-mm3-training-studio.md and ` +
      `docs/plans/2026-08-18-encoder-training-plan.md (encoder scoreboard). ` +
      `Recent thread messages, oldest first:\n\n${context}\n\n` +
      `Scragnog has asked you (from his console — this request is NOT visible in the thread) to interject ` +
      `in the conversation now. Compose ONE message that lands naturally in the discussion above` +
      (note ? `, steering toward this: ${note}` : '') +
      `. Do not mention being asked to post; just say the thing.`;
    const { text } = await runClaude(prompt, channelId);
    // channel.send, not reply — there is no message to reply to.
    let rest = text.trim() || '(empty)';
    while (rest.length > 0) {
      await channel.send(rest.slice(0, 1900));
      rest = rest.slice(1900);
      if (rest.length > 3 * 1900) break; // runaway cap
    }
    return text;
  } finally {
    busy.delete(channelId);
  }
}

http.createServer((req, res) => {
  if (req.method !== 'POST' || req.url !== '/interject') {
    res.writeHead(404).end();
    return;
  }
  let body = '';
  req.on('data', d => { body += d; });
  req.on('end', async () => {
    try {
      const j = body ? JSON.parse(body) : {};
      const channelId = j.channelId || [...ALLOWED_CHANNELS][0];
      const text = await interject(channelId, j.note ?? '');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, posted: text.slice(0, 500) }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: String(e.message ?? e) }));
    }
  });
}).listen(INTERJECT_PORT, '127.0.0.1', () =>
  console.log(`[bridge] interject endpoint on http://127.0.0.1:${INTERJECT_PORT}/interject`));

client.login(TOKEN);

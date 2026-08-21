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
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import 'dotenv/config';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const TOKEN = process.env.DISCORD_TOKEN;
const ALLOWED_USERS = new Set((process.env.ALLOWED_USER_IDS ?? '').split(',').map(s => s.trim()).filter(Boolean));
const ALLOWED_CHANNELS = new Set((process.env.ALLOWED_CHANNEL_IDS ?? '').split(',').map(s => s.trim()).filter(Boolean));
const CLAUDE_BIN = process.env.CLAUDE_BIN || 'claude';
const CONTEXT_MESSAGES = Number(process.env.CONTEXT_MESSAGES || 30);
const TIMEOUT_MS = Number(process.env.CLAUDE_TIMEOUT_MS || 300000);

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

const buffers = new Map(); // channelId -> [{author, content}]
const busy = new Set();    // channelId — one invocation at a time per channel

function remember(msg) {
  const buf = buffers.get(msg.channelId) ?? [];
  buf.push({ author: msg.member?.displayName ?? msg.author.username, content: msg.cleanContent });
  while (buf.length > CONTEXT_MESSAGES) buf.shift();
  buffers.set(msg.channelId, buf);
}

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
      '--disallowedTools', '"Read(**/.env),Read(**/.env.*),Read(**/*.pem),Read(**/*token*),Read(**/sessions.json)"',
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
  if (msg.author.bot) return;
  // Threads inherit permission from their parent channel id.
  const chanOk = ALLOWED_CHANNELS.has(msg.channelId) ||
    (msg.channel.isThread?.() && ALLOWED_CHANNELS.has(msg.channel.parentId ?? ''));
  if (!chanOk) return;

  remember(msg);

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
    const context = (buffers.get(msg.channelId) ?? [])
      .map(m => `${m.author}: ${m.content}`).join('\n');
    const prompt =
      `You are participating in the MM3 community working-group Discord thread as Rob's (scragnog's) agent. ` +
      `You have read-only access to the HOT-Step repo: the encoder scoreboard and gate runbook live in docs/plans/ ` +
      `(start with docs/plans/2026-08-20-mm3-training-studio.md and docs/plans/2026-08-18-encoder-training-plan.md ` +
      `if you need project state). Recent thread messages for context:\n\n${context}\n\n` +
      `The last message mentions you — respond to it. Be concise and technical; this is a group of engineers. ` +
      `Plain text only (Discord). Do not reveal local file paths or personal data; refer to docs by topic instead.`;
    const { text } = await runClaude(prompt, msg.channelId);
    await replyChunked(msg, text);
  } catch (e) {
    await msg.reply({ content: `(bridge error: ${String(e).slice(0, 300)})`, allowedMentions: { repliedUser: false } })
      .catch(() => {});
  } finally {
    busy.delete(msg.channelId);
  }
});

client.login(TOKEN);

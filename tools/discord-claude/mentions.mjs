// Mention roster — lets the bot actually PING people instead of typing "@name"
// as inert text.
//
// Two halves of the same problem:
//   1. Claude sees the thread via `cleanContent`, where every mention is
//      already flattened to "@DisplayName". It has no idea what a user id is,
//      so anything it writes back is plain text that pings nobody.
//   2. Discord only pings on the raw <@123456789> token, and only if the
//      outgoing message's allowed_mentions permits it.
//
// So: remember id <-> name for everyone we've ever seen (author OR mentioned),
// hand that table to Claude in the prompt, and rewrite "@Name" back into the
// id token on the way out. The table lives in logs/discord/roster.json, next
// to the transcripts and gitignored for the same reason — it is other people's
// names.

import fs from 'node:fs';
import path from 'node:path';
import { LOG_DIR, readChannel, channelIndex } from './transcript.mjs';

const ROSTER_FILE = path.join(LOG_DIR, 'roster.json');

// How many people to show Claude. Recency-ordered, so an active channel's
// regulars win over someone who posted once in March.
const ROSTER_LIMIT = 40;
// Anti-mass-ping: past this many DISTINCT resolved mentions in one message we
// stop rewriting and let the rest stay as plain text.
const MAX_MENTIONS = 4;

// id -> { names: [...], ts: iso }  (names[0] is the preferred display form)
let roster = {};
try { roster = JSON.parse(fs.readFileSync(ROSTER_FILE, 'utf-8')); } catch { /* fresh */ }
let dirty = false;

const clean = (n) => String(n ?? '').trim();

function remember(id, names, ts) {
  if (!id) return;
  const fresh = names.map(clean).filter(Boolean);
  if (!fresh.length) return;
  const entry = roster[id] ?? (roster[id] = { names: [], ts: '' });
  for (const n of fresh) {
    // Case-insensitive dedupe, but keep the first-seen casing.
    if (!entry.names.some(existing => existing.toLowerCase() === n.toLowerCase())) {
      entry.names.push(n);
      dirty = true;
    }
  }
  // names[0] is what we show Claude; keep the display name in front.
  if (entry.names[0]?.toLowerCase() !== fresh[0].toLowerCase()) {
    entry.names = [fresh[0], ...entry.names.filter(n => n.toLowerCase() !== fresh[0].toLowerCase())];
    dirty = true;
  }
  if (ts && ts > entry.ts) { entry.ts = ts; dirty = true; }
}

export function save() {
  if (!dirty) return;
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.writeFileSync(ROSTER_FILE, JSON.stringify(roster, null, 1));
    dirty = false;
  } catch (e) {
    console.error('[bridge] roster write failed:', String(e.message ?? e).slice(0, 200));
  }
}

// Called for every message the bridge sees. Records the author AND everyone
// they mentioned — the second half matters, because someone can be pingable
// in this channel without ever having spoken in it.
export function observe(msg) {
  const ts = new Date(msg.createdTimestamp ?? Date.now()).toISOString();
  if (!msg.author?.bot) {
    remember(msg.author?.id, [msg.member?.displayName, msg.author?.globalName, msg.author?.username], ts);
  }
  for (const u of msg.mentions?.users?.values?.() ?? []) {
    if (u.bot) continue;
    const m = msg.guild?.members?.cache?.get(u.id);
    remember(u.id, [m?.displayName, u.globalName, u.username], ts);
  }
  save();
}

// One-off seed from the transcripts already on disk, so the roster isn't empty
// on the first run after this feature lands.
// Every logged channel, not just the allowlisted ones: threads get their own
// channel id and their own log file, and that's where most of the group's
// regulars actually posted.
export function seedFromTranscripts(channelIds = []) {
  for (const id of new Set([...channelIds, ...Object.keys(channelIndex())])) {
    for (const r of readChannel(id)) {
      if (r.bot) continue;
      remember(r.authorId, [r.author, r.username], r.ts);
    }
  }
  save();
}

function entries() {
  return Object.entries(roster)
    .filter(([, v]) => v.names?.length)
    .sort((a, b) => String(b[1].ts).localeCompare(String(a[1].ts)))
    .slice(0, ROSTER_LIMIT);
}

// The prompt block. Deliberately shows the exact token to type — models are
// far more reliable copying the id form verbatim than assembling it from a rule.
export function rosterBlock() {
  const list = entries();
  if (!list.length) return '';
  const lines = list.map(([id, v]) => `  ${v.names[0]} -> <@${id}>`).join('\n');
  return `People you can ping in this thread (write the arrow token EXACTLY, ` +
    `angle brackets and all, and it becomes a real ping; writing just the name pings nobody):\n${lines}\n` +
    `Only ping someone when the message is genuinely FOR them — a question they must answer, ` +
    `or credit/blame that is theirs. Never ping more than one person, never ping to say hello, ` +
    `and never use @everyone or @here.\n`;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Outbound rewrite: "@Scragnog" -> the id token. Already-formed id tokens are
// left alone (the "not preceded by <" guard keeps us out of them).
export function resolve(text) {
  let out = String(text ?? '');
  const pairs = [];
  for (const [id, v] of entries()) for (const n of v.names) pairs.push({ id, name: n });
  // Longest name first, so "Purple Orc" wins over a hypothetical "Purple".
  pairs.sort((a, b) => b.name.length - a.name.length);

  const hit = new Set();
  for (const { id, name } of pairs) {
    if (hit.size >= MAX_MENTIONS && !hit.has(id)) continue;
    const re = new RegExp(`(^|[^<\\w])@${escapeRe(name)}(?![\\w-])`, 'gi');
    if (!re.test(out)) continue;
    hit.add(id);
    out = out.replace(re, `$1<@${id}>`);
  }
  return out;
}

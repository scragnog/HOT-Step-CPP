// Append-only Discord transcript — the canonical record of the working-group
// channels, independent of any Claude session state.
//
// Why this exists: the bot's own `claude -p` sessions used to be the only
// record of the conversation, and they are a LOSSY one. Claude saw a rolling
// 30-message window at each ping, so anything said while it was not being
// poked was never written down anywhere, and recovering the thread meant
// regex-scraping prompt text out of ~/.claude/projects/*.jsonl. This log
// captures EVERY message in the allowed channels — bots and the bridge's own
// replies included — the moment it arrives.
//
// Location: logs/discord/<channelId>.jsonl. logs/ is already gitignored, which
// is deliberate: this is other people's chat and does not belong in the repo.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(HERE, '..', '..');
export const LOG_DIR = path.join(REPO, 'logs', 'discord');
const INDEX_FILE = path.join(LOG_DIR, 'channels.json');

export const logPath = (channelId) => path.join(LOG_DIR, `${channelId}.jsonl`);

const ensureDir = () => fs.mkdirSync(LOG_DIR, { recursive: true });

// Files are keyed by channel ID, not name: names get renamed, IDs don't.
// channels.json carries the human label so the reader can resolve "mm3" to an id.
export function channelIndex() {
  try { return JSON.parse(fs.readFileSync(INDEX_FILE, 'utf-8')); } catch { return {}; }
}

function updateIndex(rec) {
  const label = rec.parent ? `${rec.parent}/${rec.channel}` : rec.channel;
  if (!label) return;
  const idx = channelIndex();
  if (idx[rec.channelId] === label) return;
  idx[rec.channelId] = label;
  fs.writeFileSync(INDEX_FILE, JSON.stringify(idx, null, 1));
}

export function toRecord(msg) {
  const ch = msg.channel;
  const isThread = Boolean(ch?.isThread?.());
  return {
    ts: new Date(msg.createdTimestamp).toISOString(),
    id: msg.id,
    channelId: msg.channelId,
    channel: ch?.name ?? '',
    parent: isThread ? (ch.parent?.name ?? '') : '',
    author: msg.member?.displayName ?? msg.author?.username ?? 'unknown',
    authorId: msg.author?.id ?? '',
    bot: Boolean(msg.author?.bot),
    replyTo: msg.reference?.messageId ?? null,
    content: msg.cleanContent ?? msg.content ?? '',
    attachments: [...(msg.attachments?.values?.() ?? [])].map(a => a.name).filter(Boolean),
  };
}

export function readChannel(channelId) {
  try {
    return fs.readFileSync(logPath(channelId), 'utf-8').split('\n').filter(Boolean)
      .map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  } catch { return []; }
}

// Appends only messages we haven't already got. Returns how many were new.
// Dedupe is by message id against the file, so re-running backfill, or a
// restart replaying recent history, is always safe.
export function appendMany(channelId, recs) {
  if (!recs.length) return 0;
  ensureDir();
  const have = new Set(readChannel(channelId).map(r => r.id));
  const fresh = recs.filter(r => r.id && !have.has(r.id));
  if (!fresh.length) return 0;
  fs.appendFileSync(logPath(channelId), fresh.map(r => JSON.stringify(r)).join('\n') + '\n');
  updateIndex(fresh[fresh.length - 1]);
  return fresh.length;
}

// Rewrites the file in timestamp order. Backfill pages backwards, so the raw
// append order is reverse-chronological until this runs.
export function dedupeSort(channelId) {
  const byId = new Map();
  for (const r of readChannel(channelId)) byId.set(r.id, r);
  const out = [...byId.values()].sort((a, b) => a.ts.localeCompare(b.ts) || a.id.localeCompare(b.id));
  if (!out.length) return 0;
  ensureDir();
  fs.writeFileSync(logPath(channelId), out.map(r => JSON.stringify(r)).join('\n') + '\n');
  return out.length;
}

export function format(recs) {
  return recs.map(r => {
    const when = r.ts.slice(0, 16).replace('T', ' ');
    const att = r.attachments?.length ? ` (attached: ${r.attachments.join(', ')})` : '';
    return `[${when}] ${r.author}${r.bot ? ' [bot]' : ''}: ${r.content}${att}`;
  }).join('\n');
}

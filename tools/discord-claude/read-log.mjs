#!/usr/bin/env node
// Reader for the Discord transcripts in logs/discord/ — the interface agents
// working in this repo use to catch up on the working-group thread.
//
//   node tools/discord-claude/read-log.mjs --list
//   node tools/discord-claude/read-log.mjs --since 4h
//   node tools/discord-claude/read-log.mjs --channel mm3 --last 100
//   node tools/discord-claude/read-log.mjs --grep "encoder|scoreboard" --since 3d
//   node tools/discord-claude/read-log.mjs --grep bghira --context 3
//
// --channel accepts a channel id or any substring of its name. With no
// --channel, it reads whichever channel has the most recent message.

import { channelIndex, readChannel, format } from './transcript.mjs';

const argv = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : (argv[i + 1] ?? true);
};
const has = (name) => argv.includes(`--${name}`);

// "4h" / "90m" / "3d" / "2026-08-21" -> epoch ms
function since(spec) {
  if (!spec) return 0;
  const m = String(spec).match(/^(\d+)\s*([hmd])$/i);
  if (m) {
    const mult = { m: 60e3, h: 3600e3, d: 86400e3 }[m[2].toLowerCase()];
    return Date.now() - Number(m[1]) * mult;
  }
  const t = Date.parse(spec);
  return Number.isNaN(t) ? 0 : t;
}

const index = channelIndex();
const ids = Object.keys(index);

if (has('list') || !ids.length) {
  if (!ids.length) {
    console.log('No transcripts yet — logs/discord/ is empty.');
    console.log('The bridge writes them live; run `node backfill.mjs` to pull history from Discord.');
    process.exit(0);
  }
  for (const id of ids) {
    const recs = readChannel(id);
    const last = recs[recs.length - 1];
    console.log(`${id}  ${index[id].padEnd(34)} ${String(recs.length).padStart(6)} msgs  last: ${last ? last.ts.slice(0, 16).replace('T', ' ') : '-'}`);
  }
  process.exit(0);
}

const want = flag('channel');
let targets = ids;
if (want && want !== true) {
  targets = ids.filter(id => id === want || index[id].toLowerCase().includes(String(want).toLowerCase()));
  if (!targets.length) {
    console.error(`No channel matching "${want}". Known: ${ids.map(i => index[i]).join(', ')}`);
    process.exit(1);
  }
} else if (!has('all')) {
  // Default to the liveliest channel rather than merging everything.
  targets = [ids.map(id => {
    const r = readChannel(id);
    return { id, ts: r.length ? r[r.length - 1].ts : '' };
  }).sort((a, b) => b.ts.localeCompare(a.ts))[0].id];
}

let recs = targets.flatMap(id => readChannel(id)).sort((a, b) => a.ts.localeCompare(b.ts));

const from = since(flag('since'));
if (from) recs = recs.filter(r => Date.parse(r.ts) >= from);

const pattern = flag('grep');
if (pattern && pattern !== true) {
  const re = new RegExp(pattern, 'i');
  const ctx = Number(flag('context', 0)) || 0;
  const keep = new Set();
  recs.forEach((r, i) => {
    if (!re.test(r.content) && !re.test(r.author)) return;
    for (let j = Math.max(0, i - ctx); j <= Math.min(recs.length - 1, i + ctx); j++) keep.add(j);
  });
  recs = recs.filter((_, i) => keep.has(i));
}

const label = targets.length > 3 ? `${targets.length} channels` : targets.map(i => index[i]).join(' + ');
const last = Number(flag('last', 0)) || 0;
if (last && recs.length > last) recs = recs.slice(-last);

if (!recs.length) { console.log('(no matching messages)'); process.exit(0); }
console.log(`# ${label} — ${recs.length} messages, ${recs[0].ts.slice(0, 16).replace('T', ' ')} → ${recs[recs.length - 1].ts.slice(0, 16).replace('T', ' ')}\n`);
console.log(format(recs));

#!/usr/bin/env node
// One-shot history backfill for logs/discord/.
//
// Pages backwards through the Discord API (100 messages per request, which is
// the API cap) so the transcript covers the conversation from BEFORE the bridge
// started logging. This is strictly better than recovering context out of old
// Claude session files: it gets every message, not just the windows the bot
// happened to be awake for.
//
//   node tools/discord-claude/backfill.mjs            # all allowed channels
//   node tools/discord-claude/backfill.mjs --max 20000
//   node tools/discord-claude/backfill.mjs --channel <id>
//
// Safe to re-run: appendMany dedupes by message id.

import { Client, GatewayIntentBits, Partials } from 'discord.js';
import 'dotenv/config';
import { toRecord, appendMany, dedupeSort } from './transcript.mjs';

const argv = process.argv.slice(2);
const flag = (n, d = null) => { const i = argv.indexOf(`--${n}`); return i === -1 ? d : argv[i + 1]; };

const TOKEN = process.env.DISCORD_TOKEN;
const MAX = Number(flag('max', 10000));
const only = flag('channel');
const CHANNELS = only
  ? [only]
  : (process.env.ALLOWED_CHANNEL_IDS ?? '').split(',').map(s => s.trim()).filter(Boolean);

if (!TOKEN || !CHANNELS.length) {
  console.error('Missing DISCORD_TOKEN / ALLOWED_CHANNEL_IDS — see .env.example');
  process.exit(1);
}

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent],
  partials: [Partials.Channel],
});

async function drain(channel) {
  const label = channel.name ?? channel.id;
  let before, total = 0, added = 0;
  while (total < MAX) {
    const batch = await channel.messages.fetch({ limit: 100, ...(before ? { before } : {}) });
    if (!batch.size) break;
    const recs = [...batch.values()].map(toRecord);
    added += appendMany(channel.id, recs);
    total += batch.size;
    before = [...batch.values()][batch.size - 1].id;
    process.stdout.write(`\r  ${label}: scanned ${total}, new ${added}   `);
    if (batch.size < 100) break; // reached the start of the channel
  }
  const n = dedupeSort(channel.id);
  console.log(`\r  ${label}: scanned ${total}, new ${added}, transcript now ${n} messages`);
  return added;
}

client.once('clientReady', async () => {
  console.log(`[backfill] logged in as ${client.user.tag}`);
  let grand = 0;
  for (const id of CHANNELS) {
    try {
      const channel = await client.channels.fetch(id);
      // Forum/category channels are not text-based themselves but still hold
      // threads, so only the direct drain is skipped, never the threads below.
      if (channel?.isTextBased?.()) grand += await drain(channel);
      else console.log(`  ${channel?.name ?? id}: no direct messages (forum/category), threads only`);

      // Threads carry their own message history and their own channel id, so
      // they need draining separately or thread conversation is lost.
      for (const kind of ['active', 'archived']) {
        try {
          const fetched = kind === 'active'
            ? await channel.threads.fetchActive()
            : await channel.threads.fetchArchived({ limit: 100 });
          for (const th of fetched.threads.values()) grand += await drain(th);
        } catch { /* no thread perms, or not a thread-capable channel */ }
      }
    } catch (e) {
      console.log(`  ${id}: FAILED — ${String(e.message ?? e).slice(0, 160)}`);
    }
  }
  console.log(`[backfill] done — ${grand} new messages written`);
  client.destroy();
  process.exit(0);
});

client.login(TOKEN);

# discord-claude — the MM3 working-group bridge

Puts Rob's Claude into the community Discord thread: it tracks the
conversation (rolling buffer of everyone's messages) and replies when
@mentioned **by an allowlisted user**. Each reply is a headless Claude Code
session with the HOT-Step repo as cwd — it inherits CLAUDE.md, the memory
files, skills, and docs/plans (scoreboard, runbooks), restricted to
READ-ONLY tools (Read/Grep/Glob). It can cite project state; it cannot edit
files or run commands.

## Setup (one time)

1. https://discord.com/developers/applications → New Application → Bot.
   - Reset Token → copy into `.env` (`cp .env.example .env`).
   - Privileged Gateway Intents: enable **MESSAGE CONTENT INTENT**.
2. Invite it: OAuth2 → URL Generator → scope `bot`, permissions
   `View Channels`, `Send Messages`, `Read Message History`,
   `Add Reactions`, `Send Messages in Threads` → open the URL, pick the
   server (a server admin must do this).
3. Fill `.env`: allowlisted user ids + the working-group channel id
   (Developer Mode → right-click → Copy IDs). Threads under the channel are
   covered automatically.
4. `npm install` in this folder, then `npm start`. Keep it running in a
   terminal (or a scheduled task / nssm service later).

## Use

- `@<botname> <question>` from an allowlisted user → reply in-thread, with
  the last ~30 messages as context and per-channel session continuity.
- `!model` shows the current model; `!model fable|opus|sonnet|<full-id>`
  switches it (allowlisted users only).
- Non-allowlisted pings: silently ignored by design.
- Other bots: ignored unless their id is in `ALLOWED_BOT_IDS`. A listed bot
  gets `BOT_EXCHANGE_MAX` (5) consecutive replies, then 🛑 and silence until
  a human posts in the channel. Two LLM bots that can ping each other will,
  forever, and every round is a full invocation on both sides. `!model`
  stays human-only regardless: an allowlisted bot is still untrusted text.
- One invocation at a time per channel. A ping that lands mid-reply gets ⏳
  and is answered in turn once the current reply finishes; the ⏳ is removed
  when its turn comes. Past `PING_QUEUE_MAX` (5) waiting, further pings get
  🚫 and are dropped, because by then the answer would be archaeology.

## Notes / guardrails

- Replies bill headless Claude Code sessions against Rob's plan; `!model
  sonnet` is the cheap mode for chatter.
- The invocation prompt instructs: concise, technical, no local paths or
  personal data. Every ping is a FRESH headless session; `sessions.json`
  records the last session id per channel purely so a reply can be traced
  back to its transcript for token auditing.
- Posting norm change: this bot speaks AS ITSELF (clearly a bot account),
  distinct from Rob's own posts. Rob's previous "I draft, Rob posts" norm
  still applies to Rob-authored messages.
- Images and files posted in the channel are downloaded by the bridge and read
  off disk; see Attachments below. Reading an image costs tokens, so the prompt
  tells the bot to open only what the message it is answering depends on.
- Conversation memory is the last `CONTEXT_MESSAGES` (200) entries of
  `logs/discord/<channelId>.jsonl`, rebuilt on every ping. There is no
  session state to reset. Raising the number is the only way to widen the
  bot's reach, and it is cheap: 200 messages costs ~12k tokens more per call
  than 30, against a ~36k floor set by CLAUDE.md, the memory index and the
  skill list. Resuming sessions instead was measured at $7.71 a reply and
  removed on 2026-08-24; fresh sessions run ~$0.42 and stay flat.

## Personality

Edit `persona.md` in this folder — voice, boundaries, quirks. It is re-read
on EVERY reply, so changes apply live without restarting the bot. The
operating instructions (docs pointers, respond-to-ping) are appended by the
code after the persona, so a persona edit cannot break the mechanics.


## Interjecting from your PC (no ping)

With the bot running: `say.cmd` (or `say.cmd "steer note"`) posts a
contextual message into the thread — the bot fetches fresh channel history,
composes under the persona, and speaks unprompted. Local-only: the endpoint
binds 127.0.0.1:47821 (INTERJECT_PORT to change). The steer note is never
shown in the thread.


## Pinging people back (`mentions.mjs`)

The bot can tag people for real. Two things had to be true, and neither was:

1. **Claude never saw a user id.** The transcript stores `cleanContent`, where
   Discord has already flattened `<@1234>` down to `@DisplayName`. So ids are
   harvested from the live message object — the author, plus everyone *they*
   mentioned — into `logs/discord/roster.json` (gitignored, other people's
   names), seeded on startup from the transcripts already on disk. The 40
   most-recently-seen get listed in every prompt as `name -> <@id>`.
2. **Outgoing mentions were suppressed.** An `allowedMentions` object without
   `parse` kills *every* mention in the message, so even a correctly-formed id
   token was inert. Replies now send `parse: ['users']` — user mentions live,
   roles and `@everyone` structurally impossible regardless of what any
   Discord message talks the model into.

On the way out, `mentions.resolve()` rewrites any `@Name` it recognises into
the id token (longest name first, so "Purple Orc" beats "Purple"; existing
`<@id>` tokens and things like `rob@example.com` are left alone), capped at 4
distinct people per message. `persona.md` carries the etiquette: at most one
ping, and only when the message is genuinely *for* someone.

Someone who has never posted or been mentioned in a logged channel is not in
the roster and cannot be pinged. They enter it the first time either happens.

**Bots are excluded** from the roster — pinging one usually means waking
somebody else's automation. `PINGABLE_BOTS` in `mentions.mjs` is the allowlist
of deliberate exceptions; it currently holds only **ClaudeClanker**
(`1540351007316516966`), the working group's other agent. Entries there are
seeded at module load, survive the recency cut in `entries()` (so a quiet spell
can't silently kill the ping), and get tagged `(bot)` in the prompt roster with
their own etiquette paragraph. Never add this bot's own id.

Note the ping is **one-way**: `messageCreate` returns early on `msg.author.bot`,
so ClaudeClanker's answer is logged into the context buffer but does not wake
ScragBot. It'll read the reply the next time a human pings it — there is no
automatic bot-to-bot volley, by design.

## Attachments and images (`logs/attachments/`)

ScragBot can see pictures. Every image, PDF or text file posted in an allowed
channel — plus any `cdn.discordapp.com` / `media.discordapp.net` link pasted as
text — is downloaded to `logs/attachments/<channelId>/<messageId>-a0.png` as the
message arrives, and the ping prompt hands Claude the local path. The Read tool
renders images natively, so a posted loss curve arrives as a loss curve rather
than the word `image.png`.

The bridge does the fetching, not Claude. The headless session still has no
network and still runs on `Read,Grep,Glob` only, so nothing an untrusted channel
message says can turn it into a URL fetcher.

Downloads happen at arrival time by necessity: Discord CDN URLs are signed and
expire in roughly a day, so there is no fetching them later. Each ping also
re-captures over a freshly fetched batch of the last 100 messages, which is what
back-fills anything posted while the bridge was down.

What it cannot do: **audio and video**. Read cannot decode them, so they are
never downloaded — the transcript still names them, and the bot knows a track
was posted and that it cannot hear it. Ear tests stay Scragnog's job.

Knobs (all optional, `.env`): `MEDIA_MAX_BYTES` (12 MB), `MEDIA_LOOKBACK` (25
messages per batch), `MEDIA_LIST_MAX` (12 files advertised per prompt),
`MEDIA_RETENTION_DAYS` (21), `MEDIA_ALLOW_HOSTS` (Discord CDN only — widening
this hands a chat-steered session an arbitrary-URL fetcher, so don't).

Filenames on disk are derived from the message id and slot, never from the
uploaded name, so a file called `../../.env` is just `<messageId>-a0.env`.
`logs/` is gitignored; like the transcripts, these are other people's files.

## Transcripts (`logs/discord/`)

Every message in an allowed channel is appended to `logs/discord/<channelId>.jsonl`
as it arrives — other bots and ScragBot's own replies included. `logs/` is
gitignored; these are other people's messages and must not be committed.

    node read-log.mjs --list                          # channels, counts, last activity
    node read-log.mjs --since 12h                     # busiest channel, last 12 hours
    node read-log.mjs --channel all-for-one --last 200
    node read-log.mjs --all --grep "encoder|NVFP4" --context 3

`--channel` takes an id or any substring of the name. `--since` takes `90m` /
`6h` / `3d` or a date.

`node backfill.mjs [--max N]` pages history out of the Discord API to cover
anything said before logging existed, threads included. Safe to re-run — records
dedupe by message id.

The transcript is also what the bot itself reads for context, so it is the single
source of truth for "what was said": the in-memory buffer is gone, and Claude
session files are NOT a substitute (they only ever held a rolling window).

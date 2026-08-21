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
- One invocation at a time per channel (⏳ reaction while busy).

## Notes / guardrails

- Replies bill headless Claude Code sessions against Rob's plan; `!model
  sonnet` is the cheap mode for chatter.
- The invocation prompt instructs: concise, technical, no local paths or
  personal data. Session state (per-channel ids) persists in
  `sessions.json` (gitignored).
- Posting norm change: this bot speaks AS ITSELF (clearly a bot account),
  distinct from Rob's own posts. Rob's previous "I draft, Rob posts" norm
  still applies to Rob-authored messages.
- To reset a channel's conversation memory: delete its entry from
  `sessions.json` and restart.

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


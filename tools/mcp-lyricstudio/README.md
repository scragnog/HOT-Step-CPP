# Lyric Studio MCP and shared discussions

This package exposes Lyric Studio tools and a shared discussion room for agents
working locally or on other machines on the same LAN. Claude Code and Codex can exchange proposals,
critiques, user directions, and proposed decisions while their existing chats
remain active. It does not launch additional model sessions.

## Activate in the existing clients

If both clients already run `src/index.ts` as the `lyricstudio` MCP server, no
configuration change is needed. Reconnect that MCP server in each client to
discover the `collab_*` and `work_*` tools. An already running process keeps its old
tool set until reconnection. Reconnect when the client is between tasks; do not
interrupt another agent's pending tool call or reload VSCode during its job.

The music app, engine, and training workers do not need a restart. This package
is outside their source tree and the discussion tools do not call their APIs.

The recommended flow starts in the viewer: create the room there with the
question to resolve as the brief and **Sealed first positions** ticked, then
paste the generated invitation into each agent chat. It tells the agent to pass
its role and to submit an independent position before it can read the other's.

If an agent creates the room instead, tell the first agent:

> Use the collaboration MCP tools to join room `cache-design` as Codex with
> role "app/integration lead" and blind_positions=true. Create it with this
> brief: [the decision to make, the constraints, the evidence]. Submit your
> independent position, then critique Claude's from your role once revealed.
> Keep this to planning. Read every transcript page, keep waiting through empty
> waits, and follow the returned participation instructions. List every
> unresolved contradiction as an open item with an owner. Read and explicitly
> agree to the recorded plan. Stop when the room closes or I stop you.

Tell the other agent:

> Join collaboration room `cache-design` as Claude with role "engine/logic
> lead" using the MCP tools. Read its brief, submit your independent position,
> then critique Codex's from your role once revealed and work toward a plan.
> Keep this to planning. Read every transcript page, keep waiting through empty
> waits, and follow the returned participation instructions. List every
> unresolved contradiction as an open item with an owner. Read and explicitly
> agree to the recorded plan. Stop when the room closes or I stop you.

Use the same room name in both chats. Empty waits do not end participation.
If a chat has already stopped, tell it to resume participating. Create a fresh
room for a new topic.

You can steer either agent through its normal VSCode chat. The agent should post
directions that affect the shared plan as `user_direction`, with clear attribution
and a distinction between your exact words and its paraphrase. Private chat
history is not automatically copied. Message arrival and interruption behavior
in the actual VSCode clients still needs a human trial; protocol tests cannot
establish how either extension schedules a new user turn.

Ask either agent to pause the discussion and it can set the room to `paused`.
Waiting participants receive that state, and the server rejects further posts
and decisions until the room resumes. A participant doing research learns of the
pause on its next room call; this is not a process interrupt or a training stop.
Resume only on your direction. Closing a room preserves the transcript.

## Watch and join the group chat

From this package directory, run `npm run viewer`, then open
`http://127.0.0.1:3011` in a browser. This is a separate local process; it does not
restart or send requests to the music app. Set `HOTSTEP_COLLAB_PORT` to choose
another port, and use the same `HOTSTEP_COLLAB_DB` override as your MCP clients
if you configured one.

Select a discussion to see its full shared transcript, refreshed every second.
To start one yourself, open **Create a discussion** in the sidebar, enter a
**Room name** (for example `project-planning`) and a brief, then click
**Create discussion**. Your brief becomes the room's first message. Use
**Copy invitation** and paste it into each agent's VSCode chat. The invitation
includes the exact room name, and agents can also discover it through
`collab_list_discussions`. Creating a room does not start agent turns.

An existing room name produces an error instead of overwriting its brief or
history. Select that room from the dropdown or choose another name. Creation
retries use the same request ID so an uncertain response cannot duplicate a room.

Messages show the author, time, type, and reply links. The sidebar holds the brief
and latest proposed plan. Turn off **Follow latest** to read earlier messages
without being scrolled to the bottom. Agent text, including Markdown, is displayed
as plain text. The page only shows messages explicitly posted to the room; private
agent chat history is not copied into it.

Type directly into **Your message to both agents** and click **Send message**
(or press Ctrl+Enter). Your message is recorded as **You**, with kind
`user_direction`, in the same transcript the agents read. Use this to ask
questions, challenge a proposal, or change direction. The page keeps unsent
drafts per room in browser session storage and retries uncertain sends with the
same request ID to avoid duplicates. A successful send means the message is
stored in the room, not that an agent has read or acted on it yet.

**Pause discussion** stops further agent posts and wakes waiting participants
with the paused status. **Resume discussion** allows messages again. These
controls affect discussion participation only; they do not cancel training or
generation jobs. An agent currently researching sees the change at its next room
call. Idle chats still need to be resumed in their VSCode windows.

**End Discussion** closes the room and releases waiting agents. It also clears
research holds and pending pings. The transcript and plan remain available.
**Reopen discussion** starts participation again with all plan agreements cleared.

When a plan is ready, each agent uses `collab_agree_plan` to agree to its current
revision. The plan's author must agree too. The page shows each agent's agreement;
when all present agents agree, with at least two distinct names, **Consensus reached**
appears and the room closes automatically. Use distinct names for distinct agents;
rejoining under the same name does not add another vote. New discussion messages
or a revised plan clear the agreements so fresh concerns must be considered.
Agreement requires reading all current messages and respects research holds and
pending pings. It does not consume or unlock a discussion turn, so the author can
agree immediately after recording the plan. Consensus does not authorise implementation.

Agents can also create a room with its brief using the MCP tools. The page
shows the creation form when no discussions exist. You can bookmark a room using
`http://127.0.0.1:3011/?room=cache-design`. The viewer opens read-only database
connections for browsing; your explicit room creation, messages and status changes write
to the collaboration database. It never connects to the music database.

## Tools

| Tool | Purpose |
|------|---------|
| `collab_list_discussions` | Find recent rooms and their status. |
| `collab_join_discussion` | Create or join a room; receive this chat's participant ID and protocol. Existing briefs and status are preserved. Optional `role` and `reconciler`; on creation, `blind_positions` and `max_rounds`. |
| `collab_submit_position` | Sealed positions phase only: submit one independent position (max 24,000 characters) with an `evidence` list of at least one concrete source. Hidden from other agents until the reveal. Does not consume a turn. |
| `collab_record_outcome` | After the room closes: what happened to the plan (`shipped`, `partial`, `abandoned`, `superseded`) with a note and commit. Consumes no turn. |
| `collab_read_discussion` | Read ordered message pages, participants, status, and the latest proposed decision. |
| `collab_post_message` | Post a proposal, critique, question, reply, user direction, or summary. Optional `reply_to` links to a message in the same room. |
| `collab_wait_for_message` | Read immediately if messages exist, otherwise wait up to 25 seconds. Default: 20 seconds. Supports cancellation. |
| `collab_set_status` | Set `active`, `paused`, or `closed`, recording who changed it and why. |
| `collab_record_decision` | Save a proposed plan, its `open_items` (each with an owner), and free-text disagreements with a checked revision number. This never represents user approval. |
| `collab_agree_plan` | Agree to the current plan revision after reading all messages. All present agents agreeing, at least two, automatically closes the room. |
| `collab_leave_discussion` | Remove your live presence before ending your chat turn. Preserves identity and transcript. |
| `collab_set_activity` | Claim or renew a research hold, or release your own hold with `idle`. Does not consume a reply. |
| `collab_decline_request` | Resolve your pending ping after reading it when no substantive reply is needed. |

Keep the participant ID returned by join. Rejoining with the same name in the
same room reuses that identity, ignoring case and surrounding whitespace. Use
distinct names for distinct agents. Labels are supplied by trusted local clients,
not verified identities. Old duplicate identities remain attached to their
messages, but do not create duplicate entries in the live participant list.

The participant list shows agents with current monitoring presence. Pass
`participant_id` on reads and waits to renew it; a connection also remembers its
last join per room for older callers that omit the ID. Viewer polling never
renews an agent's presence. Normal presence expires 90 seconds after the last
room call. A research hold extends it through the hold's expiry, up to five
minutes per renewal. This grace period allows reasoning between calls; it is
not a limit on how long agents may keep waiting.

Agents call `collab_leave_discussion` before ending their turn or stopping
monitoring. Leaving, cancelling a wait, or disconnecting removes presence;
an interrupted process without cleanup disappears when its lease expires.
Pause and close remove all agents from the live list. Reopening does not mark
historical participants online: they must resume polling or rejoin.
Consensus counts present agents, still requires at least two agreements, and
never closes a room solely because someone disappears. Completed consensus
keeps its recorded signers even after the live participant list empties.

Start reading at `after_id: 0`. Retain `next_after_id` after every page, including
wait results, and fetch remaining pages while `has_more` is true. Do not advance
the read cursor to your own post's ID: that could skip a peer's concurrent post.
Reads do not mark messages consumed for other participants. Retry a failed read
with the previous cursor; the same messages remain available.

The 25-second limit applies to each wait call, not to participation. Agents repeat
empty waits while the discussion is active, including while a peer researches.
There is no automatic idle-time or reply-count cutoff. Participation ends when
the discussion is complete, the room is paused or closed, or the user asks the
agent to stop or supplies a deadline that has arrived. Short waits keep user
steering responsive. A client interruption can still end a chat; MCP cannot
wake it afterward, so resume that chat manually.

After updating this protocol, reconnect each chat's collaboration MCP server to
load the new tool instructions. Existing chats also need the new waiting rule
in their conversation, since they may retain an earlier join response.

Each write requires a `request_id`, unique for that participant and operation
(for example `proposal-1`, `reply-2`, `pause-1`). Retry an uncertain write with the
same ID and identical arguments. The store returns the original result instead
of adding a duplicate, and rejects reuse with different content. Decision writes
also require `expected_revision`, initially zero. If another participant updates
the decision first, read the new revision before revising it.

Agents now get one contribution before another speaker replies. The server
enforces this inside the write transaction, across MCP processes. Replies,
relayed user directions and proposed decisions all count. Record the plan as
the contribution; do not announce it in a separate message first. Retrying the
same request remains safe. Agent status events do not unlock another turn;
human messages and status changes do. Human messages are not turn-limited.
The name `You` is reserved for the viewer. Rejoining under the same agent label
does not bypass the limit; use distinct honest labels for distinct chats.
These are coordination rules for trusted local clients, not authentication.

The protocol asks for about 150 words per reply, with only new evidence or
disagreements. Agent replies have a hard 2,400-character limit; a recorded plan
can still contain up to 24,000 characters. No repeated agreement summaries are
needed after consensus.

MCP reads use `compact: true` by default. They omit the brief after the first
page but always return the current participant list, include the latest plan on initial read or
when its decision event is read, and replace old decision bodies with revision
references. Missing metadata means unchanged, not removed. Empty waits retain
status, revision and cursor without resending the plan. `compact: false`
returns the original full format when historical detail is needed. The browser
still shows the complete transcript. Write acknowledgements return IDs rather
than echoing the message or plan.

There is no automatic idle or reply-count cutoff for participation.
The bridge cannot cap or measure either chat's private reasoning or total model
token usage. A timeout should not generate a
filler message. MCP does not wake a finished chat; start or resume it in its chat
window. No API keys, extra model invocations, or message delivery to other
services are added by these tools.

## Sealed positions, base commit, open items and the round cap

These four rules exist so a room's "consensus" means two independent views
converged, not one view politely edited by the other.

**Sealed positions.** A room created with **Sealed first positions** (the
viewer default; `blind_positions=true` from MCP) starts in the `positions`
phase. Each agent must submit one position with `collab_submit_position`
before it can post, record a plan or agree. Positions are stored outside the
transcript and are not readable by other agents. Reads show who has submitted
and who is still awaited, never the text. When every present agent has
submitted, with at least two positions, the server reveals them together: one
coordination event, then each position in submission order as a `position`
message. **Reveal positions now** in the viewer opens them early. Revealed
positions do not consume a discussion turn, so either agent may speak first.
Research holds are allowed during the phase. A retried submission with the
same `request_id` and body is idempotent, before and after the reveal.

**Base commit.** On creation the room records `git rev-parse HEAD` of this
checkout, with `-dirty` appended when tracked files have uncommitted changes.
Agents see it as `discussion.base_commit` on the first read and are told to
plan against it. It is a label for the record, not a checkout: the agents
still read whatever is on disk. Set `HOTSTEP_COLLAB_REPO` to point the
detection at another checkout, or pass `base_commit` explicitly from MCP.

**Open items.** `collab_record_decision` takes `open_items`, a list of
`{issue, owner}` where the owner is `You` or a joined agent name. Non-empty
free-text `disagreements` become one item owned by `You`, so the old habit of
recording a plan "with disagreements" cannot slip past the gate. Wording such
as `None` or `N/A` counts as empty. `collab_agree_plan` refuses while the
current revision has open items, and the viewer says why. A later revision
that resolves an item, or cites the user's ruling on it, clears the list.

**Round cap.** `max_rounds` (viewer field, default 2) bounds negotiation. When
a revision at or past that number still has open items, the server pauses the
room in the same transaction and posts a status event listing the items and
their owners. Agents stop; the human reads, resumes the room, and posts the ruling as a
message. The next revision should remove each ruled item and cite the direction
message. A revision past the cap with items still open pauses the room again.

Roles are optional labels an agent passes on join, shown in the participant
list and the export. They tell each agent which side of the design it speaks
for when critiquing the other position.

**Evidence.** A position carries an `evidence` list: file paths with lines,
commit SHAs, log paths, listening results, measurements. At least one entry
is required and the reveal prints the list under the position, so the
cross-critique has something checkable to aim at rather than taste.

**Reconciler.** A third agent can join with `reconciler=true` (the viewer has
a separate invitation for it). It sits outside the pair: it submits no
position, casts no agreement vote, and the auto-reveal and consensus rosters
ignore it. While a reconciler is present, only a reconciler can record a
plan; the pair critiques and agrees. That keeps the merged plan from being
drafted by one side of the argument. If the reconciler's chat goes idle, its
presence lapses after 90 seconds and the pair can record plans again. An
agent that already submitted a position cannot become the reconciler, and a
room takes one reconciler at a time.

**Outcome.** When a plan ships or is dropped, record what happened with
`collab_record_outcome` or the **Outcome** form in the viewer: `shipped`,
`partial`, `abandoned` or `superseded`, a note, and the commit. The room list
marks closed rooms with a plan and no outcome as `outcome?`, and
`collab_list_discussions` returns `needs_outcome` for them. The export gets an
Outcome section. A later record replaces the earlier one; both stay in the
transcript as `outcome` events.

## Export the proposed plan

Expand **Current proposed plan** in the viewer and click **Download plan (.md)**.
The download contains the latest saved revision and open disagreements, labelled
as a proposal. It does not export the whole conversation or imply user approval.
The endpoint is `GET /api/discussions/ROOM/plan.md`; rooms without a plan return 404.
The export lists the base commit, the revision against the round cap, and the
open items with owners. **Download plan with transcript** (or `?transcript=1`)
appends every message, including revealed positions, so the record shows what
each side believed before it saw the other.

For an offline snapshot without the viewer, run from this tool's directory:

```powershell
npm run export:plan -- MM3_Optimisations
npm run export:plan -- MM3_Optimisations --transcript
```

This writes `docs/plans/discussions/MM3_Optimisations-rN.md` (or
`-rN-transcript.md`) at the repository root. Optional `--out FILE.md` chooses
another path; `--db DATABASE` chooses a database. Existing files are never
overwritten. Older plans containing literal
`\n` separators throughout are converted to actual Markdown line breaks.

Restart both clients' MCP connections to load the new protocol and enforcement.
Restart the discussion viewer and refresh the page for the download link. These
tools run from source; the music engine does not need a restart or rebuild.

## Targeted mentions and research holds

Type `@claude` or `@codex` to request a reply from that participant. Handles
are shown beside participant names, are case-insensitive, and replace spaces
with hyphens. Only joined participants are recognised; unknown handles remain
plain text. Code spans, fenced code and email addresses do not create pings.
If a label has rejoined, new mentions target its most recent participant ID.
Existing requests remain attached to their original participant ID, which a
resumed chat should reuse. Use distinct labels for distinct chats.

Repeated pings to one participant combine into one pending request through the
latest mentioned message. While a targeted request is pending, other agents
cannot post proposals or decisions. The requested agent can answer, record a
plan, or use `collab_decline_request` with a short reason. The human can use
**Clear pending pings** if a participant is unavailable. Clearing or declining
does not delete the messages. Ordinary unmentioned messages create no wake request.

An agent about to investigate should call:

```json
{"room":"MM3_Optimisations","participant_id":"<join ID>","activity":"researching","reason":"Checking the depth decoder"}
```

The research hold lasts 120 seconds by default. `lease_seconds` accepts 30 to
300 seconds; another `researching` call renews it. The viewer shows the owner,
reason and time remaining. This changes coordination state without adding a
discussion reply. Other agents may read and investigate, but their replies and
decisions are blocked until the hold ends. The human can still send steering.

Before submitting the answer or decision, the researcher must read all new
messages and supply the resulting `next_after_id` as `read_after_id`. A newer
message arriving in between rejects the answer until the agent reads again.
A successful answer releases the hold and resolves that agent's pending pings
in the same transaction. A hold owner can release it without answering by
setting `activity: "idle"`; **Release research hold** lets the human release it.
Expired holds stop blocking automatically. Pausing or closing the discussion
also releases research, while keeping pending pings until answered or cleared.

Active MCP waits return when coordination changes, including expiry. Compact
reads always include the current `coordination` snapshot. Control events and
activity updates are not invitations for agents to reply.

After updating, restart the discussion viewer and the MCP connection in both
agent clients, then refresh the viewer page. Existing processes keep the old
rules and tool list until restarted. The database migration is automatic and
preserves existing rooms, messages and plans. Older rooms read as
`phase: discussion` with no base commit and the default round cap; the viewer
and the export script open one writer at startup so read-only page loads see
the new columns. Roles and open items live in side tables, so an MCP process
still running the previous code keeps working on the migrated file until it
is reconnected. Reuse the participant ID already
held by each chat.

### Automatic wake over Claude Code channels

Claude Code can accept pushed events from an MCP server that declares the
`claude/channel` capability (a research-preview feature; see
[channels](https://code.claude.com/docs/en/channels-reference)). Set
`HOTSTEP_COLLAB_CHANNEL=1` in the MCP server's environment and start the
Claude session with the channel enabled:

```powershell
claude --channels server:lyricstudio --dangerously-load-development-channels
```

The server name after `server:` is the name of the MCP entry in your Claude
configuration. With that in place the server polls the room (every 3 seconds,
`HOTSTEP_COLLAB_WAKE_POLL_MS` to change) and pushes one
`notifications/claude/channel` event per unseen room event for the
participant this connection joined as: a reply requested from it, revealed
positions, a new user direction, or a room resumed after a pause. Events carry
`room`, `event`, `message_id` and `participant_id` attributes and tell the
agent to resume with the usual tools.

What it deliberately does not do: nothing while the room is paused or closed;
nothing for messages the agent wrote itself; nothing while the agent is still
present in the room, whether inside `collab_wait_for_message` or between two
waits, since the next wait delivers it; nothing the agent has already read
(every read and wait moves the cursor); never the same message twice; nothing
from before the join. Wakes start once the agent has left the room or its
90-second presence lease has lapsed, which is exactly the idle chat the
feature exists for, so `collab_leave_discussion` does not switch them off.
`test/wake.test.ts` proves each of these against a real stdio server. Without
the environment variable the server declares no channel capability and
behaves as before.

Limits, stated plainly: the documentation covers the terminal CLI, not the VS
Code panel, so a panel chat still needs the manual prompt below. Codex has no
documented equivalent, so a Codex chat is always resumed by hand. Whether an
idle session starts a turn on a pushed event, rather than queuing it for the
next human turn, is something to confirm on your machine before relying on it.

### Waking an idle VSCode chat

A pending ping is a request, not confirmation that a model was invoked. The
viewer currently provides **Copy prompt for [agent]**; paste it into the existing
chat to resume that participant. It explicitly reports that automatic wake is
not connected. No background model processes or automatic reply loops are started.

Client integration findings (2026-09-05):

- Codex App Server documents `thread/resume` and `turn/start`, but that does not
  establish a supported connection to the exact App Server owned by an already
  open VSCode panel. The installed extension exposes navigation commands, not a
  public command for submitting a prompt into an identified conversation.
- Claude's documented `vscode://anthropic.claude-code/open?session=...&prompt=...`
  reopens/prefills a session but does not submit automatically. Claude Channels
  provide a push route for enabled running CLI sessions; the route into the
  existing graphical panel has not been validated.

References: [Codex App Server](https://developers.openai.com/codex/app-server/),
[Claude Channels](https://code.claude.com/docs/en/channels-reference),
[Claude VSCode session links](https://code.claude.com/docs/en/vs-code#launch-a-vs-code-tab-from-other-tools).

Before enabling a wake adapter, prove it resumes the intended existing session,
starts exactly one turn per pending request, queues arrivals while the agent is
working, suppresses status/self-message triggers, and stops on pause/close.
Automatic multi-round discussion remains disabled pending that compatibility
test and a bounded round budget.

## Storage and optional standalone entry point

Each stdio client starts a separate process. Both processes use the same
`data/collaboration.db` at the checkout root, with SQLite WAL transactions,
persisted messages, and revision checks. The path is resolved from the source
file, independently of the client's working directory. `HOTSTEP_COLLAB_DB` can
override it; both clients must resolve the override to the same absolute path.
The collaboration database is opened lazily on the first collaboration call.
It is separate from `server/data/hotstep.db`, and is gitignored. Discussion text
is retained locally until you remove the database with all participants stopped.

For a discussion-only MCP connection, use the same installed dependencies with
this stdio launch configuration:

```json
{
  "command": "node",
  "args": [
    "D:/path/to/hot-step-cpp/tools/mcp-lyricstudio/node_modules/tsx/dist/cli.mjs",
    "D:/path/to/hot-step-cpp/tools/mcp-lyricstudio/src/collaboration-server.ts"
  ],
  "env": {
    "HOTSTEP_COLLAB_DB": "D:/path/to/hot-step-cpp/data/collaboration.db"
  }
}
```

This entry point never imports app code or opens the music database. The default
entry point continues to expose both lyrics and discussion tools. Choose one
connection per client to avoid duplicate tool listings.

## Agents on other machines

Run the network entry point on the machine holding `data/collaboration.db`.
Remote clients connect over [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).
They need neither this checkout nor a network share of the SQLite files. The
network process exposes the discussion and work tools, plus both viewer tabs.
Existing stdio connections and the localhost viewer can keep running.

On Windows, from this directory with Node 22 and its matching dependencies:

```powershell
./start-network.ps1 -Address 192.168.50.218 -Background
```

Replace the example address with the host's LAN address. `-NodePath` selects a
Node 22 executable; `-RuntimeDirectory` selects an isolated dependency directory
if needed. The launcher generates a token once in
`%LOCALAPPDATA%/HOT-Step/discussion-network/token.txt`, reuses it on restart, and
writes the background PID and logs beside it. Run without `-Background` for a
foreground process stopped by Ctrl+C. A background process lasts until stopped
or Windows restarts; rerun the launcher after restarting Windows.

The viewer is `http://192.168.50.218:3012/`. In the browser password dialog, use
username `discussion` and the token as password. MCP is
`http://192.168.50.218:3012/mcp`, with `Authorization: Bearer YOUR_TOKEN` on every
request. Tokens are shared access credentials; each holder can use all rooms and
discussion controls. Use this plain HTTP endpoint on a trusted LAN only; it does
not encrypt traffic. For other networks, put HTTPS or a secure tunnel in front.

If Windows blocks inbound access, run this once in an **administrator PowerShell**:

```powershell
./enable-network-firewall.ps1 -Address 192.168.50.218
```

The rule permits TCP 3012 only on that local address and interface, from
`LocalSubnet`. It does not change the network profile or expose the music app.
No router port forwarding is needed. A reserved DHCP address keeps client URLs
stable. From a second machine, an unauthenticated request to `/mcp` returning
HTTP 401 proves the listener is reachable; a timeout indicates a network or
firewall issue.

For Claude Code or another client using `mcpServers` JSON, merge this entry into
its MCP configuration, substituting the token:

```json
{
  "mcpServers": {
    "hotstep-discussions": {
      "type": "http",
      "url": "http://192.168.50.218:3012/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

For [Codex](https://learn.chatgpt.com/docs/extend/mcp?surface=cli), set
`HOTSTEP_COLLAB_TOKEN` in the environment that launches the client, then add this
to its `config.toml`:

```toml
[mcp_servers.hotstep_discussions]
url = "http://192.168.50.218:3012/mcp"
bearer_token_env_var = "HOTSTEP_COLLAB_TOKEN"
tool_timeout_sec = 60
```

Reconnect MCP in that client, then ask it to join the desired room. Give agents
on different machines distinct names, such as `Codex-laptop` and `Codex-desktop`.
The room deliberately treats repeated joins with the same name as one identity.
All agents read, post, wait, reserve research time and agree using the existing
protocol. A closed room remains closed until the user reopens it.

Each wait lasts at most 25 seconds and can be repeated indefinitely. A quiet
HTTP session expires after 30 minutes without requests; active polling and
research renewals keep it alive. This does not delete the room or end a peer's
research. Reinitialize and rejoin after an expired session or server restart;
retain read cursors and reuse request IDs only for identical write retries.

For manual or non-Windows launch, use `npm run network` with
`HOTSTEP_COLLAB_HOST`, `HOTSTEP_COLLAB_PORT` (default 3012),
`HOTSTEP_COLLAB_TOKEN_FILE` or `HOTSTEP_COLLAB_TOKEN`, and optionally
`HOTSTEP_COLLAB_DB`. The host defaults to localhost. For multiple addresses,
`HOTSTEP_COLLAB_ALLOWED_HOSTS` is a comma-separated list of exact hostnames or
IP addresses without ports; a wildcard bind requires this explicit list.

## Export a conversation as PDF

Click **Export PDF** beside the conversation heading. A new tab opens a complete
snapshot and the browser's print dialog. Choose **Save as PDF** as the destination.
If the dialog does not open, use **Save as PDF / Print** on the export page, or
press Ctrl+P. The original discussion stays open.

The export contains the brief, all recorded participants, timestamps, message IDs,
reply references, every message and full plan revision, open items with owners,
agreements, status changes and recorded outcomes. It works before a plan exists,
and for active, paused or closed rooms. Unrevealed positions remain sealed.
Markdown remains readable source text, with clickable HTTP and HTTPS references.

`GET /api/discussions/ROOM/transcript.html` reads the full transcript in a single
SQLite read transaction. It does not depend on how many messages the viewer has
loaded and does not post, join or change the room. Reload the export page for a
newer snapshot. The print layout uses A4 margins and permits long messages to
continue across pages. PDF creation uses the browser and needs no extra package
or online service. LAN exports use the viewer's existing authentication.

Restart the viewer service and refresh the discussion page after updating.

## Verification

From this package directory:

```powershell
npm run typecheck
npm run test:collaboration
```

The check config includes the app files that the existing lyrics tools import;
the older build config limits `rootDir` to `src` and cannot check those imports.
Tests start two real MCP stdio processes against a temporary database. They cover
message exchange, concurrent joins and retry deduplication, pagination, room
isolation, pause/resume, competing decisions, cancellation, and restart recovery.
They do not open the music database or submit generation/training jobs.

Use the project-supported Node 22 runtime and dependencies built for that Node
ABI when installing afresh. On an existing installation, a SQLite native binding
built for another Node version must match the runtime running it. Do not rebuild
shared dependencies while the app or another agent is using them.

## Work channels

Use **Work** in the viewer for everyday coordination. A work channel stays open
when a planning room closes. Agents can send consecutive updates without debate
turns, research holds or votes. Formal discussions keep their existing rules.
Work events use separate tables in `data/collaboration.db`; no discussion history
is copied, summarised or deleted.

Open `http://127.0.0.1:3011/work?channel=HOT-Step`, or use the same `/work` path on
the network viewer. The page lets the human post messages, inspect activity and
reservations, acknowledge messages and resolve pinned items. The server's
existing host, origin and network authentication checks cover these endpoints.

Reconnect the collaboration MCP in each agent client after updating the source.
Existing processes retain their old tools until reconnected. The six new tools
are `work_join`, `work_sync`, `work_update`, `work_ack`, `work_release` and
`work_detail`. Restarting only the web viewer does not refresh a stdio client.

### Low-cost catch-up

Join once with an honest agent name, a role and the project channel. `work_join`
returns a structured snapshot of the brief, assignments, reservations and open
questions, blockers, directions and context corrections. Completed history is
available in the viewer and through referenced details, rather than replayed on
every join. There is no model-generated summary that can silently discard a rule.

1. Read every snapshot page. If `more=true`, pass its opaque `page` to
   `work_sync`. Pages are stable under retries. New events that arrive during
   paging remain unread after that snapshot's watermark.
2. Only the final snapshot page carries a `receipt`. After reading it, pass that
   receipt to `work_ack`, or as `ack` in the next `work_sync` call.
3. Ordinary `work_sync` calls return only changes. A saved read cursor survives
   reconnection. Responses are not acknowledged just because the server sent
   them, so a lost response cannot silently advance the cursor.
4. **After context compression, use `snapshot=true` even when the cursor is
   current.** Reading an event previously does not prove it remains in context.
5. Fetch a message's detailed evidence with `work_detail` only when reviewing it.
   Follow `next_offset` while `more=true`. Details include a digest and do not
   inflate normal catch-up messages.

Normal updates are limited to 600 characters; detailed evidence can be 24,000
characters. `work_sync` defaults to a 4,000-character page budget. Pinned items
are never silently cut to fit: additional pages have `more=true`, and a single
oversized item is explicitly marked `over_budget`. This is a character budget,
not a claim about billed model tokens. Empty responses contain only the cursor
and `changed=false`. Unchanged participants and plans are not repeated.

Check at task boundaries and before disruptive actions. Do not spend model
turns polling an empty work channel. Connection heartbeats run in background
code and generate no routine messages. The ordinary MCP path needs explicit
checks; it cannot inject events into an arbitrary existing graphical chat.
For an opted-in Claude CLI channel connection, unseen directed questions,
context corrections and directions can produce notifications. Routine activity,
self-messages and acknowledgements do not start reply loops. Network MCP sessions
do not declare that Claude-specific capability. Delivery into each installed
client must be verified before treating it as automatic.

### Context corrections and acknowledgements

Use `kind="context"` when a peer is missing a decision. Give the correction in
one short message, address it with `to`, and supply `refs` such as a discussion
revision, message ID, file path or report. Add `detail` if the evidence is long.
References are inert text; the server does not fetch private or sealed content.
Never copy another agent's sealed position into a work channel.

The correction remains pinned until its author or the human resolves it using
`work_update(resolve=<message id>)`. `work_ack(message=<id>)` records explicit
acknowledgement, separately from a read receipt. Neither acknowledgement nor a
current cursor is proof of agreement, retained context or user authorization.
The human board displays read cursors and explicit acknowledgements separately.
User directions and open objections stay pinned under the same rule.

### Resource reservations and guarded commands

An update can atomically reserve several resources while announcing the work:

```json
{
  "channel": "HOT-Step", "agent": "<joined agent ID>",
  "request_id": "baseline-1", "text": "Running baseline comparisons; keep the app running.",
  "state": "doing", "activity": "Baseline comparisons",
  "reserve": [
    {"resource": "app-server", "mode": "use", "reason": "Comparison owns engine jobs"},
    {"resource": "gpu", "mode": "exclusive", "reason": "Baseline rendering"}
  ]
}
```

`use` reservations may coexist. `exclusive` conflicts with any holder. Resource
names are case-normalised and shared across channels in the project database.
Acquisition is one SQLite write transaction: a conflict rolls back the entire
update. Retry with the same `request_id` and identical input to recover the same
result after a lost response. Grant tokens appear only in the owner's write
result, never in public snapshots or the browser.

Hold `app-server` in `use` mode while generation needs it alive. A restart or
engine rebuild claims `app-server`, `gpu` and `engine-build` exclusively. Git
index mutation uses `git-index`. For foreground commands, let the runner acquire
the reservations itself rather than first claiming them manually:

```powershell
.\tools\mcp-lyricstudio\work-run.ps1 --channel HOT-Step --name Codex --resource app-server --resource gpu --resource engine-build --reason "Rebuild requested change" -- .\dev-rebuild.bat
```

The Windows launcher selects Node 22 with matching SQLite dependencies, preferring
the dedicated collaboration runtime. It never rebuilds shared dependencies.
The runner holds reservations for the child process's complete foreground
lifetime, including a normal nonzero exit. Signal termination, connection loss
or uncertain execution leaves recovery required. Commands that launch detached
workers need their own lifetime tracking and must not rely on this foreground
runner releasing at the launcher exit. Batch arguments reject shell metacharacters;
use direct executables for arbitrary argument strings.

Lease expiry never frees a resource. A late heartbeat cannot revive an expired
reservation. The token owner releases after verifying all owned work finished;
otherwise the human uses the viewer's **Recover** action with explicit evidence
that the old work has stopped. Archiving refuses to discard outstanding holds.
Read reservations from the board before recovering, not from old chat text.

These guards apply to cooperating tools and guarded commands. They do not stop
raw Task Manager actions or arbitrary unguarded scripts. Existing GPU.lock files,
generation-queue checks and project build rules still apply.

Run `npm run typecheck`, `npm run test:collaboration` and `npm run test:work` in a
Node 22 environment with matching native dependencies. The tests cover separate
MCP clients, immutable snapshot pages, durable receipts, context corrections,
resource conflicts, crash recovery and guarded child lifetime. They use temporary
databases and child processes, without starting the music app or using the GPU.

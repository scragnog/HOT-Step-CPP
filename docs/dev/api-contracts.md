# Generation API contracts

This page records the contracts a client relies on to render a song through the Node server
today: the request, response, error, local auth, job and media rules for `POST /api/generate`,
and the engine capability manifest. The flat request remains the frozen baseline for
frontend decoupling; the intent section documents the additive server resolver.
Baseline line references are to commit `7c711f64`.

For the full route list see the generated [HTTP API index](api.md).

## Versioning

The consumer HTTP contract has no version number yet. The internal generation envelope's
`version: 1` (`GENERATION_ENVELOPE_VERSION`, `server/src/services/backends/types.ts`) versions the
server's own job snapshot, not this API. New fields are additive; a client must ignore
response fields it does not know.

## Local auth

| Step | Contract |
|---|---|
| Get a token | `GET /api/auth/auto` returns `{ user, token }`. It creates the single local user on first use. No credentials are asked for (`server/src/routes/auth.ts:36-41`). |
| Use it | `Authorization: Bearer <token>` on routes that call `getUserId` (`auth.ts:98-102`), including `POST /api/generate`. |
| Lifetime | Tokens live in server memory only. Every server restart invalidates them; the next call answers `401 { "error": "Unauthorized" }` and the client fetches a new token from `/api/auth/auto`. |
| Unauthenticated routes | Job status (`GET /api/generate/status/:id`), cancel, queue and the media mounts below do not check the token. |

Exposure today: the Node server binds `SERVER_HOST`, default `0.0.0.0` (`server/src/config.ts:221-224`),
CORS is open (`server/src/index.ts:80`), and the Vite dev server binds `0.0.0.0:3000`
(`ui/vite.config.ts`). Only the training-worker token gate checks the caller's address
(`server/src/services/training/trainingWorkers.ts:92-96`), and only when a worker token is set.
So the local auth contract identifies the installation's single user; it does not restrict
who can reach the server. Changing that belongs to the separately approved auth and route
policy work, not to this baseline.

## Generate request

`POST /api/generate` with a JSON object body.

The route also accepts `generation-intent/1` bodies for clients moving generation
policy into Node. An intent has `{ "contract": "generation-intent/1", "params":
{ ... }, "input": { ... }, "settings": { ... } }`. `params` is the persisted generation-control state;
`input` holds the form's caption, lyrics, task and caller fields. Node fills
missing control defaults, computes the adapter stack and conditional fields,
and then overlays `input` as the existing Create caller does. Optional `settings`
carries `triggerUseFilename` and `triggerPlacement` for filename trigger fallback. Backend-specific
controls belong in `params.backendParams`; Node reads their keys, defaults and
value constraints from the active backend's capability manifest. An unknown
or invalid extension value returns `400`. The active backend still selects the
engine, and a flat legacy body keeps its existing handling. The original body
is captured before intent resolution when dev capture is enabled.

- The body is the UI's `GenerationParams` shape (`ui/src/types.ts:108-371`), camelCase, plus
  caller-specific extras. Two callers matter for the baseline:
  - Create (`ui/src/App.tsx:651-670`) merges the global parameters, the Create form and
    `source: "create"` plus a few settings (`coResident`, `cacheLmCodes`, `parallelWhisper`,
    `parallelQualityEval`, `parallelCoverArt`).
  - A written song from Lyric Studio (`_executeItem`, `ui/src/stores/audioGenQueueStore.ts:1382-1613`) builds
    its body from the album preset and sets `taskType: "text2music"` and
    `source: "lyric-studio"`.
- `withTimeout` in `ui/src/services/api.ts:208-211` adds `generationTimeoutMinutes` from Settings when the caller
  did not set it.
- The server, not the body, chooses the backend: the active backend from `/api/backends`
  (`server/src/services/generation/envelope.ts:97-127`). `backend` in the body is log-only; a
  mismatch is recorded on the job as `submittedBackendMismatch`.
- `expectedBackend` (optional, sent by the MCP server, never by the UI) makes the request fail
  with `409` when it differs from the active backend (`server/src/routes/generate.ts:70-76`).
- Unknown fields are kept. The whole body is frozen into the job's envelope as `submission`, and
  each backend's `resolveRequest` reads what it understands. Backend-specific knobs travel as
  prefixed keys (`yue2*`, `mm3*`) or in `backendParams`.
- Seeds: `seed` with `randomSeed`. When `randomSeed` is true the backend picks a seed. A
  retried attempt can be reseeded (`envelope.policy.retry.reseedOnRetry`), and the seed it
  actually used is reported per attempt in `attempts[].effective.seed`. MM3 seeds are decimal
  strings, because they are 64-bit.
- For ACE (or a body without `backend`), a timbre reference that no longer exists on disk is
  refused before queuing (`generate.ts:307-313` at the baseline commit).

## Generate response

| Case | Status | Body |
|---|---|---|
| Queued | `200` | `{ "jobId": "<uuid>", "status": "pending" }` |
| Capture-only mode (dev) | `200` | `{ "jobId": null, "status": "captured", "captureId": "<uuid>" }` |

## Errors

Every error is JSON with an `error` string, and sometimes extra fields:

| Status | When | Extra fields |
|---|---|---|
| `400` | Body is not an object, `backend` is not a string, the active backend does not support the requested operation, a missing timbre reference | none |
| `401` | Missing or stale token | none |
| `409` | `expectedBackend` differs from the active backend | `expectedBackend`, `activeBackend` |
| `500` | Request capture failed (dev only), or the active backend is not registered | none |
| `503` | Engine still booting, or paused for training preprocessing | `detail` on the boot case |

Failures after queuing are not HTTP errors. They show up in the job status.

## Jobs

| Route | Contract |
|---|---|
| `GET /api/generate/status/:id` | `{ jobId, status, stage, progress, result, error, attempts, ace_job_id, ace_phase, ace_phase_progress, batch, mm3_streaming, mm3_interleaved, mm3_duration, mm3_takes, mm3_take_seeds, mm3_ending }` (`generate.ts`, status route). `404` when the id is unknown. |
| `POST /api/generate/cancel/:id` | `{ success: true, jobId }`. Marks the job cancelled and cancels the engine job. |
| `POST /api/generate/cancel-all` | `{ success: true, cancelled }` |
| `GET /api/generate/queue` | `{ depth, running, owner, draining, current, pending }` |
| `POST /api/generate/reset-queue` | `{ success: true, cancelled, drained }` |

`status` is one of `pending`, `running`, `lm_running`, `synth_running`, `saving` (active) or
`succeeded`, `failed`, `cancelled` (terminal) (`server/src/services/generation/jobTypes.ts`).

- Jobs live in server memory. A terminal job is pruned an hour after creation.
- A server restart loses every job, queued or running. Status then answers `404`, so a
  client must treat `404` on a job it submitted as "lost", not "never existed".
- Jobs run one at a time on the shared GPU lane. YuE2 can render several queued jobs in one
  engine batch; `batch` names the lead and its members.

On success `result` carries:
- `audioUrls`, with `songIds` index-aligned (one per take)
- optional `bpm`, `duration`, `keyScale`, `timeSignature`
- per-take `durations`, `masteredAudioUrls` and `noAdapterAudioUrls`
- `timing` and `totalMs`

## Media

| Mount | Serves | Auth |
|---|---|---|
| `/audio/<file>` | Rendered audio and latents from the data directory's audio folder (`server/src/index.ts:126-135`) | none |
| `/references/<file>` | Uploaded reference audio (`index.ts:143-153`) | none |

Result URLs are root-relative (`/audio/<file>`). A client on another origin must prefix the
server origin itself; the dev UI relies on the Vite proxy for `/api`, `/audio` and `/references`.

## Stream sessions

Node keeps a session for each engine audio stream it proxies: STORM (`POST
/api/generate/storm/stream`) and MM3 (`GET /api/generate/mm3/stream/:id`). The stream response
names its session in the `X-Stream-Session` header. Types are in
`server/src/contracts/streamSessions.ts`, the client in `ui/src/services/streamSessionsApi.ts`.

| Method | Path | Does |
|---|---|---|
| `GET` | `/api/stream-sessions` | Sessions, newest first |
| `GET` | `/api/stream-sessions/:id` | Status, per-chunk analysis, recording state |
| `POST` | `/api/stream-sessions/:id/recording` | `{ action: 'start' \| 'stop' \| 'discard' }` |
| `GET` | `/api/stream-sessions/:id/recording/export?format=wav\|flac\|mp3\|opus&bitrate=` | The stopped recording as a download |

Analysis: each chunk (STORM slot, MM3 window) gets BPM, key and first onset from
`server/src/services/streamSessions/analysis.ts`, the same code the STORM player imports for
its crossfade timing, run on channel 0 at the chunk's own sample rate. The last 64 chunks are kept.

Recording: between start and stop, Node appends each chunk's samples to a file exactly as the
engine produced them. The export is that concatenation with a fresh WAV header, transcoded by
ffmpeg for FLAC, MP3 (default 320 kbps) and Opus (default 160 kbps). No client mix,
crossfade or device is in the path, so the export of an MM3 take is the song as rendered,
while STORM slots are joined end to end without the player's crossfades. Device capture
(the STORM player's MediaRecorder button) is separate and unchanged.

Lifetime and caps (`LIMITS` in `services/streamSessions/index.ts`):

- At most 16 sessions. At the cap the oldest ended session is evicted; if all are live, a new
  stream plays without a session.
- A recording stops itself at 2 GiB (`capped`); what it holds stays exportable.
- At most 2 exports are built or downloaded at once; another gets 429. Exports stream from
  disk, so a large recording is never held in memory.
- A session with no chunk for 2 hours is ended. An ended session and its recording are deleted
  30 minutes after it ends (`expiresAt`). Recordings left by an earlier server run are deleted
  when the session store first loads.
- A chunk that cannot be parsed, or that changes the audio format mid-recording, fails the
  recording (`failed`, with `error`), as does a disk write that fails after stop; discard it to
  start again. If Node cannot open a session at all, the stream plays without one.

Stop, cancel and reconnect: the session ends when its stream ends, whether it finished or was
stopped or aborted. Start fails with 409 after that, but stopping, exporting and discarding
still work until the session expires. A reconnect opens a new stream and so a new session;
recordings do not carry across.

Access follows the stream routes: installation-scoped, no per-user check.

## Fixture capture (dev only)

`POST /api/generate` can record each incoming body as a fixture, for comparing a later
server-side resolver against what the UI sends today. It is off unless the server process
was started by `dev.bat` (which sets `HOT_STEP_DEV`) with `HOTSTEP_GENERATE_CAPTURE` set in
the environment it inherits. No launcher sets the capture variable, and the end-user launchers
never set `HOT_STEP_DEV`, so it cannot turn on in an installed copy.

| Value | Effect |
|---|---|
| `record` | Record, then handle the request as normal |
| `capture-only` | Record and return `{ jobId: null, status: "captured", captureId }`. No job is created and the GPU is not touched. The UI's queue will not find the job, so expect the queue item to fail. |

Where and what it records:
- The capture is middleware ahead of the generate handler
  (`server/src/services/generation/requestCapture.ts`, mounted in `server/src/routes/generate.ts`).
  It runs before the engine-ready checks, envelope construction and any normalization, so
  every body is recorded exactly as it arrived.
- It needs the same bearer token as the handler. A missing or stale token gets
  `401 { "error": "Unauthorized" }`, and nothing is written. An authenticated `capture-only`
  request skips the engine-ready checks, so it works while the engine is down.
- It records from a deep copy and never writes to the request.
- Fixtures go to `<data dir>/dev-captures/generate/`, one JSON file each, outside git.
- Each fixture (`schema: "hotstep.generate-capture/1"`) holds:
  - `commit` and whether the checkout was `dirty`, read from git at the moment of capture
  - `caller`: the body's `source`, an optional `X-HotStep-Capture-Caller` header for
    headless harnesses, and the Referer path
  - `settings`: the active backend and the body's `backend` field
  - `seed`: `seed`, `randomSeed` and `lmSeed`
  - the redacted `body`, with the list of `redactedPaths`

What is kept out, and what cannot be:
- Request headers are not stored, with two exceptions, each kept only when it passes a strict
  check and dropped otherwise:
  - the caller header, kept only as a short identifier (letters, digits, `.`, `_`, `-`, at most
    64 characters)
  - the Referer, kept only as its path (no host, query or fragment), of at most 128 plain
    characters

  The body's `source` and `backend` are stored under `caller` and `settings` by the same rule.
- Body keys that name a credential are replaced with `"[redacted]"`. Keys are compared
  case-insensitively, ignoring `-` and `_`. A key matches when it:
  - ends in `token`; keys ending in `tokens`, such as `maxTokens`, are kept
  - or contains one of `apikey`, `secret`, `password`, `passwd`, `passphrase`,
    `authorization`, `cookie`, `credential`, `privatekey`, `accesskey`, `signingkey`,
    `clientkey` or `bearer`
- Redaction goes by key name only. Free-text values, such as the caption and lyrics, are
  stored as sent; a secret pasted into one of them would be recorded.

The same function is the comparison point for the resolved-request work: a server-resolved
request must produce a legacy-shaped body that passes through this capture before envelope
construction, so old and new can be diffed fixture against fixture.

## Engine capability manifest

`GET /api/capabilities?backend=<id>` (default: the active backend) returns the backend's
manifest, cached for 10 seconds. If the probe fails it returns an all-false manifest with
`up: false` rather than an error (`server/src/routes/backends.ts:153-188`). The shape is
`BackendCapabilities` (`server/src/services/backends/types.ts:160-177`):

| Field | Meaning |
|---|---|
| `backend` | Backend id: `ace`, `minimax-m3` or `yue2` |
| `up` | Whether the backend answered its probe |
| `core.captionSource` | `none`, `mm3-tracks` or `yue2-dataset`; gates the existing Create and Lyric Studio caption source controls |
| `core` | Model-agnostic parameters it honours: `duration { max, auto, editable? }`, `bpm`, `keyscale`, `negativePrompt`, `batch { max }`, `seed`, plus backend-declared extras |
| `features` | One boolean per feature or studio, such as `cover`, `repaint`, `stems`, `streaming`, `adapters`, `lmAdapters`, `lmAdapterSelectable`, `whisper`, `forcedAlignment` |
| `extensions` | Backend-declared knobs, rendered by the shared plugin-parameter renderer and grouped by `group` (`generation` or `lm`) and optional `section` |
| `license` | Optional licence text, shown verbatim |

`GET /api/backends` lists `{ id, displayName, resourcePool, active }` and `activeId`.
`POST /api/backends/active { id }` switches backends.

Shared code should branch on these flags, not on a backend id. The UI does not yet do this
everywhere; changes to the manifest are proposed separately before they land here.

## Caption and content resolution

`GET /api/resolve/path` returns `{ "path": "old" | "resolved" }` to an authenticated
client. The server setting `GENERATION_INTENT_PATH` defaults to `resolved`; only the exact
value `old` rolls back to the legacy client builder, and any other value (including unset)
selects resolved. Create and the Lyric Studio written-song queue read this before each
submission. A missing path endpoint on an older server selects the old path.
Changing the setting does not replay existing jobs or rewrite browser drafts. A selected
resolved path previews a typed intent and submits its returned request; preview errors stop
that submission. The queue remains browser-owned in this slice.

`POST /api/resolve/preview` builds the body Create or the Lyric Studio queue would send to
`/api/generate`, without queuing anything. Node resolves the caption, wildcards, compose
helpers, duration and the album preset overlay; the client submits the returned `request`
unchanged. It needs the bearer token (`401` without it). It answers `400` with zod `issues`
for a malformed intent, and `400` for an unknown engine or for `params.expectedBackend` naming a
different engine than the one being resolved. It answers `404` for an unknown generation.

The engine is the intent's `engine`, or the active one. Every returned `request` carries
`expectedBackend` set to that engine. So if the active engine changes between preview and
submit, `/api/generate` refuses the body with `409` instead of running content resolved for
one engine on another.

The schemas are zod 4, in `server/src/contracts/resolution.ts`; TypeScript clients import the
inferred types from there. Two intents:

| `kind` | Fields | Resolves like |
|---|---|---|
| `create` | `params` (the Create body, with `caption` and `lyrics` as typed), optional `engine`, `compose { autoExpand, loraTrigger, beatIntro, introBars }`, `captionSource` | `CreatePanel.tsx` Generate |
| `written-song` | `generationId`, `lyricsSetId` (the album it renders as), optional `sourceLyricsSetId`, `artistName`, `params` (the queue item's parameter snapshot), `settings`, `mm3Selection`, `yue2Selection`, `yue2Pick`, `yue2Defaults` | `audioGenQueueStore.ts` `_executeItem` |

The browser state those paths read from local storage arrives as explicit fields:
- the caption source choice (`captionSource`, `mm3Selection`, `yue2Selection`)
- "Use LLM duration" (`settings.useLlmDuration`, default true)
- "Use LM adapter" (`settings.useLmAdapter`, default false)
- filename trigger settings (`settings.triggerUseFilename`, `settings.triggerPlacement`)
- randomized timbre (`settings.randomizeTimbreRef`)
- the stored time signature and language
- the app flags (`settings.app`)

Node never reads or writes a stored selection. The selection writes the browser makes on the
written-song path (top-bar adapter, LM adapter, mastering reference, MM3 adapter) are
returned in `uiEffects` and are not applied.

Resolution rules, unchanged from the browser:

- **Caption, written song.**
  - On MiniMax-Music3: the song's own MM3 caption when the choice is Custom and it has one,
    else a picked album track, else the album track nearest the song's tempo, else the song's
    own MM3 caption.
  - On YuE2: when rendering as another album, a non-Custom dataset pick first. Then the
    song's own YuE2 caption, then the dataset pick. The default choice is Automatic only when
    a dataset is linked and the pick holds an adapter.
  - Anything that comes out empty, and every ACE render, uses the song's ACE caption.
- **Caption, Create.**
  - A caption source for the active engine with tracks to offer replaces the box, unless the
    choice is Custom.
  - Then wildcards are expanded, when `autoExpand` is set.
  - Then the trigger is prepended, unless the caption already starts with it as a whole word,
    and the beat intro request is appended.
- **Wildcards** use the DiT seed when the seed is fixed, zero included, and a fresh random
  seed when it is random. The seed used is in `provenance.wildcards`.
- **Duration.**
  - MiniMax-Music3: always `-1`.
  - A written song: the LLM's duration when allowed and positive, else the lyric estimate
    (90 to 360 s, BPM 120 when the song has none), else 180.
  - Create: the duration as sent.
- **Preset overlay**, written songs:
  - A preset adapter replaces the stack with one entry at the snapshot's `loraScale`
    (default 1). It also clears the global trigger words, and re-derives them from the adapter
    file name when filename triggers are on.
  - The preset LM adapter applies only with `useLmAdapter`; without it `lmAdapter` is removed.
  - On MiniMax-Music3 the preset adapter, or none, replaces the global one.
  - A preset reference track becomes the mastering reference, and also the timbre reference
    unless the snapshot names a dedicated one.

The response is `{ request, provenance, warnings, uiEffects, version }`:
- `provenance` says where the caption, duration and trigger came from, and which preset paths
  applied.
- `version` is the sha256 of the request with keys sorted. `verifyResolvedRequest(body,
  version)` (`server/src/services/generation/resolve/resolveIntent.ts`) checks that a body is
  exactly the previewed one.

Parity with the browser is checked by `npx tsx scripts/resolve-parity.mjs`, run from
`server/`.
- It runs the ports against the UI modules on generated inputs.
- When the browser capture fixtures are present, it rebuilds each one's intent and resolves it
  through the real loader, using a temporary copy of the database. It then compares the whole
  body against what the browser sent.
- Two differences are expected: the pinned `expectedBackend`, and the caption and lyrics of a
  random-seed wildcard draw. Those are checked as valid expansions instead.
- The script prints which inputs each case had to reconstruct.

## Durable audio queue

`/api/audio-queue` keeps a queue of renders in the database and submits them from the Node
process. Pending work therefore survives a closed browser and a server restart, and two
clients cannot render the same item twice. It submits through the same code as
`POST /api/generate` (`submitGeneration` in `server/src/routes/generate.ts`), so items go
through the same checks, job list and GPU lane as any other render. All routes need the
bearer token.

| Route | Does |
|---|---|
| `POST /items` | Queue `{ idempotencyKey, request, meta? }`. `201 { item, created: true }` for a new item. The same key with the same request answers `200 { item, created: false }`. The same key with a different request answers `409` |
| `GET /items[?status=]` | All items, oldest first |
| `GET /items/:id` | One item, or `404` |
| `POST /items/:id/cancel` | Cancel; `409` for a finished item |
| `POST /items/:id/retry` | Queue a failed, cancelled or interrupted item again as a new attempt; `409` otherwise |
| `GET /state`, `POST /pause`, `POST /resume` | `{ paused, maxInFlight, counts }` |
| `POST /migration/import` | Import a verified browser backup by `backupId` with `choice: hold`, `resume`, or `discard`; returns a persistent receipt with each legacy and server item id. A repeated backup id returns the same receipt. Invalid items return `400` with field paths or item ids; conflicting items return `409` |
| `POST /migration/resume-held` | Explicitly release imported held ids and resume the executor |
| `POST /migration/rollback-export` | Pause submissions and export all items, job ids, and state. Returns `409` while a submission is in flight; retry after it settles |
| `DELETE /items/:id` | Dismiss a terminal item; `409` while active |

`request` is the exact body for `/api/generate`, normally a resolve preview's `request`. It is
captured once and never re-resolved. Its engine is its `expectedBackend`. A request without one
is pinned to the engine active when it is queued.

An item is submitted only while its engine is active. After a backend switch it stays pending
with a `waiting` message, and the queue does not switch engines itself. At most `maxInFlight`
(4) items are submitted at once, so YuE2 can still render queued songs as one batch. Pause
stops new submissions; submitted items carry on.

Item `status`:

| Status | Meaning |
|---|---|
| `pending` | Not sent yet. `waiting` says why, when it is held back |
| `held` | Imported browser item awaiting the user's resume or discard choice |
| `submitting` | Claimed by the executor, with the submit in flight |
| `submitted` | Accepted; `jobId` is the generation job, followed until it ends |
| `succeeded`, `failed`, `cancelled` | The job's outcome. `result` holds the job result; `error` holds the reason |
| `interrupted` | The server stopped while the item was submitting or submitted, or its job disappeared from the generation queue. Whether it rendered is unknown |

How each transition happens:
- **Submit.** A claim (pending to submitting) is one guarded update, so a cancel racing a
  submit, or two executors, cannot both act on an item.
- **Submit refused.**
  - A `409` (engine switched) or `503` (engine not ready) puts the item back to pending with
    the reason in `waiting`.
  - Any other refusal marks it failed.
- **Cancel.**
  - A pending item is cancelled at once.
  - A submitting item is cancelled as soon as its submit returns, including its new job.
  - A submitted item cancels its job.
- **Restart.** On startup, items left submitting or submitted become `interrupted`. Their
  jobs lived in the old process's memory, so the queue cannot tell whether they rendered.
  Nothing is resubmitted automatically. Only `retry` renders an item again. It keeps the
  earlier job ids in `previousJobIds` and counts the attempt in `attempt`.

Items live in the `audio_intents` table, created on first use; the paused flag is in
`audio_queue_state`.

The browser migration saves a versioned export of both the legacy localStorage queue and the
IndexedDB queue, including item and job ids. It reads the saved copy back before importing,
downloads it, and retains the original browser records after the server receipt is stored.
New queue items then go to `/api/audio-queue`; the browser reads its status, takes and job
results. Reloads and other tabs read the same server list. To roll back, the browser pauses
the server, exports and verifies its current items, reconciles submitted and terminal job ids
into the saved browser queue, and holds pending work for an explicit resume. A submitting
item cannot be exported until its submission settles. Submitted or interrupted jobs are
never automatically replayed by the browser.

## Workflow jobs and revisions

`/api/workflows` runs multi-step studio operations in the Node process, such as a lyric refine
sequence, an approved preview or a cover analysis. A closed tab, a reconnect or a server restart
can't lose such an operation or run it twice. It also stores drafts that two clients can edit
safely. All routes need the bearer token, and every job and document belongs to one user.

A studio adds a job kind from its own module with `registerWorkflowKind` (in
`server/src/routes/workflows.ts`). A kind has an input schema (zod), a `run(ctx)` function, an
optional `maxConcurrent` (default 1) and `timeoutMs` (default 30 minutes). The kind owns the
domain policy; the service (`server/src/services/workflows/`) owns the lifecycle. Wire types are
in `server/src/contracts/workflow.ts`, and the UI client is `ui/src/services/workflowApi.ts`.

| Route | Does |
|---|---|
| `POST /jobs` | Submit `{ kind, idempotencyKey, input }`. `201 { job, created: true }` for a new job. The same key and input within (user, kind) answers `200 { job, created: false }`. The same key with another input is a `409`. An unknown kind, or input the kind's schema rejects, is a `400` with field paths |
| `GET /jobs[?kind=&status=]` | The user's jobs, newest first, at most 200 |
| `GET /jobs/:id[?after=N]` | `{ job, events, gap }`: the events after sequence `N` |
| `GET /jobs/:id/events[?after=N]` | Server-sent events: a snapshot frame `{ type: 'snapshot', job, gap }`, then `{ type: 'event', event }` frames with `id:` set to the event's sequence, so `Last-Event-ID` resumes. The stream ends at the job's current final status; a retried job's stream carries on past the earlier attempt's |
| `POST /jobs/:id/cancel` | Acknowledge a cancel. `409` for a finished job |
| `POST /jobs/:id/retry` | Run a failed, cancelled or interrupted job again as a new attempt; `409` otherwise |
| `POST /documents` | Create `{ kind, data }` at revision 1 |
| `GET /documents?kind=`, `GET /documents/:id` | Read documents |
| `PUT /documents/:id` | `{ expectedRevision, data }`. Lands only if the stored revision still matches; otherwise `409 { error, currentRevision }` |
| `DELETE /documents/:id?expectedRevision=N` | Delete, with the same check |

Job `status`:

| Status | Meaning |
|---|---|
| `pending` | Accepted, not started |
| `running` | Its step is running in this process. `cancelRequested` may be set |
| `succeeded`, `failed`, `cancelled` | The outcome. `result` holds what the step returned; `error` holds the reason |
| `interrupted` | The server stopped while the job was running. Its last step may or may not have finished |

How each transition happens:
- **Submit.** The input is validated by the kind's schema and captured once. The job runs from
  that copy, never from client state.
- **Start.** A claim (pending to running) is one guarded update. Jobs of one kind start oldest
  first, up to its `maxConcurrent`.
- **Cancel.**
  - A pending job is cancelled at once and never runs.
  - A running job gets `cancelRequested`, a `cancel-requested` event, an aborted `ctx.signal`,
    and its audio items cancelled. It ends `cancelled` at once, without waiting for its step.
    Whatever the step returns afterwards is discarded.
- **Timeout.** A run past `timeoutMs` is aborted and the job fails at once, whether or not
  the step listens to its signal. Its slot is freed for the next job.
- **Fencing.** Each run holds a claim id. A step's events, audio items and completion count
  only while that claim is current. A step that ignored its signal, or a step from the old
  process after a restart and retry, can't write to the job.
- **Restart.** On startup, jobs left running become `interrupted` with a message. Nothing that
  started is rerun automatically. Pending jobs never started, so they start normally. Only
  `retry` runs a job again; it counts the attempt in `attempt`.
  An interrupted job's audio items are left to the audio queue, which reports their own state.

Audio never runs in a workflow step. `ctx.audio.enqueue(key, request)` queues a captured
`/api/generate` body on the durable audio queue, which owns GPU admission, under the key
`workflow:<jobId>:<key>`. `ctx.audio.wait(id)` resolves when that item finishes. A retried job
gets the same audio items back instead of rendering twice, unless the kind includes
`ctx.attempt` in the key.

Each state change is also an event, numbered from 1 per job with no holes. A step adds its own
with `ctx.emit(type, data)` (at most 64 KB each). Only the newest 500 events per job are kept.
A reader whose cursor fell behind that window gets `gap: true` and should rebuild its view
from `job`. `followJob` in the UI client reconnects after a drop and resumes from the last
event it saw.

Revisions use one guarded update (`WHERE revision = ?`). Of two writes based on the same
revision, exactly one lands, and the other gets the current revision. `bumpRevision` applies
the same check to a domain table with a `revision` column (`workflow_documents`,
`builder_projects`). Call it inside the transaction that makes the domain change, so both land
or neither does.

Jobs and events live in `workflow_jobs` and `workflow_events`, documents in
`workflow_documents`, all created on first use.

## UI transport client

`ui/src/services/httpClient.ts` is the one configurable client for every route above: a base
URL (default `/api`) and a separate `mediaRoot` (default `''`, same origin) for `mediaUrl()`,
bearer auth, `AbortSignal` cancellation, JSON `get/post/patch/put/delete`, and a multipart
`upload()` (XHR, so it can report progress fetch cannot). It is the transport underneath
`api.ts`; studios with their own client file (`lireekApi.ts`, `trainingApi.ts`,
`stemStudioApi.ts`, and the other per-studio clients) keep their current fetch wrappers and move
onto it incrementally in later slices, not in one pass.

Two ways to read an SSE route, picked by whether the route checks the bearer token:
- `eventSourceUrl(path)` + `sharedEventSource.ts` for routes that don't (model-manager download
  progress, logs) — `EventSource` cannot set a header, so this is honest about carrying no auth
  rather than smuggling a token into the query string, which `getUserId` (`routes/auth.ts`)
  would not read anyway.
- `streamEvents(path, onData, opts)` for routes that do — a `fetch` with `Authorization`,
  reading the response body and splitting `data:` frames the same way `workflowApi.ts`'s
  `followJob` already does for `/api/workflows`. `onData` gets each frame's raw payload; this
  method does no protocol-specific parsing of its own. Cancellable via `opts.signal`, unlike
  `sharedEventSource`'s auto-reconnecting streams.

Every non-OK response throws `ApiError(status, message, body, currentRevision?, reason?)` — a
superset of `workflowApi.ts`'s `WorkflowRequestError`, so a revisioned caller (document/job
writes, song-builder sections) can switch to this client later without losing the stale-revision
or unsupported-version detail. `message` is the server's `error` field, or `API error: <status>`
when the body is not JSON. A caller that already does `catch (e) { ... e.message }` needs no
change, since `ApiError extends Error`.

Compatibility: this client changes no request body, header or route — it is a drop-in
replacement for the fetch calls it wraps. `api.ts`'s own `get/post/patch/del` now delegate to it;
`generateApi.selectedPath` keeps its 404-means-legacy-server fallback. Call sites with their own
error shape (`builderOp`'s `WorkflowRequestError`, the post-processing endpoints' `alreadyProcessed`
flag, `vstApi.updateChain`'s no-throw `.then(r => r.json())`) are left untouched rather than
folded in and risking a silent behavior change; they are candidates for a later slice once each
one's special case has an equivalent on `ApiError` or its own typed subclass.

### Shared contracts for legacy routes

Most domains below already have a `server/src/contracts/*.ts` module the client imports types
from. Two routes that predate that pattern had drifted into two independently hand-kept
copies of the same shape — exactly the bug class shared contracts exist to prevent:

- **Auth** (`server/src/routes/auth.ts`, `server/src/contracts/auth.ts`). `GET /api/auth/auto`
  returns `AutoLoginResponse` (`{ user, token }`); `GET /me` returns `MeResponse` (`{ user }`,
  `401`/`404` on a missing/unknown token); `POST /setup` and `PATCH /username` both return
  `UsernameUpdateResponse` (`{ user, token }` — username changes mint a fresh token). `AuthUser`
  is the `users` table row (`id`, `username`, `bio`, `avatar_url`, `banner_url`, `created_at`).
  `ui/src/types.ts`'s `User`/`AuthState` now re-export `AuthUser` instead of keeping their own
  copy.
- **Model manager** (`server/src/routes/modelManager.ts`,
  `server/src/contracts/modelManager.ts`). `GET /registry` returns `ModelRegistryResponse`
  (`{ packs, files, modelsDir, variant, cudaMajor }`); `files` are `RegistryFile` — a catalogue
  `RegistryFileEntry` (`id`, `role`, `displayName`, `scale`, `variant`, `quant`, `sizeBytes`,
  `sha256?`, `companions?`, `sm?`, `family?`, ...) plus this install's `installed`/`outdated`.
  `POST /download` and `POST /download/:id/resume` both return `DownloadStartResponse`
  (`{ jobId }`); `GET /downloads` streams `{ jobs: DownloadJob[] }` frames (unauthenticated, via
  `eventSourceUrl`/`sharedEventSource`); `POST /download/:id/cancel` and
  `DELETE /files/:filename` are untyped `{ ok: boolean }` acks, unchanged. `ui/src/types.ts`
  previously kept a second, hand-written `RegistryFile`/`StarterPack` pair that was missing
  `repoPath`, `sha256` and `companions`, and a `ModelRegistry` missing `variant` and `cudaMajor`
  — nothing currently reads those gaps, but a future consumer would have silently gotten
  `undefined`. It now re-exports the server's types instead of keeping its own.

The remaining legacy domains predated the contracts pattern entirely — one ad-hoc inline type in
`api.ts`, no server-side counterpart. Each now has a `server/src/contracts/<domain>.ts`; the
route returns against it (`satisfies`, type-only, no logic or validation change) and `api.ts`
imports the same type instead of its own copy. Errors across all of them are the ordinary
`{ error: string }` shape `ApiError` already unwraps — only the success payloads are listed here.

- **Health** (`contracts/health.ts`). `GET /health` returns `HealthResponse`: `status`,
  `version`, `commit`, `dirty`, `engineBuiltAt` (ace-server.exe's mtime, or `null`), nested
  `aceServer`/`server`/`engine` status, and `clients` (open SSE-holding tabs). `api.ts`'s
  previous inline type was missing `version`, `commit`, `dirty`, `engineBuiltAt` and `clients`,
  and marked `engine` optional when the route always sends it. `GET /presence` is a raw SSE
  keepalive with no JSON frames and has no response type.
- **Shutdown** (`contracts/shutdown.ts`, mounted at `/api/shutdown`). No request body. `POST
  /api/shutdown` always returns `200 { success: true, message }` (`ShutdownResponse`) before
  tearing the process down 300ms later. `POST /api/shutdown/restart` returns `200
  { success: true, message }` (`RestartResponse`) after writing the `.restart-requested`
  marker for the loop wrapper to pick up; a write failure is `500 { error }`.
- **Settings** (`contracts/settings.ts`). `GET /settings/env` takes no input and returns
  `EnvResponse` (`{ values, restartKeys }`, unset keys backfilled with their resolved default).
  `POST /settings/env` takes `{ values: Record<string, string> }`; keys outside
  `EXPOSED_ENV_KEYS` or non-string values are silently dropped rather than rejected, and an
  empty/missing/non-object `values` is `400 { error }`; on success it returns
  `EnvUpdateResponse` (`{ updated, restartRequired }`, `restartRequired` true if any updated key
  is in `RESTART_REQUIRED_KEYS`), or `500 { error }` if writing `.env` fails. `GET
  /settings/gpus` takes no input and returns `GpusResponse` (`{ gpus: GpuInfo[] }`, empty array
  when `nvidia-smi` is unavailable), where `GpuInfo` re-exports `services/gpuDevices.ts`'s
  existing `NvidiaGpu` rather than a third copy of the same shape.
- **Profiles** (`contracts/profiles.ts`). A profile is `ProfileFile` (`{ name, saved_at, data }`
  — the same shape as an exported preset JSON, so one can be dropped into the profiles folder by
  hand). `GET /profiles` takes no input, returns `ListProfilesResponse`. `GET /profiles/:name`
  returns a bare `ProfileFile`, `404 { error }` if missing. `POST /profiles` takes
  `{ name: string, data: object }`; a missing/blank `name` or a `data` that isn't a plain
  object is `400 { error }`; on success it overwrites any existing profile of that name and
  returns `SaveProfileResponse` (`{ ok, name, saved_at }`), or `500 { error }` on a write
  failure. `PATCH /profiles/:name` takes `{ newName: string }`; a missing/blank `newName` is
  `400 { error }`, a missing source profile is `404 { error }`, an existing profile already at
  `newName` is `409 { error }` (renaming onto itself is a no-op success, not a conflict); on
  success it returns `RenameProfileResponse` (`{ ok, name }`), or `500 { error }` on a write/
  unlink failure. `DELETE /profiles/:name` takes no body, `404 { error }` if missing, else
  `DeleteProfileResponse` (`{ ok, deleted }`) or `500 { error }` on an unlink failure.
- **Mastering** (`contracts/mastering.ts`, bearer-token auth on every route). `POST
  /mastering/upload-reference` is multipart with a `file` field; no bearer token is
  `401 { error }`, no file is `400 { error }`; non-WAV/MP3 formats are transcoded to WAV on
  upload (ffmpeg missing or conversion failure surfaces as `500 { error }`); on success it
  returns `UploadReferenceResponse` (`{ name, path, url }`). `GET /mastering/references` takes
  no input and always returns `200 { references: [] }` on a read failure rather than erroring.
  `DELETE /mastering/references/:name` needs a bearer token (`401` if missing), `404 { error }`
  if the file doesn't exist, `400 { error }` if path resolution would escape the references
  directory, else `DeleteReferenceResponse` (`{ ok: true }`). `POST /mastering/run` needs a
  bearer token (`401`) and body `{ songId: string, referenceName: string }` (missing either is
  `400 { error }`); `404 { error }` for an unknown `songId`, a missing audio file on disk, or a
  missing reference file; a mastering-binary failure is `500 { error }`; on success it returns
  `RunMasteringResponse` (`{ ok, masteredUrl, songId }`) and writes `masteredUrl` to the song
  row.
- **Adapters** (`contracts/adapters.ts`). `GET /adapters/browse?path=&filter=` (both optional;
  `filter` is `adapters`/`audio`/`trainingAudio`, anything else means unfiltered) returns
  `BrowseResponse` (`{ current, entries: BrowseEntry[] }`); an unresolvable `path` is
  `400 { error }`, a missing/non-directory path is `404` with `BrowseErrorResponse` (the same
  shape plus `error`), and a read failure mid-listing is `500` with `BrowseErrorResponse` — the
  entries array stays present either way so a caller that skips the status code still gets
  something to render. `POST /adapters/scan` takes `{ folder: string }`; a missing/non-string
  `folder` or a missing/non-directory path is a quiet `200 { files: [] }` rather than an error,
  and so is any scan exception — always `200`. `GET /adapters/lm?folder=` (optional; defaults to
  every configured planner-adapter root) returns `LmAdaptersResponse`
  (`{ root, adapters: LmAdapterEntry[], error? }`) — also always `200`; a scan failure sets
  `error` rather than changing status, so the picker can still show whatever root it resolved.
  `LmAdapterEntry` carries the per-adapter eval sidecar (`evalScore`, `evalVerdict`).
  `api.ts`'s previous inline copy of `LmAdapterEntry` marked `lmSize`/`run`/`trigger`/
  `triggerPosition` optional; the route always sends them, now reflected in the shared type.
- **VST** (`contracts/vst.ts`). `VstPlugin` and `ChainEntry` used to be declared once in the
  route file and copied again in `api.ts` (as `VstPlugin`/`VstChainEntry`) — now one definition,
  re-exported from `api.ts` under its existing names so nothing importing from there changes.
  `GET /vst/scan` takes no input; a missing `vst-host.exe` is `503 { error, hint }`, a scan
  failure is `500 { error }`, else `ScanPluginsResponse` (`{ plugins }`). `GET /vst/chain` takes
  no input and always returns `ChainConfig` (`GetChainResponse`) — an empty `{ plugins: [] }` if
  no chain file exists yet or it fails to parse. `PUT /vst/chain` takes `{ plugins: ChainEntry[]
  }`; a non-array `plugins` is `400 { error }`, else it saves and echoes back `ChainConfig`
  (`UpdateChainResponse`). `POST /vst/gui` takes `{ pluginPath: string, uid?: string }`; a
  missing `pluginPath` is `400 { error }`, a missing `vst-host.exe` is `503 { error }`, else
  `GuiResponse` (`{ ok, pid }`). `POST /vst/process` takes `{ inputPath: string, outputPath:
  string }`; missing either is `400 { error }`, a missing `vst-host.exe` is `503 { error }`, a
  missing `inputPath` file is `404 { error }`, a processing failure is `500 { error }`, else
  `ProcessResponse` (`{ ok, skipped: true }` when no plugin is enabled — input copied through
  unprocessed — else `{ ok, elapsed }`). Monitor sub-routes, all under `vst-host.exe`'s control
  file/status file pair: `POST /vst/monitor/start` takes `{ trackPath: string }` — missing is
  `400`, an unresolved track file is `404`, a missing exe is `503`, an empty enabled-plugin chain
  is `400`, else `MonitorStartResponse` (`{ ok, pid, plugins }`). `POST /vst/monitor/stop` takes
  no input, always `200` — `MonitorStopResponse` (`{ ok, wasRunning }`). `POST
  /vst/monitor/switch` takes `{ trackPath: string }` — missing is `400`, no monitor running is
  `400`, an unresolved track is `404`, else `MonitorSwitchResponse` (`{ ok: true }`). `GET
  /vst/monitor/status` takes no input, always `200` — `MonitorStatusResponse`
  (`{ running, paused, pid, position, duration }`); `api.ts`'s previous inline type was missing
  `paused`, which the route always sends. `POST /vst/monitor/seek` takes `{ position: number }`
  — a non-number `position` or no monitor running is `400`, else `MonitorSeekResponse`
  (`{ ok, position }`). `POST /vst/monitor/pause` and `/vst/monitor/resume` take no input, `400`
  if no monitor is running, else `MonitorPauseResponse`/`MonitorResumeResponse` (`{ ok: true }`).
  `POST /vst/monitor/restart` takes no input; no track currently loaded or a missing exe is
  `400`/`503`, an empty enabled-plugin chain is `400`, else `MonitorRestartResponse`
  (`{ ok, pid, plugins }`).
- **Seeds** (`contracts/seeds.ts`). Every route takes an optional `?subdir=` query param
  scoping to a subfolder (the UI currently always uses the flat default). `GET /seeds` returns
  `ListSeedsResponse`, inlining each seed's metadata. `GET /seeds/favorites` returns
  `ListFavoritesResponse` (favorites outside the flat default dir are silently skipped — it
  reads without a `subdir`). `GET /seeds/random` is `404 { error }` if the dir has no seeds or
  the picked file can't be read, else `RandomSeedResponse`. `GET /seeds/:name` is `404 { error
  }` if missing, else `GetSeedResponse`. `POST /seeds` takes `{ name: string, seed: number,
  description?: string, tags?: string[], subdir?: string }`; a missing/blank `name` or a
  non-numeric `seed` is `400 { error }`; `seed` is clamped to `[0, Number.MAX_SAFE_INTEGER]` and
  `description` truncated to 500 chars; a write failure is `500 { error }`; on success it
  returns `SaveSeedResponse` (`{ ok, name, seed }`). `DELETE /seeds/:name` is `404 { error }` if
  missing, `500 { error }` on an unlink failure, else `DeleteSeedResponse` (`{ ok, deleted }`)
  and also drops the name from favorites. `POST /seeds/:name/favorite` takes no body, always
  `200` — `ToggleFavoriteResponse` (`{ ok, name, favorite }`, flipping membership in the
  favorites list).
- **Download resume** (`contracts/modelManager.ts`, `POST
  /model-manager/download/:jobId/resume`). No body; a resume failure (unknown/non-resumable job)
  is `400 { error }`, else `DownloadStartResponse` (`{ jobId }`) — same contract as the initial
  `POST /download`.

Compatibility for all of the above: every change here is a type annotation (`satisfies` on an
existing `res.json(...)` call, or a `: Type` on an existing variable) and an import swap on the
client side. No route logic, validation, status code or response field changed; the point was
closing the gap between what each route actually sends and what the client's type claimed it
sends, not altering either.

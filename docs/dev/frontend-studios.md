# Studio workflows for frontend clients

Resolve runtime paths first using the [data folder configuration](config.md#dev-checkouts).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Auto-Gen, Cover, Repaint, Stem Builder's layer render and Lyric Studio
batches are all `WorkflowKind`s on the one generic envelope at
`/api/workflows`
([contracts/workflow.ts](../../server/src/contracts/workflow.ts),
[routes/workflows.ts](../../server/src/routes/workflows.ts)). Auto-Gen is the
UI's name for the studio the code, its kinds (`insta-*`) and its source tag
(`insta-gen`) still call Insta-Gen. Per-kind wire shapes live in
[contracts/studioWorkflows.ts](../../server/src/contracts/studioWorkflows.ts);
the services under `services/workflows/` and `services/lireek/` import those
schemas rather than redefining them, so the published shapes are the ones the
server validates against. Cover's YuE2 score and Build steps, and stem
separation, are plain Express routes (not `WorkflowKind`s), each in its own
section below. Create's control dictionary is on
[frontend-create-controls.md](frontend-create-controls.md); this page does not
cover it.

## The shared envelope

Every `/api/workflows` route requires `Authorization: Bearer <token>`; a
missing or invalid token is 401 before any kind runs.

**Submit:** `POST /api/workflows/jobs` with `{ kind, idempotencyKey, input }`.
`input` is validated against that kind's schema; a validation failure is 400
with `{ error, issues: [{ path, message }] }`, and an unknown kind is 400.
Resubmitting the same `idempotencyKey` with the same `input` returns the
existing job (200); a different `input` under the same key is 409. A fresh
submit is 201. The submit answers `{ job: WorkflowJob, created }`; every other
job route returns `{ job }` (or `{ jobs: [...] }` for the list, `GET
/api/workflows/jobs[?kind=&status=]`), carrying `status`, `result`, `error`,
`attempt` and the cursor fields below.

**Poll or stream:** `GET /api/workflows/jobs/:id?after=N` returns `{ job,
events, gap }` — events are only those after cursor `N`; `gap: true` means
some were dropped from the bounded window and the client should rebuild from
`job` instead of trusting the event list. `GET /api/workflows/jobs/:id/events`
is the SSE form of the same thing: a `snapshot` frame, then live `event`
frames with `id: <seq>`; a reconnect with `Last-Event-ID` (or `?after=`)
resumes without loss or repeats. The stream closes itself once the job
reaches a final status.

**Cancel and retry:** `POST /api/workflows/jobs/:id/cancel` acknowledges a
pending or running job (409 for one already final); a step that ignores the
abort signal still ends `cancelled` once it returns, and its result is
discarded. `POST /api/workflows/jobs/:id/retry` only accepts `failed`, `cancelled` or
`interrupted` jobs (a job the server left running across a restart becomes
`interrupted`, never auto-rerun) and reuses any audio queue item the earlier
attempt already produced.

**Documents:** a handful of kinds persist a revisioned draft between steps
(Auto-Gen's preview, Cover's draft) using the same plain documents the
client can read directly: `GET /api/workflows/documents/:id` returns `{
document }`; `PUT .../documents/:id` takes `{ expectedRevision, data }` and
is 409 with `{ error, currentRevision }` on a stale revision; `DELETE
.../documents/:id?expectedRevision=N` returns `{ removed: true }`. Clients
normally only read these; the kind itself writes them. These routes refuse,
with 400, the typed document kinds other domains serve (studio drafts and
the playlist, preference presets).

## Auto-Gen (`insta-preview`, `insta-direct`, `insta-approve`)

`insta-preview` and `insta-direct` share [`instaInputSchema`](../../server/src/contracts/studioWorkflows.ts)
(caption/genres/lyricMode, an LLM subject for `lyrics-ai` mode, `engineParams`
passed straight to the engine, and `expectedBackend` pinning the active
backend, a 409 if it changed before the job could run). Every key except
`systemPrompt` is required, `model` included; an empty `model` takes the
provider's default model. `lyricMode` is `instrumental`, `lyrics` or
`lyrics-ai`; `lyrics-ai` requires `provider`, and a non-empty `subject` unless
`randomSubject` is set.

- `insta-preview` resolves caption/lyrics/metadata without rendering and
  returns `{ documentId, revision, result: InstaResult }`. It also creates
  the `insta-preview` document with body `{ input, result, edits: { lyrics:
  result.lyrics, caption: result.caption } }` — `edits` starts as a copy of
  `result`, not `result` itself.
- `insta-direct` resolves and renders in one job: `{ request, audioIntentId,
  audio }`, where `audio` is the finished `/api/audio-queue` item's result.
- `insta-approve` renders from the document's `edits`, not its `result` — to
  change what renders, edit before approving: `GET
  /api/workflows/documents/:id` for the current `{ input, result, edits }`
  and `revision`; `PUT .../documents/:id` with `{ expectedRevision: revision,
  data: { input, result, edits: { lyrics, caption } } }` (the route replaces
  `data` wholesale, so resend `input`/`result` unchanged); then
  `insta-approve` with `{ documentId, revision: <the PUT's new revision> }`
  (409 `Stale preview revision N` on a mismatch). It renders the same as
  `insta-direct`, using the approved `edits` in place of `result`.

Per-item errors: none of these are batched, so a failure is the job's own
`error` field, not a per-item list. No explicit cancellation point beyond the
shared `/cancel`; retrying `insta-direct`/`insta-approve` re-enqueues a fresh
audio item only if the earlier attempt never got one far enough to reuse.

## Cover (`cover-open`, `cover-caption`, `cover-transcribe`, `cover-render`, `cover-stems`, `cover-build-save`)

A Cover session opens a `cover-draft` document and runs through it. Every
step after `cover-open` takes `{ documentId, revision }` and is 409 `Stale
cover draft revision N` on a mismatch, or `Cover source changed` if the
source asset's hash no longer matches what the draft was opened against.

- `cover-open({ assetId, cached? })` analyzes the source (metadata + BPM/key)
  unless `cached` supplies both, and returns `{ documentId, revision,
  metadata, analysis }`. `cover-open({ score: { abc } })` (instead of
  `assetId`, never both) opens a score-only draft for an imported MusicXML
  score: no recording, already approved with that score. Only a `yue2`
  `cover-render` and Build read it; every other step is 400 `Cover draft has
  no asset id`, and an `ace` render is 400 `A score-only source covers with
  YuE2 only`.
- `cover-stems({ separationId, controls: [{ index, volume, muted }] })` mixes
  a finished stem separation down on the server (the stem mixer's volumes and
  mutes) and stores the WAV as an audio asset: `{ assetId, audioUrl, filename
  }`. Open a draft on that asset with `cover-open`. 404 for an unknown
  separation.
- `cover-caption({ ...ref, artistId, provider, model, force })` resolves a
  style caption (cached from an existing generation/profile unless `force`)
  and returns `{ documentId, revision, caption }`.
- `cover-transcribe({ ...ref, force })` resolves a YuE2 ABC score for the
  source and returns `{ documentId, revision, abc, scoreSource? }`. The score
  is not yet "approved", and a new transcription clears any earlier approval.
  The bundled UI approves with `POST /api/yue2-cover/drafts/:id/approve-score`
  and `{ revision, abc }`, which checks the source and the score (non-empty,
  at most 64 KB) and writes both `abc` and `approvedAbc`; it answers `{
  document, sourceId, sourceLabel, abc }`, or 409 on a stale revision. A `PUT
  /api/workflows/documents/:id` that sets `approvedAbc` also works.
- `cover-render({ ...ref, expectedBackend, engineParams, title, lyrics,
  caption, instrumental, controls, ... })` renders on `ace` or `yue2`. `yue2`
  is 409 `Review and approve the score first` if the draft has no
  `approvedAbc`, and 400 if `controls.pairMode === 'pair'` without both
  adapter halves set. On `yue2` an optional `prepare` (the Prepare document,
  [contracts/yue2Prepare.ts](../../server/src/contracts/yue2Prepare.ts))
  must be current and built on the approved score, and must agree with
  `instrumental` (409 otherwise); its exported lyrics replace `lyrics`. Empty
  lyrics without `instrumental` are 400, and so is a backend without covers.
  On `ace`, an optional `stems` (the same `{ separationId, controls }`)
  covers the separation's stems mixed down on the server instead of the
  draft's own audio; a failed mix fails the job rather than falling back. On
  `yue2`, `controls.guide` carries the Follow source settings (audio sources
  only). Returns `{ request, audioIntentId, audio, elapsedSeconds }`, the Auto-Gen
  render shape plus the render time.

- `cover-build-save({ sessionId })` is Build's Save song: it renders the
  session's kept takes once as an ordinary song, from the params the session
  already holds (no body from the client). 409 `Keep at least one take` when
  nothing is kept, 404 for another user's or an unknown session. Result as
  `cover-render`.

### YuE2 score and Build routes (`/api/yue2-cover`)

Cover's YuE2 Prepare and Build steps are plain routes under `/api/yue2-cover`
([routes/yue2Cover.ts](../../server/src/routes/yue2Cover.ts)). Every one needs
the token; a refusal is `{ error }`. A Build session keeps its own revision.
Every `/build/:id/*` command except `cancel` and `preview` names the revision
the client saw, and is 409 when it is stale; the session rules are in
[architecture.md](architecture.md#yue2-build-sessions).

| Routes | Purpose |
|---|---|
| `GET /readiness` | Whether lead-sheet transcription can run. A supplied ABC does not need it. |
| `POST /transcriptions`, `GET` and `DELETE /transcriptions/:jobId` | A lead sheet for a source (`sourceAudioUrl` or `songId`, optional `abc`, `force`). It answers 200 `{ status: 'done', abc, sections, ... }` from a supplied ABC or a matching dataset sidecar, otherwise 202 `{ status: 'queued', jobId, ... }` to poll or cancel. |
| `POST /drafts/:id/approve-score`, `POST /drafts/:id/save-score-details` | Approve a `cover-draft`'s score (above); write the BPM and key read from a score to the matching dataset song. |
| `POST /score/remeter`, `POST /score/shift-bars` | Previews that store nothing: 2/4 to 4/4 (`{ abc, pickup }`) and bar lines moved by whole beats (`{ abc, beats }`). 422 when the score cannot be changed that way. |
| `POST /sections/review`, `/sections/match`, `/sections/plan-score`, `/sections/suggest`, `/sections/save-dataset` | Prepare's section tools: list a score's sections and check them against lyrics; match or plan them from where the source vocal sings each lyric block; suggest lyric assignments for a Prepare document; save lyrics, score, BPM or key to the matching dataset song. |
| `POST /source-metadata` | The matching dataset song's BPM, key, instrumental flag, lyrics and score, when the source is one (`{ matched: false }` otherwise). |
| `POST /drift/:songId` | Measure how far a saved YuE2 cover drifted from its score; the result is cached on the song. |
| `POST /build`, `GET /build/:id`, `DELETE /build/:id` and the `/build/:id/*` commands | Build by section: create a session from `{ cover }` (the same input `cover-render` takes, on `yue2`, with a draft reference; the server writes the params and checks them as a render would) or, as Create's Build tab does, from `{ params }` (and optional `lyrics`), then `generate`, `cancel`, `keep`, `undo`, `crop`, `trim`, `preview`, `lyrics`, `occurrence-lyrics`, `settings` and `plan`; `DELETE /build/:id/takes/:take` drops one take. `POST /build/lyrics-check` checks a lyrics draft before a session exists. |

## Repaint and Lego layer (`repaint-render`, `layer-render`)

`layer-render` is the Stem Builder studio's render. Both take a `source`
discriminated on `kind: 'asset' | 'song'`, pinned by `expectedUrl`: a 409
`Source changed; select it again` / `Source song changed; select it again`
if the asset or song's current URL no longer matches what the client picked.
Both also take `expectedBackend` and `engineParams`, and are 409 when the
active backend changed or 400 when it lacks the `repaint` or `lego` feature.
Neither has a draft document: a retry resends the original `input`.

- `repaint-render` adds `regionStart`/`regionEnd` (checked against the
  source's actual duration at run time — 400 `Region exceeds source
  duration`, or `Region end must follow start`, not at submit), `lyrics`,
  `styleCaption`, `sourceName` (the render's title is `<sourceName>
  (Repaint)`), `repaintMode` (`conservative` | `balanced` | `aggressive`) and
  `crossfadeFrames`.
- `layer-render` adds `trackName` (one of the twelve stem names), `caption`
  and `buildModel`, which must match a plain Base DiT checkpoint
  (`^acestep-v15-(?:xl-)?base-`) — 400 otherwise.

Both return `{ request, audioIntentId, audio }`.

## Lyric Studio batch (`lyric-batch`)

One job kind, not six. `lyric-batch`'s `input` is `{ items: [...], profiles?,
histories?, concurrency? }`: 1 to 10,000 items, each a discriminated union on
`type` (`profile`, `generate`, `refine`, `fetch`, `render` or
`preflight-error`). The client does not build these items. It posts its
requests to one of two Lyric Studio routes, which capture them
(`captureLyricItems`/`captureRenderItems` in
[services/lireek/lyricWorkflow.ts](../../server/src/services/lireek/lyricWorkflow.ts))
and submit the job:

- `POST /api/lireek/workflow-batches` with `{ items, concurrency?,
  idempotencyKey? }`: 1 to 1000 requests of `{ type, targetId, provider,
  model?, count? }` for `profile` (a lyrics set), `generate` (a profile) or
  `refine` (a generation), plus `extraInstructions`, `userSubject` and
  `noThink` for `generate`. A `fetch` request is `{ type: 'fetch', artist,
  album?, maxSongs? }` (default 50, at most 200). `count` (1 to 50) repeats a
  request, and the expanded batch holds at most 10,000 items.
- `POST /api/lireek/workflow-renders` with `{ intents, idempotencyKey? }`: 1
  to 200 written-song intents to render. The job's kind is `lyric-render`,
  the same input and runner with no limit on jobs at once, so a render never
  waits behind another job and songs submitted one at a time reach the audio
  queue together. Each queued render carries `meta.view` (the song, its
  artist and album) for the queue row, and `meta.lyricLink`
  (`{ generationId, lyricsSetId }`): the audio queue lists the recording
  under its song when the item succeeds, on whichever attempt, so a render
  retried from the queue is listed too. Capture also stores the `view` on
  each render item, and a render's `preflight-error` item keeps its
  `intent` and `view`. The bundled UI sends a single render as a batch of
  one.

Both need the token and answer `{ job, created }`: 201 for a new job, 200
when the idempotency key repeats it (a key is made up when none is sent), 409
when the key was used for a different batch, 400 for a malformed request.
Capture resolves each request against the library, stamps a source revision
for staleness checks, and turns anything that fails to resolve into a
`preflight-error` item instead of failing the whole submit.

The job's `result` is `{ results: [{ index, status: 'done' | 'error', value?,
error? }] }`, one entry per submitted item, in order. This is the per-item
error reporting the other kinds don't need. `value`'s shape depends on the
item: `{ id, ... }` for `profile`, `generate` and `refine`, `{ artist_id,
lyrics_set_id, songs_fetched }` for `fetch`, `{ audioIntentId, generationId,
jobId }` for `render`. A `preflight-error` item always reports `error` and
never runs. A render that failed before it reached the audio queue (a
  preflight error, a stale source, a refused enqueue) carries `unqueued: true`
  in its result entry: no queue row shows that error, so a client shows it
  from the job, and submitting the item's `intent` again is its Retry. Items whose source changed since submission (revision mismatch
against the live row) fail with `Source changed since submission` (`Written
song or preset changed since submission` for a render) rather than running
against stale data. Items run in lanes, one per artist with each render a
lane of its own, in order within a lane, so each song sees the subjects, keys
and titles the songs before it picked; `concurrency` (1 to 16, default 1)
sets how many lanes run at once. The job emits `item-start`, `item-result`
and `item-error` events, and `phase` and `chunk` events while an LLM writes.
When the batch holds only `render` (and `preflight-error`) items, the renders
are queued onto the audio queue together before any of them are awaited, so
a full-render batch is admitted as one unit. Cancellation and retry use the
shared job endpoints; retry skips items a prior attempt already completed
(recorded per `(jobId, index)`), so a partial batch resumes rather than
restarting from item 0.

## Stem separation

Unlike the kinds above, stem separation is plain Express on
[routes/stemStudio.ts](../../server/src/routes/stemStudio.ts) and
[routes/supersep.ts](../../server/src/routes/supersep.ts) — no `/api/workflows`
envelope, no auth (this is a local single-user app), no job document. Wire
shapes are in
[contracts/stemSeparation.ts](../../server/src/contracts/stemSeparation.ts);
both route files import the request schemas from there.

### Stem Separator (`/api/stem-studio`)

These are the Stem Separator studio's routes (the code calls it Stem Studio).
They run their own pipeline and keep every result on disk under
`<data root>/library/stems/<jobId>/`, independent of ace-server's job pool.

- `POST /extract` with `{ sourceAudioUrl, sourceFileName?, tracks, style?,
  lyrics?, ditSettings? }` — `tracks` is 1+ names from the twelve
  `STEM_TRACK_NAMES` (also used by `layer-render`); an unknown name is 400
  `Invalid track names: <names>`, listing every bad one, not just the first.
  Runs each track as a sequential DiT generation against the source and
  returns `{ id }` immediately — the pipeline keeps running after the
  response; poll it.
- `POST /supersep` with `{ sourceAudioUrl, sourceFileName?, level? }` runs the
  neural separator in-process and saves every stem the engine returns,
  including hidden debug stems the UI never lists. Also returns `{ id }`
  immediately.
- `GET /:jobId/progress` returns `{ status, progress, currentTrack,
  completedStems, totalTracks, warning?, error?, sepMessage? }`. Progress is
  phase-weighted for SuperSep (separation 0-80%, saving 80-100%) and
  per-track for Extract. A job whose process restarted mid-run is not
  resumed: it is gone from memory, so this 404s rather than reporting a
  phase forever; a job that finished before the restart still reports `done`
  by reading `_meta.json` off disk.
- `GET /:jobId/result` returns `{ id, type, stems: [{ trackName, category?,
  audioUrl, durationSec, index, sizeBytes, stage? }] }`; 404 `Job not found
  or not complete` until the job has a `_meta.json`. `category`/`stage` are
  present only for SuperSep stems.
- `GET /:jobId/stem/:trackName` streams that stem's WAV. `GET
  /:jobId/download-all` streams every completed stem as one ZIP.
- `GET /jobs` lists every job on disk (not the in-memory map, so it survives
  a restart), newest first. `GET /stats` returns `{ totalBytes, jobCount,
  stemCount }` for the Settings page.
- `DELETE /:jobId` cancels it if still running and deletes its directory;
  404 if the job directory never existed. `DELETE /all` cancels every
  running job and wipes `<data root>/library/stems/` entirely — used by "Clear All Stems"
  in Settings. There is no per-item error list for either pipeline: a
  failure is the job's own `status: 'failed'` and `error` string.

### SuperSep proxy (`/api/supersep`)

A separate, thinner route used by Cover Studio's splitter: no job
persistence, Node forwards to ace-server's own `/supersep/*` endpoints. The
six handlers below do NOT all relay the response the same way; each is
documented on its own because Node reshapes some of them:

- `POST /separate?level=0..5` with `{ audioUrl }` — reads that file off disk
  (`/references/...` or `/audio/...`), converts it to an engine-compatible
  format (400 `Audio conversion failed: ...` if that fails), and forwards
  the bytes to ace-server. 400 `audioUrl required in request body` if
  missing. A 2xx from ace-server comes back as its own JSON; a non-2xx comes
  back with ace-server's status and `{ error: <ace-server's response text> }`.
- `GET /:jobId/progress` always replies 200 with ace-server's decoded JSON
  body (`{ status, progress, message, error? }`) — Node does not check
  ace-server's HTTP status here, so an upstream error still arrives as a 200.
- `POST /:jobId/release` discards ace-server's response body entirely and
  always replies with Node's own `{ ok }`, at 200 if ace-server's reply was
  2xx or ace-server's status otherwise. Tells ace-server to drop a finished
  job from its resident pool (it never evicts on its own).
- `GET /:jobId/result` forwards ace-server's status and JSON body unchanged
  in both directions — 2xx `{ stems: [...] }` or the non-2xx error as-is.
- `GET /:jobId/stem/:index` proxies a single stem's WAV by its numeric
  index, not its name. On success it streams the WAV with a Node-generated
  `Content-Disposition`; on failure it does NOT forward ace-server's error
  body — it replies with ace-server's status and a fixed `{ error: 'Failed
  to fetch stem' }`.
- `POST /recombine` forwards the request body as-is — Node applies no
  schema. ace-server's real shape (`engine/tools/hot-step-server.cpp`,
  `/supersep/recombine`) is `{ id, stems?: [{ index, volume?, muted? }] }`:
  400 `Missing id` without `id`; an out-of-range or missing `index` in a
  `stems` entry is silently skipped, not rejected; 404 `Job not found`; 409
  `Job not complete` if that job hasn't finished separating. On a non-2xx,
  Node forwards ace-server's status and JSON body unchanged; on success it
  streams the resulting WAV (`Content-Type: audio/wav`, no
  `Content-Disposition`). See
  [contracts/stemSeparation.ts](../../server/src/contracts/stemSeparation.ts)'s
  `supersepRecombineRequestSchema` for the documented shape — Node does not
  validate against it, ace-server does.

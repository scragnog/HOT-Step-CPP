# Studio workflows for frontend clients

Insta-Gen, Cover, Repaint, Lego (stem layer) and Lyric Studio batches are all
`WorkflowKind`s on the one generic envelope at `/api/workflows`
([contracts/workflow.ts](../../server/src/contracts/workflow.ts),
[routes/workflows.ts](../../server/src/routes/workflows.ts)). Per-kind wire
shapes live in
[contracts/studioWorkflows.ts](../../server/src/contracts/studioWorkflows.ts);
the services under `services/workflows/` and `services/lireek/` import those
schemas rather than redefining them, so this page and the code cannot drift
apart silently. Stem separation and Create's full control dictionary are
published on their own pages; this one does not cover them.

## The shared envelope

Every route below requires `Authorization: Bearer <token>`, like the rest of
the API; a missing or invalid token is 401 before any kind runs.

**Submit:** `POST /api/workflows/jobs` with `{ kind, idempotencyKey, input }`.
`input` is validated against that kind's schema; a validation failure is 400
with `{ error, issues: [{ path, message }] }`. Resubmitting the same
`idempotencyKey` with the same `input` returns the existing job (200);
a different `input` under the same key is 409. A fresh submit is 201. The
response and every other job route return `{ job: WorkflowJob }` (or `{ jobs:
[...] }` for the list), carrying `status`, `result`, `error`, `attempt` and
the cursor fields below.

**Poll or stream:** `GET /api/workflows/jobs/:id?after=N` returns `{ job,
events, gap }` — events are only those after cursor `N`; `gap: true` means
some were dropped from the bounded window and the client should rebuild from
`job` instead of trusting the event list. `GET /api/workflows/jobs/:id/events`
is the SSE form of the same thing: a `snapshot` frame, then live `event`
frames with `id: <seq>`; a reconnect with `Last-Event-ID` (or `?after=`)
resumes without loss or repeats. The stream closes itself once the job
reaches a final status.

**Cancel and retry:** `POST /api/workflows/jobs/:id/cancel` acknowledges a
pending or running job; a step that ignores the abort signal still ends
`cancelled` once it returns, and its result is discarded. `POST
/api/workflows/jobs/:id/retry` only accepts `failed`, `cancelled` or
`interrupted` jobs (a job the server left running across a restart becomes
`interrupted`, never auto-rerun) and reuses any audio queue item the earlier
attempt already produced.

**Documents:** a handful of kinds persist a revisioned draft between steps
(Insta-Gen's preview, Cover's draft) using the same plain documents the
client can read directly: `GET /api/workflows/documents/:id` returns `{
document }`; `PUT .../documents/:id` takes `{ expectedRevision, data }` and
is 409 with `{ error, currentRevision }` on a stale revision; `DELETE
.../documents/:id?expectedRevision=N` returns `{ removed: true }`. Clients
normally only read these — the kind itself writes them.

## Insta-Gen (`insta-preview`, `insta-direct`, `insta-approve`)

`insta-preview` and `insta-direct` share [`instaInputSchema`](../../server/src/contracts/studioWorkflows.ts)
(caption/genres/lyricMode, an LLM subject for `lyrics-ai` mode, `engineParams`
passed straight to the engine, and `expectedBackend` pinning the active
backend — a 409 if it changed before the job could run). `lyricMode:
'lyrics-ai'` requires `provider`, and a non-empty `subject` unless
`randomSubject` is set.

- `insta-preview` resolves caption/lyrics/metadata without rendering, saves
  them as an `insta-preview` document, and returns `{ documentId, revision,
  result: InstaResult }`. Edit `result.lyrics`/`result.caption` client-side,
  then approve.
- `insta-direct` resolves and renders in one job: `{ request, audioIntentId,
  audio }`, where `audio` is the finished `/api/audio-queue` item's result.
- `insta-approve` takes `{ documentId, revision }` (409 `Stale preview
  revision N` if the document moved), rebuilds the render request from the
  document's result and — if the client supplied edited lyrics/caption when
  the document was created — those edits, then renders the same as
  `insta-direct`.

Per-item errors: none of these are batched, so a failure is the job's own
`error` field, not a per-item list. No explicit cancellation point beyond the
shared `/cancel`; retrying `insta-direct`/`insta-approve` re-enqueues a fresh
audio item only if the earlier attempt never got one far enough to reuse.

## Cover (`cover-open`, `cover-caption`, `cover-transcribe`, `cover-render`)

A Cover session opens a `cover-draft` document and runs through it. Every
step after `cover-open` takes `{ documentId, revision }` and is 409 `Stale
cover draft revision N` on a mismatch, or `Cover source changed` if the
source asset's hash no longer matches what the draft was opened against.

- `cover-open({ assetId, cached? })` analyzes the source (metadata + BPM/key)
  unless `cached` supplies both, and returns `{ documentId, revision,
  metadata, analysis }`.
- `cover-caption({ ...ref, artistId, provider, model, force })` resolves a
  style caption (cached from an existing generation/profile unless `force`)
  and returns `{ documentId, revision, caption }`.
- `cover-transcribe({ ...ref, force })` resolves a YuE2 ABC score for the
  source and returns `{ documentId, revision, abc, scoreSource? }`. The score
  is not yet "approved" — the client sets the draft's `approvedAbc` via `PUT
  /api/workflows/documents/:id` before rendering on `yue2`.
- `cover-render({ ...ref, expectedBackend, engineParams, title, lyrics,
  caption, instrumental, controls, ... })` renders on `ace` or `yue2`. `yue2`
  is 409 `Review and approve the score first` if the draft has no
  `approvedAbc`, and 400 if `controls.pairMode === 'pair'` without both
  adapter halves set. Returns `{ request, audioIntentId, audio }` like
  Insta-Gen's render.

## Repaint and Lego layer (`repaint-render`, `layer-render`)

Both take a `source` discriminated on `kind: 'asset' | 'song'`, pinned by
`expectedUrl` — a 409 `Source changed; select it again` /
`Source song changed; select it again` if the asset or song's current URL no
longer matches what the client picked. Neither has a draft document: a retry
resends the original `input`.

- `repaint-render` adds `regionStart`/`regionEnd` (checked against the
  source's actual duration at run time — 400 `Region exceeds source
  duration`, not at submit), `lyrics`, `styleCaption`, `repaintMode`
  (`conservative` | `balanced` | `aggressive`) and `crossfadeFrames`.
- `layer-render` adds `trackName` (one of the twelve stem names) and
  `buildModel`, which must match a plain Base DiT checkpoint
  (`^acestep-v15-(?:xl-)?base-`) — 400 otherwise.

Both return `{ request, audioIntentId, audio }`.

## Lyric Studio batch (`lyric-batch`)

One job kind, not six. `lyric-batch`'s `input` is `{ items: [...] }`, 1-200
entries, each a discriminated union on `type`: `profile`, `generate`,
`refine`, `fetch`, `render` or `preflight-error`. The client does not build
these items directly — it calls the existing Lyric Studio capture helpers
(`captureLyricItems`/`captureRenderItems` in
[services/lireek/lyricWorkflow.ts](../../server/src/services/lireek/lyricWorkflow.ts)),
which resolve each requested operation against the library, stamp a source
revision for staleness checks, and turn anything that fails to resolve into
a `preflight-error` item instead of failing the whole submit.

The job's `result` is `{ results: [{ index, status: 'done' | 'error', value?,
error? }] }`, one entry per submitted item, in order — this is the per-item
error reporting the other kinds don't need. `value`'s shape depends on the
item: `{ id, ... }` for `profile`/`generate`, `{ artist_id, lyrics_set_id,
songs_fetched }` for `fetch`, `{ audioIntentId, generationId, jobId }` for
`render`. A `preflight-error` item always reports `error` and never runs.
Items whose source changed since submission (revision mismatch against the
live row) fail with `Source changed since submission` rather than running
against stale data. `render` items are queued onto the audio queue together
before any of them are awaited, so a full-render batch is admitted as one
unit. Cancellation and retry use the shared job endpoints; retry skips items
a prior attempt already completed (recorded per `(jobId, index)`), so a
partial batch resumes rather than restarting from item 0.

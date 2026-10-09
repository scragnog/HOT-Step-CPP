# Building a replacement frontend

Everything a new client needs to drive HOT-Step over HTTP, without reading `ui/src` or any
execution service. It links the shared contracts and worked call sequences; it does not
repeat them. [`server/scripts/frontend-reference/`](../../server/scripts/frontend-reference/README.md)
is a headless reference client built from exactly these contracts, exercising the real Node
routes end to end against a fake engine and fake workers, never `ui/src`.

This page proves the contracts are enough to build and orchestrate a client under fake
dependencies. It does not qualify trainer/engine correctness, audio quality, or human
acceptance of the running app; those are separate gates, listed at the end.

## Origin, auth and media

Node serves one HTTP origin. In dev, Vite proxies `/api`, `/audio` and `/references` from
`http://localhost:3000` to the Node server on `:3001` ([HTTP API index](api.md)); a client on
another origin must configure that one server origin and prefix every relative media URL
the server returns (`/audio/<file>`, `/references/<file>`) with it.

Auth is a single local user, not a multi-tenant login:

1. `GET /api/auth/auto` returns `{ user, token }`, creating the user on first call. No
   credentials.
2. Send `Authorization: Bearer <token>` on routes that need it.
3. Tokens live in server memory only. A server restart invalidates every token; the next
   authenticated call answers `401 { error: 'Unauthorized' }`, so a client fetches a new
   token and retries.

Full detail, including which routes need no token at all (status, cancel, media mounts), is
in [Local auth](api-contracts.md#local-auth).

## Discovery

[`docs/dev/api.md`](api.md) is the generated index of every route, grouped by mount prefix,
rebuilt from the actual `router.<verb>()` calls and `index.ts` mounts — never out of date with
the code that defines it. It names the file but not the request/response shape; those are in
the domain pages this one links below.

`GET /api/capabilities?backend=<id>` reports what the active backend actually supports
(`core`, `features`, `extensions`), so a client can branch on capability flags instead of a
hardcoded backend id. See [Engine capability manifest](api-contracts.md#engine-capability-manifest).

## Errors and versioning

Every error is JSON with an `error` string, occasionally extra fields (`expectedBackend`,
`currentRevision`, `issues` from zod). See [Errors](api-contracts.md#errors) for the generate
route's table; the same shape (plain `error`, sometimes `currentRevision` on a `409`) recurs
across the revisioned document routes (playlist, drafts, presets).

There is no HTTP contract version number. New response fields are additive; a client must
ignore ones it does not recognise. A document or job has its own `revision`/`schemaVersion`,
which is a conflict and upgrade mechanism for that one row, not an API version
([Versioning](api-contracts.md#versioning)).

## Jobs: polling and resuming

Three independent job mechanisms exist. Picking the wrong one for a given route is the most
common mistake a new client makes:

| Mechanism | Routes | Survives a restart? | Resume |
|---|---|---|---|
| Legacy generation job | `/api/generate/status/:id`, `/cancel/:id` | No — lost from memory. `404` means "gone", not "never existed" | None; resubmit |
| Durable audio queue | `/api/audio-queue/*` | Yes — items persist in the database | Items left `submitting`/`submitted` become `interrupted`; only an explicit `POST /items/:id/retry` runs them again |
| Workflow jobs/documents | `/api/workflows/*` | Yes | `GET /jobs/:id/events?after=N` resumes an SSE stream with `Last-Event-ID`; an interrupted job needs `POST /jobs/:id/retry` |

None of these replay or continue an interrupted run by itself. A client must poll or
reconnect, see the terminal or interrupted state, and explicitly retry; nothing reruns
automatically. See [Jobs](api-contracts.md#jobs), [Durable audio queue](api-contracts.md#durable-audio-queue)
and [Workflow jobs and revisions](api-contracts.md#workflow-jobs-and-revisions).

## Create's queue ownership

A fresh Create submission goes through one of two paths, and a new client must pick the
right one rather than assume either. The browser decides with one `localStorage` key,
`lireek-audio-queue-owner-v1` (`ui/src/stores/audioGenQueueStore.ts`):

| Value | Meaning | Submission path |
|---|---|---|
| Missing or any value other than `server`/`migrating` | Browser queue (the default) | The browser holds the queue client-side and submits each item directly to legacy `POST /api/generate`, polling `GET /api/generate/status/:id` and cancelling with `POST /api/generate/cancel/:id` |
| `server` | Durable Node ownership | The browser submits through [`/api/audio-queue`](api-contracts.md#durable-audio-queue) instead; Node holds the queue and survives a browser close or server restart |
| `migrating` | Handoff in progress | Submissions are refused (`Audio queue migration is in progress`) until the migration guard resolves it to `server` (success) or back to `browser` (failure) |

A client that has never run the migration, or that reads an unset/unrecognised key, is on
the browser path — it is never silently treated as server-owned. The key, its default and
its three values are an existing browser decision this documentation does not change; a new
client does not migrate another client's queue, and nothing here is a contract to add a
fourth value or move the default.

Both submission paths are exercised with fake execution in `client.test.ts`: the legacy
direct-submit path and the resolved durable-queue path (`client.test.ts:58`, resolve ->
`/api/audio-queue` -> a finished generation). Node cannot prove which `localStorage` value a
real browser would read or write; that remains a UI-side behavioural gate, not something an
HTTP-only client can demonstrate for itself.

## Import and export

Covered in full in [Media for frontend clients](frontend-media.md): upload an asset, import
it into the library, resolve what to export, download each resolved URL. Per-item requests
(import, export) report success or error per item, in order; one bad item never stops the
rest.

## The full catalogue

Every audited workflow group, its contract page and its reference-client coverage:

| Group | Contracts | Reference coverage |
|---|---|---|
| Create | [Create's control dictionary](frontend-create-controls.md), [Generate request](api-contracts.md#generate-request) | `client.test.ts` |
| Insta-Gen | [Studio workflows](frontend-studios.md#insta-gen-insta-preview-insta-direct-insta-approve) | `client.test.ts` |
| Lyric Studio | [Studio workflows](frontend-studios.md#lyric-studio-batch-lyric-batch) | `client.lyric.test.ts`, `client.lyric.chain.test.ts` (full chain, fixture orchestration — live LLM/Genius unverified, see "What this page cannot prove") |
| Cover | [Studio workflows](frontend-studios.md#cover-cover-open-cover-caption-cover-transcribe-cover-render) | `client.test.ts` (open only), `client.cover.test.ts` (full chain) |
| Repaint | [Studio workflows](frontend-studios.md#repaint-and-lego-layer-repaint-render-layer-render) | `client.test.ts` |
| Stem separation | [Studio workflows](frontend-studios.md#stem-separation) | `client.test.ts` |
| Stem Builder | [Studio workflows](frontend-studios.md#repaint-and-lego-layer-repaint-render-layer-render) | `client.test.ts` |
| Song Builder | [Library, playlist, drafts and Song Builder](frontend-library.md#song-builder) | `client.library.test.ts` |
| ACE training | [Training for frontend clients](frontend-training.md) | `client.training.test.ts`, `client.trainingRuns.test.ts` |
| MM3 training | [Training for frontend clients](frontend-training.md) | `client.training.test.ts`, `client.trainingRuns.test.ts` |
| YuE2 training | [Training for frontend clients](frontend-training.md) | `client.training.test.ts`, `client.trainingRuns.test.ts` |
| Library | [Library, playlist, drafts and Song Builder](frontend-library.md#songs) | `client.library.test.ts` |
| Playlists | [Library, playlist, drafts and Song Builder](frontend-library.md#playlist) | `client.library.test.ts` |
| Studio drafts | [Library, playlist, drafts and Song Builder](frontend-library.md#studio-drafts) | `client.library.test.ts` |
| Presets | [Preference presets](frontend-presets.md) | `client.library.test.ts` |
| Import/export | [Media for frontend clients](frontend-media.md#importing-audio) | `client.library.test.ts` |
| Streaming/recording | [Media for frontend clients](frontend-media.md#live-audio-streams) | `client.streaming.test.ts`, `client.mm3stream.test.ts` |
| Settings/backends | [Engine capability manifest](api-contracts.md#engine-capability-manifest) | `client.library.test.ts` |

Restart interruption, where it applies, is exercised separately in `client.restart.test.ts`
against the real training-job and workflow-job persistence, not simulated.

## What this page cannot prove

- **Lyric Studio's real LLM provider and Genius calls.** `generate`/`refine` call a real LLM
  provider and `fetch` calls the real Genius API; `lyricWorkflow.ts`'s
  `overrideLyricWorkflowDeps()` lets `client.lyric.chain.test.ts` exercise the full
  fetch → profile → generate → refine → render chain, a mixed-batch per-item failure, a
  stale-revision refine, and a cancel-and-retry, all for real against the production routes
  and persistence — but Genius and the LLM themselves are fixtures. Whether the real Genius
  API or a real LLM provider returns usable output is unverified by this harness.
  `client.lyric.test.ts` separately proves auth failure, rejected input, an idempotency
  conflict, and that cancelling an already-finished or unknown job is refused (409/404).
- **Browser-only behaviour.** Which `localStorage` value a real browser reads, writes or
  migrates; crossfade/gapless playback; device-capture recording. These need a browser, not
  an HTTP client.
- **Remote training workers.** Worker registration, token auth and mirrored runs are a
  separate worker-to-Node contract, not covered by a reference HTTP client acting as the
  single local user.
- **Runtime and listening quality.** A fake engine proves route/contract orchestration under
  fake dependencies. It does not prove the real engine renders correctly, or that the result
  sounds right. That is a human acceptance gate, run against the real app with `dev.bat`.

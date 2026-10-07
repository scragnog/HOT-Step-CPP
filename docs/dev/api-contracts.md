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
| `core` | Model-agnostic parameters it honours: `duration { max, auto, editable? }`, `bpm`, `keyscale`, `negativePrompt`, `batch { max }`, `seed`, plus backend-declared extras |
| `features` | One boolean per feature or studio, such as `cover`, `repaint`, `stems`, `streaming`, `adapters`, `lmAdapters`, `lmAdapterSelectable`, `whisper`, `forcedAlignment` |
| `extensions` | Backend-declared knobs, rendered by the shared plugin-parameter renderer and grouped by `group` (`generation` or `lm`) and optional `section` |
| `license` | Optional licence text, shown verbatim |

`GET /api/backends` lists `{ id, displayName, resourcePool, active }` and `activeId`.
`POST /api/backends/active { id }` switches backends.

Shared code should branch on these flags, not on a backend id. The UI does not yet do this
everywhere; changes to the manifest are proposed separately before they land here.

## Caption and content resolution

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

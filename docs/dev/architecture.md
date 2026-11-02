# Architecture

Resolve runtime paths first using the [data folder configuration](config.md#dev-checkouts).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

HOT-Step CPP is a desktop app for local music generation. It has three tiers: a C++
inference engine, a Node server that owns the database and orchestrates the engine, and a
React UI in the browser. This page shows how they fit together and where each feature lives
in the source tree. Read it before you pick a file to edit.

## The three tiers

| Tier | Stack | Location | Role |
|---|---|---|---|
| Engine | C++17, CUDA/Vulkan/Metal/CPU through GGML | `engine/` | Inference binaries. `ace-server` is the HTTP server the app talks to. Other tools: `ace-train`, `ace-caption`, `ace-midi`, `mastering`, `mp3-codec`, `neural-codec`, `quantize`, `vst-host`, and the upstream CLIs `ace-lm`, `ace-synth`, `ace-understand` |
| Server | Node 20 to 24, TypeScript, Express, better-sqlite3 | `server/src/` | Job queue, SQLite, file storage, spawns and restarts the engine, serves the built UI |
| UI | React 19, Vite, TanStack Query, Zustand, Tailwind | `ui/src/` | Browser frontend, Query server snapshots and client stores, one component folder per studio |

```
Browser
  |  dev (agents):  http://localhost:3000  (Vite HMR, proxies /api /audio /references to :3001)
  |  prod (end users): http://localhost:3001
  v
Node server   server/src/index.ts   Express on 0.0.0.0:3001
  |-- /api/*         route files in server/src/routes/
  |-- /audio         <data dir>/audio         (generated songs)
  |-- /references    <data dir>/references    (mastering reference tracks)
  |-- /listening     ear-test score sheets under _experiments/_LISTENING/ (loopback only)
  |-- /*             ui/dist, when it exists, with an SPA fallback to index.html
  |-- SQLite         <data dir>/hotstep.db
  |-- runs as needed: ace-train, ace-caption, ace-midi, mastering, mp3-codec,
  |                   vst-host, whisper-cli, Essentia, sd-cli, ffmpeg
  v  HTTP, server/src/services/aceClient.ts
ace-server    engine/tools/hot-step-server.cpp   127.0.0.1:8085
  |-- ACE-Step 1.5      /lm /synth /understand /vae /warm /job ...
  |-- MiniMax-Music3    /mm3/*
  |-- YuE2              /yue2/*
  |-- Stable Audio 3    /sa3/props /sa3-synth
  `-- SuperSep, Spectral Lifter, PP-VAE re-encode, SA3 refine
```

| Service | Default port | Set by |
|---|---|---|
| Node server | 3001 | `SERVER_PORT` |
| Vite dev server | 3000 | hard-coded in `ui/vite.config.ts` and `dev.bat` — the address agents use |
| ace-server | 8085 | `ACESTEPCPP_PORT` |

The `<data dir>` is `<data root>/library/` when there is a data folder, and the `DATA_DIR`
default `server/data/` in a checkout that has none. [config.md](config.md) has every
variable, including how `DATA_DIR` resolves.

## UI server state

`ui/src/main.tsx` provides one HMR-stable QueryClient above AuthProvider and App.
Query owns server lists, details, capabilities and job snapshots. Zustand,
external stores and React state retain forms, playback, selections, unsaved
edits and workflow execution. Editors can seed a working copy from a snapshot;
background reads must preserve unsaved typing. Compatibility projections select
the captured cache entry and must not become independent fetch owners.

The [server-state contract](ui-server-state.md) defines the client, keys,
transport signatures, retry policy and SSE bridge. Keys contain API base,
resource, id and normalized filters, plus an opaque auth epoch and actual
backend/worker only where relevant. Credentials stay out of keys. A scheduled
training read captures its worker with its key and passes that target and
Query's AbortSignal into the transport. Local run/review/mirror records omit
worker scope. An auth change cancels and removes the previous epoch's queries.

Default freshness is 30 seconds and retention is five minutes. Focus does not
refetch; reconnect does. Reads retry network/5xx failures at most twice, never
aborts or 4xx. Mutations never retry automatically. Active job polling normally
runs every 1.5 seconds, stops at terminal status and backs off failed cycles up
to 30 seconds. Resource overrides and runtime steps live in each area's README:

| Area | Snapshot ownership and retained client work |
|---|---|
| [Global models](../../ui/src/queries/global-models/README.md) | Plugins, capabilities, VRAM, catalogues, model downloads, profiles, seeds, references and VST snapshots; parameter forms and selection intent stay local |
| [Training data](../../ui/src/queries/training-data/README.md) | `queries/training-data/queries.ts` owns datasets, detail, jobs, preparation status, capabilities, workers, pipelines and batches; the training store projects only the selected worker/dataset and retains pending edits |
| [Training runs](../../ui/src/queries/training-runs/README.md) | Worker status/jobs, local run/review/score/preparation records; charts/logs retain SSE reducers, form fields and playback stay local |
| [SA3 previews](../../ui/src/queries/sa3-previews/README.md) | One run's SA3 checkpoint preview list, keyed under training-runs by worker, dataset and run; it polls while a cell is queued or rendering, and the render command never retries |
| [Studios](../../ui/src/queries/studios/README.md) | MIDI jobs/notes, extraction/separation, projects, workflow snapshots and shared songs/plugins/models; composition, audio scheduling and explicit execution stay local |
| [Cover and lyrics](../../ui/src/queries/cover-lyrics/README.md) | Catalogues, sources, profiles, presets, drafts, playlists and workflow snapshots; lyric edits and serial execution stay local |
| [Shell and queue](../../ui/src/queries/shell-queue/README.md) | Library, recent songs, settings, preferences, browse results, generation status and queue projections; routing, playback, migration and completion effects stay client-owned |

Commands capture their target before awaiting, then update or invalidate related
list/detail keys after success. A failed command leaves the old snapshot visible.
Deletion cancels old reads before the command and again after success, removes
the accepted rows and invalidates. Canceling observation aborts an HTTP read;
canceling a server job requires its explicit command. The shared write helpers are in
`queries/writes.ts` (`writeAndInvalidate`, `useUpload`), and `queries/invalidation.ts`
holds the invalidation manager they call.

JSON reads and commands go through one axios instance (`services/axiosClient.ts`). Its
request interceptor adds the bearer token, and its response interceptor turns every
failure into an `ApiError` and decides the toast: failed GETs stay silent, other failures
toast unless the request opts out with `X-Skip-Global-Error`. Uploads stay on XHR for
progress events and SSE streams stay on fetch. Uploads and streams that use
`sseFailure()` in `services/httpClient.ts` report failures through the same handler.

SSE remains the push channel. Shared log/download/training connections and
workflow streams keep their existing reducers. Scoped bridges reconcile snapshots
on events and reconnect, coalesce bursts and discard late events after teardown.
Replay cannot repeat a submission or completion effect. MIDI's note stream
retains its own live-note reducer and closes on unmount.

The generation executor keeps serialization, IndexedDB persistence, manual rows
and legacy server/browser ownership rules. Its existing projection loop schedules
Query reads; it is not replaced by a second poller. Queue reads suppress Query
retries because the executor already handles failures. Delivery is claimed only
after the song read succeeds and occurs once per page. This cache migration adds
no server durability or permission changes. Remaining integration gaps are
explicitly recorded in the audit below.

## UI server-state integration audit

Audited implementation `a9748d5d`, including the foundation's historical
manifest, current production service imports, direct fetches, stream openers,
intervals and imperative store reads. Test files and test fakes are excluded
from the production scan. The historical manifest remains with its foundation
owner in [ui-server-state.md](ui-server-state.md#consumer-manifest); the
reconciliation below records current dispositions without editing that owner's
file. Historical counts describe files, not remaining HTTP calls.

### Requests to area owners

R1-R6 code reconciliation is complete in HOTSTEP-107 (`8909d3b3` and
`d5a47ded`), HOTSTEP-108 (`a7a4536e`, accepted after Builder review) and
HOTSTEP-109 (`0b000e03`, awaiting Tester review). Runtime acceptance remains
with HOTSTEP-106; these closures do not claim live request-count reductions.

| Request | Owner | Resolution |
|---|---|---|
| R1 | Engineer, training-data; Maker, training-runs | Closed in `8909d3b3` and `0b000e03`. Preparation follow and run panels share local `yue2-preparation` and `yue2-preparation-job` reads. Sequencing, terminal stop, captured selection guards and cancellation remain with the executor |
| R2 | Engineer, training-data | Closed in `8909d3b3` and `d5a47ded`. Browse, scan preview, caption files, export preview and local audition revision reads enter Query. The training-data README classifies command-time reads. Caption forms seed once per selection visit, including A to pending B to A under StrictMode |
| R3 | Engineer, shell-queue; Maker 2, cover-lyrics | Closed in `8909d3b3` and `a7a4536e`. `SONG_VIEWS` includes Cover's feed and both recent-song feeds; Cover writes/deletes and workflow events call the shared invalidator. HOTSTEP-108 passed Builder review |
| R4 | Engineer, foundation; Maker and Maker 2, area adoption | Closed in `8909d3b3`, `a7a4536e` and `0b000e03`. Shared GPU, adapter-scan and workflow keys use the documented canonical reads and wire shapes. Workflow jobs retain auth and `after` filters; events invalidate all offsets for the captured job. Training commands/events also invalidate shared preparation snapshots |
| R5 | Engineer, shell-queue; Maker, studios | Closed in `8909d3b3` and `0b000e03`. Peaks and player sidecars enter shared Query entries; Repaint uses the same LRC key and `fetchLrc` read. Track variants share the sidecar URL, missing files remain null, and crop/retranscription invalidates sidecars. Audio blobs and decoding remain media transports |
| R6 | Engineer, foundation | Closed in `8909d3b3`. The foundation manifest records current dispositions and additional consumers, publishes the shared contracts and corrects the song-detail auth note. The key retains an epoch for sign-out cleanup. At the time its read sent no bearer token; since the axios transport (`18897e6c`) it does |

Files in R2 are under `ui/src/components/training-studio/`; other paths above
are under `ui/src/`. The preparation context read at submission, recipe/preset
resolution and scan commands used to construct a submission can remain explicit
execution operations when documented as such. That does not exempt repeated
preparation detail/job polling or a displayed server snapshot.

### Surviving fetches, streams and timers

At the audit, all remaining direct fetch sites fell into the groups below. A service
transport does not independently own a snapshot: its caller's Query options supply that
ownership. R1, R2 and R5 now follow this rule through the shared reads.

Later changes are not re-audited here. The axios transport (`18897e6c`) moved every
service file's JSON calls and uploads onto `httpClient.ts`, so the raw fetches left under
`services/` are the transport itself and three stream readers (assistant chat, the
lireek SSE helper and workflow events). `components/compare-studio/useAbEngine.ts`
(media transport) and `components/storm/DriftPanel.tsx` were added after the audit
and fetch directly; they have no disposition yet. Rows below that name raw fetches in
service files, upload components, MetadataEditorModal or `queries/studios/lyrics.ts`
describe the audit, not the current code.

| Paths under `ui/src/` | Disposition and reason |
|---|---|
| `services/httpClient.ts`, `api.ts`, `trainingApi.ts`, `dataFolderApi.ts`, `audioQueueApi.ts`, `workflowApi.ts`, `preferencesApi.ts`, `trainingOperations.ts` | Shared/domain HTTP transports. Query owns read snapshots; POST/PUT/PATCH/DELETE commands stay explicit. Workflow event fetch is a resumable stream |
| `services/{midiStudioApi,stemStudioApi,supersepApi,inspireApi,streamSessionsApi}.ts` | Raw fetches that remain are upload/start/control/cancel/delete commands. GET snapshots feed studios Query options; progress waiters use Query observers |
| `services/{lireekApi,lyricWorkflowApi,coverWorkflowApi}.ts` | Command and streaming transports; lireek and draft read wrappers enter cover-lyrics Query. Approval/save operations are mutations |
| `services/exportImportApi.ts` | Import, export resolution and profile validation commands; downloaded artifacts are media transport |
| `queries/studios/lyrics.ts` | Shared `lrc` key and canonical `fetchLrc` read, also used by player consumers |
| `App.tsx` | Timeout-bound health probes are liveness transport for offline/restart recovery, not cached feature data. Extract-kick/shutdown/restart are commands. Log presence is a shared stream. The health interval and animated-dot interval retain those purposes |
| Cover, MIDI, Repaint, Stem source/builder and shared LatentImport components | Multipart audio/latent upload commands; Cover's analyze POST is an explicit analysis command |
| Details MetadataEditorModal, SettingsPanel, DataFolderPanel | Cover upload and restart commands; displayed settings use Query |
| Global bar SeedManagerDrawer, Yue2LmAdapterDropdown; `stores/{backendStore,globalParamsStore}.ts` | Seed writes, adapter import, model/backend selection and active-model mirror commands; random seed GET chooses a new value explicitly |
| `components/library/CoverArtPromptModal.tsx` | Prompt-preview POST seeds editable command input; generation POST starts once; job status uses Query |
| `components/lyric-studio/CuratedProfileModal.tsx` | Streaming profile-build command; completion invalidates source snapshots |
| `hooks/useStreamAudio.ts`, `stores/mm3StreamStore.ts` | Audio/text stream transport, control and stop commands; recording interval is a local clock |
| `components/shared/StemMixer.tsx`, queue audio fetch in `stores/audioGenQueueStore.ts` | Binary audio transport/decoding, not server metadata snapshots |
| `stores/discoStore.ts` | Per-track disco sidecar decoded into an audio-reactive playback buffer; media transport |
| `audio/peaksClient.ts`, player lyric readers, LRC utilities | Shared Query peaks, LRC and word-timing snapshots; utilities supply canonical transports |
| `components/training-studio/sendAuditionToCustomGen.ts` | Local Query dataset revision read before the explicit draft command |
| `services/sharedEventSource.ts` | Shared EventSource transport with subscription teardown |
| `components/midi-studio/MidiPlayer.tsx` | Live-note EventSource and 200 ms note-buffer flush timer; terminal notes use Query |
| `components/midi-studio/midiSynth.ts`, `components/training-studio/TrainingRunStats.tsx`, `utils/heapWatch.ts` | WebAudio scheduling, elapsed-time display and heap sampling intervals; no server reads |
| `stores/audioGenQueueStore.ts` | Executor only: serialized execution loops and one-second status reuse. The server projection cadence lives in `queries/shell-queue/projection.ts` (1.5 s active, about 10 s idle) |
| `hooks/usePolling.ts` | Removed, with no surviving network callers |

Imperative `getState`/`getSnapshot` reads were also checked. Global-parameter
assembly, profile application, playback-engine snapshots and form selections
are client reads. `backendStore` capability/model fields and VST chain fields
are Query compatibility projections; `fetchModels` routes through Query.
Create's backend refusal, Cover/Lyric adapter defaults and queue parameter
capture consume those projections without starting a second read owner.
Training's PipelineStageProgress adopts the running stage into the studio's one
job projection, which captures worker/selection; its job refresh enters Query.
Stream metrics, preview buffers and pending sample edits remain
reducers/client state.
The preparation executor and run panels share local Query detail/job snapshots.

### Verification and remaining acceptance

After `0b000e03`, `npx tsc -b` and all 277 UI tests pass, including shared
adapter scan, GPU/preparation and LRC request tests and workflow replay-offset
invalidation. The docs check passes with the same 11 screenshot warnings.

At the original audited implementation, `npx tsc -b` passed in `ui/`. The full UI suite
passed 265 tests, zero failures/skips, using:

```powershell
node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-force-exit --test-timeout=60000 "src/**/*.test.ts"
```

The force-exit flag bounds retained app handles; the per-test timeout bounds a
stuck test. Existing queue tests register their CSS loader.
`node tools/docs/check-docs.mjs` passed after these doc edits, with 11 existing
missing-screenshot warnings. Passing unit checks does not
establish live request reductions. R1-R6 code closures and review owners are recorded above. The assigned runtime job on
port 3000 still owns baseline comparisons, simultaneous consumers, tab return,
delayed MIDI deletion and worker-switch replies, download completion, VST failed
load retention and generation progress/delivery. This docs slice starts no app.
Unavailable worker/GPU/download cases and human visual/listening feedback must
remain explicit acceptance gaps.

## How the server runs the engine

`server/src/services/aceEngineProcess.ts` owns the `ace-server` child process. On startup
`index.ts` calls `startAceServer()`, which passes `--models`, `--host` and `--port`, plus
`--adapters`, `--user-plugins`, `--keep-loaded`, `--noise-profile`, `--draft-lm`,
`--vae-chunk`, `--vae-overlap` when their settings apply. The full list is in
[config.md](config.md#flags-passed-to-ace-server).

- An abnormal exit respawns the engine after 3 seconds. Three crashes inside 30 seconds
  stop the respawns and mark the engine not ready. The usual cause is a missing DLL next to
  the binary. After giving up, the engine starts again by itself when a Model Manager
  download queue drains (`retryIfGaveUp`).
- An empty models folder is not a crash. `ace-server` prints `ERROR: no models found` and
  exits 1; the server spends none of the crash budget, reports `engine.noModels` on
  `/api/health`, and starts the engine when a download queue drains or a finished `.gguf`
  appears in the models folder (checked every 5 seconds).
- Training jobs stop the engine on purpose to free the GPU (`stopAceServer`) and start it
  again afterwards (`restartAceServer`). A deliberate stop never triggers a respawn.
- MiniMax-Music3 and YuE2 keep their model selection inside the engine process, so a
  restart loses it. `index.ts` registers restore hooks through `onEngineRestarted` that put
  the saved selection back after every respawn. Stable Audio 3 needs none: its DiT choice
  is saved in SQLite and sent with each request.
- In a portable Windows CUDA build, the first launch downloads the cuBLAS and cudart DLLs
  from Hugging Face before it starts the engine. If that fails, the engine starts on CPU.

Because of the respawn logic, killing `ace-server` from outside while Node is running
starts it again straight away. Rebuild the engine with `dev-rebuild.bat`, which asks Node
to shut down first. See [building.md](building.md#engine-rebuild-rules).

## A generation request, end to end

1. The UI builds a camelCase parameter object from the Create panel and the global bar
   (`ui/src/stores/globalParamsStore.ts`, `ui/src/types.ts`) and posts it to
   `POST /api/generate` through `ui/src/services/api.ts`. For Create and written-song
   requests the client queue (`ui/src/stores/audioGenQueueStore.ts`) first posts the
   intent to `POST /api/resolve/preview` (`server/src/routes/resolve.ts`), which returns
   the exact `/api/generate` body with `expectedBackend` set, unless
   `GENERATION_INTENT_PATH=old`. When the server owns the queue, that body is enqueued on
   `/api/audio-queue` instead of posted directly. [api-contracts.md](api-contracts.md)
   has the shapes.
2. `server/src/routes/generate.ts` checks that the engine is ready, builds a job envelope
   that names the backend (`services/generation/envelope.ts`), and queues the job. Jobs run
   one at a time on a shared GPU lane (`services/generation/gpuLane.ts`). An optional
   `backendOverride` runs one job on a named backend without switching the active one, and
   an `expectedBackend` that no longer matches the backend the job would run on is refused
   with 409.
3. The route hands the job to the backend's `generate()`. For ACE-Step that is
   `services/backends/ace/generate.ts`:
   - `translateParams()` turns the UI params into a snake_case `AceRequest`
     (`services/generation/translateParams.ts`, type in `services/aceClient.ts`).
   - The LM phase posts to the engine's `/lm`, unless the LM cache
     (`services/generation/lmCache.ts`) already has codes for these LM params.
   - The synth request is rebuilt as `{ ...aceReq, <LM output fields> }`. Fields the Node
     server adds do not survive the `/lm` round trip, so never copy the LM response forward.
     The `generation-request-flow` skill explains this trap in detail.
   - Each track goes to `/synth`, as JSON or multipart when source audio or latents are
     attached.
   - The WAV or MP3 is saved, auto-trim and the post-processing chain run
     (`services/generation/postProcessing.ts`), and a row is inserted into `songs`.
4. The engine accepts each `/lm` or `/synth` as an async job and returns a job id. The
   server polls `GET /job?id=<id>` and fetches the result with `GET /job?id=<id>&result=1`.
   The polling loop is `services/generation/pollUntilDone.ts`, which also holds the stall
   and wall-clock watchdogs. YuE2 supplies the submitted NAR chunk frame cap,
   or 9000 frames for an uncapped song. During NAR and the completed semantic
   handoff, the silence allowance is 400 ms per frame plus 2 minutes, with a
   15-minute floor and a 9000-frame ceiling. Actual phase/step changes reset
   the timer; repeated running responses do not. YuE2 decode and output phases
   (`vae_decode`, legacy `vae`, and `encoding_output`) can be silent for a whole
   batch, so they use the user-configured wall-clock timeout alone. Downstream
   post-processing starts after this polling loop returns. Other phases retain
   their existing inactivity limits.
5. The UI polls `GET /api/generate/status/:id` until the job finishes.

`ace-server` answers HTTP on one thread. While it runs DiT, VAE or adapter work it cannot
answer at all, so a slow `/health` in the middle of a generation is expected. The client
timeouts in `aceClient.ts` are long for that reason.

## Backends

The server can drive five model families through one interface. The registry is
`server/src/services/backends/registry.ts`, and the interface (`EngineBackend`) is in
`services/backends/types.ts`. Registration order is the backend toggle's order.

| Id | Display name | Server code | Engine code |
|---|---|---|---|
| `ace` (default) | ACE-Step 1.5 | `services/backends/ace/` | `engine/src/` core pipeline |
| `minimax-m3` | MiniMax-Music3 | `services/backends/minimax/` | `engine/src/minimax/`, `/mm3/*` |
| `yue2` | YuE2 | `services/backends/yue2/` | `engine/src/yue2/`, `/yue2/*` |
| `sa3` | Stable Audio 3 | `services/backends/sa3/` | `engine/src/sa3-*.h`, `/sa3/props`, `/sa3-synth` |
| `heartmula` | HeartMuLa | `services/backends/heartmula/` | `engine/src/heartmula/`, `/heartmula/*` |

Each backend reports a capability manifest (duration limits, whether it has an LM, plugins,
adapters and so on). The UI reads it from `GET /api/capabilities` and hides controls a
backend does not support (`ui/src/stores/backendStore.ts`,
`ui/src/components/shared/BackendCapabilityGate.tsx`). The active backend id is stored in
the SQLite `settings` table under `active_backend_id` and switched with
`POST /api/backends/active`. User-facing detail is in [../user/backends.md](../user/backends.md).

### YuE2 build sessions

Build by section keeps a session file per build under `<audioDir>/takes/<id>/`
(`server/src/services/yue2Takes.ts`). Its shape and rules are defined once, in the
header of `server/src/services/backends/yue2/coverArrangement.ts`; read that before
changing sessions.

- **Version 2 sessions.** A session carries `version`, a `revision` and an
  `arrangement` (source sections `s0..`, the planned output occurrences `o0..`). A file
  without a version is upgraded once on first open by `migrateLegacySession`.
- **Revision.** It counts changes to what the next take would be sent: the plan, the
  lyrics and the settings. Every take records the revision and the parent (the
  spine it resumed). Every mutation (generate, keep, undo, crop, trim, delete a take,
  lyrics, occurrence lyrics, settings, plan) names the revision the page saw, and a
  mismatch is refused with 409. Keep also needs the take's revision and parent to match.
  Keep and undo never change the revision, so a take made from an earlier spine is
  keepable again after undoing back to it.
- **Settings.** `POST /api/yue2-cover/build/:id/settings` changes the cover's voices,
  tempo, key, key shift, CFG scale and AR strength. It works only before the first keep
  and while no takes are pending, and bumps the revision.
- **Delete.** `DELETE /api/yue2-cover/build/:id` removes the session folder. It is
  refused with 409 while takes or a build-so-far render are running, or while a Save song
  render for the build is queued or running.
- **Build lyrics.** The Build page owns its lyrics. `POST /api/yue2-cover/build` takes
  an optional `lyrics` that replaces `params.lyrics`, and
  `POST /api/yue2-cover/build/:id/lyrics` replaces them later: it rewrites
  `params.lyrics` and the stored engine request together (the route rebuilds the
  request from the new params, as a render would), bumps the revision, and is refused
  while takes are pending. Lyrics of sections already kept are locked, so kept audio
  stays valid; text after them is free. Prepare lyrics and dataset sidecars are never
  written. Until a session exists the text is a draft in the Cover Studio state,
  checked by `POST /api/yue2-cover/build/lyrics-check`.
- **Arrangement.** `POST /api/yue2-cover/build/:id/plan` replaces the occurrences not
  yet built: `upcoming` lists source section ids (a new occurrence), upcoming
  occurrence ids (kept identity) or `improv:<bars>` (2, 4, 8 or 16 bars), and `finished`
  says the plan ends the song. Kept occurrences never move. An improv occurrence adds
  that many bars of rests labelled Improv to the score, and its take composes them
  without the score. A plan edit moves takes whose own occurrences it leaves alone to the
  new revision, so they stay keepable; a take that composed an ending stays keepable only
  while the plan still ends after it. While the plan is open, a take's prompt adds the source
  section after the last planned one (`lookaheadSection`) and stops a few seconds into
  it, so the seam is composed without the user choosing what follows. An occurrence
  can carry its own sung lines (`POST /api/yue2-cover/build/:id/occurrence-lyrics`,
  unbuilt occurrences only), which `composeArrangement` writes in place of its block's.
  Pairing sections with lyric tags counts empty tags, so of two same-kind sections the
  instrumental one keeps Prepare's empty tag. The session keeps its score and lyrics in source order; every take and
  Save song sends them written out in plan order (`composeArrangement`), with any lyric
  schedule rebuilt from that text, or dropped when re-pairing the text would give a
  block to the wrong occurrence. A take's `from`, `to` and `starts` index occurrences;
  alignment, the take length cap and undo use the same indexes. The source order
  sends the stored text unchanged. Repeats make the prompt longer; a request past the
  model context fails with the engine's error in the session's `error`.

## Feature map

Every API mount is registered in `server/src/index.ts`. Route files are in
`server/src/routes/`, services in `server/src/services/`, UI folders in
`ui/src/components/`. The full endpoint list is in [api.md](api.md).

| Feature | API mount | Route file | Main services | UI | Engine side |
|---|---|---|---|---|---|
| Create (main generation) | `/api/generate` | `generate.ts` | `backends/*`, `generation/*`, `aceClient.ts`, `autoTrim.ts` | `create/`, `global-bar/` | `/lm`, `/synth`, `/mm3/synth`, `/yue2/synth`, `/sa3-synth` |
| Intent resolution | `/api/resolve` | `resolve.ts` | `generation/resolve/` | `services/api.ts` (`generateApi`), `stores/audioGenQueueStore.ts` | none |
| Durable audio queue | `/api/audio-queue` | `audioQueue.ts` | `audioQueue/intentQueue.ts`, which submits through the same path as `/api/generate` | `services/audioQueueApi.ts`, `stores/audioGenQueueStore.ts` | none of its own |
| Workflow jobs and documents | `/api/workflows` | `workflows.ts` | `workflows/` (`workflowJobs.ts`, `revisions.ts` and each studio's job kinds) | `services/workflowApi.ts`, `queries/studios/workflows.ts` | none of its own |
| Backend toggle and capabilities | `/api` (`/backends`, `/capabilities`) | `backends.ts` | `backends/registry.ts` | `global-bar/BackendToggle.tsx`, `stores/backendStore.ts` | `/mm3/props`, `/yue2/props`, `/sa3/props` |
| Auto-Gen | `/api/inspire`, `/api/workflows` (`insta-*` kinds) | `inspire.ts` | `aceClient.ts`, `lireek/llm/registry.ts`, `workflows/instaGenWorkflow.ts` | `insta-gen/` | `/lm` in inspire mode |
| Lyric Studio | `/api/lireek` | `lireek.ts`, `lireek/*` | `lireek/` (prompts, slop detector, LLM providers, the `lyric-batch` workflow kind) | `lyric-studio/` | none, uses external LLMs |
| Library | `/api/songs` | `songs.ts` | `audioCrop.ts`, `disco-analyzer.ts`, `library/importTrack.ts`, `library/exportZip.ts`, `generation/rePostProcess.ts` | `library/`, `details/`, `player/`, `playlist/` | none |
| Library export and import | `/api/export-import` | `exportImport.ts` | `exportImport/`, `library/importTrack.ts` | `services/exportImportApi.ts`, `library/` | none |
| Studio drafts and playlist snapshots | `/api/studio-drafts` | `studioDrafts.ts` | `studioDrafts/` | `services/studioDraftsApi.ts`, `services/studioDraftMirror.ts`, `shared/StudioDraftPicker.tsx` | none |
| Named presets | `/api/preferences` | `preferences.ts` | `preferences/` | `services/preferencesApi.ts`, `services/presetCollection.ts` | none |
| Player waveform peaks | `/api/audio` | `audio.ts` | `audio/peaks.ts` | `ui/src/audio/peaksClient.ts` | none |
| Models dropdown | `/api/models` | `models.ts` | `aceClient.ts` (`GET /props`) | `global-bar/ModelsDropdown.tsx` | `model-registry.h`, `model-store.cpp` |
| Model Manager (downloads) | `/api/model-manager` | `modelManager.ts` | `modelDownloadService.ts`, `server/src/data/model-registry.json` | `model-manager/` | none |
| DiT adapters (LoRA, LoKr) | `/api/adapters` | `adapters.ts` | `adapters/stMetadata.ts`, `generation/adapterSections.ts` | `global-bar/AdaptersDropdown.tsx` | `adapter-merge.h`, `adapter-runtime.h` |
| Stem Studio | `/api/stem-studio` | `stemStudio.ts` | `aceClient.ts`, `audioConvert.ts` | `stem-studio/` | `/synth` (extract), `/supersep/*` |
| SuperSep proxy | `/api/supersep` | `supersep.ts` | `audioConvert.ts` | `stem-studio/`, Cover Studio's stem split (`cover-studio/SourcePanel.tsx`, `stores/coverStemsStore.ts`), `services/supersepApi.ts`, `shared/StemMixer.tsx` | `supersep.cpp` |
| Stem Builder | `/api/workflows` (`layer-render` kind), plus `/api/upload` and `/api/songs` | `workflows.ts` | `workflows/repaintLayerWorkflow.ts` | `stem-builder/`, `services/repaintLayerWorkflowApi.ts` | the `lego` task on `/synth` |
| Repaint Studio | `/api/workflows` (`repaint-render` kind) | `workflows.ts` | `workflows/repaintLayerWorkflow.ts`, `generation/sourceAudio.ts` | `repaint-studio/`, `services/repaintLayerWorkflowApi.ts` | `sampler-repaint.h` |
| Cover Studio | `/api/generate`, `/api/analyze`, `/api/yue2-cover`, `/api/workflows` (`cover-*` kinds) | `generate.ts`, `analyze.ts`, `yue2Cover.ts` | `generation/sourceAudio.ts`, Essentia, `workflows/coverWorkflow.ts`, `yue2Cover.ts`, `yue2Takes.ts`, `backends/yue2/cover*.ts` | `cover-studio/` | cover task types in `request.h`; `/yue2/synth` and `ace-train yue2-sheet` for YuE2 covers |
| Song Builder | `/api/builder` | `songBuilder.ts` | `songBuilder/sectionOps.ts` (section renders run as a workflow kind), `db/database.ts` | `song-builder/` | seed-latent path of `/synth` |
| STORM (continuous stream) | `/api/generate/storm/*` | `generate.ts` | `storm/slot.ts`, `storm/drift.ts`, `streamSessions/`, `aceClient.ts` | `storm/` | `/synth`, `/yue2/synth`, `/sa3-synth` for ACE, YuE2 and SA3 slots |
| Stream sessions and recordings | `/api/stream-sessions` | `streamSessions.ts` | `streamSessions/` | `services/streamSessionsApi.ts`, `stores/mm3StreamStore.ts`, `hooks/useStreamAudio.ts` | none of its own |
| Compare Studio | `/api/compare` | `compare.ts` | `compare/` (`store.ts`, `analyzer.ts`) | `compare-studio/`, `queries/compare/`, `services/compareApi.ts` | none |
| MIDI Studio | `/api/midi-studio` | `midiStudio.ts` | `muscriptor.ts`, `midiParser.ts` | `midi-studio/` | `ace-midi` binary |
| Training Studio | `/api/training` | `training.ts` | `training/*` | `training-studio/` | `ace-train`, `ace-caption`, `/understand` |
| Training workers (training on another PC) | `/api/workers` (controller), `/api/training/worker` (worker) | `workers.ts` | `training/trainingWorkers.ts`, `training/workerUpdate.ts`, `training/yue2Mirror.ts` | `training-studio/WorkersPanel.tsx`, `training-studio/TrainingWorkerBar.tsx` | none |
| Cover art images | `/api/cover-art` | `coverArt.ts` | `coverArt/*` | `library/CoverArtPromptModal.tsx`, `global-bar/CoverArtDropdown.tsx` | `sd-cli` (stable-diffusion.cpp), not the music engine |
| Mastering | `/api/mastering` | `mastering.ts` | `generation/postProcessing.ts` | `global-bar/PostProcessingDropdown.tsx` | `mastering`, `mp3-codec` binaries |
| VST3 chain | `/api/vst` | `vst.ts` | none, runs `vst-host` | `global-bar/VstChainDropdown.tsx`, `stores/vstChainStore.ts` | `tools/vst-host.cpp` |
| Lua plugins | `/api/plugins` | `plugins.ts` | `aceClient.ts` (`GET /plugins`) | `global-bar/PluginControls.tsx`, `hooks/usePluginRegistry.ts` | `lua-plugin.h`, `lua-plugin-registry.h`, `hot-step-sampler.h` |
| Assistant | `/api/assistant` | `assistant.ts` | `lireek/llm/registry.ts`, `server/src/data/assistant-knowledge.md` | `assistant/`, `services/assistantApi.ts` | none |
| Settings (.env editor, data folder) | `/api/settings` | `settings.ts`, `settingsDataFolder.ts` | `config.ts` (`reloadEnvConfig`), `gpuDevices.ts`, `downloadSave.ts`, `server/src/dataroot/pointer.ts` | `settings/` | none |
| Update notices | `/api/updates` | `updates.ts` | `updates/` (`checker.ts`, `install.ts`) | `settings/UpdatesPanel.tsx`, `services/updatesApi.ts` | none |
| Parameter profiles | `/api/profiles` | `profiles.ts` | none | `global-bar/ProfilesModal.tsx` | none |
| Seeds | `/api/seeds` | `seeds.ts` | none | `global-bar/SeedManagerDrawer.tsx` | none |
| Uploads (audio, latents) | `/api/upload` | `upload.ts` | `latentFormat.ts` | `shared/LatentImport.tsx`, studio upload fields | none |
| Downloads and export | `/api/download` | `download.ts` | `audioMetadata.ts`, `downloadSave.ts` (`DOWNLOAD_DIR`) | `player/`, `library/` | `mp3-codec` |
| Logs, VRAM, terminal | `/api/logs` | `logs.ts` | `logger.ts` | `terminal/`, `shared/VramIndicator.tsx` | `/vram`, `/models/loaded`, `/models/unload` |
| Health, shutdown, restart | `/api/health`, `/api/shutdown` | `health.ts`, `shutdown.ts` | `server/src/engineState.ts`, `aceEngineProcess.ts` | `App.tsx` (liveness), `shared/NoModelsBanner.tsx` | `/health` |
| Auth (local single user) | `/api/auth` | `auth.ts` | `db/database.ts` | none | none |
| BPM and key analysis | `/api/analyze` | `analyze.ts` | Essentia binary (`ESSENTIA_BIN`) | `cover-studio/` | none |
| DAW facade (versioned API for DAW clients) | `/api/v1/daw` | `dawV1.ts` | `daw/` (`dawFacade.ts` calls `/api/generate` over loopback) | none | none of its own |
| Ear-test score sheets | `/listening` (not under `/api`, loopback only) | `listening.ts` | none | none, the pages live in `_experiments/_LISTENING/` | none |

`POST /api/shutdown/restart` writes the restart marker (`RESTART_MARKER_PATH` in
`config.ts`): `.restart-requested` at the repo root in a checkout, `restart-requested` in
the profile folder in a release. The loops in `LAUNCH.bat`, `server/restart-loop.cmd`,
`dev.sh`, `launch.sh` and the release launchers see it and start the server again. While a
data-folder change is pending, a restart is refused with 409 if work is still running.

## Engine source layout

The engine is a fork of acestep.cpp. [engine.md](engine.md) covers its internals; this is the
map of top-level pieces.

| Path | What it is |
|---|---|
| `engine/tools/hot-step-server.cpp` | The `ace-server` binary: HTTP routes, request parsing, job queue. `engine/tools/ace-server.cpp` is upstream's version, kept for reference and not compiled |
| `engine/src/pipeline-lm.cpp`, `qwen3-lm.h`, `metadata-fsm.h` | LM phase: caption enrichment, metadata and audio codes |
| `engine/src/pipeline-synth*.cpp`, `dit.h`, `dit-graph.h` | DiT synthesis |
| `engine/src/hot-step-sampler.h` | HOT-Step's sampling loop. Solver, scheduler and guidance dispatch go through here, not upstream's `dit-sampler.h` |
| `engine/src/hot-step-params.h` | The sideband struct (`g_hotstep_params`) that carries HOT-Step-only request fields into the sampler |
| `engine/src/lua-plugin.h`, `lua-plugin-registry.h` | Lua plugin host. Scans `engine/plugins/`, the repo-root `plugins/` overlay, then the `--user-plugins` folder when one is passed, for solvers, schedulers, guidance and postprocess plugins |
| `engine/src/adapter-merge.h`, `adapter-runtime.h`, `adapter-cancel.h`, `lokr-*.h` | DiT adapter loading, merge and runtime modes |
| `engine/src/model-store.h`, `model-store.cpp`, `model-registry.h` | Model discovery and VRAM residency |
| `engine/src/vae*.h` | VAE encode and decode on GGML |
| `engine/src/dit-trt.h`, `lm-trt.h`, `hot-step-sampler-trt.h` | TensorRT paths |
| `engine/src/supersep.cpp`, `spectral-lifter.h`, `sa3-*.h` | Stem separation, the Spectral Lifter, and Stable Audio 3: the refiner (`/sa3-refine`) and, from pure noise, the `sa3` backend's `/sa3-synth` |
| `engine/src/hot-step-families.h` | Registry of the extra model families hosted by `ace-server` |
| `engine/src/minimax/` | MiniMax-Music3 port and its `/mm3/*` routes (`mm3-server.h`, `mm3-job.h`) |
| `engine/src/yue2/` | YuE2 port and its `/yue2/*` routes (`yue2-server.h`), plus the SheetSage lead-sheet model |
| `engine/src/moss/` | MOSS-Music audio encoder used by `ace-caption` |
| `engine/src/train/` | Training code behind `ace-train` (ACE-Step DiT and LM, MM3, YuE2, SA3, HeartMuLa) |
| `engine/tools/ace-train.cpp` | Training CLI. Subcommands include `preprocess`, `train-dit`, `train-lm`, `mm3-lm-train`, `mm3-train-dit`, `yue2-joint-train`, `yue2-sheet`, `sa3-train`, `heartmula-prepare`, `heartmula-train` and others |
| `engine/plugins/` | Built-in Lua solvers, schedulers and guidance modes. See [plugins-authoring.md](plugins-authoring.md) |
| `engine/ggml` | Submodule pinned to the HOT-ggml fork. See [ggml-fork.md](ggml-fork.md) |

Four upstream files carry `#include` hooks into HOT-Step code (`pipeline-synth-ops.cpp`,
`model-store.h`, `dit.h`, `fsq-detok.h`). `engine/verify-hooks.ps1` checks them along with the
`hot-step-server.cpp` hooks and the HOT-ggml changes.

## Server source layout

| Path | What it is |
|---|---|
| `server/src/index.ts` | Entry point: logger, DB init, route mounts, static serving, engine bootstrap, shutdown |
| `server/src/config.ts` | Every environment variable and its default. See [config.md](config.md) |
| `server/src/routes/` | One Express router per feature |
| `server/src/services/` | Feature logic. `aceClient.ts` wraps the engine HTTP API; `aceEngineProcess.ts` owns the child process |
| `server/src/services/backends/` | Backend registry and the four backend implementations |
| `server/src/services/generation/` | Queue, envelopes, param translation, intent resolution (`resolve/`), LM cache, post-processing |
| `server/src/services/workflows/` | Durable workflow jobs, revisioned documents and the per-studio job kinds |
| `server/src/services/training/` | Dataset Studio and Training Studio runners |
| `server/src/services/lireek/` | Lyric Studio and the LLM provider registry |
| `server/src/db/` | SQLite schema and access (`database.ts`, `lireekDb.ts`) |
| `server/src/contracts/` | Request and response types shared with the UI, many with zod schemas. The UI imports some of them at runtime, which is why the Docker UI stage copies this folder |
| `server/src/dataroot/` | Data folder choice, pointer, transfers and the read-only `locateData()` inspector. See [config.md](config.md#release-data-folder) |
| `server/src/data/` | Runtime data shipped with the app, such as `model-registry.json` and `assistant-knowledge.md`. The release workflow copies this folder whole |
| `server/src/engineState.ts` | Engine ready flag, boot status message, the no-models flag and the engine's compute device |

## UI source layout

| Path | What it is |
|---|---|
| `ui/src/App.tsx` | View routing. Views: create (default), `insta-gen`, `lyric-studio`, `cover-studio`, `stem-studio`, `stem-builder`, `song-builder`, `storm`, `midi-studio`, `training-studio`, `compare-studio` (URL `/compare`), `repaint`, `library`, `settings` |
| `ui/src/components/<studio>/` | One folder per studio or panel. `global-bar/` is the parameter bar shared by all generation views; `shared/` holds reusable pieces |
| `ui/src/stores/` | Zustand stores. `globalParamsStore.ts` holds generation params; others cover playback, backends, streaming, training, VST chain, post-processing |
| `ui/src/services/` | HTTP clients. `httpClient.ts` is the shared client: JSON goes through the axios instance in `axiosClient.ts`, multipart uploads through XHR, SSE through `streamEvents()`, and every failure becomes the `ApiError` in `apiError.ts`. `api.ts` is the main caller (base `/api`). Studios with large surfaces keep their own client file (`lireekApi.ts`, `trainingApi.ts`, `stemStudioApi.ts` and so on), built on `httpClient.ts` |
| `ui/src/queries/` | TanStack Query client (`client.ts`), key factory, retry and polling policy (`contract.ts`), SSE bridge, write helpers and one folder per area. See [UI server state](#ui-server-state) |
| `ui/src/hooks/` | Shared hooks such as `useCapabilities.ts` and `usePluginRegistry.ts` |
| `ui/src/i18n/locales/` | UI strings. `en.json` is the source for control labels |

## Data and logs

- SQLite database: `<data dir>/hotstep.db`. Lyric Studio tables live in the same file.
  Tables include `songs`, `playlists`, `artists`, `lyrics_sets`, `profiles`, `generations`,
  `settings`, `album_presets`, `builder_projects`, `training_datasets`, the `compare_*`
  tables and others; the core schema is in `server/src/db/database.ts`. Some services
  create their own tables: `generation_jobs` (`generation/jobLedger.ts`), `audio_intents`
  (`audioQueue/intentQueue.ts`), `workflow_jobs` and `workflow_events`
  (`workflows/workflowJobs.ts`), `workflow_documents` (`workflows/revisions.ts`),
  `audio_assets` (`assets/audioAssets.ts`) and `lyric_workflow_items`
  (`lireek/lyricWorkflow.ts`).
- `workflow_documents` holds typed, schema-versioned, optimistically-revisioned client state:
  named VST/scale/AI-continue/YuE2-joint presets, STORM tuning, ordered playlist snapshots and
  Create/Cover/Repaint/STORM/Stem drafts (`server/src/services/workflows/revisions.ts`,
  `server/src/services/preferences/`, `server/src/services/studioDrafts/`). A browser's legacy
  `localStorage` copy is imported into it once, by exact key and content hash
  (`document_import_receipts`), never overwritten afterward. The playlist stores ordered
  snapshots independently of `playlists` and `playlist_songs`, so queue item IDs and variant
  URLs need no live song row. `server/src/services/studioDrafts/` validates draft fields and
  browser import keys. Existing `cover-draft` and `training-audition-create` IDs remain
  readable through the handoff adapter. The UI's selected draft ID stays local. Create, Cover
  and Repaint mirror their autosaved fields through `ui/src/services/studioDraftMirror.ts`.
  Each editor keeps its own draft id and expected revision; the per-browser pointer
  `hs-studioDraft:<studio>` only seeds a newly opened editor. A 409 creates a new draft and
  never overwrites the other writer's content. `StudioDraftPicker` loads a draft after a
  confirm if the editor has unsaved edits. Create's request content comes from
  `ui/src/components/create/createContent.ts`, shared by Generate and draft resume.
- Files under the data dir: `audio/` (outputs, with YuE2 build sessions in `audio/takes/`),
  `references/`, `vst/` (`chain.json`, `states/`), `lyrics/` (exports), `training/`,
  `daw_tmp/` (DAW facade uploads and exports).
- A git checkout may also have a repo-root `data/` folder. It is not the live data dir.
- Logs: `<data root>/logs/YYYY-MM-DD_HH-MM-SS/` per session, holding `ace_engine.log`,
  `node_console.log`, `generations/gen_<uuid>_<task>.log` and `training/<kind>_<jobId>.log`
  (written live by `pushEvent` in `labelingQueue.ts` plus each runner's `trainerLine`). The newest folder is the
  current session. Server-generated log timestamps use the machine's local time
  and include its UTC offset, matching the session directory's clock.

## Related

- [building.md](building.md)
- [config.md](config.md)
- [engine.md](engine.md)
- [api.md](api.md)
- [../user/backends.md](../user/backends.md)
- [plugins-authoring.md](plugins-authoring.md)

## Release storage boundary

`server/src/dataroot/bootstrap.ts` resolves the release data folder before consumers open the database, logs or children. `HOT_STEP_ROOT` selects release mode; without it the checkout keeps its existing paths until Settings gives it a data folder ([Dev checkouts](config.md#dev-checkouts)). The profile pointer and pending-transfer journal are outside both app and data folders. The release launcher and shutdown route share a profile `restart-requested` marker; dev retains its checkout marker.

App-owned `server/data/` contains bundled catalogues and assistant knowledge. Managed library, models, adapters, settings, logs and downloaded tools live under the effective data root. External overrides remain external. See [Configuration](config.md#release-data-folder).

## Consumer reconciliation ledger

This ledger covers 187 historical manifest rows and 23 additional
current files found through service imports, raw fetches, stream/interval
openers or imperative store reads at `a9748d5d`. Rows name the owning area,
not the owner of every imported transport. The area README tables above
supply resource details; R1-R6 resolutions are recorded above. Query consumers may also contain
client forms, explicit commands or streams. A type-only import needs no fetch
migration. The historical test fake is retained here only to reconcile its row.

| File under `ui/src/` | Area | Current disposition |
|---|---|---|
| `App.tsx` | shell-queue | Query library; liveness transport, commands, log stream |
| `audio/peaksClient.ts` | shell-queue | Query shared `peaks`; invalidated by `forgetPeaks` |
| `components/assistant/AssistantPanel.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/ArtistSettingsPanel.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/CoverArtistGrid.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/CoverStudio.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/SourcePanel.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/Yue2Build.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/Yue2CoverPanel.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/Yue2CoverScore.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/create/AiGenerateModal.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/create/CreatePanel.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/create/Mm3ComposeButton.tsx` | shell-queue | Area service/store adapter; Query reads, explicit commands/client state |
| `components/create/trainingCreateDraft.ts` | shell-queue | Explicit one-shot form handoff; command-time revision transport |
| `components/details/MetadataEditorModal.tsx` | shell-queue | Area service/store adapter; Query reads, explicit commands/client state |
| `components/details/Yue2CoverDrift.tsx` | shell-queue | Area service/store adapter; Query reads, explicit commands/client state |
| `components/global-bar/AdaptersDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/CoverArtDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/GenerationDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/GlobalParamBar.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/Mm3LmAdapterDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/ModelsDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/MonitorBar.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/PostProcessingDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/ProfilesModal.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/SeedManagerDrawer.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/Yue2LmAdapterDropdown.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/insta-gen/InspirePreview.tsx` | studios | Client/types/props or scoped store projection; no independent HTTP read |
| `components/insta-gen/InstaGenPanel.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/library/CoverArtPromptModal.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/library/ImportTracksModal.tsx` | shell-queue | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/AlbumGrid.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/AlbumHeader.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistGrid.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistPageSidebar.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistSidebar.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/CuratedProfileModal.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/LyricStudioV2.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/PresetSettingsModal.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/ProfilesTab.tsx` | cover-lyrics | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/PromptEditor.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/ProviderSelector.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/QueuePanel.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/RecordingsTab.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/RenderAcrossModal.tsx` | cover-lyrics | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/SourceLyricsTab.tsx` | cover-lyrics | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/WrittenSongsTab.tsx` | cover-lyrics | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/playlistStore.ts` | cover-lyrics | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/useAudioGeneration.ts` | cover-lyrics | Area service/store adapter; Query reads, explicit commands/client state |
| `components/midi-studio/MidiPlayer.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/midi-studio/MidiStudio.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/midi-studio/midiSynth.ts` | studios | Client WebAudio scheduling |
| `components/model-manager/ModelManagerModal.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/model-manager/useDownloadStream.ts` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/model-manager/useModelRegistry.ts` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/player/LyricsBar.tsx` | shell-queue | Query shared `lrc` and `lyrics-json` |
| `components/player/SectionMarkers.tsx` | shell-queue | Query shared `lrc` and `lyrics-json` |
| `components/player/TrimControls.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/repaint-studio/RepaintStudio.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/settings/DataFolderPanel.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/settings/SettingsPanel.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/FileBrowserModal.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/LatentImport.tsx` | shell-queue | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/PresetSyncNotices.tsx` | shell-queue | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/StemMixer.tsx` | shell-queue | Binary media transport and client mixer |
| `components/shared/StudioDraftPicker.tsx` | shell-queue | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/UnifiedRecentSongs.tsx` | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/VramIndicator.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/song-builder/SongBuilder.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/song-builder/followSection.ts` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-builder/RecentBuilds.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-builder/StemBuilder.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/RecentExtractions.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/SourceSelector.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/StemStudio.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/TrackSelector.tsx` | studios | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/AiContinuePresetModal.tsx` | studios | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/StormLiveControls.tsx` | studios | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/StormPage.tsx` | studios | Query consumer; forms/commands/streams stay client-owned |
| `components/terminal/TerminalPanel.tsx` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/AuditionCard.tsx` | training-data | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/AuditionPlayer.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/BatchImportWizard.tsx` | training-data | Query shared browse and training-data scan-preview; explicit pipeline command |
| `components/training-studio/BuildPanel.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/DatasetAssetChips.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/FolderPicker.tsx` | training-data | Query shared browse; obsolete navigation cancelled |
| `components/training-studio/LabelPanel.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/LmAdapterPicker.tsx` | training-data | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Mm3RunsPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Mm3TrainCard.tsx` | training-runs | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/MonitorPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/NewDatasetWizard.tsx` | training-data | Query scan-preview; explicit create command |
| `components/training-studio/PipelineStageProgress.tsx` | training-data | Query job; selected-worker projection |
| `components/training-studio/PreprocessOptionsForm.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/PreprocessPanel.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/PreprocessVariantCard.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/RefinePanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/ReviewPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/SampleDrawer.tsx` | training-data | Query caption snapshots seed local edits per visit; save commands update cache |
| `components/training-studio/SampleGrid.tsx` | training-data | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/SampleRow.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/SendToLyricStudio.tsx` | training-data | Query export preview seeds form; explicit export command |
| `components/training-studio/TrainDitForm.tsx` | training-runs | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/TrainLmForm.tsx` | training-runs | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/TrainPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/TrainingRunStats.tsx` | training-runs | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/TrainingWorkerBar.tsx` | training-data | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/WorkersPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2AitkTrainCard.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2ArTrainCard.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2BatchPanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2ClearPreparedCard.tsx` | training-data | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2LadderReview.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2OptimisePanel.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2OptimizerFields.tsx` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/Yue2SplitChart.tsx` | training-data | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/Yue2TrainCard.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2TrainStages.tsx` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/sendAuditionToCustomGen.ts` | training-data | Local Query dataset revision read; explicit draft command |
| `components/training-studio/useMm3Status.ts` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/useYue2ArStatus.ts` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/useYue2Status.ts` | training-runs | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/yue2ChartData.ts` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/yue2JointCommand.ts` | training-data | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/yue2JointPresets.ts` | training-data | Client/types/props or scoped store projection; no independent HTTP read |
| `context/AuthContext.tsx` | foundation | Query consumer; forms/commands/streams stay client-owned |
| `hooks/useAssistantActions.ts` | studios | Client/types/props or scoped store projection; no independent HTTP read |
| `hooks/useCapabilities.ts` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `hooks/useEventSource.ts` | global-models | Shared stream/reducer; no snapshot fetch |
| `hooks/usePluginRegistry.ts` | global-models | Query consumer; forms/commands/streams stay client-owned |
| `hooks/usePolling.ts` | shell-queue | Removed |
| `hooks/useStreamAudio.ts` | studios | Audio stream/commands; client recording clock |
| `queries/contract.ts` | foundation | Query owner/policy; area contract applies |
| `queries/keys.ts` | foundation | Query owner/policy; area contract applies |
| `queries/sseBridge.ts` | foundation | Query owner/policy; area contract applies |
| `services/api.ts` | foundation | Transport; area Query caller owns reads, commands/streams explicit |
| `services/assistantApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/audioQueueApi.ts` | shell-queue | Transport; area Query caller owns reads, commands/streams explicit |
| `services/coverWorkflowApi.ts` | cover-lyrics | Transport; area Query caller owns reads, commands/streams explicit |
| `services/dataFolderApi.ts` | shell-queue | Transport; area Query caller owns reads, commands/streams explicit |
| `services/exportImportApi.ts` | shell-queue | Transport; area Query caller owns reads, commands/streams explicit |
| `services/httpClient.ts` | foundation | Transport; area Query caller owns reads, commands/streams explicit |
| `services/inspireApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/lireekApi.ts` | cover-lyrics | Query-backed reads; commands/working-copy reducers |
| `services/lyricWorkflowApi.ts` | cover-lyrics | Transport; area Query caller owns reads, commands/streams explicit |
| `services/midiStudioApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/preferencesApi.ts` | shell-queue | Transport; area Query caller owns reads, commands/streams explicit |
| `services/presetCollection.ts` | shell-queue | Query-backed reads; commands/working-copy reducers |
| `services/repaintLayerWorkflowApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/settingDocument.ts` | shell-queue | Query-backed reads; commands/working-copy reducers |
| `services/sharedEventSource.ts` | foundation | Transport; area Query caller owns reads, commands/streams explicit |
| `services/stemStudioApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/streamSessionsApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/studioDraftMirror.ts` | cover-lyrics | Query-backed reads; commands/working-copy reducers |
| `services/studioDraftsApi.ts` | cover-lyrics | Query-backed reads; commands/working-copy reducers |
| `services/supersepApi.ts` | studios | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingApi.ts` | foundation | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingOperations.ts` | training-data | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingPreparationApi.ts` | training-data | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingRecipesApi.ts` | training-data | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingReviewApi.ts` | training-runs | Transport; area Query caller owns reads, commands/streams explicit |
| `services/workflowApi.ts` | shell-queue | Transport; area Query caller owns reads, commands/streams explicit |
| `services/yue2CoverApi.ts` | cover-lyrics | Query-backed reads; commands/working-copy reducers |
| `stores/audioGenQueueStore.ts` | shell-queue | Query reads; client executor/scheduler and media transport |
| `stores/audioQueueStoreTestFake.ts` | shell-queue | Test fake; excluded from production |
| `stores/backendStore.ts` | global-models | Query exact-key projections and selection commands |
| `stores/coverStemsStore.ts` | cover-lyrics | Query separation waiter; client execution/mixer |
| `stores/discoStore.ts` | shell-queue | Media-sidecar playback buffer transport |
| `stores/globalParamsStore.ts` | global-models | Client forms/persistence and active-model mirror command |
| `stores/mm3StreamStore.ts` | shell-queue | Stream reducer and explicit execution |
| `stores/postProcessStore.ts` | shell-queue | Query observer; explicit command and completion reducer |
| `stores/streamingStore.ts` | cover-lyrics | Query-backed recovery; stream execution reducer |
| `stores/trainingStore.ts` | training-data | Query projection and commands; shared preparation detail/job reads |
| `stores/vstChainStore.ts` | global-models | Query projections/presets and editable command state |
| `utils/downloadTrack.ts` | shell-queue | Explicit export command and download transport |
| `utils/heapWatch.ts` | shell-queue | Client heap sampling timer |
| `utils/lrcUtils.ts` | shell-queue | Canonical transport for shared Query `lrc` |
| `utils/wordLrcUtils.ts` | shell-queue | Canonical transport for shared Query `lyrics-json` |
| `utils/yue2CaptionSource.ts` | shell-queue | Query snapshot and captured command assembly |
| `components/global-bar/VstChainDropdown.tsx` (added) | global-models | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/SongActionsMenu.tsx` (added) | shell-queue | Query consumer; forms/commands/streams stay client-owned |
| `queries/cover-lyrics/reads.ts` (added) | cover-lyrics | Query owner/policy; area contract applies |
| `queries/cover-lyrics/songs.ts` (added) | cover-lyrics | Query owner/policy; area contract applies |
| `queries/cover-lyrics/workflows.ts` (added) | cover-lyrics | Query owner/policy; area contract applies |
| `queries/global-models/options.ts` (added) | global-models | Query owner/policy; area contract applies |
| `queries/global-models/presets.ts` (added) | global-models | Query owner/policy; area contract applies |
| `queries/shell-queue/preferences.ts` (added) | shell-queue | Query owner/policy; area contract applies |
| `queries/shell-queue/queue.ts` (added) | shell-queue | Query owner/policy; area contract applies |
| `queries/shell-queue/settings.ts` (added) | shell-queue | Query owner/policy; area contract applies |
| `queries/shell-queue/songs.ts` (added) | shell-queue | Query owner/policy; area contract applies |
| `queries/studios/lyrics.ts` (added) | studios | Query owner/policy; area contract applies |
| `queries/studios/midi.ts` (added) | studios | Query owner/policy; area contract applies |
| `queries/studios/resources.ts` (added) | studios | Query owner/policy; area contract applies |
| `queries/studios/shared.ts` (added) | studios | Query owner/policy; area contract applies |
| `queries/studios/workflows.ts` (added) | studios | Query owner/policy; area contract applies |
| `queries/training-data/queries.ts` (added) | training-data | Query owner/policy; area contract applies |
| `queries/training-runs/events.ts` (added) | training-runs | Query owner/policy; area contract applies |
| `queries/training-runs/hooks.ts` (added) | training-runs | Query owner/policy; area contract applies |
| `queries/training-runs/queries.ts` (added) | training-runs | Query owner/policy; area contract applies |
| `stores/playbackStore.ts` (added) | shell-queue | Client/types/props or scoped store projection; no independent HTTP read |
| `utils/mm3CaptionSource.ts` (added) | shell-queue | Query-backed lireek read; client caption assembly |
| `utils/paramProfiles.ts` (added) | shell-queue | Client/types/props or scoped store projection; no independent HTTP read |

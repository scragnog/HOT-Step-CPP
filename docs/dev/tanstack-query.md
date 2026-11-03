# TanStack Query — architecture, status, and worklist

`@tanstack/react-query` ^5.104.3, audited against `ui/` on 2026-10-27. This page
is the decision record and the operational reference: what shipped, what is
open, the rules the implementation follows, the layer map, the sanctioned
exceptions and the audit method.

Companion documents (they own the detail this page does not repeat):

- [ui-server-state.md](ui-server-state.md) — policy contract: server vs client
  state, key schema, consumer manifest.
- [architecture.md — UI server state](architecture.md#ui-server-state) — the
  short system-level summary and area map.

## Status

Each phase is the work that resolved one of the original audit findings (F1–F5,
last column). F4 keeps its numbered anti-pattern ledger: #1–#13 closed, #14–#17
in the sanctioned table below.

| Workstream | Scope | Findings | Outcome |
|---|---|---|---|
| **P1** — deprecation sweep | `fetchQuery` → `queryClient.query()`; ESLint guard | F1 | **Done.** Zero non-test `fetchQuery` |
| **P2** — transport + writes | One axios transport, one error notifier, `writeAndInvalidate` / `useUpload`, one invalidation owner | F2, F5 | **Done.** SSE stays on fetch; one interceptor with GET silent and status-mapped copy; `invalidateQueries` is ESLint-guarded |
| **P3** — component mutations | `useMutation` at named write sites; double-toast sweep | F2 | **Done.** Component writes converted; store writes on `writeAndInvalidate`; `runCommand` → `trainingWritePlan`. Long-lived job/stream flows stay imperative by design |
| **P3.1** — mutation consistency | Legacy studio CRUD handlers | F2 | **Done.** Lyric Studio CRUD, prompts, presets and song deletes converted |
| **P4** — reads and timers | Storm poll/control, disco wait, VST poll, scans | F3 | **Done.** Uploads on `useUpload`; polls on `refetchInterval`/`pollWhile`; disco on `waitForQuery`; scans/previews converted. No component-owned network interval remains outside the health probe and the fixed-cadence exceptions recorded below |
| **P5** — env form | `useEnvSettingsForm` over `useQuery` | — | **Done.** Draft/seed/save lifecycle + draft-safety tests |
| **P6** — composers and dedup | One factory per entity, duplicate factories removed | F4 | **Done.** F4 #1–#13 closed; #14–#17 sanctioned |
| **P7** — structural | Training SSE second owner, audio-queue executor state | F3 remainder | **Done.** 7.2 (queue projection) and 7.1(a–d) (one job cache, one bridge owning the job stream, store keeps only the selected id) shipped |

**Verification (2026-11-03):** `tsc -b` clean; UI suite **467 pass / 0 fail /
0 skip** of 467; whole-`src` ESLint **826 findings**, all pre-existing categories
and zero project-guard hits; `npm run build` ✓;
`node tools/docs/check-docs.mjs` → `ok`.

## Architecture

```
[ UI components ]      consume query state, trigger mutations
       ↕
[ TanStack Query ]     cache, dedup, retry, isPending/isError
       ↕                queryManager is the only invalidation owner
[ Axios transport ]    one instance; auth + global error interceptors
       ↕
[ REST API ]           SSE is a separate push channel into the cache
```

The module-by-module map is the [layer map](#layer-map) below. The key schema,
key rules and shared-key table live in [ui-server-state.md#keys](ui-server-state.md#keys).
The table below is the part that changes when a new resource class appears.

## Layer map

### Foundation

| File | Role |
|---|---|
| `client.ts` | One HMR-stable `QueryClient`: `staleTime` 30 s, `gcTime` 5 min, focus refetch off, reconnect refetch on, retry ≤ 2 for network/5xx, mutations never retried. Mounted once in `main.tsx`. |
| `contract.ts` | Retry predicate, abort detection, `pollWhile()` (1.5 s active, doubling to 30 s while failing). |
| `keys.ts` | Five-slot key schema `['hs', resource, scope, id, filters]`, `serverKey`/`resourceKey`/`scopeOf`, auth session epoch, shared keys. |
| `invalidation.ts` | The **only** file allowed to call `invalidateQueries`; exposes `queryManager` groups, the event map and `invalidateEvents`. |
| `writes.ts` | `writeAndInvalidate(client, fn, plan)` for non-React stores, plus `useUpload()`. |
| `sseBridge.ts` | SSE frames → coalesced invalidations, in-place `setQueryData` updates, `once` ledger, optional per-frame callbacks. |
| `wait.ts` | `waitForSnapshot` / `waitForQuery` — the non-React `QueryObserver` waiter: canonical keys, `pollWhile`, auth-epoch guard, duplicate suppression (`deliverRepeats` opts back in). |
| `workflowFollow.ts` | `followWorkflowJob` — the one job-follow implementation: abort plumbing, auth-epoch guard, 50 ms coalesced invalidation. |
| `client.ts` + `keys.ts` + `AuthContext.tsx` | A session-epoch change cancels and drops every auth-scoped query from the previous token. |

### Resource slices

| Area | Covers |
|---|---|
| `shell-queue/` | App shell, library, songs, queue, settings, lyrics sidecars, preferences, file browse |
| `global-models/` | Models, adapters, plugins, VRAM, backends, capabilities, presets |
| `studios/` | Extraction, separation, workflow, song-builder, inspire/assistant, MIDI |
| `cover-lyrics/` | Lyric Studio (lireek), cover workflows, sidecar snapshots |
| `training-data/` | Datasets, samples, preparation, training reads |
| `training-runs/` | Jobs, review, scores, ladder, workers, status polls, run SSE |
| `compare/` | Compare sessions, take analysis, diff trace, library search |

## The pattern by entity type

The same shape is used for every resource class:

| Entity type | Pattern | Examples |
|---|---|---|
| Collection + detail | One factory each; detail key carries the id | `songsOptions`/`songOptions`, `trainingQueries.datasets`/`dataset`, `runQueries.mm3Runs` |
| Singleton / snapshot | One factory, no id | `envOptions`, `pluginsOptions`, `healthOptions`, `trainingQueries.capabilities` |
| Job / status | Factory + `pollWhile`, `staleTime: 0` | `trainingQueries.job`, `runQueries.workerUpdate`, `postProcessStatusOptions`, `sa3PreviewOptions` |
| Form-seeding read | `query()` with `staleTime: 0`, never observed | `readSong`, `readAudioQueue`, `trainingQueries.sampleYue2`, `readSnapshot` |
| Cross-area resource | `SHARED` name + `sharedKeys`; the owner exports the factory, other areas re-export or call it | `sharedKeys.songs`, `sharedKeys.workflowJob`, `gpusOptions`, `browseOptions` |
| Store / service read | Imperative `query()` through the area's `read*` helper | `shell-queue/queue.ts`, `preferences.ts`, `cover-lyrics/reads.ts` |
| Write in React | One `useMutation`; `onSuccess` invalidates via `queryManager` | Component writes, `useEnvSettingsForm`, `useReferenceTracks` |
| Write outside React | `writeAndInvalidate(client, fn, plan)` | `trainingStore`, queue/store commands |
| Stream-backed | `bridgeEvents` / `followWorkflowJob` map frames to the cache | downloads, workflows, training run events |

**Key naming.** `training-data` declares its names once in `TD` and
`training-runs` in `TR`; shell-queue, global-models and studios export one
named factory per resource; cover-lyrics uses `snapshotOptions(domain, path)`
for the generic wire-snapshot case, and sa3-previews keeps `sa3PreviewKeys`.
All of these produce the same five-slot key, and every factory returns
`queryOptions(...)` (a few compose one with a spread for a per-resource
override). `Worker` is defined once in `keys.ts` and shared by both training
areas.

## Conventions

1. **Reads:** declare an options factory in the owning area; consume it with
   `useQuery({ ...options, enabled })`. Never hand-build a key or `queryFn` in a
   component.
2. **Writes in React:** one `useMutation` per command, invalidating in
   `onSuccess` through `queryManager`. Use `silent: true` only when the
   component owns the error display.
3. **Writes outside React:** `writeAndInvalidate(client, fn, plan)`; the plan
   is mandatory.
4. **Invalidation:** never call `invalidateQueries` outside
   `queries/invalidation.ts`; add a `queryManager` group instead. Enforced by
   `ui/eslint.config.js` (tests exempt).
5. **Polling:** use `refetchInterval` / `pollWhile` or `waitFor*`. Do not add
   fetching `setInterval`s. A fixed cadence is acceptable only when a missed
   tick is worthless — record it below.
6. **Errors:** reads render an inline failure state; commands are notified by
   the transport exactly once.
7. **Keys:** to move an entry to a new id/scope, add a composer in `keys.ts`;
   never splice a tuple by index.
8. **Optimistic updates:** snapshot in `onMutate`, restore in `onError`.
9. **One-shot form reads:** a read that only seeds a form or wizard uses
   `queryClient.query()` with `staleTime: 0` and is never observed.
10. **Imperative orchestration:** long-lived job and stream flows stay outside
    React with their own request and progress state. Training commands
    invalidate through `trainingWritePlan`.
11. **Deletes:** cancel the old reads before the command and again after
    success, then remove the accepted rows.
12. **Hooks:** factory-in-component is the default. A named hook exists only
    when the query lifecycle or derived state is shared (`useRunJob`,
    `useJobSelection`, `useReferenceTracks`, `useVram`), or the hook owns
    non-query work (`useDownloadStream`, `useCaptionFile`,
    `useEnvSettingsForm`). Single-consumer wrappers may sit next to their
    component; move them into `queries/<area>/` when a second consumer appears.

### Sanctioned exceptions (change the doc before “fixing” one)

| Exception | Why it stays |
|---|---|
| SSE streams (`sharedEventSource.ts`, `useEventSource.ts`, `bridgeEvents`) | Axios cannot stream response bodies; the bridge fills the same cache entries. |
| Media blobs and peaks (`StemMixer`, `useAbEngine`, `useStreamAudio`, `audio/peaksClient`) | Binary playback, not JSON server state. |
| `/api/health` probes (`App.tsx`) | Liveness transport that may reload the page; it must not pollute the cache. |
| Sidecar text reads (`utils/lrcUtils.ts`, `utils/wordLrcUtils.ts`, `LyricsBar`, `SectionMarkers`) | Text payloads outside the JSON transport, keyed through `sharedKeys`; absent and failed both mean "no lyrics". |
| Queryable POST (`yue2CoverApi.ts`, `cover-lyrics/cover` for `/source-metadata` and `/sections/review`) | Read-only and idempotent; `filters` carry only answer-changing fields (`songId`/`sourceAudioUrl`, `abc`/`lyrics`), never hints such as `sourceLabel` or `force`. |
| `useDownloadStream` disabled observer (`components/model-manager/useDownloadStream.ts`, `global-models/downloads`) | Deliberate SSE cache slot: the download bridge replaces the whole entry with each validated frame, so there is no snapshot to fetch. |
| `waitForSnapshot` / `waitForQuery` | Services and stores cannot use hooks; the observer reuses the same cache and polling policy. |
| Fixed-cadence `refetchInterval` (Directed Drift 2 s, VST GUI 1.5 s) | A missed tick is replaced by the next one, so `pollWhile` backoff only adds latency. |
| `Sa3TrainCard` local status factory | F4#17 until SA3 moves into `training-data`. |
| `Yue2CoverDrift`, `useCoverArtGeneration` follow-up | The result is module state or a `cover-art-updated` event, not a cache read. |
| `BackendToggle.tsx:32`, `MonitorBar.tsx:29` side-effect observers | The observer keeps the entry fetched and the monitor polled; the store projection (`backendStore`, `vstChainStore`) owns the visible state. |
| `NoModelsBanner` | Derived banner; an error must not claim "no models". |
| `SettingsPanel` health/worker reads | Ancillary status; the env hook renders its own line. |

### Errors

The axios interceptor is the only failure notifier: GETs never toast (reads
render `isError` / `QueryError`), aborts never surface, and a command passes
`silent: true` when it renders its own failure line. Status mapping: no
response → unreachable, one deduped warn on 401, translated `api.error.*` for
403/404/5xx, otherwise the server's `error` field. Uploads
stay on XHR for progress but route failures through the same handler; their
timeout is a silence watchdog, not `xhr.timeout`.

## Open work

P1–P7 are closed. The read-failure sweep closed on 2026-10-27, the
`compareStore` migration closed the same day, the audio-queue projection timer
(P7.2) closed the same day, and P7.1(a–d) closed on 2026-10-28.

The remaining work is P8, the `queryClient.query()` cleanup. It is proposed, not
started: a sanity check on 2026-10-28 confirmed two duplicate fetch owners, found
one dead read, and rejected four candidates that must stay. The roadmap, the
phase order, and the per-site verdicts are in
[the P8 appendix](#appendix-p8-reactive-read-migration).

Compare has its own `queries/compare/` slice: read factories for
sessions/detail/search/analysis/diff, mutation hooks for every command with
optimistic scoring and snapshot rollback, and list membership invalidation on
archive/auto-archive transitions. `analysis`/`diff` keep a five-minute freshness
because they are expensive and cached server-side, and song search keeps the
previous page while the next one loads. The hand-rolled store was deleted.

## P7 notes

The status table and history above record that P7 is closed. The lasting
invariants: `bridgeRunEvents` merges `job`/`progress`/`status` frames into
`trainingQueries.job`, `useTrainingJobProjection()` is the only bridge into that
entry and the only opener of the active job's stream, and `trainingStore` keeps
only `activeJobId`, the worker, and the sample/log/preview/metric reducers. Two
chart readers keep their own subscription to the same URL — `Yue2SplitChart`
and `Yue2AitkTrainCard` — because they reduce the job's `metric` frames into
chart state; both stop at a terminal status and neither writes the job cache.
The one command that writes that entry is `cancelJob`: the server can close the
stream before its terminal frame lands, so the store writes `cancelled` itself.

`useAudioQueueProjection()` owns the queue read with `ACTIVE_POLL_MS` while
active and the idle cadence otherwise;
the executor keeps submission scheduling, engine-park rules and its 2.5 s job
read.

**Watch:** MM3 and YuE2 status routes share the frame vocabulary. A job stream
that 404s after a server restart is synthesized as failed by the bridge's error
probe (only on confirmed 404; transient errors leave the cached state for the
poll to recover). Once a job reaches terminal status the SSE bridge tears down
so the EventSource does not reconnect and replay forever.

## Verification

Guard greps, from `ui/` — each prints only the exception it names:

```sh
grep -rn "fetchQuery" src --include=*.ts --include=*.tsx | grep -v '\.test\.'         # empty
grep -rn "invalidateQueries" src --include=*.ts --include=*.tsx | grep -v '\.test\.'  # queries/invalidation.ts only
grep -rn "new FormData(" src/services --include=*.ts                                  # httpClient.upload only
grep -rn "fetch(" src/services --include=*.ts                                         # transport internals + SSE only
```

Imperative reads in components are blocked by ESLint, not a grep: `npx eslint
src` must report no `no-restricted-syntax` hits from the component rule, and the
exemption list in `ui/eslint.config.js` is the audited set.

Inventory for the layer map and the [P8 appendix](#appendix-p8-reactive-read-migration):

```sh
grep -rn "useQuery(" src --include=*.ts --include=*.tsx | grep -v '\.test\.' | wc -l
grep -rn "useMutation(" src --include=*.ts --include=*.tsx | grep -v '\.test\.' | wc -l
grep -rn "\.query(" src --include=*.ts --include=*.tsx | grep -v '\.test\.'
```

Re-run them after a structural change and update this block in the same commit;
the totals below are a dated snapshot, not a permanent truth.

**2026-11-03:** `tsc -b` clean; UI suite **467 pass / 0 fail / 0 skip** of 467;
whole-`src` ESLint **826 findings**, all pre-existing categories and zero
project-guard hits; `npm run build` ✓; `node tools/docs/check-docs.mjs` → `ok`;
`useQuery` 147, `useMutation` 122, `queryClient.query()` on 62 lines in 29 files.

The P8 appendix's five-way split is the 45-site inventory measured on 2026-10-28;
re-measure before starting a phase rather than trusting either figure.

## Phase history

The outcome index is the status table above; this is the dated log.

| Date | Milestone |
|---|---|
| 2026-10-20 | Audit written (F1–F5). |
| 2026-10-21 | P1 and P2 shipped. |
| 2026-10-22 | P3 and the post-P3 write-site audit shipped. |
| 2026-10-23/24 | Point-in-time read, write and imperative audits closed. |
| 2026-10-25 | Last P3 stragglers (extract-kick, quit/restart) moved onto the transport. |
| 2026-10-27 | HeartMuLa re-audit and ESLint invalidation guard; P5 and P6 closed; P7 design recorded. Entity-pattern audit (`Worker` unified in `keys.ts`, `TR` names centralised, factories return `queryOptions`, last five `localOptions` usages moved), §4.12 read-failure sweep, §4.13 compare migration, P7.2, and P7.1(a–b). |
| 2026-10-28 | P7.1(c–d) closed; P8 measured and sanity-checked. |
| 2026-11-02 | P4–P7 re-audited. Terminal streams (the projection and both YuE2 chart readers) close at a terminal status, the job-loss probe fires only on a confirmed 404, and the adoption test's mock matches the real job payload, removing its intermittent failure. |

## Appendix: P8 reactive read migration

**Status:** Proposed, measured 2026-10-28. Medium priority: duplicated requests
and store copies that drift from the cache.

`queryClient.query()` appeared 45 times in non-test code in 20 files when P8 was
measured on 2026-10-28; it now appears on 62 lines in 29 files (§Verification).
Most of those calls are correct: services and stores cannot use hooks. The problem is a
subset that duplicates a read the UI already observes through `useQuery`, so the
same key is fetched twice and the store's copy can disagree with the cache.

Those 45 measured call sites split five ways: 3 component reads to observe (P8.2), 3
deferred (P8.4), 4 duplicate or dead (P8.1), 7 Training Studio store snapshots
(P8.3), and 28 sanctioned non-React flows (listed at the end).
`InstaGenPanel.tsx:157` is a fourth component read, but it enters through
`instaWorkflowApi.getDocument`, so it is not in the count. Each phase's table is
the site list; this appendix does not repeat it.

Each phase updates the counts in [§Verification](#verification) and the phase
note here, uses `queryManager` for every invalidation, and takes its reads from
the owning factory. A phase is one commit so it can be reverted alone.

### Verified duplicates

Suspected duplicates were sanity-checked on 2026-10-28 before any removal. Same
key, same data, store readers, and side effects were checked per candidate.

| Resource | Imperative read | Reactive owner | Verdict |
|---|---|---|---|
| pipelines | `trainingStore.loadPipelines` (`stores/trainingStore.ts:1087`) | `MonitorPanel.tsx:247` | Duplicate. The store field has no reader. |
| YuE2 batches | `trainingStore.loadYue2Batches` (`:1117`) | `Yue2BatchPanel.tsx:43`, `useHeartmulaController.ts:36` | Duplicate. One store reader remains, the batch wizard. |
| train-lm / train-dit status | `loadTrainLmStatus` / `loadTrainDitStatus` (`:862`, `:987`) | `plugins/ace.tsx:130-132` | Not a duplicate. Different keys and different answers. See P8.1. |
| backend model catalogue | `backendStore.fetchModels` (`:304`) | `backendStore` projection | Not a duplicate. Callers read the projection right after `await`. |
| backend capabilities / list | `fetchCapabilities` (`:294`), `fetchBackends` (`:273`) | projections | Not duplicates. Warm-up plus `loading`/`error`. |
| VST chain | `vstChainStore.loadChain` (`:179`) | `vst-chain` projection | Not a duplicate. Only trigger of the initial fetch. |
| training job probe | `trainingStore.ts:633` | job query | Keep. Removal can flash a synthesized failure. |

### P8.0 Baseline

Record the checks and totals in [§Verification](#verification) before editing,
plus the current `queryClient.query()` count. Today: the 462 pass / 0 fail /
1 skip suite, a clean `tsc -b`, zero project-guard findings, docs `ok`, and 64
matching lines in 29 files.

### P8.1 Delete the confirmed duplicates

1. `loadPipelines` (`:1087`) plus the `pipelines` and `pipelinesLoading` fields.
   No component, test, or store action reads either field.
   `MonitorPanel.tsx:247` owns `trainingQueries.pipelines(worker)` with its own
   poll. The dataset invalidation at `:1091` only ran after a pipeline command,
   never during that poll, and `refreshAfterJob` plus
   `MonitorPanel.tsx:260-266` already cover the dataset list. Drop the check,
   then confirm by hand that datasets refresh after a stage finishes.
2. `loadYue2Batches` (`:1117`) plus the `yue2Batches` field, after
   `Yue2AitkBatchWizard.tsx:20` moves to
   `useQuery({ ...trainingQueries.yue2Batches(worker), enabled: open })`. Drop
   the explicit re-reads in `startYue2Batch`, `addToYue2Batch`,
   `setTrainingWorker` and `workerCommand`; keep their invalidates.
3. `vstChainStore.pollMonitorStatus` (`:337`). No caller; `MonitorBar.tsx:29`
   owns the query and the store projection mirrors it.

`vstChainStore.scanPlugins` (`:170`) also drops its own read, because
`VstChainDropdown.tsx:42` observes `vstScanOptions`. It must still await the
refetch to clear `scanning`, so it becomes a manager-owned `refetchQueries`, not
a plain delete.

Why the rejected candidates stay:

- `loadTrainLmStatus` / `loadTrainDitStatus` are a latent bug, not a duplicate.
  No caller passes the optional query, so `lastTrainLmQuery` and
  `lastTrainDitQuery` stay undefined and the store always reads empty filters
  (server default size and dataset slug). `plugins/ace.tsx:130-132` reads the
  same factory with the form's `lmSize` and `adapterName`, a different key and
  answer, while `AuditionCard` depends on the store copy for `adapterExists`,
  `adapterDir`, `variantKey` and `lmSize`. Fix it by giving both consumers one
  shared key or by passing the panel's parameters into the loader, and delete
  the dead `lastTrain*Query` state in the same change.
- `backendStore.fetchModels` is load-bearing:
  `PresetSettingsModal.tsx:129-131` reads `models[YUE2]` right after
  `selectModels` resolves, and `audioGenQueueStore.ts:1015-1017` fetches when
  the catalogue is absent before `yue2PickAtEnqueue`. An invalidate does not
  refetch an inactive query, so those reads would go stale.
- `fetchCapabilities` is warm-up after a backend switch and
  `backendStore.query.test.ts` pins its semantics. `fetchBackends` is awaited at
  boot and is the error surface for an advisory read.
- `loadChain` is the only trigger of the initial `vst-chain` fetch; the
  projection never fetches. Rename it to `prefetchChain` if the intent needs to
  be clearer.

Exit: the deleted symbols do not appear in `rg`; the batch wizard fetches
batches itself; Monitor, a "Train on" switch, the YuE2 batch panel and the
monitor bar each produce one request per key.

### P8.2 Component one-shots become `useQuery`

| File | Change |
|---|---|
| `PipelineStageProgress.tsx:66` | Observe `trainingQueries.job(worker, jobId)` with `enabled: !!jobId`, then adopt on `data`. It currently races `useActiveTrainingJob()`. |
| `SendToLyricStudio.tsx:54` | `useQuery({ ...tq.lyricStudioPreview(worker, datasetId), enabled: open })`; seed the form once from `data`. |
| `BatchImportWizard.tsx:166` | `useQuery(tq.browse(dir, 'trainingAudio'))`, `enabled: open && !!dir`. Rows stay client state. |
| `InstaGenPanel.tsx:157` | Replace the `instaWorkflowApi.getDocument` recovery read with `documentOptions(id)`. |

`BatchImportWizard.tsx:134` stays sequential on purpose: 20 folders must not open
20 concurrent recursive reads. Revisit with `useQueries` only.

### P8.3 Training Studio read migration

The largest phase; removes the store's remaining server snapshots.

1. Add `useTrainingCapabilities()` over `trainingQueries.capabilities(worker)`
   (it cannot reuse the backend-scoped `useCapabilities`), move the training
   components off `store.capabilities` (`:571`), then delete the field and
   loader.
2. `PreprocessPanel` observes `preprocessStatus(worker, id)` (`:822`) with
   `pollWhile`; delete the loader and field; the two effects at `:73-84` become
   one terminal invalidate.
3. `AuditionCard` observes `trainingQueries.auditions(worker, id)` (`:1011`);
   delete `loadAuditions` and the field.
4. Move `datasets` (`:585`) and `detail` (`:603`) to their factories, leaving
   the store with `selectedDatasetId` and `trainingWorker`, as P7.1 did for jobs.
   This touches `DatasetList`, `BatchImportWizard`, `Yue2AitkBatchWizard`,
   `Yue2AitkTrainCard`, `TrainPanel`, `PreprocessPanel`, `RefinePanel`,
   `BuildPanel`, `SendToLyricStudio`, `DatasetDetail`, `SampleRow`,
   `TrainingStudio` and `plugins/ace.tsx`. Ship item 4 as its own commit.
5. Move `trainLmStatus` (`:862`) and `trainDitStatus` (`:987`) last. Fix the key
   first: no caller passes the filters, so both still read the server default.

Exit: no store snapshot is also observed by `useQuery`, and one key has one
owner.

### P8.4 Remaining reads and contract reconciliation

1. Add `audioGenerationsOptions(genId, lyricsSetId)` and use `useQueries` in
   `RecordingsTab.tsx:122` for the fallback statuses.
2. Convert `ProviderSelector.getCachedProviders` to `getQueryData` plus an
   `ensureQueryData`-style refresh inside the handler, or list it in the
   exception table.
3. Regenerate the counts in [§Verification](#verification), and fold the final
   keep list into the
   [sanctioned exceptions](#sanctioned-exceptions-change-the-doc-before-fixing-one).
4. **Landed 2026-11-02:** the ESLint rule beside the `fetchQuery` ban blocks
   `.query()` under `src/components/`. The audited exemption list lives in
   `ui/eslint.config.js`; shrink it as sites move to `useQuery`, and add a file
   only with an entry here.

Exit: the inventory matches `rg`, the exception table lists every remaining call
site, and `npx eslint src` reports zero project-guard hits.

### Sanctioned flows (28)

These keep their imperative read. The final exception table gets this list.

- Binary and text payloads: `audio/peaksClient.ts:52`,
  `utils/yue2CaptionSource.ts:214`.
- Service or executor reads: `services/yue2CoverApi.ts:210`,
  `services/supersepApi.ts:122`, `services/stemStudioApi.ts:181`,
  `queries/shell-queue/queue.ts:125-146` (six executor reads),
  `queries/shell-queue/preferences.ts:44,58`,
  `queries/global-models/presets.ts:23`, `queries/cover-lyrics/reads.ts:28`,
  `queries/cover-lyrics/workflows.ts:27,28`, `queries/studios/workflows.ts:13,14`.
- Store flows: `trainingStore.ts:633`, `:1624`, `:1675`, `:1679`;
  `backendStore.ts:273`, `:294`, `:304`; `vstChainStore.ts:179`.
- Pure handler: `components/training-studio/sendAuditionToCustomGen.ts:14`.

### Definition of done

1. No `queryClient.query()` in component render paths; the handler-only
   `ProviderSelector.getCachedProviders` is converted or listed.
2. Every remaining store read is non-React by necessity and named in the
   sanctioned table.
3. No key has two fetch owners. Verify through Monitor, a "Train on" switch, a
   backend switch, and opening the VST chain.
4. `tsc -b` clean, UI suite green, `npm run build` succeeds, the guard count is
   zero, `check-docs` prints `ok`, and the inventory numbers are regenerated.

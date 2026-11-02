# Shell and generation queue server state

The shell-queue area covers the app shell (`App.tsx`), the library, Create,
details, Settings, player, playlist and sidebar, the shared components and
stores not assigned elsewhere, and the generation queue. It uses the client,
keys and policy in `../client.ts`, `../keys.ts` and `../contract.ts`.
Selections, forms, playback, the queue's sequencing and its legacy
owner/migration rules stay client state.

## Library

shell-queue owns the canonical `sharedKeys.songs()` and `sharedKeys.song(id)`
reads (`songs.ts`). `App.tsx` observes the library; the Recent Songs column
observes `shell-queue/recent-songs`, one entry per source filter. Both are
auth-scoped and replace the old component state and module-level cache.

Every song write patches the cached rows at once and then invalidates every
song view (`SONG_VIEWS`, defined in `../keys.ts`): library, details, this
area's recent feed, studios' `studios/recent-songs` and Cover's
`cover-lyrics/cover-songs` (the last two invalidated only, never edited here).

| Write | Follow-up |
|---|---|
| Generation completes (`song-created`, queue `onSongCreated`), import | Prepend to the library, invalidate |
| Rename, metadata edit, crop, retranscribe | Patch or invalidate |
| Delete, bulk delete, Recent Songs delete | Drop rows, remove the detail, invalidate (cancels a list read already in flight) |
| Cover art finished, post-processing done or reverted | Patch library, detail and recent feed, then invalidate (cancels an older read in flight) |
| Nuke generations | Empty the library, invalidate |
| Server comes back (shared `/api/logs` stream reopens) | Invalidate every song view |

## Generation queue

The executor in `stores/audioGenQueueStore.ts` keeps its sequencing,
migration fencing, manual rows and completion effects. Its server reads go
through `queue.ts`: queue items, job status, workflow job, song rows, song ids
and the submit path, each by `query()`. Loops reading the same thing at
once share one request. A job status younger than one second, or a queue
list younger than one poll for the enqueue waiter, is taken from the cache.
These reads never retry; each loop already decides what a failure means.
Submits, cancels, retries, dismissals and migration commands stay direct
calls and run once.

`useAudioQueueProjection()` (in `projection.ts`, mounted once in `App.tsx`)
is the only queue poller: every 1.5 seconds while any row waits or runs, about
every 10 seconds when the queue is idle (to see work another tab queued). A
local command refreshes at once. A refresh asked for while one is in flight
runs again straight after it, so an enqueue that races the poll still appears
at once.

A finished render's song is delivered to the library once per page. A
path claims the delivery only after its song read succeeds, so a failed
read leaves it to the other path. The
projection's `song-created` event and the enqueue's own waiter used to both
deliver a server-owned Create render, which ran `handleSongCreated` (and the
disco stem extraction it starts) twice.

## Other reads

| Resource | Key | Notes |
|---|---|---|
| Post-processing run | `shell-queue/post-process` | Query observer, 1.5 s while running, stops at a terminal status; one completion event |
| Cover art job | `shell-queue/cover-art-job` | Observer, 2 s, stops at a terminal status, five-minute safety stop |
| Report log tails | `shell-queue/report-logs` | Fresh each visit; the Report an issue tab redacts and trims them |
| Last failed job | `shell-queue/report-failed-job` | Auth; fresh each visit; newest failed audio queue item |
| Server version | `shell-queue/health` | Settings footer |
| `.env` values | `shell-queue/env` | Fresh each load; seeds the editable form |
| GPUs, stem storage | shared `gpus`, `shell-queue/stem-stats` | Clearing stems updates then invalidates |
| Data folder | `shell-queue/data-folder` | Fresh each visit; refetched after a cancel or a failed command |
| Update notice | `shell-queue/updates` | Fresh each visit; polled every 2 s while a check started elsewhere (startup, another tab) is still running; Check now writes its result into the entry |
| File browser listing | shared `browse` | Fresh each navigation; only the newest navigation lands |
| Adapter folder scan | shared `adapter-scan` | Fresh per click |
| YuE2 caption source | `training-data/yue2-caption-source` | Fresh; read by `utils/yue2CaptionSource.ts`, which keeps the error surface and local cache |
| Workflow job (queue) | shared `workflow-job`, `after` 0 | Fresh per read, no retry |
| Waveform peaks | shared `peaks` | `audio/peaksClient.ts`; never stale, `forgetPeaks` removes a rewritten file's entry, failures uncached |
| Lyric sidecars | shared `lrc`, `lyrics-json` (`lyrics.ts`) | One entry per base render; missing caches null; crop (`TrimControls.tsx`) and retranscribe (`SongActionsMenu.tsx`) refresh them through `queryManager.sidecars.all` |
| YuE2 lead sheet | shared `score-abc` (`lyrics.ts`) | One entry per base render, read when a song actions menu opens; missing or the SPA shell caches null |
| Presets and settings documents | `shell-queue/presets`, `shell-queue/setting` | Default adapters for `PresetCollection` and `SettingDocument`; fresh reads, dropped when unread (gcTime 0), writes invalidate |

Create reads two other areas' snapshots through their exported options, so
both views share one cache entry: the YuE2 dataset list
(`trainingQueries.datasets`, with the worker captured in the key) and the
Insta-Gen provider list (`inspireProvidersOptions`). Mastering references use
global-models' `referencesOptions`, and upload or delete invalidates that key.

## Consumer dispositions

| Consumer | Disposition |
|---|---|
| `App.tsx` | query (library, reconnect bridge); `/api/health` liveness probes with a timeout stay transport; extract-kick, shutdown and restart are mutations; `/api/logs` presence is a stream |
| `audio/peaksClient.ts` | query (shared `peaks`), read by the playback engine outside React |
| `components/create/AiGenerateModal.tsx` | query (studios providers); inspire and random subject are commands |
| `components/create/CreatePanel.tsx` | query (training-data datasets); YuE2 plan is a command |
| `components/create/Mm3ComposeButton.tsx` | mutation (compose) |
| `components/create/trainingCreateDraft.ts` | transport: one-shot handoff read applied to the form once |
| `components/details/MetadataEditorModal.tsx` | mutation: the save runs through `writeAndInvalidate` with the `song:written` event, the cover upload through `useUpload` |
| `components/details/Yue2CoverDrift.tsx` | mutation (measure) |
| `components/library/CoverArtPromptModal.tsx` | query (job); prompt preview is a one-shot pre-fill of an editable field; generate is a mutation |
| `components/library/ImportTracksModal.tsx` | mutation; `song-created` invalidates |
| `components/player/LyricsBar.tsx`, `SectionMarkers.tsx` | query (shared `lrc`, `lyrics-json`) |
| `components/player/TrimControls.tsx` | mutation; invalidates song views and sidecars |
| `components/settings/AdapterReferencesAction.tsx` | mutation (album preset adapter-reference cleanup); `lireekApi` invalidates the lireek snapshots after it |
| `components/settings/DataFolderPanel.tsx` | query; preview, schedule, cancel and restart are commands |
| `components/settings/SettingsPanel.tsx` | query (health, GPUs, stem stats); the editable env form is `useEnvSettingsForm`; saves, purges and restart are mutations |
| `components/settings/UpdatesPanel.tsx` | query (`shell-queue/updates`); Check now is a `useMutation` that writes its result into the entry |
| `components/shared/FileBrowserModal.tsx` | query |
| `components/shared/LatentImport.tsx` | mutation (upload) |
| `components/shared/PresetSyncNotices.tsx` | query through `PresetCollection` |
| `components/shared/StemMixer.tsx` | transport: audio file blobs |
| `components/shared/StudioDraftPicker.tsx` | consumer of cover-lyrics' draft service; its Query adoption is that owner's |
| `components/shared/UnifiedRecentSongs.tsx` | query |
| `hooks/usePolling.ts` | deleted; it had no callers |
| `services/audioQueueApi.ts`, `dataFolderApi.ts`, `preferencesApi.ts`, `updatesApi.ts`, `workflowApi.ts` | transport; reads take `ReadOptions` |
| `services/exportImportApi.ts` | mutation (imports, export resolve, profile validation) |
| `services/presetCollection.ts`, `settingDocument.ts` | query (default adapters) |
| `stores/audioGenQueueStore.ts` | query (all server reads); commands direct; `lireekApi` preset reads and health probes stay transport (cover-lyrics' client; liveness) |
| `stores/discoStore.ts` | transport: static per-track disco JSON |
| `stores/mm3StreamStore.ts` | stream |
| `stores/postProcessStore.ts` | query |
| `utils/downloadTrack.ts` | mutation (export download) |
| `utils/heapWatch.ts` | client timer |
| `utils/lrcUtils.ts`, `wordLrcUtils.ts` | transport: canonical sidecar reads for the shared keys |
| `utils/mm3CaptionSource.ts` (not in the manifest: dynamic import) | transport: reads cover-lyrics' `lireekApi.getLyricsSet` |
| `utils/yue2CaptionSource.ts` | query |

## Runtime acceptance handoff

Run on `http://localhost:3000` in the assigned runtime job. This slice did not
launch the app or claim network counts.

| View/resource | Expected endpoints and cadence |
|---|---|
| Any view | `GET /api/songs` once per session and stale remount; again after a song write or a server reconnect |
| Recent Songs column | `GET /api/songs/recent?limit=50[&source=...]` once per source while fresh; after song writes |
| Server-owned queue, active | `GET /api/audio-queue/items` every 1.5 s; `GET /api/generate/status/:id` per running job, shared by every loop following it |
| Server-owned queue, idle | `GET /api/audio-queue/items` about every 10 s |
| Browser-owned queue | Status reads per running job at the existing 1.5 s and 2.5 s loops; no projection poll |
| Post-processing re-run | `GET /api/songs/:id/postprocess?jobId=` every 1.5 s, one at a time, stops at done or failed |
| Cover art (library menu) | `GET /api/cover-art/generate/:jobId` every 2 s, stops at done or failed |
| Settings | `/api/health`, `/api/settings/env`, `/api/settings/gpus`, `/api/stem-studio/stats`, `/api/settings/data-folder`, `/api/updates/status` on open |
| Presets (Storm AI-continue) | `/api/preferences/presets/:family`, `/api/preferences/settings/:family` on load and after writes |

1. Open Library, Create and the Recent Songs column together. Count one
   `/api/songs` and one recent read per source; switch views and come back
   within 30 seconds without new reads.
2. Queue three Create renders (server-owned queue). Confirm one
   `POST /api/audio-queue/items` each, list reads every 1.5 s while they run,
   one status request per running job per tick, and the idle cadence after the
   last finishes. Stage, progress and generation time look as before.
3. When a render finishes, the library and Recent Songs show it once; with
   disco stem extraction on, `POST /api/songs/:id/extract-kick` fires once per
   song.
4. Delete a song while a library refetch is slow (DevTools throttling): the
   row stays gone. Rename, crop and edit metadata: library and details update.
5. Stop and restart the server: after the offline overlay clears (page reload)
   or the logs stream reopens, the library refetches once.
6. Run post-processing and cover art on a track: polling stops at the end, the
   mastered badge or cover appears once, no repeated completion.
7. Exercise browser-owned queue migration banners and manual Cover/Lyric rows
   if available; the queue suites cover these paths, runtime is unverified.

Focused tests: `shellQueue.test.ts` (shared reads, write invalidation,
delete-during-read, cache patches, status read sharing and no retry, waiter
cache reuse, post-process terminal stop, preference read-back) and
`stores/audioGenQueueStore.delivery.test.ts` (one submission, one delivery,
enqueue racing the projection read). Run the UI suite with:

```powershell
node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-timeout=60000 "src/**/*.test.ts"
npx tsc -b
```

# UI server state

TanStack Query owns every server snapshot the UI fetches: lists, details,
statuses, capabilities, job progress. Client state stays where it is: forms,
playback, drafts, generation params and workflow execution live in Zustand or
React context. Server-sent event streams stay as the push channel and drive
the cache through the bridge below. They never become a second fetch owner.

This page is the contract every area builds on. The code lives in
`ui/src/queries/` (`client.ts`, `contract.ts`, `keys.ts`, `sseBridge.ts`,
`invalidation.ts`, `writes.ts`) and the transports in `ui/src/services/`
(`api.ts`, `trainingApi.ts`, `httpClient.ts`, `axiosClient.ts`,
`sharedEventSource.ts`). The audit and migration plan that tracks where the
code still falls short of this contract is
[tanstack-query.md](tanstack-query.md).

## The client

One `QueryClient`, exported as `queryClient` from `queries/client.ts`.
`main.tsx` mounts `QueryClientProvider` outside `AuthProvider` and `App`. The
client survives Vite HMR through `import.meta.hot.data`, so an edit never
builds a second cache with its own observers and polls. Tests build their own
with `createQueryClient()`. Devtools are not installed.

| Policy | Value |
|---|---|
| `staleTime` | 30 s |
| `gcTime` | 5 min |
| `refetchOnWindowFocus` | off |
| `refetchOnReconnect` | on |
| Query retry | at most 2, network errors and 5xx only (`retryQuery`) |
| Retry delay | 500 ms doubling, capped at 4 s (`retryDelay`) |
| Never retried | aborts, every 4xx |
| Mutation retry | off, always |

These are project choices, not library defaults. Override per resource only.
Active jobs and VRAM poll at `ACTIVE_POLL_MS` (1.5 s) through `pollWhile`:

```ts
refetchInterval: pollWhile<JobStatus>(job => job?.status === 'running'),
```

`pollWhile` stops for good once the predicate fails (terminal stop). A poll
cycle is one fetch plus its retries. While cycles keep failing, the interval
doubles per consecutive failed cycle (3 s, 6 s, 12 s, 24 s, then capped at
30 s), and the next success resets it to 1.5 s. Capability probes keep their
existing down manifest behaviour.

## Keys

Every key is a five-slot tuple built by `serverKey()`:

```
['hs', resource, scope, id, filters]
```

| Slot | Meaning |
|---|---|
| `resource` | a shared name (below) or `<area>/<name>` |
| `scope.api` | the API base of `apiClient`, `/api` today |
| `scope.session` | present only on keys whose request carries the bearer token |
| `scope.worker` | training target: a worker name, or `null` for this PC |
| `scope.backend` | generation engine, only where data differs by engine |
| `id` | one item, or `null` for a list or snapshot |
| `filters` | normalized query parameters, `{}` when none |

Shared local resources omit `worker` and `backend`. A key never holds a
credential: `session` is an opaque counter. `normalizeFilters` drops
`undefined` and `''` and sorts the keys, so equal requests share an entry.

Build an invalidation prefix with `resourceKey(resource, scope?, id?)`.
Matching is per slot, so `resourceKey('training-data/datasets', { worker: 'A' })`
hits every dataset list on worker A and nothing on B or this PC. Pass the
prefix to `queryManager` (`queries/invalidation.ts`): `queryManager.invalidate(client,
...keys)`, `queryManager.filter(client, filter)` for a predicate, or a named
group such as `queryManager.songs.views(client)`. `invalidation.ts` is the only
file that calls `QueryClient.invalidateQueries`.

### Capture the target in the key

A query function reads its target back from its own key with `scopeOf()`,
never from a mutable global. Whatever the selection does after scheduling, a
request goes where its cache entry says it went.

```ts
const datasetsQuery = (worker: string | null) => queryOptions({
  queryKey: serverKey('training-data/datasets', { worker }),
  queryFn: ({ queryKey, signal }) => listDatasets({ worker: scopeOf(queryKey)!.worker, signal }),
});
```

### Auth sessions

`AuthProvider` calls `setAuthToken(token)` (`services/axiosClient.ts`) and
`startAuthSession(queryClient, token)` before the new token reaches any
consumer, and exposes the result as `sessionEpoch` on `useAuth()`. The
transport reads the bearer from that accessor, so no read or command passes a
token. A changed token bumps the epoch, cancels the old epoch's in-flight
auth-scoped queries and removes them. Public keys are untouched. Build auth
keys with `{ auth: true }` and read `sessionEpoch` from `useAuth()` in the
component, so it re-renders when the epoch moves.

### Shared resources

More than one area reads these. The contract is the key, the read and the
cached shape, not another area's hooks. Any area may query a shared resource
directly with its canonical key and canonical read below, so no Stage 2 slice
waits on another. Several observers on one key still share one request and
one cache entry.

The cached value is the canonical read's result, exactly. Reshape it with
`select` in the observer, never inside the `queryFn`: two `queryFn`s on one
key that return different shapes would overwrite each other. The owner is the
area that may change the read or the shape. It announces a change to every
consumer before making it, and adds invalidation for the events it handles.

| Key | Canonical read | Cached shape | Owner |
|---|---|---|---|
| `sharedKeys.songs()` | `songApi.list({ signal })` | `{ songs: Song[] }`, normalized | shell-queue |
| `sharedKeys.song(id)` | `songApi.get(id, { signal })` | `{ song: Song }`, normalized | shell-queue |
| `sharedKeys.plugins()` | `apiClient.get('/plugins', { signal })` | raw registry JSON | global-models |
| `sharedKeys.capabilities(backend)` | `apiClient.get('/capabilities?backend=<id>', { signal })` | raw JSON | global-models |
| `sharedKeys.vram()` | `apiClient.get('/logs/vram', { signal })` | raw JSON | global-models |
| `sharedKeys.loadedModels()` | `apiClient.get('/logs/models-loaded', { signal })` | raw `{ loaded }` | global-models |
| `sharedKeys.gpus()` | `settingsApi.getGpus({ signal })` | raw `GpusResponse` | shell-queue |
| `sharedKeys.adapterScan(folder)` | `adapterApi.scan(folder, { signal })` | raw scan response | global-models |
| `sharedKeys.browse(dir, filter?)` | `adapterApi.browse(dir, filter, { signal })` | raw `BrowseResponse` | shell-queue |
| `sharedKeys.workflowJob(id, after = 0)` | `apiClient.get('/workflows/jobs/<id>?after=<after>', { signal })` | `WorkflowReplay` | shell-queue |
| `sharedKeys.workflowDocument(id)` | `apiClient.get('/workflows/documents/<id>', { signal })` | `{ document: WorkflowDocument }` | shell-queue |
| `sharedKeys.preparation(id)` | `getYue2Preparation(id, { signal })` | `Yue2PreparationSummary` | training-data |
| `sharedKeys.preparationJob(id)` | `getYue2PreparationJob(id, { signal })` | job summary or null | training-data |
| `sharedKeys.peaks(audioUrl)` | `GET /api/audio/peaks?url=<audioUrl>`, 4xx/5xx thrown | raw `Peaks`, may carry `unsupported` | shell-queue |
| `sharedKeys.lrc(sidecarUrl(audio, '.lrc'))` | `fetchLrc(audio, { signal })` | LRC text, or null when missing | shell-queue |
| `sharedKeys.lyricsJson(sidecarUrl(audio, '.lyrics.json'))` | `fetchLyricsJson(audio, { signal })` | `LyricsJson`, or null when missing | shell-queue |
| `sharedKeys.scoreAbc(sidecarUrl(audio, '.score.abc'))` | `fetchScoreAbc(audio, { signal })` | YuE2 ABC lead sheet text, or null when missing | shell-queue |

Workflow keys always carry `after` in their filters, 0 for the whole history,
so a resumed read never shares an entry with a full one. Both workflow reads
throw `ApiError` (from `apiClient`), so the retry policy applies to them as to
any other read. `workflowApi` remains the command transport. The `adapterScan`
read is a POST, but a read-only one: run it on explicit request
(`enabled: false` plus `refetch`, or `query()` with `staleTime: 0`).
Sidecar ids are the sidecar URL, not the audio URL, so a mastered or
cache-busted variant shares the base render's entry. Peaks are keyed by the
audio URL as played; `forgetPeaks(url)` removes the entry when a file is
rewritten in place. Peaks use `staleTime: Infinity` and a 30-minute `gcTime`,
since only a rewrite changes them.

Song rows appear in several areas' feeds. `SONG_VIEWS` in `keys.ts` lists
every one of them: the two shared song keys, `shell-queue/recent-songs`,
`studios/recent-songs` and `cover-lyrics/cover-songs`. Any song write, delete,
completion or reconnect calls `queryManager.songs.views(client)` (or names the
`song:written` event in a `writeAndInvalidate` plan), or passes `SONG_VIEWS`
as a reconnect list, rather than naming feeds itself. An area may also edit
its own feed in place first. `queryManager.sidecars.all(client)` refreshes
lyric sidecars after a crop or retranscription.

`sharedKeys.songs()` is auth-scoped: read `token` and `sessionEpoch` from
`useAuth()`, build the key in render and gate the query on the token
(`enabled: !!token`). The token never goes in the key or the `queryFn`; the
transport adds the bearer itself. `sharedKeys.song(id)` also carries the
epoch, so the entry is dropped at sign-out. Since the axios request
interceptor attaches the bearer to every request, its read,
`songApi.get(id, { signal })`, carries the token too. The workflow keys are
auth-scoped like the songs list. Polling cadence stays per observer
(`refetchInterval`), so a consumer that wants faster VRAM updates sets its own
interval without touching the owner's options.

Each area keeps its own options, hooks and event mappings in
`ui/src/queries/<area>/`, with resources named `<area>/<name>`. The areas are
`training-data`, `training-runs`, `global-models`, `cover-lyrics`, `studios`
and `shell-queue`. `queries/sa3-previews/` is a folder of its own, but its
resource, `training-runs/sa3-previews`, belongs to the training-runs area.

## Transports

`ApiClient` (`httpClient.ts`) sends every JSON request through one axios
instance (`axiosClient.ts`), with a 60 s default timeout that slow routes
raise with `ClientOptions.timeout`. Its request interceptor adds the bearer
from the auth session accessor. Its response interceptor is the one place a
failure becomes an `ApiError` and gets logged or toasted (`handleFailure`):
GET failures never toast, since reads show their error inline and retry per
the policy above; aborts never surface; a request can opt out with
`ClientOptions.silent`. Multipart `upload()` stays on XHR for its progress
events and `streamEvents()` stays on fetch, and both report failures through
the same `handleFailure`. The filters and status messages are in
[tanstack-query.md](tanstack-query.md#errors).

Every read in `api.ts` takes a trailing `opts?: ReadOptions` (`{ signal }`).
Positional callers are unchanged, songs are still normalized, and errors are
still `ApiError` with `status` and the server's message.

`trainingApi.ts` routes by target:

- Functions on the "Train on" base take a trailing `TrainingReadOptions`
  (`{ worker, signal }`) for reads or `TrainingTarget` (`{ worker }`) for
  mutations. `worker: undefined` keeps the current selection, `null` is this
  PC, a name is that worker.
- Local-only routes (run records, review, scores, cleanup, mirrored previews)
  keep their local target and take `ReadOptions` on reads.
- `/api/workers` reads take `ReadOptions`; the worker is already an argument.
- `jobStreamUrl`, `sampleAudioUrl` and `previewAudioUrl` take an optional
  `TrainingTarget`, so a stream or player opened for worker A stays on A.

Training errors are now `ApiError` too, with the same message as before, so
retry policy can tell a 404 from a 503. Other domain clients get signals in
their owning area's slice.

Reads throw on failure. Never catch an error into an empty result inside a
`queryFn`: an empty list in the cache looks like real data and stays for
`staleTime`.

## Mutations

- Capture the target when the mutation is created and pass it explicitly
  (`TrainingTarget`). Never read the selection inside `onSuccess`.
- No automatic retry. A submit happens once, when the user asks.
- On success, invalidate the related lists and details for the captured scope.
  Outside a component, `writeAndInvalidate(client, command, plan)`
  (`queries/writes.ts`) runs the command and then the plan: `events(...)` for
  named events, `keysPlan(...)` for keys the write already knows. Uploads use
  the `useUpload()` mutation, which wraps `apiClient.upload` with progress and
  abort.
- Mutations take no signal. Cancelling a read aborts that read; it never
  cancels a server job. Stopping a job is its own explicit mutation.
- No generic get-or-submit query functions.
- Do not copy query data into unscoped Zustand state. A compatibility selector
  reads the cache by a captured key and never fetches on its own.

## Event streams

`bridgeEvents(queryClient, options)` subscribes through `sharedEventSource`,
one connection per URL per tab, next to the existing log, text and audio
reducers. The area maps each parsed event to effects:

| Effect | Behaviour |
|---|---|
| `invalidate` | keys to refetch; bursts coalesce to one invalidation per key per 50 ms |
| `update` | in-place edit, applied only when the entry exists and the updater returns a value |
| `once` | a side effect keyed by id, run at most once per client across replays and remounts |

`reconnectInvalidate` lists the authoritative snapshots to refetch after the
stream drops and reopens. The first open does nothing, because the server
replays its buffer on connect. The caller captures its scope when it starts
the bridge: the URL and every key `map` builds come from that capture. The
returned teardown unsubscribes and drops pending invalidations, and nothing
the old stream delivers afterwards touches the cache. Coalesced invalidations
flush through `invalidateKeys` in `invalidation.ts`.

```ts
useEffect(() => bridgeEvents(queryClient, {
  url: jobStreamUrl(jobId, { worker }),
  reconnectInvalidate: [resourceKey('training-runs/jobs', { worker })],
  map: (e: JobEvent) => e.type === 'done'
    ? { invalidate: [resourceKey('training-runs/jobs', { worker })],
        once: { id: `${worker}:${jobId}:done`, run: notifyDone } }
    : undefined,
}), [jobId, worker]);
```

## Sanctioned exceptions

The single exception list is owned by
[tanstack-query.md §Sanctioned exceptions](tanstack-query.md#sanctioned-exceptions-change-the-doc-before-fixing-one).
Three of its rows started here as architectural patterns: the disabled query as
a stream's cache slot, the queryable POST, and the side-effect-only observer.
New code copies one only for the same reason, and changing a row starts in that
table.

## Consumer manifest

Every non-test file under `ui/src` that imports a service module, calls
`fetch(` directly, opens a stream or runs an interval, at base `0ae4bf36`
plus this foundation. It was generated by scanning sources, so it counts
imports inside `services/` and type-only imports too. That is why it lists
more files than the 127 importers in the original inventory.

The list is append-only. A file added later gets a new row; a migrated file
keeps its row and its disposition changes. Areas report dispositions in their
slice hand-off, and the integration stage reconciles them here. Disposition
values: `open`, `query` (reads owned by Query), `transport`, `mutation`,
`stream` or `client` (a surviving call with its reason), `none` (nothing to
migrate).

Columns: services imported, raw `fetch(` calls, stream openers, intervals.

Dispositions were reconciled against the integration audit's ledger in
[architecture.md](architecture.md#consumer-reconciliation-ledger) at
`a9748d5d`, then updated for the R1, R2 and R5 fixes. A row now holds the
ledger's wording in place of the single values above. Counts in the first 187
rows are still those at `0ae4bf36`. The rows after them are the 23 files the
audit found, then files added since, counted when appended. The last 25 rows
were appended at `be3c85f8` (2026-10-23) for files that met the criteria but
had no row. Of the listed files only `hooks/usePolling.ts` no longer exists.

| File | Area | Services | fetch | SSE | Timers | Disposition |
|---|---|---|---|---|---|---|
| `App.tsx` | shell-queue | api, sharedEventSource | 7 | 1 | 2 | Query library; liveness transport, commands, log stream, queue projection observer |
| `audio/peaksClient.ts` | shell-queue | - | 1 | - | - | Query: shared `peaks`; failures uncached, `forgetPeaks` removes the entry (R5) |
| `components/assistant/AssistantPanel.tsx` | studios | assistantApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/ArtistSettingsPanel.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/CoverArtistGrid.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/CoverStudio.tsx` | cover-lyrics | api, coverWorkflowApi, lireekApi, studioDraftMirror, supersepApi, workflowApi, yue2CoverApi | 5 | - | 2 | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/SourcePanel.tsx` | cover-lyrics | api, supersepApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/Yue2Build.tsx` | cover-lyrics | yue2CoverApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/cover-studio/Yue2CoverPanel.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/Yue2CoverScore.tsx` | cover-lyrics | yue2CoverApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/create/AiGenerateModal.tsx` | shell-queue | inspireApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/create/CreatePanel.tsx` | shell-queue | api, studioDraftMirror, trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/create/Mm3ComposeButton.tsx` | shell-queue | lireekApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/create/trainingCreateDraft.ts` | shell-queue | trainingOperations | - | - | - | Explicit one-shot form handoff; command-time revision transport |
| `components/details/MetadataEditorModal.tsx` | shell-queue | api | 1 | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/details/Yue2CoverDrift.tsx` | shell-queue | yue2CoverApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/global-bar/AdaptersDropdown.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/CoverArtDropdown.tsx` | global-models | - | 3 | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/GenerationDropdown.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/GlobalParamBar.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/Mm3LmAdapterDropdown.tsx` | global-models | - | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/ModelsDropdown.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/MonitorBar.tsx` | global-models | - | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/PostProcessingDropdown.tsx` | global-models | api | 4 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/ProfilesModal.tsx` | global-models | api, exportImportApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/SeedManagerDrawer.tsx` | global-models | - | 5 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/global-bar/Yue2LmAdapterDropdown.tsx` | global-models | - | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/insta-gen/InspirePreview.tsx` | studios | inspireApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/insta-gen/InstaGenPanel.tsx` | studios | api, inspireApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/library/CoverArtPromptModal.tsx` | shell-queue | - | 3 | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/library/ImportTracksModal.tsx` | shell-queue | exportImportApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/AlbumGrid.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/AlbumHeader.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistGrid.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistPageSidebar.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/ArtistSidebar.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/CuratedProfileModal.tsx` | cover-lyrics | lireekApi | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/LyricStudioV2.tsx` | cover-lyrics | lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/PresetSettingsModal.tsx` | cover-lyrics | lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/ProfilesTab.tsx` | cover-lyrics | lireekApi, lyricWorkflowApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/PromptEditor.tsx` | cover-lyrics | lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/ProviderSelector.tsx` | cover-lyrics | lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/QueuePanel.tsx` | cover-lyrics | api, lireekApi, lyricWorkflowApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/RecordingsTab.tsx` | cover-lyrics | api, lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/RenderAcrossModal.tsx` | cover-lyrics | lireekApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/lyric-studio/SourceLyricsTab.tsx` | cover-lyrics | lireekApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/lyric-studio/WrittenSongsTab.tsx` | cover-lyrics | lireekApi, lyricWorkflowApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/playlistStore.ts` | cover-lyrics | studioDraftsApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/lyric-studio/useAudioGeneration.ts` | cover-lyrics | lireekApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/midi-studio/MidiPlayer.tsx` | studios | midiStudioApi | - | 1 | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/midi-studio/MidiStudio.tsx` | studios | api, midiStudioApi | 1 | - | 2 | Query consumer; forms/commands/streams stay client-owned |
| `components/midi-studio/midiSynth.ts` | studios | - | - | - | 1 | Client WebAudio scheduling |
| `components/model-manager/ModelManagerModal.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/model-manager/useDownloadStream.ts` | global-models | api | - | 1 | - | Query consumer; forms/commands/streams stay client-owned |
| `components/model-manager/useModelRegistry.ts` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/player/LyricsBar.tsx` | shell-queue | - | 1 | - | - | Query: shared `lrc` and `lyrics-json` (R5) |
| `components/player/SectionMarkers.tsx` | shell-queue | - | 1 | - | - | Query: shared `lrc` and `lyrics-json` (R5) |
| `components/player/TrimControls.tsx` | shell-queue | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/repaint-studio/RepaintStudio.tsx` | studios | api, repaintLayerWorkflowApi, studioDraftMirror, workflowApi | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/settings/DataFolderPanel.tsx` | shell-queue | dataFolderApi | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/settings/SettingsPanel.tsx` | shell-queue | api, lireekApi, stemStudioApi | 2 | - | - | Query consumer; env draft/save/restart lifecycle delegated to `useEnvSettingsForm` |
| `components/shared/FileBrowserModal.tsx` | shell-queue | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/LatentImport.tsx` | shell-queue | - | 1 | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/PresetSyncNotices.tsx` | shell-queue | presetCollection | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/StemMixer.tsx` | shell-queue | - | 1 | - | - | Binary media transport and client mixer |
| `components/shared/StudioDraftPicker.tsx` | shell-queue | studioDraftMirror, studioDraftsApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/shared/UnifiedRecentSongs.tsx` | shell-queue | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/VramIndicator.tsx` | global-models | - | 3 | - | 2 | Query consumer; forms/commands/streams stay client-owned |
| `components/song-builder/SongBuilder.tsx` | studios | api, workflowApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/song-builder/followSection.ts` | studios | workflowApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-builder/RecentBuilds.tsx` | studios | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-builder/StemBuilder.tsx` | studios | api, repaintLayerWorkflowApi, workflowApi | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/RecentExtractions.tsx` | studios | stemStudioApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/SourceSelector.tsx` | studios | - | 2 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/StemStudio.tsx` | studios | api, stemStudioApi, supersepApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/stem-studio/TrackSelector.tsx` | studios | stemStudioApi, supersepApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/AiContinuePresetModal.tsx` | studios | preferencesApi, presetCollection, settingDocument | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/StormLiveControls.tsx` | studios | inspireApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/storm/StormPage.tsx` | studios | - | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/terminal/TerminalPanel.tsx` | global-models | - | 1 | 1 | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/AuditionCard.tsx` | training-data | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/AuditionPlayer.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/BatchImportWizard.tsx` | training-data | api, trainingApi | - | - | - | Query: shared `browse`, `training-data/scan-preview` (R2); explicit pipeline command |
| `components/training-studio/BuildPanel.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/DatasetAssetChips.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/FolderPicker.tsx` | training-data | api | - | - | - | Query: shared `browse`, superseded navigation cancelled (R2) |
| `components/training-studio/LabelPanel.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/LmAdapterPicker.tsx` | training-data | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Mm3RunsPanel.tsx` | training-runs | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Mm3TrainCard.tsx` | training-runs | trainingApi, trainingRecipesApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/MonitorPanel.tsx` | training-runs | trainingApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/NewDatasetWizard.tsx` | training-data | trainingApi | - | - | - | Query: `training-data/scan-preview`, superseded scan cancelled (R2); explicit create command |
| `components/training-studio/PipelineStageProgress.tsx` | training-data | trainingApi | - | - | - | Query job; selected-worker projection |
| `components/training-studio/PreprocessOptionsForm.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/PreprocessPanel.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/PreprocessVariantCard.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/RefinePanel.tsx` | training-runs | trainingApi, trainingReviewApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/ReviewPanel.tsx` | training-runs | stemStudioApi, trainingApi, trainingReviewApi | - | - | 3 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/SampleDrawer.tsx` | training-data | trainingApi | - | - | - | Query: `training-data/sample-mm3`/`sample-yue2` seed the editor once per key; saves update the entry (R2) |
| `components/training-studio/SampleGrid.tsx` | training-data | trainingApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/SampleRow.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/SendToLyricStudio.tsx` | training-data | trainingApi | - | - | - | Query: `training-data/lyric-studio-preview` seeds the form (R2); explicit export command |
| `components/training-studio/TrainDitForm.tsx` | training-runs | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/TrainLmForm.tsx` | training-runs | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/TrainPanel.tsx` | training-runs | trainingApi, trainingRecipesApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/TrainingRunStats.tsx` | training-runs | - | - | - | - | Active job read from the job cache via `useActiveTrainingJob`; no independent HTTP read |
| `components/training-studio/TrainingWorkerBar.tsx` | training-data | trainingApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/WorkersPanel.tsx` | training-runs | trainingApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2AitkTrainCard.tsx` | training-runs | api, preferencesApi, sharedEventSource, trainingApi, trainingOperations, trainingPreparationApi | - | 1 | 5 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2ArTrainCard.tsx` | training-runs | trainingApi, trainingRecipesApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2BatchPanel.tsx` | training-runs | trainingApi | - | - | 2 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2ClearPreparedCard.tsx` | training-data | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2LadderReview.tsx` | training-runs | trainingApi, trainingReviewApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2OptimisePanel.tsx` | training-runs | trainingApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2OptimizerFields.tsx` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/Yue2SplitChart.tsx` | training-data | sharedEventSource, trainingApi | - | 1 | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/Yue2TrainCard.tsx` | training-runs | trainingApi, trainingRecipesApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/Yue2TrainStages.tsx` | training-runs | trainingPreparationApi | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/sendAuditionToCustomGen.ts` | training-data | trainingApi, trainingOperations | 1 | - | - | Query: local `training-data/dataset` revision read (R2); explicit draft command |
| `components/training-studio/useMm3Status.ts` | training-runs | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/useYue2ArStatus.ts` | training-runs | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/useYue2Status.ts` | training-runs | trainingApi | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/training-studio/yue2ChartData.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/yue2JointCommand.ts` | training-data | trainingApi, trainingOperations, trainingPreparationApi | - | - | - | Area service/store adapter; Query reads, explicit commands/client state |
| `components/training-studio/yue2JointPresets.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `context/AuthContext.tsx` | foundation | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `hooks/useAssistantActions.ts` | studios | assistantApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `hooks/useCapabilities.ts` | global-models | - | - | - | 1 | Query consumer; forms/commands/streams stay client-owned |
| `hooks/useEventSource.ts` | global-models | sharedEventSource | - | 2 | - | Shared stream/reducer; no snapshot fetch |
| `hooks/usePluginRegistry.ts` | global-models | - | 1 | - | - | Query consumer; forms/commands/streams stay client-owned |
| `hooks/usePolling.ts` | shell-queue | - | - | - | 1 | Removed |
| `hooks/useStreamAudio.ts` | studios | analysis | 4 | - | 1 | Audio stream/commands; client recording clock |
| `queries/contract.ts` | foundation | httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/keys.ts` | foundation | httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/sseBridge.ts` | foundation | sharedEventSource | - | - | - | Query owner/policy; area contract applies |
| `services/api.ts` | foundation | httpClient, workflowApi | 8 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/assistantApi.ts` | studios | - | 2 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/audioQueueApi.ts` | shell-queue | - | 2 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/coverWorkflowApi.ts` | cover-lyrics | workflowApi | 2 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/dataFolderApi.ts` | shell-queue | - | 1 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/exportImportApi.ts` | shell-queue | api | 3 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/httpClient.ts` | foundation | - | 4 | 2 | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/inspireApi.ts` | studios | workflowApi | 9 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/lireekApi.ts` | cover-lyrics | - | 2 | - | - | Query-backed reads; commands/working-copy reducers |
| `services/lyricWorkflowApi.ts` | cover-lyrics | workflowApi | 1 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/midiStudioApi.ts` | studios | - | 8 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/preferencesApi.ts` | shell-queue | workflowApi | 1 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/presetCollection.ts` | shell-queue | preferencesApi, workflowApi | - | - | - | Query-backed reads; commands/working-copy reducers |
| `services/repaintLayerWorkflowApi.ts` | studios | workflowApi | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/settingDocument.ts` | shell-queue | preferencesApi, workflowApi | - | - | - | Query-backed reads; commands/working-copy reducers |
| `services/sharedEventSource.ts` | foundation | - | - | 2 | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/stemStudioApi.ts` | studios | - | 8 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/streamSessionsApi.ts` | studios | - | 2 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/studioDraftMirror.ts` | cover-lyrics | studioDraftsApi | - | - | - | Query-backed reads; commands/working-copy reducers |
| `services/studioDraftsApi.ts` | cover-lyrics | workflowApi | - | - | - | Query-backed reads; commands/working-copy reducers |
| `services/supersepApi.ts` | studios | - | 5 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingApi.ts` | foundation | httpClient | 3 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingOperations.ts` | training-data | trainingApi | 1 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingPreparationApi.ts` | training-data | trainingOperations | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingRecipesApi.ts` | training-data | trainingOperations | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/trainingReviewApi.ts` | training-runs | trainingApi, trainingOperations | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/workflowApi.ts` | shell-queue | - | 2 | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/yue2CoverApi.ts` | cover-lyrics | - | 1 | - | - | Query-backed reads; commands/working-copy reducers |
| `stores/audioGenQueueStore.ts` | shell-queue | api, audioQueueApi, lireekApi, workflowApi | 1 | - | - | Query reads and the queue-snapshot reducer; client executor/scheduler and media transport |
| `stores/audioQueueStoreTestFake.ts` | shell-queue | preferencesTestFake | - | - | - | Test fake; excluded from production |
| `stores/backendStore.ts` | global-models | - | 5 | - | - | Query exact-key projections and selection commands |
| `stores/coverStemsStore.ts` | cover-lyrics | supersepApi | - | - | - | Query separation waiter; client execution/mixer |
| `stores/discoStore.ts` | shell-queue | - | 1 | - | - | Media-sidecar playback buffer transport |
| `stores/globalParamsStore.ts` | global-models | - | 1 | - | - | Client forms/persistence and active-model mirror command |
| `stores/mm3StreamStore.ts` | shell-queue | - | 1 | - | - | Stream reducer and explicit execution |
| `stores/postProcessStore.ts` | shell-queue | api | - | - | 1 | Query observer; explicit command and completion reducer |
| `stores/streamingStore.ts` | cover-lyrics | lireekApi, lyricWorkflowApi | - | - | - | Query-backed recovery; stream execution reducer |
| `stores/trainingStore.ts` | training-data | trainingApi, trainingOperations, trainingPreparationApi | - | - | - | Query projection and commands; preparation follow reads shared `yue2-preparation`/`-job` (R1) |
| `stores/vstChainStore.ts` | global-models | api, preferencesApi, presetCollection | 4 | - | - | Query projections/presets and editable command state |
| `utils/downloadTrack.ts` | shell-queue | exportImportApi | - | - | - | Explicit export command and download transport |
| `utils/heapWatch.ts` | shell-queue | - | - | - | 1 | Client heap sampling timer |
| `utils/lrcUtils.ts` | shell-queue | - | 1 | - | - | Transport: canonical read for shared `lrc`; parsing is client (R5) |
| `utils/wordLrcUtils.ts` | shell-queue | - | 1 | - | - | Transport: canonical read for shared `lyrics-json`; lookup is client (R5) |
| `utils/yue2CaptionSource.ts` | shell-queue | - | 1 | - | - | Query snapshot and captured command assembly |
| `components/global-bar/VstChainDropdown.tsx` | global-models | api | - | - | - | Query consumer; forms/commands/streams stay client-owned |
| `components/shared/SongActionsMenu.tsx` | shell-queue | - | - | - | - | Query consumer (shared `score-abc` once open); forms/commands/streams stay client-owned |
| `queries/cover-lyrics/reads.ts` | cover-lyrics | httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/cover-lyrics/songs.ts` | cover-lyrics | api, httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/cover-lyrics/workflows.ts` | cover-lyrics | httpClient, workflowApi | - | - | - | Query owner/policy; area contract applies |
| `queries/global-models/options.ts` | global-models | httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/global-models/presets.ts` | global-models | preferencesApi, workflowApi | - | - | - | Query owner/policy; area contract applies |
| `queries/shell-queue/preferences.ts` | shell-queue | preferencesApi, workflowApi | - | - | - | Query owner/policy; area contract applies |
| `queries/shell-queue/queue.ts` | shell-queue | api, audioQueueApi, httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/shell-queue/projection.ts` | shell-queue | - | - | - | - | Query owner: dynamic `refetchInterval` for the server audio queue; `applyServerItems` reconciles snapshots |
| `queries/shell-queue/settings.ts` | shell-queue | api, dataFolderApi, stemStudioApi | - | - | - | Query owner/policy; area contract applies |
| `queries/shell-queue/songs.ts` | shell-queue | api | - | - | - | Query owner/policy; area contract applies |
| `queries/studios/lyrics.ts` | studios | httpClient | 1 | - | - | Query owner/policy; area contract applies |
| `queries/studios/midi.ts` | studios | midiStudioApi | - | - | - | Query owner/policy; area contract applies |
| `queries/studios/resources.ts` | studios | api, assistantApi, inspireApi, stemStudioApi, streamSessionsApi, supersepApi | - | - | - | Query owner/policy; area contract applies |
| `queries/studios/shared.ts` | studios | api, httpClient | - | - | - | Query owner/policy; area contract applies |
| `queries/studios/workflows.ts` | studios | httpClient, workflowApi | - | - | - | Query owner/policy; area contract applies |
| `queries/training-data/queries.ts` | training-data | api, trainingApi, trainingPreparationApi | - | - | - | Query owner/policy; area contract applies |
| `queries/training-runs/events.ts` | training-runs | trainingApi | - | - | - | Query owner/policy; area contract applies |
| `queries/training-runs/hooks.ts` | training-runs | trainingApi | 1 | - | - | Query owner/policy; `useTrainingJobProjection` owns the active job's stream and cache, `useActiveTrainingJob` reads that cache, and the YuE2 chart cards keep their own metric-only subscription |
| `queries/training-runs/queries.ts` | training-runs | api, trainingApi, trainingOperations, trainingPreparationApi, trainingRecipesApi, trainingReviewApi | - | - | - | Query owner/policy; area contract applies |
| `stores/playbackStore.ts` | shell-queue | - | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `utils/mm3CaptionSource.ts` | shell-queue | - | - | - | - | Query-backed lireek read; client caption assembly |
| `utils/paramProfiles.ts` | shell-queue | - | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `queries/shell-queue/lyrics.ts` | shell-queue | - | 1 | - | - | Query owner for shared `lrc`/`lyrics-json` (R5); owner and transport for shared `score-abc` |
| `components/compare-studio/CompareStudio.tsx` | compare | compareApi | - | - | - | Query consumer: sessions, session detail and library search; create/delete are mutations |
| `components/compare-studio/DiffTrace.tsx` | compare | compareApi | - | - | - | Props only; the parent owns the diff query |
| `components/compare-studio/SessionView.tsx` | compare | compareApi | - | - | - | Query consumer: per-take `analysis` and the active `diff`; archive/delete are mutations |
| `components/compare-studio/TakeRow.tsx` | compare | compareApi | - | - | - | Query consumer: score and remove-take mutations over the session cache |
| `components/compare-studio/useAbEngine.ts` | compare | - | 1 | - | - | Binary audio transport for the A/B engine's takes |
| `components/cover-studio/PrepareLyrics.tsx` | cover-lyrics | yue2CoverApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/cover-studio/PrepareStructure.tsx` | cover-lyrics | yue2CoverApi | - | - | 1 | Client/types/props; the interval loops a recording's playback window, no server reads |
| `components/cover-studio/SectionPlayback.tsx` | cover-lyrics | - | - | - | 2 | Client audio playback timers; no server reads |
| `components/midi-studio/MidiExportPanel.tsx` | studios | midiStudioApi | - | - | - | Query consumer (`studios/midi-notes`); export and ABC lead-sheet commands, export invalidates notes and jobs |
| `components/settings/AdapterReferencesAction.tsx` | shell-queue | lireekApi | - | - | - | Explicit cleanup command; `lireekApi` invalidates the lireek snapshots after it |
| `components/settings/UpdatesPanel.tsx` | shell-queue | updatesApi | - | - | - | Query: `shell-queue/updates`, polled every 2 s while a check runs; Check now is a mutation that writes its result into the entry |
| `components/storm/DriftPanel.tsx` | studios | - | - | - | - | Query consumer: 2 s `stormDriftOptions` snapshot poll while streaming; the four drift control POSTs are silent commands ([tanstack-query.md](tanstack-query.md) P4) |
| `components/training-studio/Sa3TrainCard.tsx` | training-runs | api, sa3PreviewApi, trainingApi | - | - | - | Query consumer: component-local `training-data/sa3-status`, `training-runs/sa3-previews`, `global-models/sa3-adapters` and audition `shell-queue/generation-status`; start, cancel, render and audition are mutations; invalidates through `queryManager` ([tanstack-query.md](tanstack-query.md) F4#17) |
| `components/training-studio/plugins/ace.tsx` | training-runs | trainingApi, trainingRecipesApi | - | - | - | Query consumer (ACE LM/DiT status and recipes); scoped training-store actions submit |
| `components/training-studio/plugins/chips.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/trainGates.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/trainingPluginTypes.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `components/training-studio/trainingTargets.ts` | training-data | trainingApi | - | - | - | Client/types/props or scoped store projection; no independent HTTP read |
| `queries/sa3-previews/queries.ts` | training-runs | sa3PreviewApi | - | - | - | Query owner/policy; area contract applies |
| `queries/writes.ts` | foundation | httpClient | - | - | - | Query owner/policy: shared write helpers (`writeAndInvalidate`, `useUpload`) |
| `services/axiosClient.ts` | foundation | apiError | 1 | - | - | Transport: the axios instance, its fetch adapter and the global error catch |
| `services/compareApi.ts` | compare | httpClient | - | - | - | Transport; `queries/compare` owns the reads and commands |
| `services/sa3PreviewApi.ts` | training-runs | httpClient | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `services/updatesApi.ts` | shell-queue | httpClient | - | - | - | Transport; area Query caller owns reads, commands/streams explicit |
| `queries/compare/queries.ts` | compare | compareApi | - | - | - | Query owner for sessions, session detail, picker search, analysis and diff |
| `queries/compare/writes.ts` | compare | compareApi | - | - | - | Query owner: compare commands and the optimistic score cache follow-up |
| `components/training-studio/useHeartmulaController.ts` | training-data | trainingApi | - | - | - | Query consumer (`training-data/heartmula-status`, `heartmula-previews`); prepare/train/batch are commands, the preview retry is a mutation |
| `components/settings/useEnvSettingsForm.ts` | shell-queue | api | - | - | - | Query consumer (`shell-queue/env`, GPUs) and mutation owner (save/restart); draft seeded only while clean, `queryManager.env.values` invalidates |

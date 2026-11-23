# Cover and lyric server state

This area uses the stable client and canonical key factory in `../client.ts`
and `../keys.ts`. The area owns the raw wire snapshots for lireek, cover
transcription/build sessions, studio drafts and lyric workflows. Editable
lyrics, score edits, form fields, playback and workflow execution stay local.
The shared studio-drafts client also serves Create, Repaint and Insta-Gen;
their readers share these keys without importing area hooks.

## Keys and reads

`snapshotOptions(domain, path, options?, staleTime?)` splits a local
API path into pathname and normalized search filters. Keys use
`cover-lyrics/{lireek,cover,drafts,workflow}`, the pathname as id, the captured
API base and the auth epoch whenever a session token is set: the transport
sends the bearer whenever one exists, so the key's auth scope follows the same
accessor (`getAuthToken()`). Tokens never appear in keys. The cached shape is
the endpoint's raw JSON. The existing
public lireek wrappers retain their array-to-object projections and positional
arguments; every read now accepts trailing `ReadOptions` and consumes its
signal. Failed HTTP reads throw `ApiError`, including status and revision
conflict details. Commands run once, outside query functions, and invalidate
related snapshots after success. Draft revision/conflict logic is unchanged. Leaving an editor or changing its
auth token cancels the pending load and prevents a late draft from being applied.

Cover's filtered `/songs?source=cover-studio` response has a separate normalized
`cover-lyrics/cover-songs` key. It is not the full-library wire shape. Successful
song deletion cancels the old list, changes cached rows only after the server
accepts deletion, cancels again and invalidates. Completion and song writers
also call foundation's `queryManager.songs.views` (through `songsChanged()` in
`reads.ts`), refreshing canonical `songs`, `song`, `shell-queue/recent-songs`,
`studios/recent-songs` and the Cover feed.
The render waiter invalidates these views when a `cover-render` job succeeds,
including renders completed without a stream subscriber.
The reference browser uses `global-models/references`; the LM adapter picker
uses `global-models/lm-adapters` with the same folder filter as the global bar.
Legacy generation snapshots use `shell-queue/generation-status` and the captured
job id, so queue and recording readers share the status request.

## Polling and streams

Query owns transcription, section-build and awaited cover/generation status
polls. They run every 1.5 seconds while active, stop at terminal status and use
foundation's capped 30-second error backoff. An awaited execution may retain
an observer while the view is absent, preserving the serial generation queue.
Observer teardown aborts a read and never implicitly cancels a server job.
Explicit user cancellation keeps its existing server command. Generation
completion side effects run once and the existing generation timeout remains.

Text/audio streams and lyric workflow sequencing remain. The workflow bridge
captures the auth scope and job id, reconciles on reconnect and coalesces
snapshot/event invalidation over 50 ms. The existing stream reader suppresses
replayed event sequences. Old-session streams cannot deliver effects into a
new session. The streaming queue remains a client execution reducer, not a
second HTTP snapshot cache. Cover stems use the studios-owned Query separation
waiter; a local epoch also rejects a late start response after clear/replace.

Workflow jobs use `sharedKeys.workflowJob(id, after)` with an auth epoch and
an explicit `after`, including zero. Their cached shape is raw `WorkflowReplay`.
Documents use `sharedKeys.workflowDocument(id)` and cache `{ document }`.
Events invalidate every `after` variant for the captured job and all documents
in its auth scope, plus every song view, every lireek snapshot and that
scope's drafts. No `cover-lyrics/workflow-job` or `workflow-document` cache
remains. The workflow list keeps its area key and `kind` filter.

This area has no audio lyric-sidecar reader or cache. Its editable lyrics and
dataset metadata are different resources. Audio sidecars use foundation's
`sharedKeys.lrc(sidecarUrl(audio, '.lrc'))` and `sharedKeys.lyricsJson(...)`:
raw text or word timings, with null for a missing sidecar. Crop/retranscription
invalidation is owned by the shared sidecar contract, not the editable lyrics.

## Consumer manifest

Paths are under `src/` unless noted. Type-only imports and form-only children
remain API consumers in the historical import count, but own no server reads.

| Consumer | Disposition |
|---|---|
| `cover-studio/CoverStudio.tsx` | Observes bulk artists, all lyrics sets and all presets; observes readiness/transcription; source metadata keyed by captured input; cover/legacy jobs observed by Query; source/caption/lyric edits and serialized execution remain local |
| `cover-studio/SourcePanel.tsx` | Canonical reference list query, enabled while browser opens; upload invalidates |
| `cover-studio/Yue2Build.tsx` | Session query keyed by captured session id; polls while pending; successful commands update their returned session |
| `cover-studio/ArtistSettingsPanel.tsx`, `CoverArtistGrid.tsx`, `Yue2CoverPanel.tsx`, `Yue2CoverScore.tsx`, `PrepareLyrics.tsx`, `PrepareStructure.tsx`, `SectionPlayback.tsx` | Types, forms, score/audio display only; the Prepare and section players' intervals time audio playback windows |
| `lyric-studio/LyricStudioV2.tsx` | Observes artists, selected artist's albums, album detail, bulk queue lists and presets; URL restore guarded against navigation; image enrichment stays a command, attempted once per resource identity |
| `lyric-studio/ProviderSelector.tsx` | Query providers, shared across both picker variants; imperative provider fallback uses the same Query snapshot |
| `lyric-studio/PromptEditor.tsx` | Prompt query; editable buffer local; save/reset refresh and reset uses the fresh returned value |
| `lyric-studio/PresetSettingsModal.tsx` | Preset and MM3-adapter queries; hydrate editable form for captured album; no callback-identity-driven fetch loop |
| `lyric-studio/QueuePanel.tsx` | Preset/all-generation queries, canonical folder-scoped LM adapters; counts derived from one generation snapshot; submissions stay explicit |
| `lyric-studio/RenderAcrossModal.tsx` | Bulk sets/presets/generations queries; chosen target and song remain form state |
| `lyric-studio/RecordingsTab.tsx` | Cancellable Query-backed recording reads and canonical generation statuses; grouped playback projection local; old album responses cannot replace the selected group |
| `lyric-studio/CuratedProfileModal.tsx` | Query-backed source reads; curated build remains a streaming command and invalidates on completion |
| `lyric-studio/ProfilesTab.tsx`, `SourceLyricsTab.tsx`, `WrittenSongsTab.tsx` | Receive authoritative parent detail props; mutations/stream results reconcile via lireek/workflow cache; source-caption helpers owned by shell-queue keep their compatibility projections |
| `lyric-studio/useAudioGeneration.ts` | Intent capture and enqueue stay commands; preset lookup uses the same Query-backed client |
| `stores/playlistStore.ts` | Editable playlist, pending commands and revision conflict reducer retained; authoritative playlist/import snapshots use shared draft queries |
| `lyric-studio/StreamingPanel.tsx`, `InlineAudioQueue.tsx` | Client execution/progress display only |
| `lyric-studio/ArtistGrid.tsx`, `AlbumGrid.tsx`, `AlbumHeader.tsx`, `ArtistSidebar.tsx`, `ArtistPageSidebar.tsx` | Types/forms/parent snapshots; provider picker delegates to its Query owner |
| `stores/coverStemsStore.ts` | Separation execution/mixer state retained; Query owns separation snapshots; late submission guard prevents resurrection |
| `stores/streamingStore.ts` | Existing run/queue reducers; Query-backed list/get recovery; stream bridge handles replay/invalidation |
| Other cover/lyric files | Local score parsing, file input, URL helpers, resize handlers or prop-only presentation; no HTTP snapshot owner |

Surviving raw fetches are the `/analyze` command (`CoverStudio.tsx`), curated
profile streaming (`CuratedProfileModal.tsx`) and the SSE text consumer
(`consumeSSE` in `services/lireekApi.ts`). Uploads go through
`apiClient.upload()`, and score approval/save commands through `apiClient`.
They never run as get-or-submit query functions. The only component intervals
left in either studio are the audio playback timers in `PrepareStructure.tsx`
and `SectionPlayback.tsx`; none of them polls the server. Drafts and playlists
deliberately hydrate an editable working copy; Query remains their sole HTTP
read/cache owner. External model and caption helpers remain with their row
owners.

## Runtime acceptance handoff

No app was launched by this slice. Tester owns API/network acceptance on
`http://localhost:3000`; human visual/listening acceptance remains separate.

| Entry/resource | Expected requests and cadence |
|---|---|
| Cover catalogue cold entry | `/api/lireek/artists` once, `/lyrics-sets` once, `/presets` once; zero per-set `/preset` reads and zero per-artist lyrics-set scans |
| Cover return within 30 seconds | Zero additional catalogue requests while fresh; a stale return makes at most one request per catalogue key |
| Cover restored artist | Bulk preset hydration only; zero caption/render/transcription submissions on entry/return |
| Readiness | `/api/yue2-cover/readiness` once when stale and YuE2 view is enabled |
| Source metadata | POST `/api/yue2-cover/source-metadata`, read-only snapshot keyed by source input; no job starts |
| Transcription | `/api/yue2-cover/transcriptions/:id` every 1.5 seconds while queued/running; terminal stops |
| Section build | `/api/yue2-cover/build/:id` every 1.5 seconds while pending; idle session has no interval |
| Cover workflow | `/api/workflows/jobs/:id?after=0` every 1.5 seconds while active; terminal stops |
| Legacy render | `/api/generate/status/:id`, shared with shell queue, every 1.5 seconds while active; terminal stops |
| Lyric entry | `/api/lireek/artists` and `/presets` once per stale key; selected artist adds one filtered `/lyrics-sets` and selected album one `/full-detail` |
| Lyric queue/render-across | One bulk `/lyrics-sets`, `/profiles`, `/presets`, `/generations/all` per enabled stale key, shared with other consumers |
| Preset editor | `/lyrics-sets/:id/preset` only for the opened album; optional `/api/training/lyrics-sets/:id/mm3-adapters` |
| Prompts/providers/references | One request per stale key when enabled, no interval |
| Draft picker/load | `/api/studio-drafts/drafts?studio=...` and `/drafts/:id`, per captured studio/id; no job replay on load |
| Lyric workflow recovery | `/api/workflows/jobs?kind=lyric-batch`, then captured job reads/events; reconnect invalidates snapshots |

1. Compare cold Cover entry and fresh return with HOTSTEP-96's baseline of
   690 list reads and 826 per-set preset reads. Check the three catalogue
   endpoints and confirm no per-id fan-out. Repeat with a restored artist in
   ACE and YuE2 modes and verify zero submission POSTs from remount alone.
2. Delay source A metadata/transcription, switch to B, release A. B's source,
   lyrics and score stay selected. Repeat with a slow section-build session
   and a session switch. Clear stems while their start request is delayed.
3. Open a preset, edit without saving and trigger unrelated renders. The form
   stays edited. Save/delete the preset, build/delete a profile, save/reset a
   prompt and save/load a named draft. Reopen other consumers and check the
   affected snapshots refresh without discarding unrelated lyric edits.
4. Drop and reconnect a lyric workflow stream; replay its terminal frame.
   Confirm one completion effect and no repeated start/render submission.
   Switch away/back during active work and confirm one observer/stream per key.
5. Delay a cover list, delete a row and release the old list. It stays deleted.
   Refuse a delete and confirm the row remains. Finish a cover and verify the
   library, recent songs and cover list all refresh. Check generation progress,
   elapsed time, completion and the next serial queue item remain unchanged.
6. Observe idle/terminal jobs for ten seconds and confirm no extra polls.
   Simulate repeated 5xx cycles and verify capped backoff. Record unavailable
   GPU, worker, download or listening cases as unverified.

Checks for this slice use `npx tsc -b` and the full UI suite with the server
loader. `--test-force-exit` is needed when imported app modules retain handles:

```powershell
node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-force-exit --test-timeout=60000 "src/**/*.test.ts"
```

# Studio server state

The studios area uses the client, keys and policy in `../client.ts`,
`../keys.ts` and `../contract.ts`. Query owns read snapshots. Source choices,
composer fields, lyric edits, mixer controls, playback and workflow execution
remain client state. Read functions in the seven owned studio clients accept
trailing `ReadOptions` and pass the signal to the shared HTTP client.

## MIDI polling and deletion

`midiJobsOptions` fetches the list and the progress of active jobs as one
cancellable query cycle. Repeated reads share the in-flight promise. It polls
every 1.5 seconds while any job is queued or transcribing, and stops for done,
failed, cancelled or empty lists. Consecutive failed cycles back off through
the foundation policy, capped at 30 seconds. The download status query uses
the same policy while a model downloads. There are no component poll timers.

Deletion cancels the list and the selected job's notes before the request,
then cancels again after success to cover a poll started during the delete.
Only successful deletion filters the list and removes the notes cache. The
list is invalidated on either outcome. Query cancellation prevents a delayed
old response from putting the deleted row back, including a transport that
resolves after abort. Failed deletion preserves the cached row and reports
the error. Commands do not retry automatically.

Finished piano-roll notes use a query. Live notes retain their SSE reducer,
batch flush and WebAudio transport. Open, error and terminal events reconcile
the job snapshot. EventSource and its flush timer close on unmount.

## Other consumers

| Consumer | Disposition |
|---|---|
| `midi-studio/MidiStudio.tsx` | Status/jobs queries and canonical shared songs; upload/transcribe/token/download/delete commands |
| `midi-studio/MidiPlayer.tsx` | Notes query; live note SSE and playback remain streams/client state |
| `midi-studio/MidiExportPanel.tsx` | Notes query (tempo grid, channels); MIDI re-export and ABC lead-sheet commands, and a re-export invalidates the notes and jobs |
| `midi-studio/midiSynth.ts` | WebAudio scheduling only |
| `stem-studio/StemStudio.tsx` | Model query, Query-observed extraction progress and result; editable mixer remains local |
| `stem-studio/SourceSelector.tsx` | Filtered recent-song query; upload command |
| `stem-studio/RecentExtractions.tsx` | Extraction list query; successful delete updates and invalidates it |
| `stem-studio/TrackSelector.tsx` | Constants and form choices only |
| `stem-builder/StemBuilder.tsx` | Model query and workflow snapshots; source/layer composition stays local |
| `stem-builder/RecentBuilds.tsx` | Selects from canonical shared songs; delete cancels, updates after success and invalidates |
| `song-builder/SongBuilder.tsx` | Project list/detail cache; drafts stay local, commands refresh the captured project |
| `song-builder/followSection.ts` | Existing sequenced workflow stream reconciles project snapshots; obsolete projects cannot receive the view |
| `repaint-studio/RepaintStudio.tsx` | Shared songs, source-keyed lyrics, workflow snapshots; region/lyric drafts stay local |
| `insta-gen/InstaGenPanel.tsx` | Providers, prompt, workflow job/document and shared song detail queries; editable preview remains a form |
| `insta-gen/InspirePreview.tsx` | Form props and types only |
| `storm/StormPage.tsx` | Observes the canonical plugins key/read directly; dropdown projection uses `select` |
| `storm/StormLiveControls.tsx` | LLM generation commands and audio controls |
| `storm/DriftPanel.tsx` | Observes `stormDriftOptions()` with a 2 s `refetchInterval` while the stream plays; drift/control posts are silent commands |
| `compare-studio/CompareStudio.tsx` | Reads `compareQueries.sessions`/`session`/`songSearch`; create and delete are mutations in `queries/compare/writes.ts` |
| `compare-studio/SessionView.tsx`, `TakeRow.tsx` | Session detail, per-take analysis and the active diff through `queries/compare`; score, remove-take, archive and delete are mutations |
| `compare-studio/DiffTrace.tsx` | Props only; the parent owns the diff query |
| `compare-studio/useAbEngine.ts` | Take audio fetched as media for the A/B engine; no snapshot |
| `storm/AiContinuePresetModal.tsx` | Uses shared `PresetCollection`/`SettingDocument`; their Query adoption belongs to shell-queue's owned service files |
| `assistant/AssistantPanel.tsx` | Provider query; chat remains a streaming command |
| `hooks/useStreamAudio.ts` | STORM stream/control/stop commands, analysis commands, WebAudio and device recording timers |
| `hooks/useAssistantActions.ts` | Applies client settings only |

`shared.ts` keeps the exact canonical `{ songs }` wire shape. Models use the
global-models options so model installation invalidates studio selectors too.
The recent-song endpoint has a distinct response and filtered key, with no
worker/backend scope because it is a local library resource. `SONG_VIEWS`
includes `resourceKey('studios/recent-songs')`, so every song write in the
shell invalidates it.

Workflow reads use `sharedKeys.workflowJob(id, after)` and
`sharedKeys.workflowDocument(id)`, with the auth session and captured id. Their
existing stream reducer keeps event sequencing and replay suppression.
Snapshots and replay events invalidate the captured job, document and songs
keys with 50 ms burst coalescing, including every `after` variant of the job
and all `SONG_VIEWS`; terminal completion flushes the pending
timer. A delayed stream from an obsolete auth epoch cannot deliver effects
or invalidate the new epoch. Recovered repaint/layer subscriptions abort on
unmount without cancelling the server job or clearing its recovery pointer.

Repaint lyrics use `sharedKeys.lrc(sidecarUrl(audio, '.lrc'))` and the
canonical `fetchLrc` read, sharing player reads across track variants.
Missing sidecars cache null. Crop and retranscription invalidate that entry.

The extraction/separation wait helpers use Query observers rather than their
own polling loops. They detach at a terminal status, nonretryable read error,
explicit observation abort or configured timeout. Active execution may keep
an observer while its view is absent. The legacy inspire helper retains its
four-minute timeout. Caller-supplied extraction/separation intervals still
work; their default is now 1.5 seconds. Unused exported read entrypoints stay
transports for their future owning consumers.

Shared preset and draft services remain with shell-queue and cover-lyrics
respectively. This slice does not duplicate their readers or change their
revision/conflict logic. These two shared-service migrations must be included
when the integration owner reconciles the foundation manifest.

## Runtime acceptance handoff

Run on `http://localhost:3000` in the assigned runtime job. This slice does not
launch the app or claim visual/network acceptance.

| View/resource | Expected endpoints and cadence |
|---|---|
| MIDI idle | `/api/midi-studio/status`, `/jobs`, each once when stale; no interval after a terminal/empty list |
| MIDI active | `/api/midi-studio/jobs` and `/:id/progress` for active rows, one nonoverlapping cycle per 1.5 s; capped error backoff |
| MIDI download | `/api/midi-studio/status` every 1.5 s while downloading; stop when complete |
| MIDI preview | `/:id/notes` for completed rows; `/:id/stream` for live notes, closed on unmount |
| MIDI/Repaint/Recent builds | `/api/songs`, same canonical auth-scoped cache; no independent intervals |
| Stem source picker | `/api/songs/recent?limit=50&source=...`, key per normalized filter, enabled while open |
| Stem/model selectors | `/api/models`, shared with global-models; its owner controls availability polling |
| Extraction | `/api/stem-studio/jobs`; `/:id/progress` every 1.5 s while active, then `/:id/result`; terminal list invalidation |
| Separation helper | `/api/supersep/:id/progress` every 1.5 s while running, then `/:id/result` |
| Song Builder | `/api/builder/projects`, `/projects/:id`; workflow variant/end/reconnect reload the captured project, no independent interval |
| Repaint/Layer/Insta-Gen | `/api/workflows/jobs/:id` and `/documents/:id`; existing `/jobs/:id/events` stream drives invalidation |
| Insta-Gen | `/api/inspire/llm/providers`, `/llm/prompt`; prompt save/reset invalidates its query |
| Assistant | `/api/assistant/providers`; `/chat` remains a streaming POST |
| Storm | Shared `/api/plugins` registry; audio `/api/generate/storm/stream`, `/control`, `/stop` remain command/stream traffic; the Drift panel polls `GET /api/generate/storm/control?streamId=` every 2 s while playing |

1. Delay a MIDI list/progress response beyond 1.5 seconds. Confirm one cycle
   remains in flight, delete its row while delayed, release the response, and
   confirm the row stays absent. Repeat with a refused delete and confirm the
   row remains with an error.
2. Transcribe and download to terminal status. Confirm the corresponding
   polling stops. Switch away/back and check one observer/stream per live
   resource and reuse of fresh snapshots, with no repeated submission.
3. Open MIDI/Repaint/Recent builds alongside shared consumers. Count shared
   songs and plugins requests by key and compare with the recorded baseline.
4. Save/reset an Insta-Gen prompt, delete an extraction/build, finish a render
   and reopen Song Builder. Confirm the affected list/detail refreshes.
5. Delay project/source A, switch to B and release A. Confirm B stays selected.
   Drop/reconnect a workflow stream, replay its terminal event and confirm no
   extra command is submitted. Check generation stage/progress/completion.
6. Keep STORM running through control changes and check stream playback is
   unchanged. Human listening and visual feedback remain part of acceptance.

Focused tests live beside these options. Run the full UI suite with the
existing server tsx loader and the CSS registration already in queue tests:

```powershell
node --import ../server/node_modules/tsx/dist/loader.mjs --test --test-timeout=60000 "src/**/*.test.ts"
npx tsc -b
```

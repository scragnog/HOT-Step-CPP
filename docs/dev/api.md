# HTTP API index

The Node server on port 3001 exposes everything the UI does as JSON over HTTP. Agents run the
app with `dev.bat` detached, so they reach these routes through `http://localhost:3000` — the
Vite dev server proxies `/api`, `/audio` and `/references` to 3001. This page is
the generated index of every route, grouped by mount prefix, with the file that defines it.
It is rebuilt by `node tools/docs/build-docs.mjs` from the mounts in `server/src/index.ts`
and the `router.<verb>(...)` calls in `server/src/routes/`. For request and response shapes,
open the route file; each handler is short and the types live next to it.

The C++ engine has its own HTTP API on port 8085, used only by the Node server. Its request
format is documented in [engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md).

Conventions:

- Long-running work (generation, training, separation, transcription) is submitted with a
  `POST`, returns a job id, and is polled or streamed over Server-Sent Events from a sibling
  `GET` route.
- Routes under `/api/lireek` belong to Lyric Studio (its internal name).
- Routes under `/api/builder` belong to Song Builder.
- Backend-specific generation lives behind the same `/api/generate` routes; the active
  backend is chosen with the `/api/backends` routes.

YuE2 cover transcription uses authenticated `/api/yue2-cover` routes. Submit either an
uploaded `sourceAudioUrl` under `/references/` or a library `songId` to
`POST /transcriptions`. Audio must be at most 100 MB and 10 minutes. A queued response
contains `jobId`, `sourceId`, and `sourceLabel`. Poll `GET /transcriptions/:jobId` for the
existing training job status and progress; when it is `done`, the response includes
`abc` and the same source identity. `DELETE` cancels through the training job queue.
`GET /readiness` reports whether the optional SheetSage2 model is installed. Supplying
non-empty `abc` in the POST returns it immediately with the source identity and needs
no transcriber. Pass that `abc` as `yue2Abc` and the `sourceId`/`sourceLabel` as
`yue2Cover` when submitting an ordinary YuE2 generation. Transcription always
returns the full score, including detected chord symbols. On generation submit,
`yue2Cover.keepChords` defaults to false. When false, the server strips inline
chord symbols from the rendered `yue2Abc` snapshot; the cover provenance keeps
the reviewed full score and the choice. The song metadata records both that
provenance and the ABC actually rendered.

`POST /drift/:songId` measures a saved YuE2 cover on request. It aligns that
song's audio with the lyrics captured at generation, matches sung lyric blocks
to like-labelled score sections in order, and returns each section's expected
first Vocal note through section end, its boundary span, sung span, start offset
in bars from the note onset, unscored reason, mean absolute offset,
and the first section more than one bar off. Unmatched sung lyric blocks are
listed. The live route checks word confidence but has no vocal stem to compare;
`stemChecked` is false. The result is cached on the song; a matching
request returns it without another alignment. If the rendered score has no
tempo, the source score's tempo is used and the response flags the fallback.

A YuE2 generation with a supplied score (and lyrics) ties each sung lyric
block to the first Vocal note of its score section, using the same score
clock and section matching as the drift metric. `yue2LyricSchedule: "bias"` is
the default (an ear test went from 2/11 on-time sections to 11/11 with it,
`RESEARCH/YUE2_ALIGNMENT_GATE_SET.md:1064`): while composing, the engine adds
`yue2LyricScheduleBias` (default -4) to the attention on a section's lyric
tokens until that note. `"mask"` stays selectable (uses -inf instead) but is
never the default. `yue2LyricSchedule: "off"` turns it off, current or saved.
`yue2LyricScheduleAbc: true` hides the section's label and music lines as
well (off by default). Field lines such as `M:`, `Q:`, `K:` and `V:` stay
visible. `yue2LyricScheduleLeadSec` reveals a section that many seconds early
(default 0). `yue2LyricScheduleBehind: n` also hides sections more than n
sections back (default -1, never). The schedule needs a score with a `Q:`
tempo, lyrics with `\n` line endings, CFG 1 (the engine's own default for
Chain of Thought melody/full; an explicit `yue2CfgScale` other than 1 skips
the schedule with a note instead, rendering at the requested CFG) and one
prompt per batch; a request that can't build one (missing score/lyrics, no
tempo, unlabelled sections, no matching timed lyric block, ...) renders
without it instead of
failing the job — the notes say why. Scheduled jobs share an engine batch
only with jobs that have the identical prompt. Lyric blocks with no timed
section are never hidden, and the job notes name them. The engine logs each
section's prompt rows as `[YuE2-C6]`.
`yue2-probe --schedule-check <request.json>` verifies a request's mapping
without a GPU.

`POST /source-metadata` accepts the same source selector and returns dataset
sidecar `lyrics`, `bpm`, `key`, `isInstrumental` and any saved `abc` when a
unique source match exists; `matched: false` keeps the usual studio path.
Matching uses the source's canonical path first, then an exact content hash
for copied library or uploaded audio. `POST /transcriptions` returns a saved
dataset ABC immediately with `scoreSource: "dataset"` and no job. Pass
`force: true` to transcribe again; a successful result updates `<stem>.abc`
next to the dataset audio. A cover generation records `lyricsSource:
"dataset-sidecar"` until the lyrics or Instrumental choice is edited.

Completed transcription responses, supplied-ABC responses and saved dataset
scores include `sections: [{ label, startBar }]`. The bar number starts at 1
and counts the Vocal voice. `POST /sections/review` accepts `{ abc, lyrics }`
and returns those sections, a non-blocking count/order verdict for the lyric
`[Tag]` lines, and `insertedLyrics` with the score tags in order. Existing
lyric lines stay under the first inserted tag.

Cover transcription first tries with the engine running. If SheetSage2 fails to load,
the job waits for any active render, stops the engine, retries once, and restarts the
engine before the next render begins. Other transcription errors fail without a retry.
For development testing, set `HOTSTEP_YUE2_COVER_FORCE_FALLBACK=1` while running
`dev.bat` (`HOT_STEP_DEV=1`) to take this path even when the first load succeeds.

Total: <!-- generated:start route-count -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
331 routes
<!-- generated:end route-count -->

<!-- generated:start routes -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
### `/api/auth`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/auth/auto` | `server/src/routes/auth.ts` |
| `POST` | `/api/auth/setup` | `server/src/routes/auth.ts` |
| `GET` | `/api/auth/me` | `server/src/routes/auth.ts` |
| `PATCH` | `/api/auth/username` | `server/src/routes/auth.ts` |
| `POST` | `/api/auth/logout` | `server/src/routes/auth.ts` |

### `/api/songs`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/songs` | `server/src/routes/songs.ts` |
| `GET` | `/api/songs/ids` | `server/src/routes/songs.ts` |
| `GET` | `/api/songs/recent` | `server/src/routes/songs.ts` |
| `GET` | `/api/songs/:id` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/import` | `server/src/routes/songs.ts` |
| `PATCH` | `/api/songs/:id` | `server/src/routes/songs.ts` |
| `DELETE` | `/api/songs/:id` | `server/src/routes/songs.ts` |
| `DELETE` | `/api/songs` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/bulk-delete` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/nuke-generations` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/crop` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/retranscribe` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/postprocess` | `server/src/routes/songs.ts` |
| `GET` | `/api/songs/:id/postprocess` | `server/src/routes/songs.ts` |
| `DELETE` | `/api/songs/:id/postprocess` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/extract-kick` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/analyze-disco` | `server/src/routes/songs.ts` |

### `/api/generate`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/generate/yue2/plan` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate` | `server/src/routes/generate.ts` |
| `GET` | `/api/generate/status/:id` | `server/src/routes/generate.ts` |
| `GET` | `/api/generate/mm3/stream/:id` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/cancel/:id` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/cancel-all` | `server/src/routes/generate.ts` |
| `GET` | `/api/generate/queue` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/reset-queue` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/storm/stream` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/storm/stop` | `server/src/routes/generate.ts` |
| `POST` | `/api/generate/storm/control` | `server/src/routes/generate.ts` |
| `GET` | `/api/generate/storm/control` | `server/src/routes/generate.ts` |

### `/api/models`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/models` | `server/src/routes/models.ts` |
| `GET` | `/api/models/health` | `server/src/routes/models.ts` |
| `GET` | `/api/models/pp-vae` | `server/src/routes/models.ts` |
| `GET` | `/api/models/stablestep` | `server/src/routes/models.ts` |
| `GET` | `/api/models/stablestep/adapters` | `server/src/routes/models.ts` |

### `/api/health`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/health/presence` | `server/src/routes/health.ts` |
| `GET` | `/api/health` | `server/src/routes/health.ts` |

### `/api/shutdown`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/shutdown` | `server/src/routes/shutdown.ts` |
| `POST` | `/api/shutdown/restart` | `server/src/routes/shutdown.ts` |

### `/api/mastering`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/mastering/upload-reference` | `server/src/routes/mastering.ts` |
| `GET` | `/api/mastering/references` | `server/src/routes/mastering.ts` |
| `DELETE` | `/api/mastering/references/:name` | `server/src/routes/mastering.ts` |
| `POST` | `/api/mastering/run` | `server/src/routes/mastering.ts` |

### `/api/download`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/download/:id` | `server/src/routes/download.ts` |

### `/api/adapters`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/adapters/browse` | `server/src/routes/adapters.ts` |
| `POST` | `/api/adapters/scan` | `server/src/routes/adapters.ts` |
| `GET` | `/api/adapters/lm` | `server/src/routes/adapters.ts` |

### `/api/logs`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/logs` | `server/src/routes/logs.ts` |
| `GET` | `/api/logs/vram` | `server/src/routes/logs.ts` |
| `GET` | `/api/logs/models-loaded` | `server/src/routes/logs.ts` |
| `POST` | `/api/logs/models-unload` | `server/src/routes/logs.ts` |

### `/api/lireek`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/lireek/slop-scan` | `server/src/routes/lireek.ts` |
| `POST` | `/api/lireek/purge` | `server/src/routes/lireek.ts` |
| `POST` | `/api/lireek/purge-generations` | `server/src/routes/lireek.ts` |
| `POST` | `/api/lireek/purge-profiles` | `server/src/routes/lireek.ts` |
| `GET` | `/api/lireek/prompts` | `server/src/routes/lireek.ts` |
| `PUT` | `/api/lireek/prompts/:name` | `server/src/routes/lireek.ts` |
| `DELETE` | `/api/lireek/prompts/:name` | `server/src/routes/lireek.ts` |
| `GET` | `/api/lireek/recent-songs` | `server/src/routes/lireek.ts` |
| `GET` | `/api/lireek/artists` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/artists/create` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/artists/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/artists/:id/refresh-image` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/artists/:id/set-image` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/lyrics-sets` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/create` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/lyrics-sets/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/lyrics-sets/:id/full-detail` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/lyrics-sets/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/lyrics-sets/:id/songs/:index` | `server/src/routes/lireek/crudRoutes.ts` |
| `PUT` | `/api/lireek/lyrics-sets/:id/songs/:index` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/:id/refresh-image` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/:id/set-image` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/:id/add-song` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/fetch-lyrics` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/search-song-lyrics` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/profiles` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/profiles/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/profiles/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/generations` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/generations/all` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/generations/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `PATCH` | `/api/lireek/generations/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/generations/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/generations/:id/export` | `server/src/routes/lireek/crudRoutes.ts` |
| `POST` | `/api/lireek/generations/:id/audio` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/generations/:id/audio` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/lyrics-sets/:id/rendered-as` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/audio-generations/:id` | `server/src/routes/lireek/crudRoutes.ts` |
| `PATCH` | `/api/lireek/audio-generations/resolve` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/lyrics-sets/:id/preset` | `server/src/routes/lireek/crudRoutes.ts` |
| `PUT` | `/api/lireek/lyrics-sets/:id/preset` | `server/src/routes/lireek/crudRoutes.ts` |
| `DELETE` | `/api/lireek/lyrics-sets/:id/preset` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/presets` | `server/src/routes/lireek/crudRoutes.ts` |
| `GET` | `/api/lireek/providers` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/:id/build-profile` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/lyrics-sets/:id/build-profile-stream` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/profiles/recalculate-stats` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/profiles/:id/generate` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/profiles/:id/generate-stream` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/generations/:id/refine` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/generations/:id/refine-stream` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/artists/:id/curated-profile` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/artists/:id/curated-profile-stream` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/artists/:id/generate-caption` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/skip-thinking` | `server/src/routes/lireek/llmRoutes.ts` |
| `POST` | `/api/lireek/mm3/compose` | `server/src/routes/lireek/mm3Routes.ts` |
| `POST` | `/api/lireek/mm3/parse-brief` | `server/src/routes/lireek/mm3Routes.ts` |
| `GET` | `/api/lireek/mm3/corpus-info` | `server/src/routes/lireek/mm3Routes.ts` |

### `/api/vst`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/vst/scan` | `server/src/routes/vst.ts` |
| `GET` | `/api/vst/chain` | `server/src/routes/vst.ts` |
| `PUT` | `/api/vst/chain` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/gui` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/process` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/start` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/stop` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/switch` | `server/src/routes/vst.ts` |
| `GET` | `/api/vst/monitor/status` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/seek` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/pause` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/resume` | `server/src/routes/vst.ts` |
| `POST` | `/api/vst/monitor/restart` | `server/src/routes/vst.ts` |

### `/api/analyze`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/analyze` | `server/src/routes/analyze.ts` |
| `POST` | `/api/analyze/metadata` | `server/src/routes/analyze.ts` |

### `/api/upload`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/upload/audio` | `server/src/routes/upload.ts` |
| `POST` | `/api/upload/latent` | `server/src/routes/upload.ts` |
| `POST` | `/api/upload/cover-image` | `server/src/routes/upload.ts` |

### `/api/supersep`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/supersep/separate` | `server/src/routes/supersep.ts` |
| `GET` | `/api/supersep/:jobId/progress` | `server/src/routes/supersep.ts` |
| `POST` | `/api/supersep/:jobId/release` | `server/src/routes/supersep.ts` |
| `GET` | `/api/supersep/:jobId/result` | `server/src/routes/supersep.ts` |
| `GET` | `/api/supersep/:jobId/stem/:index` | `server/src/routes/supersep.ts` |
| `POST` | `/api/supersep/recombine` | `server/src/routes/supersep.ts` |

### `/api/settings`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/settings/env` | `server/src/routes/settings.ts` |
| `POST` | `/api/settings/env` | `server/src/routes/settings.ts` |
| `GET` | `/api/settings/gpus` | `server/src/routes/settings.ts` |

### `/api/model-manager`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/model-manager/registry` | `server/src/routes/modelManager.ts` |
| `POST` | `/api/model-manager/download` | `server/src/routes/modelManager.ts` |
| `GET` | `/api/model-manager/downloads` | `server/src/routes/modelManager.ts` |
| `POST` | `/api/model-manager/download/:jobId/cancel` | `server/src/routes/modelManager.ts` |
| `POST` | `/api/model-manager/download/:jobId/resume` | `server/src/routes/modelManager.ts` |
| `DELETE` | `/api/model-manager/files/:filename` | `server/src/routes/modelManager.ts` |

### `/api/stem-studio`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/stem-studio/extract` | `server/src/routes/stemStudio.ts` |
| `POST` | `/api/stem-studio/supersep` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/:jobId/progress` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/:jobId/result` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/:jobId/stem/:trackName` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/:jobId/download-all` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/jobs` | `server/src/routes/stemStudio.ts` |
| `DELETE` | `/api/stem-studio/all` | `server/src/routes/stemStudio.ts` |
| `DELETE` | `/api/stem-studio/:jobId` | `server/src/routes/stemStudio.ts` |
| `GET` | `/api/stem-studio/stats` | `server/src/routes/stemStudio.ts` |

### `/api/assistant`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/assistant/chat` | `server/src/routes/assistant.ts` |
| `GET` | `/api/assistant/providers` | `server/src/routes/assistant.ts` |

### `/api/plugins`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/plugins` | `server/src/routes/plugins.ts` |
| `POST` | `/api/plugins/reload` | `server/src/routes/plugins.ts` |

### `/api/inspire`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/inspire` | `server/src/routes/inspire.ts` |
| `GET` | `/api/inspire/status/:id` | `server/src/routes/inspire.ts` |
| `POST` | `/api/inspire/cancel/:id` | `server/src/routes/inspire.ts` |
| `POST` | `/api/inspire/llm` | `server/src/routes/inspire.ts` |
| `GET` | `/api/inspire/llm/providers` | `server/src/routes/inspire.ts` |
| `POST` | `/api/inspire/llm/subject` | `server/src/routes/inspire.ts` |
| `GET` | `/api/inspire/llm/prompt` | `server/src/routes/inspire.ts` |
| `PUT` | `/api/inspire/llm/prompt` | `server/src/routes/inspire.ts` |
| `DELETE` | `/api/inspire/llm/prompt` | `server/src/routes/inspire.ts` |

### `/api/cover-art`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/cover-art/status` | `server/src/routes/coverArt.ts` |
| `POST` | `/api/cover-art/download` | `server/src/routes/coverArt.ts` |
| `POST` | `/api/cover-art/download/cancel` | `server/src/routes/coverArt.ts` |
| `GET` | `/api/cover-art/download/progress` | `server/src/routes/coverArt.ts` |
| `POST` | `/api/cover-art/prompt-preview` | `server/src/routes/coverArt.ts` |
| `POST` | `/api/cover-art/generate` | `server/src/routes/coverArt.ts` |
| `GET` | `/api/cover-art/generate/:jobId` | `server/src/routes/coverArt.ts` |

### `/api/seeds`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/seeds` | `server/src/routes/seeds.ts` |
| `GET` | `/api/seeds/favorites` | `server/src/routes/seeds.ts` |
| `GET` | `/api/seeds/random` | `server/src/routes/seeds.ts` |
| `GET` | `/api/seeds/:name` | `server/src/routes/seeds.ts` |
| `POST` | `/api/seeds` | `server/src/routes/seeds.ts` |
| `DELETE` | `/api/seeds/:name` | `server/src/routes/seeds.ts` |
| `POST` | `/api/seeds/:name/favorite` | `server/src/routes/seeds.ts` |

### `/api/profiles`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/profiles` | `server/src/routes/profiles.ts` |
| `GET` | `/api/profiles/:name` | `server/src/routes/profiles.ts` |
| `POST` | `/api/profiles` | `server/src/routes/profiles.ts` |
| `PATCH` | `/api/profiles/:name` | `server/src/routes/profiles.ts` |
| `DELETE` | `/api/profiles/:name` | `server/src/routes/profiles.ts` |

### `/api/builder`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/builder/projects` | `server/src/routes/songBuilder.ts` |
| `GET` | `/api/builder/projects/:id` | `server/src/routes/songBuilder.ts` |
| `POST` | `/api/builder/projects` | `server/src/routes/songBuilder.ts` |
| `PATCH` | `/api/builder/projects/:id` | `server/src/routes/songBuilder.ts` |
| `DELETE` | `/api/builder/projects/:id` | `server/src/routes/songBuilder.ts` |
| `POST` | `/api/builder/projects/:id/sections` | `server/src/routes/songBuilder.ts` |
| `PATCH` | `/api/builder/sections/:id` | `server/src/routes/songBuilder.ts` |
| `DELETE` | `/api/builder/sections/:id` | `server/src/routes/songBuilder.ts` |

### `/api/midi-studio`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/midi-studio/status` | `server/src/routes/midiStudio.ts` |
| `POST` | `/api/midi-studio/hf-token` | `server/src/routes/midiStudio.ts` |
| `POST` | `/api/midi-studio/models/:size/download` | `server/src/routes/midiStudio.ts` |
| `POST` | `/api/midi-studio/transcribe` | `server/src/routes/midiStudio.ts` |
| `GET` | `/api/midi-studio/jobs` | `server/src/routes/midiStudio.ts` |
| `GET` | `/api/midi-studio/:jobId/progress` | `server/src/routes/midiStudio.ts` |
| `GET` | `/api/midi-studio/:jobId/stream` | `server/src/routes/midiStudio.ts` |
| `GET` | `/api/midi-studio/:jobId/notes` | `server/src/routes/midiStudio.ts` |
| `GET` | `/api/midi-studio/:jobId/file` | `server/src/routes/midiStudio.ts` |
| `DELETE` | `/api/midi-studio/:jobId` | `server/src/routes/midiStudio.ts` |

### `/api/training`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/training/capabilities` | `server/src/routes/training.ts` |
| `GET` | `/api/training/scan-preview` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets` | `server/src/routes/training.ts` |
| `GET` | `/api/training/jobs` | `server/src/routes/training.ts` |
| `GET` | `/api/training/jobs/:jobId` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/jobs/:jobId` | `server/src/routes/training.ts` |
| `GET` | `/api/training/jobs/:jobId/stream` | `server/src/routes/training.ts` |
| `GET` | `/api/training/previews/:previewId/:slot` | `server/src/routes/training.ts` |
| `POST` | `/api/training/pipeline` | `server/src/routes/training.ts` |
| `GET` | `/api/training/pipeline` | `server/src/routes/training.ts` |
| `GET` | `/api/training/pipeline/:id` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/pipeline/:id` | `server/src/routes/training.ts` |
| `POST` | `/api/training/pipeline/:id/pause` | `server/src/routes/training.ts` |
| `POST` | `/api/training/pipeline/:id/resume` | `server/src/routes/training.ts` |
| `POST` | `/api/training/yue2-batch` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-batch` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-batch/:id` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/yue2-batch/:id` | `server/src/routes/training.ts` |
| `POST` | `/api/training/yue2-batch/finish` | `server/src/routes/training.ts` |
| `POST` | `/api/training/yue2-batch/:id/items` | `server/src/routes/training.ts` |
| `POST` | `/api/training/yue2-batch/:id/pause` | `server/src/routes/training.ts` |
| `POST` | `/api/training/yue2-batch/:id/resume` | `server/src/routes/training.ts` |
| `GET` | `/api/training/defaults` | `server/src/routes/training.ts` |
| `GET` | `/api/training/active-models` | `server/src/routes/training.ts` |
| `PUT` | `/api/training/active-models` | `server/src/routes/training.ts` |
| `PUT` | `/api/training/defaults` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id` | `server/src/routes/training.ts` |
| `PATCH` | `/api/training/datasets/:id` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/rescan` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/datasets/:id` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/prepared-data` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/datasets/:id/prepared-data` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/samples/bulk` | `server/src/routes/training.ts` |
| `PATCH` | `/api/training/datasets/:id/samples/:sampleId` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/samples/:sampleId/${kind}` | `server/src/routes/training.ts` |
| `PUT` | `/api/training/datasets/:id/samples/:sampleId/${kind}` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/samples/:sampleId/audio` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/label` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/enhance/genius` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/enhance/caption` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/enhance/yue2-caption` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-captions-missing` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/build` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/dataset-json` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/lyric-studio` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/lyric-studio` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/preprocess` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/preprocess` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/datasets/:id/preprocess/:variantKey` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/mm3` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-codes` | `server/src/routes/training.ts` |
| `GET` | `/api/training/mm3/preview` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-train-lm` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/mm3-runs` | `server/src/routes/training.ts` |
| `GET` | `/api/training/lyrics-sets/:id/mm3-adapters` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-resume-lm` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-preprocess` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-train` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-joint-train` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-optimise` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-optimise/base-loss` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-joint-prepare` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-joint-prepare` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-runs` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-joint-runs` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-joint-preset` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-joint-previews` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-review` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-review-complete` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-cleanup-plan` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-cleanup` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/datasets/:id/yue2-joint-runs/:jobId` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-rung-scores` | `server/src/routes/training.ts` |
| `PUT` | `/api/training/datasets/:id/yue2-rung-scores` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-album-score` | `server/src/routes/training.ts` |
| `PUT` | `/api/training/datasets/:id/yue2-album-score` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-rung-scores/export` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-joint-previews/render` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-joint-previews/audio` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-ar` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-tokenize` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-sheet` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-sheet` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-sheet/:name` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-stems` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-align` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-ar-train` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-ar-runs` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-dataset-captions` | `server/src/routes/training.ts` |
| `GET` | `/api/training/yue2-adapter-captions` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/train-lm` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/train-lm` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/train-dit` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/train-dit` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/ls-generations` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/audition` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/audition` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/samples/:sampleId/audition` | `server/src/routes/training.ts` |

### `/api/audio`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/audio/peaks` | `server/src/routes/audio.ts` |

### `/api/yue2-cover`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/yue2-cover/readiness` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/source-metadata` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/review` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/match` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/save-dataset` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/drift/:songId` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/transcriptions` | `server/src/routes/yue2Cover.ts` |
| `GET` | `/api/yue2-cover/transcriptions/:jobId` | `server/src/routes/yue2Cover.ts` |
| `DELETE` | `/api/yue2-cover/transcriptions/:jobId` | `server/src/routes/yue2Cover.ts` |

### `/api`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/backends` | `server/src/routes/backends.ts` |
| `POST` | `/api/backends/active` | `server/src/routes/backends.ts` |
| `GET` | `/api/backends/models` | `server/src/routes/backends.ts` |
| `POST` | `/api/backends/models` | `server/src/routes/backends.ts` |
| `GET` | `/api/capabilities` | `server/src/routes/backends.ts` |
| `GET` | `/api/mm3/planks` | `server/src/routes/backends.ts` |
| `GET` | `/api/mm3/plank-meta` | `server/src/routes/backends.ts` |
| `GET` | `/api/mm3/plans` | `server/src/routes/backends.ts` |
| `GET` | `/api/mm3/plan-meta` | `server/src/routes/backends.ts` |
| `DELETE` | `/api/mm3/plans/:file` | `server/src/routes/backends.ts` |
| `GET` | `/api/mm3/lm-adapters` | `server/src/routes/backends.ts` |
| `POST` | `/api/yue2/import-adapter` | `server/src/routes/backends.ts` |

### `/listening`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/listening/:folder/{*splat}` | `server/src/routes/listening.ts` |
| `POST` | `/listening/:folder/scores` | `server/src/routes/listening.ts` |
<!-- generated:end routes -->

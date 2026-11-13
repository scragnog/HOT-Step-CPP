# HTTP API index

The Node server on port 3001 exposes everything the UI does as JSON over HTTP. Agents run the
app with `dev.bat` detached, so they reach these routes through `http://localhost:3000` — the
Vite dev server proxies `/api`, `/audio` and `/references` to 3001. This page is
the generated index of every route, grouped by mount prefix, with the file that defines it.
It is rebuilt by `node tools/docs/build-docs.mjs` from the mounts in `server/src/index.ts`
and the `router.<verb>(...)` calls in `server/src/routes/`. For request and response shapes,
open the route file; each handler is short and the types live next to it.

Request, response, error, auth, job and media contracts for generation, the dev-only
request capture, and the engine capability manifest are in
[Generation API contracts](api-contracts.md).

The C++ engine has its own HTTP API on port 8085, used only by the Node server. Its request
format is documented in [engine/docs/ARCHITECTURE.md](../../engine/docs/ARCHITECTURE.md).

MM3 cover encoding uses `POST /api/mm3/cover/encode` with `sourceAudioUrl`, optional
`sourceName`, and `launder: true` for the Cover Studio path. API callers can omit
`launder` for plain encoding. Laundering accepts `launderSteps` (integer 1..1000,
default 30), `launderSeed` (safe integer >= -1, default 42; -1 chooses a random
seed), and `launderCaption` (source description, at most 64,000 characters). Blank
caption uses "a full song with vocals and instruments". The older `caption` field
is an alias when `launderCaption` is absent. A byte-identical source with matching
launder settings reuses its saved plank as `{ plank, reused: true }`; `force: true`
bypasses reuse. Otherwise the response is `{ jobId, plank }`;
poll `/api/training/jobs/:jobId` until done. The plank sidecar saves the actual
steps, seed and caption, copied into cover song facts as `mm3PlankLaunderSteps`,
`mm3PlankLaunderSeed` and `mm3PlankLaunderCaption`. LM quant remains Q8_0.
The plank sidecar also records `sourceSizeBytes`, `sourceSha256` and
`launderRequestedSeed` for reuse, while `launderSeed` records the actual random
seed when the request used -1.

`POST /api/mm3/cover/source-prep` takes `{ sourceAudioUrl, wantLyrics,
wantCaption, bpm?, key? }`. The two `want` fields are booleans, at least one true.
It returns `{ jobId }`. Poll the training job endpoint for terminal status and
`result: { lyrics?, mm3Caption? }`. Work uses Music-Flamingo on the same serial
GPU queue as cover encoding, stopping the engine before captioning and restoring
it afterwards. Missing Music-Flamingo returns Model Manager guidance.
`GET /api/mm3/cover/launder-models` includes `flamingo: boolean` alongside missing
launder model files. `POST /api/yue2-cover/source-metadata` also returns
`mm3Caption` from the matched dataset track's `<stem>.mm3.txt`, when present.

A laundered encode also writes `<id>.guide` (the encoder's top 32 codes per frame)
and `<id>.mm3hiddens` (the launder's renderer condition) beside the plank, and
`GET /api/mm3/planks` reports them as `guide` and `state`. The Cover Studio render
workflow's `mm3Cover` takes `mode` (`pin`, `follow`, `follow-depth`,
`follow-stream`, `product`, `blend`; absent = `pin`) with `strength`,
`candidates`, `stateFix` (`off`, `depth`, `stream`, `both`), `weight` and `blend`
(percent). Every mode but `pin` sends the plank's guide instead of the plank, and
returns 409 asking for a Re-encode when the guide (or, for `blend`, the state) is
missing. The generation params behind it are `mm3GuideStateFix` (0 to 3),
`mm3GuideProduct`, `mm3GuideSeconds` (trim), and `mm3BlendPath` with
`mm3BlendAmount` (0 to 1), all resolved inside the plank folder.

An MM3 generation can follow a source guide instead of a plank (experimental,
API only). `mm3GuidePath` names a `.guide` file in the plank folder, written by
`ace-train mm3-codes` or `mm3-launder` with `--guide-k N`: the encoder's top N
codes per frame. The planner then chooses among those candidates.
`mm3GuideScale` (0 to 1, engine default 0.5) sets how often the source wins.
`mm3GuideK` uses fewer candidates than the file holds, and `mm3GuideOutside`
(-100 or lower) restricts the draw to the candidates. `mm3GuideGatePath` names a
raw f32 file in the same folder, one multiplier on the scale per frame. The
song is as long as the guide, so natural endings, takes and the length floor
are skipped. A loaded plank wins over a guide, and a guide wins over a saved plan.

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
clock and section matching as the drift metric. `yue2LyricSchedule: "off"` is
the default — condition strength 1 with the old soft-by-default ruined lyric
adherence. `"bias"` stays selectable (an ear test went from 2/11 on-time
sections to 11/11 with it, `RESEARCH/YUE2_ALIGNMENT_GATE_SET.md:1064`): while
composing, the engine adds `yue2LyricScheduleBias` (default -4) to the
attention on a section's lyric tokens until that note. `"mask"` also stays
selectable (uses -inf instead). Neither is ever the default; an explicit
choice (current or saved) always wins.
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
and returns those sections and a non-blocking count/order verdict for the
lyric `[Tag]` lines. `POST /sections/save-dataset` writes `lyrics`, or `bpm`
with `key` (such as `"B Minor"`), into the matched dataset song's sidecar and
marks those label fields as user-set.

The YuE2 Prepare flow keeps one `yue2Prepare` document per draft (contract:
`server/src/contracts/yue2Prepare.ts`, shared with the UI): the ABC, stable section
ids, lyric blocks, an assignment map, `lyricRevision` and two confirmations, the
structure hash and the structure hash with the lyric revision.
`POST /sections/suggest` takes `{ prepare, useRecording? }` plus the source fields
and returns tag-and-order assignment suggestions stamped with
`{ sourceKey, structureHash, lyricRevision }`, so the UI drops a reply that arrives
after the score or lyrics changed. With `useRecording: true` it aligns the vocal
first and falls back to the instant suggestions with `recordingError` set. It never
rewrites the score.

Submissions are checked by one shared validator (`preparedPair` and
`preparedParams`). The cover-render workflow input takes `input.prepare` (YuE2 only)
and renders the lyrics the document exports. Build session start and
`POST /api/generate` take `params.yue2Prepare` along with `yue2Abc`, `lyrics` and
`instrumental`; the document is dropped before queueing. Each answers 409 when the
document is malformed, a confirmation is stale, the score is not the confirmed one,
a section or lyric block is unaccounted for, the Instrumental flag contradicts the
lyrics, or the lyrics differ from the exported ones. Submissions without the
document are unchanged.

Cover transcription first tries with the engine running. If SheetSage2 fails to load,
the job waits for any active render, stops the engine, retries once, and restarts the
engine before the next render begins. Other transcription errors fail without a retry.
For development testing, set `HOTSTEP_YUE2_COVER_FORCE_FALLBACK=1` while running
`dev.bat` (`HOT_STEP_DEV=1`) to take this path even when the first load succeeds.

Total: <!-- generated:start route-count -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
493 routes
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
| `POST` | `/api/songs/export-zip` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/nuke-generations` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/:id/crop` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/lyrics/offset` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/lyrics/split` | `server/src/routes/songs.ts` |
| `POST` | `/api/songs/lyrics/timings` | `server/src/routes/songs.ts` |
| `GET` | `/api/songs/retranscribe/model` | `server/src/routes/songs.ts` |
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

### `/api/updates`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/updates/status` | `server/src/routes/updates.ts` |
| `POST` | `/api/updates/check` | `server/src/routes/updates.ts` |

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
| `GET` | `/api/download/:id/lrc` | `server/src/routes/download.ts` |
| `POST` | `/api/download/:id/lrc/save` | `server/src/routes/download.ts` |
| `GET` | `/api/download/:id` | `server/src/routes/download.ts` |
| `POST` | `/api/download/:id/save` | `server/src/routes/download.ts` |

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
| `GET` | `/api/logs/snapshot` | `server/src/routes/logs.ts` |
| `GET` | `/api/logs/vram` | `server/src/routes/logs.ts` |
| `GET` | `/api/logs/models-loaded` | `server/src/routes/logs.ts` |
| `POST` | `/api/logs/models-unload` | `server/src/routes/logs.ts` |

### `/api/lireek`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/lireek/workflow-batches` | `server/src/routes/lireek.ts` |
| `POST` | `/api/lireek/workflow-renders` | `server/src/routes/lireek.ts` |
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
| `POST` | `/api/lireek/presets/update-adapter-references` | `server/src/routes/lireek/crudRoutes.ts` |
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
| `GET` | `/api/vst/gui/open` | `server/src/routes/vst.ts` |
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
| `GET` | `/api/settings/data-folder` | `server/src/routes/settings.ts` |
| `POST` | `/api/settings/data-folder/preview` | `server/src/routes/settings.ts` |
| `POST` | `/api/settings/data-folder/schedule` | `server/src/routes/settings.ts` |
| `POST` | `/api/settings/data-folder/cancel` | `server/src/routes/settings.ts` |
| `GET` | `/api/settings/env` | `server/src/routes/settings.ts` |
| `POST` | `/api/settings/env` | `server/src/routes/settings.ts` |
| `GET` | `/api/settings/gpus` | `server/src/routes/settings.ts` |
| `GET` | `/api/settings/engine-workers` | `server/src/routes/settings.ts` |

### `/api/settings/backup`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/settings/backup/export` | `server/src/routes/settingsBackup.ts` |
| `POST` | `/api/settings/backup/preview` | `server/src/routes/settingsBackup.ts` |
| `POST` | `/api/settings/backup/import` | `server/src/routes/settingsBackup.ts` |

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
| `POST` | `/api/builder/projects/:id/sections/generate` | `server/src/routes/songBuilder.ts` |
| `POST` | `/api/builder/sections/:id/choose` | `server/src/routes/songBuilder.ts` |
| `POST` | `/api/builder/sections/:id/stop` | `server/src/routes/songBuilder.ts` |
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
| `POST` | `/api/midi-studio/:jobId/export` | `server/src/routes/midiStudio.ts` |
| `POST` | `/api/midi-studio/:jobId/abc` | `server/src/routes/midiStudio.ts` |
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
| `POST` | `/api/training/heartmula-batch` | `server/src/routes/training.ts` |
| `POST` | `/api/training/mm3-batch` | `server/src/routes/training.ts` |
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
| `POST` | `/api/training/datasets/:id/enhance/normalise-lyrics` | `server/src/routes/training.ts` |
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
| `GET` | `/api/training/datasets/:id/mm3-previews` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-previews/render` | `server/src/routes/training.ts` |
| `DELETE` | `/api/training/datasets/:id/mm3-runs/:run` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-review-complete` | `server/src/routes/training.ts` |
| `GET` | `/api/training/mm3-rung-scores/export` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-train-lm` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/mm3-runs` | `server/src/routes/training.ts` |
| `GET` | `/api/training/lyrics-sets/:id/mm3-adapters` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/mm3-resume-lm` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-preprocess` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-train` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/yue2-joint-train` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/yue2-joint-holdout` | `server/src/routes/training.ts` |
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
| `PATCH` | `/api/training/datasets/:id/yue2-joint-runs/:jobId/params` | `server/src/routes/training.ts` |
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
| `GET` | `/api/training/datasets/:id/sa3-train` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/sa3-previews` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/sa3-previews/render` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/sa3-previews/audio` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/sa3-train` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/heartmula-status` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/heartmula-prepare` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/heartmula-train` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/heartmula-checkpoint` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/heartmula-previews` | `server/src/routes/training.ts` |
| `POST` | `/api/training/datasets/:id/heartmula-previews/render` | `server/src/routes/training.ts` |
| `GET` | `/api/training/datasets/:id/heartmula-previews/audio` | `server/src/routes/training.ts` |
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
| `POST` | `/api/yue2-cover/drafts/:id/approve-score` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/drafts/:id/save-score-details` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/score/remeter` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/score/shift-bars` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/review` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/match` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/plan-score` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/suggest` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/sections/save-dataset` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/drift/:songId` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/lyrics-check` | `server/src/routes/yue2Cover.ts` |
| `GET` | `/api/yue2-cover/build/:id` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/generate` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/cancel` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/keep` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/undo` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/crop` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/preview` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/trim` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/lyrics` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/occurrence-lyrics` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/settings` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/build/:id/plan` | `server/src/routes/yue2Cover.ts` |
| `DELETE` | `/api/yue2-cover/build/:id` | `server/src/routes/yue2Cover.ts` |
| `DELETE` | `/api/yue2-cover/build/:id/takes/:take` | `server/src/routes/yue2Cover.ts` |
| `POST` | `/api/yue2-cover/transcriptions` | `server/src/routes/yue2Cover.ts` |
| `GET` | `/api/yue2-cover/transcriptions/:jobId` | `server/src/routes/yue2Cover.ts` |
| `DELETE` | `/api/yue2-cover/transcriptions/:jobId` | `server/src/routes/yue2Cover.ts` |

### `/api/resolve`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/resolve/path` | `server/src/routes/resolve.ts` |
| `POST` | `/api/resolve/preview` | `server/src/routes/resolve.ts` |

### `/api/audio-queue`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/audio-queue/items` | `server/src/routes/audioQueue.ts` |
| `GET` | `/api/audio-queue/items` | `server/src/routes/audioQueue.ts` |
| `GET` | `/api/audio-queue/items/:id` | `server/src/routes/audioQueue.ts` |
| `DELETE` | `/api/audio-queue/items/:id` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/items/:id/cancel` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/items/:id/retry` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/migration/import` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/migration/resume-held` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/migration/rollback-export` | `server/src/routes/audioQueue.ts` |
| `GET` | `/api/audio-queue/state` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/pause` | `server/src/routes/audioQueue.ts` |
| `POST` | `/api/audio-queue/resume` | `server/src/routes/audioQueue.ts` |

### `/api/workflows`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/workflows/jobs` | `server/src/routes/workflows.ts` |
| `GET` | `/api/workflows/jobs` | `server/src/routes/workflows.ts` |
| `GET` | `/api/workflows/jobs/:id` | `server/src/routes/workflows.ts` |
| `POST` | `/api/workflows/jobs/:id/cancel` | `server/src/routes/workflows.ts` |
| `POST` | `/api/workflows/jobs/:id/retry` | `server/src/routes/workflows.ts` |
| `GET` | `/api/workflows/jobs/:id/events` | `server/src/routes/workflows.ts` |
| `POST` | `/api/workflows/documents` | `server/src/routes/workflows.ts` |
| `GET` | `/api/workflows/documents` | `server/src/routes/workflows.ts` |
| `GET` | `/api/workflows/documents/:id` | `server/src/routes/workflows.ts` |
| `PUT` | `/api/workflows/documents/:id` | `server/src/routes/workflows.ts` |
| `DELETE` | `/api/workflows/documents/:id` | `server/src/routes/workflows.ts` |

### `/api/preferences`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/preferences/presets/yue2-joint/resolve` | `server/src/routes/preferences.ts` |
| `GET` | `/api/preferences/presets/:family` | `server/src/routes/preferences.ts` |
| `POST` | `/api/preferences/presets/:family` | `server/src/routes/preferences.ts` |
| `PUT` | `/api/preferences/presets/:family/:id` | `server/src/routes/preferences.ts` |
| `DELETE` | `/api/preferences/presets/:family/:id` | `server/src/routes/preferences.ts` |
| `POST` | `/api/preferences/presets/:family/import` | `server/src/routes/preferences.ts` |
| `GET` | `/api/preferences/settings/:family` | `server/src/routes/preferences.ts` |
| `PUT` | `/api/preferences/settings/:family` | `server/src/routes/preferences.ts` |
| `POST` | `/api/preferences/settings/:family/import` | `server/src/routes/preferences.ts` |

### `/api/studio-drafts`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/studio-drafts/playlist` | `server/src/routes/studioDrafts.ts` |
| `POST` | `/api/studio-drafts/playlist/commands` | `server/src/routes/studioDrafts.ts` |
| `GET` | `/api/studio-drafts/handoffs/:id` | `server/src/routes/studioDrafts.ts` |
| `GET` | `/api/studio-drafts/drafts` | `server/src/routes/studioDrafts.ts` |
| `POST` | `/api/studio-drafts/drafts` | `server/src/routes/studioDrafts.ts` |
| `GET` | `/api/studio-drafts/drafts/:id` | `server/src/routes/studioDrafts.ts` |
| `PUT` | `/api/studio-drafts/drafts/:id` | `server/src/routes/studioDrafts.ts` |
| `DELETE` | `/api/studio-drafts/drafts/:id` | `server/src/routes/studioDrafts.ts` |
| `POST` | `/api/studio-drafts/import` | `server/src/routes/studioDrafts.ts` |

### `/api/stream-sessions`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/stream-sessions` | `server/src/routes/streamSessions.ts` |
| `GET` | `/api/stream-sessions/:id` | `server/src/routes/streamSessions.ts` |
| `POST` | `/api/stream-sessions/:id/recording` | `server/src/routes/streamSessions.ts` |
| `GET` | `/api/stream-sessions/:id/recording/export` | `server/src/routes/streamSessions.ts` |

### `/api/export-import`

| Method | Path | Defined in |
|---|---|---|
| `POST` | `/api/export-import/assets` | `server/src/routes/exportImport.ts` |
| `POST` | `/api/export-import/exports/resolve` | `server/src/routes/exportImport.ts` |
| `POST` | `/api/export-import/imports` | `server/src/routes/exportImport.ts` |
| `POST` | `/api/export-import/profiles/validate` | `server/src/routes/exportImport.ts` |

### `/api/compare`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/compare/songs` | `server/src/routes/compare.ts` |
| `GET` | `/api/compare/sessions` | `server/src/routes/compare.ts` |
| `POST` | `/api/compare/sessions` | `server/src/routes/compare.ts` |
| `GET` | `/api/compare/sessions/:id` | `server/src/routes/compare.ts` |
| `DELETE` | `/api/compare/sessions/:id` | `server/src/routes/compare.ts` |
| `POST` | `/api/compare/sessions/:id/takes` | `server/src/routes/compare.ts` |
| `DELETE` | `/api/compare/sessions/:id/takes/:songId` | `server/src/routes/compare.ts` |
| `PUT` | `/api/compare/sessions/:id/scores` | `server/src/routes/compare.ts` |
| `POST` | `/api/compare/sessions/:id/archive` | `server/src/routes/compare.ts` |
| `GET` | `/api/compare/analysis/:songId` | `server/src/routes/compare.ts` |
| `GET` | `/api/compare/diff` | `server/src/routes/compare.ts` |

### `/api/v1/daw`

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/v1/daw/status` | `server/src/routes/dawV1.ts` |
| `POST` | `/api/v1/daw/understand` | `server/src/routes/dawV1.ts` |
| `GET` | `/api/v1/daw/understand/:id` | `server/src/routes/dawV1.ts` |
| `POST` | `/api/v1/daw/generate` | `server/src/routes/dawV1.ts` |
| `GET` | `/api/v1/daw/jobs/:id` | `server/src/routes/dawV1.ts` |
| `GET` | `/api/v1/daw/jobs/:id/audio` | `server/src/routes/dawV1.ts` |
| `POST` | `/api/v1/daw/jobs/:id/cancel` | `server/src/routes/dawV1.ts` |

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
| `GET` | `/api/mm3/cover/launder-models` | `server/src/routes/backends.ts` |
| `POST` | `/api/mm3/cover/source-prep` | `server/src/routes/backends.ts` |
| `POST` | `/api/mm3/cover/encode` | `server/src/routes/backends.ts` |
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

### `/api/training/ops` (training operation domains)

| Method | Path | Defined in |
|---|---|---|
| `GET` | `/api/training/ops/recipes` | `server/src/services/training/recipes/operations.ts` |
| `POST` | `/api/training/ops/recipes/resolve` | `server/src/services/training/recipes/operations.ts` |
| `GET` | `/api/training/ops/recipes/:family` | `server/src/services/training/recipes/operations.ts` |
| `GET` | `/api/training/ops/review/ladder/:datasetId/:runId` | `server/src/services/training/review/operations.ts` |
| `POST` | `/api/training/ops/review/finish-batch` | `server/src/services/training/review/operations.ts` |
| `POST` | `/api/training/ops/review/decision` | `server/src/services/training/review/operations.ts` |
| `POST` | `/api/training/ops/review/select` | `server/src/services/training/review/operations.ts` |
| `POST` | `/api/training/ops/review/draft` | `server/src/services/training/review/operations.ts` |
| `GET` | `/api/training/ops/review/draft/:id` | `server/src/services/training/review/operations.ts` |
| `POST` | `/api/training/ops/review/cleanup` | `server/src/services/training/review/operations.ts` |
| `GET` | `/api/training/ops/preparation/context/:datasetId` | `server/src/services/training/yue2PreparationOperations.ts` |
| `POST` | `/api/training/ops/preparation` | `server/src/services/training/yue2PreparationOperations.ts` |
| `GET` | `/api/training/ops/preparation` | `server/src/services/training/yue2PreparationOperations.ts` |
| `GET` | `/api/training/ops/preparation/:id` | `server/src/services/training/yue2PreparationOperations.ts` |
| `GET` | `/api/training/ops/preparation/:id/job` | `server/src/services/training/yue2PreparationOperations.ts` |
| `POST` | `/api/training/ops/preparation/:id/pause` | `server/src/services/training/yue2PreparationOperations.ts` |
| `POST` | `/api/training/ops/preparation/:id/resume` | `server/src/services/training/yue2PreparationOperations.ts` |
| `POST` | `/api/training/ops/preparation/:id/retry` | `server/src/services/training/yue2PreparationOperations.ts` |
| `POST` | `/api/training/ops/preparation/:id/cancel` | `server/src/services/training/yue2PreparationOperations.ts` |
<!-- generated:end routes -->

## DAW facade notes (`/api/v1/daw`)

The facade in `server/src/services/daw/` evens out two backend quirks so DAW clients see
the same job shape on every backend:

- **Model choice (ACE-Step).** `POST /generate` takes an optional `model`: `"turbo"`,
  `"base"` or an exact installed DiT file name (the `dit_models` list in `GET /status`).
  `turbo` and `base` pick an installed model of that kind, preferring the loaded one when
  it matches. The choice is sent as the generate `ditModel`. An unknown name, or a kind
  with nothing installed, is 400 `unknown_model` listing what is installed. `lego` still
  needs a base model: `model: "turbo"` with `lego` is 400 `bad_request`, and an exact
  non-base name is 422 `base_model_required`. `model` wins over the older `dit_model`
  field. The `POST /generate` response and `GET /jobs/:id` report `dit_model` (null when
  the loaded model was left as is), and `GET /jobs/:id` reports `eta_backend`
  (`ace_turbo` or `ace_base` follows the chosen model).
- **Stable Audio 3 takes.** SA3 renders one take per job and ignores `takes`. When the
  active backend is `sa3` and `takes` is above 1, the facade submits that many single-take
  jobs back to back under one facade job id. Each take gets its own seed: `seed + i` when a
  seed is given, otherwise a fresh random seed per take. `GET /jobs/:id` reports overall
  progress across all takes plus `take_count` and `takes_done`, and lists each finished
  take with its seed. `GET /jobs/:id/audio?take=n` works for every finished take (409 while
  it is still rendering). `POST /jobs/:id/cancel` cancels the take in progress and skips
  the rest; a take that already finished is never cancelled. A status poll that fails is
  retried 3 times (0.5, 1, 2 s). If a take fails after at least one take succeeded, the
  job ends in state `partial`: it lists the finished takes (all `ready`, audio
  downloadable) and sets `error` to the failure. If no take succeeded, the job is `failed`.
- **YuE2 length.** YuE2 ignores `duration_s` and renders roughly 51 to 68 s. When the
  active backend is `yue2` and `duration_s` (or `bars`) is set, the float32 WAV export cuts
  each take to that length with a 10 ms fade-out. Trimmed takes report the requested
  `duration_s` and the original length in `trimmed_from_s` (null when no trim was needed).
  A render shorter than requested is returned as is, never padded.
- **ACE length.** ACE sizes its output from the LM's audio codes, so a text2music render
  can land a little off `duration_s` (for example 9.44 s for 10 s). For `ace` text2music
  takes the WAV export fits the audio to exactly `duration_s`: short renders are padded
  with silence, long ones are cut, with a 10 ms fade at the original end in both cases.
- **Time estimates.** `GET /status` returns `eta` (the active backend, with `backend` and
  `loaded` keys) and `eta_by_backend` for `ace_turbo`, `ace_base`, `sa3` and `yue2`. Each
  entry is `{ base_s, per_take_s, per_audio_s, cold_load_s, source }`. Estimate =
  `base_s + per_take_s * (takes - 1) + per_audio_s * duration_s`, plus `cold_load_s` when
  that backend is not the one last run (SA3 and YuE2 do not both fit in 16 GB, so any
  switch counts as cold). Defaults (`source: "default"`) come from the 2026-10-02 warm
  bench on a 4070 Ti Super; each finished facade job pulls its backend's estimate 30%
  toward the observed wall time (EMA, in memory only) and marks it `"measured"`.
  `POST /generate` and `GET /jobs/:id` return `eta_s`, the remaining seconds (estimate
  minus elapsed, floor 0, 0 once finished).

## MM3 cover encode (`/api/mm3/cover/encode`)

`POST /api/mm3/cover/encode` takes `{ sourceAudioUrl, sourceName?, launder?, caption? }` and returns
`{ jobId, plank }` at once; poll `GET /api/training/jobs/:jobId` until it ends. The engine
is stopped for the run and restarted after it. `launder` defaults to false, which runs
`ace-train mm3-codes` as before. `launder: true` runs `ace-train mm3-launder` instead
(roughly real time, about 14.5 GB of VRAM), re-rendering the source under `caption` (a
neutral caption when blank), and is 400 when its models are missing;
`GET /api/mm3/cover/launder-models` returns `{ missing: string[], flamingo: boolean }` for that check. Planks
from this route record `laundered` in their sidecar (`GET /api/mm3/plank-meta`), plus the
`rec7` and `encoder` file names when laundered.

# HOT-Step MCP (generation + training)

An MCP server exposing HOT-Step CPP's **generation** and **training dataset/job**
APIs. Generation: submit, poll, cancel, inspect the queue, switch backends,
look up a saved song — all three backends (ACE-Step 1.5, MiniMax-Music3, YuE2)
go through the same eight tools, none of them backend-specific. Training:
list/create/rescan datasets, run the labeling pipeline's three stages, list
and control jobs, then prepare, start and inspect adapter training for ACE
(LM and DiT), MiniMax-Music3 (LM) and YuE2 (joint): the thirteen `train_*` tools
below, all wrappers over `server/src/routes/training.ts`.

Everything here talks to the running app over its HTTP API
(`server/src/routes/`) — nothing reads the database or engine state directly.
That is also why this package cannot drive training or generation while the
app itself isn't running: there is no server process to call.

## Security posture — read this before exposing HOTSTEP_URL beyond loopback

The app's HTTP API has **no real authentication**. `GET /api/auth/auto`
hands out a token to anyone who asks, no credentials involved — that is the
app's existing single-user, local-machine design, not something this MCP
server adds or weakens further. Anyone who can reach `HOTSTEP_URL` can submit
generations and start labeling/caption/build jobs that occupy the GPU (MOSS,
`/understand`) or run for a long time. Keep `HOTSTEP_URL` pointed at loopback unless you have
deliberately put something else (a firewall, a tunnel with its own auth) in
front of the app.

## Setup

```
cd tools/mcp-hotstep
npm install
```

Registered in the repo root's `.mcp.json` as `hotstep`. Env:

| Var | Default | Meaning |
|---|---|---|
| `HOTSTEP_URL` | `http://127.0.0.1:3001` | Base URL of the running HOT-Step server. |

Agents run the app with `dev.bat` and work through the Vite dev server, not the
raw Node port: override `HOTSTEP_URL` to `http://127.0.0.1:3000` in `.mcp.json`'s
`hotstep` env block, then reconnect the `hotstep` MCP server in your client — same
rule as any MCP server: a connected session keeps its old tool set until
reconnected. (That reconnect is also required after editing this package's source.)

## Scripts

```
npm run typecheck   # tsc -p tsconfig.check.json — src/ and test/
npm test            # tsx --test test/*.test.ts
npm start           # run the server directly (stdio) — normally the client does this
```

## Tools

### `gen_backends`
Lists the registered backends (`ace`, `minimax-m3`, `yue2`), which one is
active, the active backend's capability manifest (`GET /api/capabilities`),
and its model catalogue (`GET /api/backends/models`). Call this before
`gen_submit` if you're not sure what's active or what it supports.

### `gen_configure`
`{ backend?, model? }` — switch the active backend and/or select its models.
Switching releases the **outgoing** backend's VRAM before anything new loads
(fire-and-forget; the switch itself answers immediately). Model selection is
engine *state* for MiniMax-Music3 (per-role quants: `lm`, `depth`, `cond`,
`dit`, `voc`) and YuE2 (`lm`, `vae_variant`) — this is where you set it. ACE
has no engine-state model selection at all: its model names travel on every
`gen_submit` call instead (its `ditModel`/`lmModel` args), so this route
answers `501` for ACE, which the tool reports as the reason rather than an
error to retry.

### `gen_submit`
`{ backend, operation?, caption, lyrics?, instrumental?, duration?, seed?,
batchSize?, title?, ditModel?, lmModel?, options? }` → `POST /api/generate`. Returns
`{ jobId, status }` immediately; poll with `gen_status` or `gen_wait`.

- `backend` is sent as both `backend` (the existing log-only field) and the
  new `expectedBackend`. If the active backend has moved on since you last
  checked `gen_backends` (someone else switched it), the submit is refused
  with `409` instead of silently running on the wrong backend.
- `operation` becomes `taskType`, defaulting to `text2music`. Checked against
  a per-backend allow-list before any HTTP call — ACE additionally supports
  `cover`, `repaint`, `lego`, `extract`, `complete`, `cover-nofsq`; the other
  two backends support `text2music` only.
- **Duration behaviour differs per backend** and the API gives no error for
  getting this wrong, so it's worth knowing: ACE (max 600s) treats `duration`
  as a *target* the LM aims for. MiniMax-Music3 (max 300s) treats it as a
  *ceiling only* — the model's own stop token usually ends it sooner. YuE2
  ignores `duration` entirely; it is always model-ended.
- `ditModel`/`lmModel` are ACE-only and independent — ACE has separate DiT
  and LM catalogues, and a name from one is not valid in the other
  (`translateParams.ts` maps them to `synth_model`/`lm_model`, and the engine
  rejects an LM name it can't find in the LM bucket). Neither does anything
  for MiniMax-Music3 or YuE2 — use `gen_configure` for those.
- `options` is a bag of backend-specific knobs sent as **top-level** HTTP
  fields, not nested. MiniMax-Music3 only reads fields prefixed `mm3` (e.g.
  `mm3Steps`); YuE2 only reads fields prefixed `yue2`; anything else is
  silently ignored by those two. ACE reads plain unprefixed names (e.g.
  `guidanceScale`, `inferenceSteps`, `negativePrompt`). An `options` key that
  collides with one of this tool's own typed args (`caption`, `lyrics`,
  `duration`, `seed`, `batchSize`, `title`, `backend`, `expectedBackend`,
  `taskType`, `ditModel`, `lmModel` — and ACE's caption aliases `prompt`,
  `songDescription`, `style`) is **rejected before any HTTP call**, since it
  would otherwise silently override what the typed arg asked for.
- **Never retried on a 401.** Every other tool retries once automatically
  after a fresh login; submit does not, because silently resending a
  generation body after a relogin risks training/rendering twice on the
  GPU. A submit that 401s returns the failure; the *next* call (a manual
  retry of `gen_submit`, or any other tool) logs in fresh.

### `gen_status`, `gen_cancel`, `gen_queue`
Thin wrappers over `GET /api/generate/status/:id`, `POST
/api/generate/cancel/:id`, `GET /api/generate/queue`.

### `gen_wait`
`{ jobId, maxSeconds? }` — polls status until it reaches `succeeded`,
`failed`, or `cancelled`, or the budget runs out, whichever comes first.
Default budget **45s**, hard max **900s**. A budget above the default needs
your **MCP client's own request timeout raised to match** — the client will
otherwise give up waiting on this call before the budget does, even though
the job and the poll loop are both still fine.

Returns `{ jobId, outcome, status }`:
- `outcome` is `"done"` (reached a terminal state), `"budget"` (the time
  budget ran out first), or `"cancelled"` (this tool call itself was
  cancelled by the client).
- `status` is the last status response actually obtained — or `null` if none
  was, which happens when the budget/cancel fires before any poll completes
  (e.g. a slow login, or an abort mid-request). A poll cut short by the
  budget or the signal never fabricates a status and never overwrites the
  last real one.

If the tool call itself is cancelled from the client side, `gen_wait`
**leaves the job running** on the server — it never calls cancel on your
behalf. Use `gen_cancel` for that.

### `gen_song`
`{ songId }` → `GET /api/songs/:id`. Returns the song's metadata plus
`absoluteAudioUrl` (its `audio_url` joined to `HOTSTEP_URL`). Never returns
audio bytes. Also includes `audioFilePath` — a real filesystem path — but
**only** when `HOTSTEP_URL` resolves to loopback (127.0.0.1/localhost), since
otherwise the path wouldn't resolve on the machine asking for it. That path
is reconstructed from this repo's own `DATA_DIR` resolution (mirrored in
`src/tools.ts` — see the design note below, `server/src/config.ts` is never
imported), so it is only correct if this MCP process sees the same
`DATA_DIR` the running server does — true for the default, unmodified setup.

### `train_capabilities`
`GET /api/training/capabilities`, returned whole: engine readiness, and
essentia/genius/LLM/MOSS/preprocess/train-lm/train-dit availability.

### `train_datasets`
`GET /api/training/datasets`, trimmed per row to `id`, `name`, `slug`,
`sourceDir`, `sampleCount` and the on-disk asset flags (`assets` —
labeled/built/tensor-variant/adapter state per backend from
`datasetAssets.ts`) — not the full row, which also carries label/caption
settings only the UI needs. Use `train_dataset` for one dataset in full.

### `train_dataset`
`{ id }` → `GET /api/training/datasets/:id`, returned whole.

### `train_dataset_create`
Mirrors `CreateDatasetInput` (`server/src/services/training/types.ts:600`):
`{ name, sourceDir, recursive?, customTag?, tagPosition?, genreRatio?,
defaultArtist?, defaultAlbum?, defaultGenre?, defaultLanguage? }` →
`POST /api/training/datasets`. Returns `201` with the created dataset.

### `train_dataset_rescan`
`{ id }` → `POST /api/training/datasets/:id/rescan`. Refuses with `409`
("A job is already running for this dataset") while a job is active for
this dataset.

### `train_dataset_label`
One tool, three stages, all async (answer `202` with a `jobId` — poll with
`train_wait` or `train_job`): `{ datasetId, stage, options? }`.
- `stage: "label"` posts `options` as `LabelOptions` (`types.ts:644`) to
  `/datasets/:id/label` — Essentia BPM/key, Genius lyrics, LLM caption; which
  steps run is in `options`.
- `stage: "caption"` posts `options` as `CaptionOptions` (`types.ts:672`) to
  `/datasets/:id/enhance/caption` — re-caption with a specific provider.
- `stage: "build"` posts `options` as `{ outputPath? }` to
  `/datasets/:id/build` — write `dataset.json`.

`options` is forwarded to the route **verbatim** — it is not re-typed here,
so a field the route gains later needs no change on this side. All three
refuse with `409` `"A job is already running for this dataset"` while one is
active. `label` additionally refuses with `409`
`"MOSS and the /understand step need the engine, which a training job owns.
Wait for it, or caption with a cloud provider (Gemini) instead."` when the
engine is held and MOSS/understand was asked for; `caption` refuses with
`409` `"MOSS needs the engine, which a training job owns. Wait for it, or
caption with a cloud provider (Gemini) instead."` under the same condition —
the two routes word it slightly differently.

### `train_jobs`, `train_job`
`{ datasetId? }` → `GET /api/training/jobs`, optionally filtered.
`{ jobId, cancel? }` → `GET /api/training/jobs/:jobId`, or `DELETE` (cancel)
when `cancel: true`.

### `train_wait`
Same contract as `gen_wait` (see above), against
`GET /api/training/jobs/:jobId`: `{ jobId, maxSeconds? }` →
`{ jobId, outcome, status }`. Terminal statuses are `TrainingJobStatus`'s
(`types.ts:50`) `done`/`failed`/`cancelled` — note `done`, not `succeeded`
(that's generation's vocabulary, not training's). Shares its polling loop
(`waitForJob` in `src/tools.ts`) with `gen_wait` rather than duplicating it.

### `train_prepare`
`{ datasetId, backend, stage?, <field object>, options? }`. Starts the data
stage training needs and returns the route's response, which carries the
`jobId`; wait with `train_wait`. Only the field object for the chosen
backend/stage may be sent. `options` is a passthrough for anything the typed
object does not name; a key it does name is rejected before any HTTP call.
Not retried on `401`.

The field objects appear in the MCP tool list as plain objects, to keep that
list small. The tool validates each one against its strict schema before any
HTTP call: an unknown key, a wrong type or a value outside an enum comes back
as an error naming the field. `train_fields` serves the same lists as the
tables below to a client that can't read this file.

| backend | stage | Route | Field object | Answers |
|---|---|---|---|---|
| `ace` | none | `POST /datasets/:id/preprocess` | `ace` | `202 { jobId }` |
| `mm3` | none | `POST /datasets/:id/mm3-codes` | `mm3` | `200 { jobId, kind }` |
| `yue2` | `preprocess` | `POST /datasets/:id/yue2-preprocess` | `yue2Preprocess` | `200 { jobId, kind, outDir, ... }` |
| `yue2` | `joint-prepare` | `POST /datasets/:id/yue2-joint-prepare` | `yue2JointPrepare` | `202 { jobId, manifest, ... }` |

`ace` and `mm3` need a built dataset. MiniMax-Music3 training also needs
per-track `.mm3.txt` captions (`train_dataset_label` stage `caption`). YuE2
`joint-prepare` imports the existing cache stages (latents, codec ids, lead
sheets) into the joint-training dataset on the CPU; its `manifest` is what
`train_start` `yue2-joint` takes as `dataset`. Every route refuses with `409`
"A job is already running for this dataset" while one is active.

The defaults below are what each route resolves an absent field to, read from
the handler in `server/src/routes/training.ts`, not from the comments in
`types.ts` (several of those have drifted). Where a route clamps silently
instead of answering `400` (`yue2-preprocess`, `mm3-train-lm`), an
out-of-range value becomes the default without an error.

#### `ace`

| Field | Type | Route default and notes |
|---|---|---|
| `ditModel` | string | DiT base the tensors (and so every adapter trained from them) are bound to. Default: the DiT selected in the Models tab (PUT /api/training/active-models), else the first BF16 DiT. 400 if unknown or none installed. |
| `vaeModel` | string | Default: first BF16 VAE, else the first VAE. |
| `textEncoder` | string | Default: first text encoder. |
| `sampleIds` | string[] | Default: every non-excluded sample whose file exists. |
| `maxDuration` | number | Seconds per song. Default 600; 0 = no truncation; < 0 is a 400. |
| `normalize` | `peak` \| `none` | Default 'peak'. |
| `targetDb` | number | Default -1.0; -60..0. |
| `dtype` | `f32` \| `bf16` | Default 'f32'. |
| `compat` | `hotstep` \| `sidestep` | Default 'hotstep'. |
| `maxCaptionTokens` | number | Default 512; 16..4096. |
| `maxLyricTokens` | number | Default 2048; 16..4096. |
| `vaeChunk` | number | Latent frames. Default 384; >= 64. |
| `vaeOverlap` | number | Latent frames. Default 48; >= 0 and < vaeChunk. |
| `overwrite` | boolean | Default false. |
| `stopEngine` | boolean | Stop ace-server for the job. Default true. |
| `outputDir` | string | Default `<tensors>/<slug>/<variantKey>`. Must be a subdirectory of the dataset's tensors root (400 otherwise). |

#### `mm3`

| Field | Type | Route default and notes |
|---|---|---|
| `maxDuration` | number | Seconds per song. Default: no cap (anything not > 0 also means no cap). |
| `launder` | boolean | Default false. true writes the SEPARATE laundered cache (mm3-codes-laundered/) via ace-train mm3-launder; train with launder: true to use it. |

#### `yue2Preprocess`

| Field | Type | Route default and notes |
|---|---|---|
| `vaeVariant` | `standard` \| `legacy` | Default 'standard'. |
| `captionMode` | `ace` \| `yue2` \| `txt` \| `default` \| `none` | Default 'yue2' when the source folder has sidecars with a caption or lyrics, else 'none'. 'default' needs defaultCaption; 'txt' needs acknowledgeSidecarFormat (400 otherwise). |
| `defaultCaption` | string | The caption for every clip under captionMode 'default'. Default ''. |
| `acknowledgeSidecarFormat` | boolean | Required true for captionMode 'txt': that mode feeds the whole ACE .txt sidecar in as the style prompt. |
| `decode` | `auto` \| `ffmpeg` | Default 'auto'. 'ffmpeg' is a 400 without an ffmpeg binary. |
| `loudnessLufs` | number | Default: absent = the engine's -14 LUFS. 0 = off; else -40..0 (400 outside). Part of the cache key: a change re-encodes. |
| `clipSeconds` | number | Default 10; 1..60. |
| `tileFrames` | number | Default 750; 1..100000. |
| `haloFrames` | number | Default 20; 12..4096. |
| `only` | string | Default '' (all files). |
| `limit` | number | Default 0 (no limit); 0..100000. |
| `force` | boolean | Default false. |

#### `yue2JointPrepare`

| Field | Type | Route default and notes |
|---|---|---|
| `models` | object | Model file paths. Default: resolved from the YuE2 latent manifest and the installed models (see train_runs readiness.jointPrepare.defaults). The route also takes a repeatable `model` ["vae=FILE", ...] form; send that through options. Keys: `vae`, `semantic`, `sheetsage`. |
| `legacyManifest` | string | The YuE2 latent manifest from the preprocess stage. Default: this dataset's own; any other dataset's is a 400. The route also accepts it as `manifest`. |
| `checkpoint` | string | Recorded as provenance only. Default: the ConvRot checkpoint if installed, else the LM GGUF. |
| `tokenizer` | string | Default: the YuE2 LM GGUF (it carries the text vocabulary). |
| `output` | string | Default: a fresh `aitk-prepared-<timestamp>` dir beside the latent manifest. Must not exist yet. |
| `lyricTiming` | boolean | Default true. |

### `train_start`
`{ datasetId, backend, <field object>, options? }`. Starts a run and returns
the route's response, which carries the `jobId`; wait with `train_wait`,
cancel with `train_job` `cancel: true`. Same field-object and `options` rules
as `train_prepare`. Never retried on `401`, because a resend could start a
second run.

| backend | Route | Field object | Needs |
|---|---|---|---|
| `ace-lm` | `POST /datasets/:id/train-lm` | `aceLm` | `train_prepare` `ace` |
| `ace-dit` | `POST /datasets/:id/train-dit` | `aceDit` | `train_prepare` `ace` |
| `mm3-lm` | `POST /datasets/:id/mm3-train-lm` | `mm3Lm` | `train_prepare` `mm3`, `.mm3.txt` captions |
| `yue2-joint` | `POST /datasets/:id/yue2-joint-train` | `yue2Joint` | `train_prepare` `yue2` (both stages), or `autoPrepare` |

`yue2-joint`: the tool always sends `trainingMethod: "aitk"` and rejects a
caller that tries to set it; the route never picks Legacy implicitly. The
route does **not** auto-prepare by default. A fresh run needs `dataset` unless
`autoPrepare: true` is sent (the Training Studio form sends it), and a resume
never auto-prepares. `resumeRunId` and `resumeStep` go together: the tool
rejects one without the other, since the route ignores a lone `resumeStep`
and would start a fresh run. `GET /api/training/defaults` has sections for
label, preprocess, train-lm and train-dit only, nothing for YuE2.

`ace-lm` resumes by default when its target chain is one leg, which it is at
the default `targetLoss` (see `initAdapter`). `initAdapter: ""` does **not**
force scratch there, because the route treats an empty string as absent. Use a
new `adapterName` or a multi-leg `targetLossStages`. `ace-dit` resumes the
newest run of the adapter name unless `initAdapter: ""` is sent. On either,
a resume takes the adapter shape (type, rank, alpha, LoKr dims) from the
source run and ignores the request's.

`mm3-lm`: `hra: true` on its own is dropped silently and the run trains
HOT-PiZZA, because `pissa` defaults on and takes precedence. Send
`pissa: false` with it.

#### `aceLm`

| Field | Type | Route default and notes |
|---|---|---|
| `variantKey` | string | Preprocess variant to train from. Default: the newest. Unknown is a 400. |
| `lmSize` | `0.6B` \| `1.7B` \| `4B` | Default '4B'. |
| `lmModel` | string | LM GGUF. Default: the BF16 model matching lmSize. Unknown is a 400. |
| `ditModel` | string | DiT whose FSQ tokenizer extracts the codes. Default: the variant's own DiT. |
| `adapterName` | string | Default: the dataset slug. [A-Za-z0-9._-]{1,64}. |
| `epochs` | number | Hard cap per leg. Default 150; 1..200. |
| `targetLoss` | number | Final target. Default 4.0; 0..20; 0 = no auto-stop. |
| `targetLossStages` | number[] | Explicit target chain, strictly descending (0 only last), one ace-train leg each. Default: the 2.0 and 1.5 rungs above targetLoss, then targetLoss — so [4.0], one leg, at the default targetLoss. |
| `initAdapter` | string | Resume source: a run dir or 'latest'. Default: scratch when the chain has more than one leg; with ONE leg (the default chain) 'latest' — the newest non-calibrated run of this adapter name, or scratch if none. '' counts as absent (training.ts:5592), so it does NOT force scratch on a one-leg chain: use a new adapterName or a multi-leg targetLossStages. A resume collapses the chain to [targetLoss] unless targetLossStages is sent. |
| `adapterType` | `lora` \| `lokr` | Default 'lora'. On any resume (including the default 'latest' one) adapterType, rank, alpha, the lokr* fields, weights and pissa/hotPizza are ignored: the engine adopts them from the source run. |
| `rank` | number | Default 16; 1..256. |
| `alpha` | number | Default 32; 1..1024. |
| `lokrDim` | number | Default 128; 4..4096 (lokr only). |
| `lokrAlpha` | number | Default 128. |
| `lokrFactor` | number | Default 6; -1 or 2..64 (lokr only). |
| `lokrDecomposeBoth` | boolean | Default true. |
| `optimizer` | `adamw` \| `muon` \| `prodigy` | Default 'adamw' (the route's resolution; types.ts's 'prodigy' comment is stale). |
| `muonLrScale` | number | Default 20; 0.001..1000. |
| `muonNsSteps` | number | Default 5; 1..20. |
| `learningRate` | number | Default 0.0001; > 0 and <= 1. |
| `gradAccum` | number | Default 2; 1..64. |
| `gradClip` | number | Default 1.0; 0..100; 0 disables. |
| `warmupRatio` | number | Default 0.05; 0..0.5. |
| `weightDecay` | number | Default 0.01; 0..1. |
| `maxLen` | number | Default 0 = auto-fit from free VRAM; else 512..16384. |
| `seed` | number | Default 42. |
| `milestoneStep` | number | Default 0 = milestones off; 0..5. |
| `milestoneKeep` | number | Default 6; 0..64. |
| `lowVram` | `auto` \| `on` \| `off` | Default 'auto'. |
| `attnHeadBlock` | number | Default 0 = engine picks; 0..128. > 0 with a flash attnBackend is a 400. |
| `chunk` | number | Default 0 = engine default (128); else 16..1024. |
| `attnBackend` | `exact` \| `flash` \| `flash-f32` | Default 'flash'. Coerced to 'exact' when the artist token is on and prefixN > 0, which is the default (token on → prefixN 8). |
| `weights` | `f32-window` \| `bf16` | Default 'bf16'. bf16 + bwd 'mm' is a 400. |
| `bwd` | `outprod` \| `mm` | Default 'outprod'. |
| `batch` | number \| `auto` | Must be 1 (the default); anything else is a 400 — micro-batching is not built. |
| `captionDropout` | number | Default 0.3; 0..1. Needs a trigger word (the variant's custom_tag): the default silently drops to 0 without one, an explicit value is a 400. |
| `regEvery` | number | Prior-preservation cadence. Default 0 = off; >= 2 when on. An explicit value with no other 600 s corpus available is a 400. |
| `regTopk` | number | Default 64; 1..256. |
| `regSongs` | number | Default 24; 1..200. |
| `regTeacher` | `cached` \| `live` | Default 'cached'. |
| `regCorpora` | `auto` \| string[] | Default 'auto' (up to 6 other artists preprocessed at 600 s). An array must be existing lm_codes.jsonl paths under data/training/tensors/. |
| `calibrate` | boolean | Run post-training calibration. Default false. |
| `calibrateRepoint` | boolean | Default true. |
| `stages` | `extract` \| `train` \| `export`[] | Default all three. |
| `artistToken` | string | Learned artist token. Default ON, named after the adapter; '' switches it off. |
| `artistTokenK` | number | Default 32; clamped to 1..256. |
| `artistTokenLr` | number | Default 0.005; clamped to 0..1. |
| `prefixN` | number | KV prefix columns. Default 8 when the artist token is on, else 0; clamped to 0..64. |
| `lossOnCot` | boolean | Default true. |
| `order` | `shuffle` \| `fixed` | Default 'shuffle'. |
| `overwrite` | boolean | Re-extract every song. Default false. |
| `stopEngine` | boolean | Default true. |
| `rslora` | boolean | Default false. |
| `dora` | boolean | Default false. The LoRA-family methods resolve silently by precedence dora > hira > loha > pissa > hra; rslora combines with any of the first four, and hra + rslora is a 400. All are LoRA-only and dropped silently under lokr. |
| `hira` | boolean | Default false. |
| `loha` | boolean | Default false. |
| `pissa` | boolean | Default false. |
| `hotPizza` | boolean | Default false. Implies pissa. |
| `hra` | boolean | Default false. |
| `loraPlusRatio` | number | Default 1 (off). |

#### `aceDit`

| Field | Type | Route default and notes |
|---|---|---|
| `variantKey` | string | Preprocess variant to train from. Default: the newest. Unknown is a 400. |
| `adapterName` | string | Default: the dataset slug. [A-Za-z0-9._-]{1,64}. |
| `adapterType` | `lora` \| `lokr` | Default 'lokr'. Changes the defaults of learningRate, weightDecay and lossWeighting. On a resume (the default whenever a previous run exists) the engine adopts the adapter shape (type, rank, alpha, lokr*, layers, targetMlp, pissa) from the source run, but those three defaults still follow THIS field. |
| `initAdapter` | string | Resume source: a run dir or 'latest'. Default 'latest' — the newest non-calibrated run of this adapter name, or scratch if none. '' forces scratch. |
| `epochs` | number | Hard cap. Default 500; 1..2000. |
| `targetLoss` | number | Default 0.3; 0..20; 0 = no auto-stop. |
| `rank` | number | Default 128; 1..256. Used by lora, but range-checked (400) for either type, as are the lokr* fields. |
| `alpha` | number | Default 256; 1..1024. |
| `lokrDim` | number | Default 512; 4..4096. |
| `lokrAlpha` | number | Default 512; 0..8192; 0 = dim. |
| `lokrFactor` | number | Default 6; -1 or 2..64. |
| `lokrDecomposeBoth` | boolean | Default true. |
| `targetMlp` | boolean | Default true. |
| `dora` | boolean | Default false. hra + rslora is a 400. |
| `hira` | boolean | Default false. |
| `loha` | boolean | Default false. |
| `pissa` | boolean | Default false. |
| `hra` | boolean | Default false. |
| `rslora` | boolean | Default false. |
| `loraPlusRatio` | number | Default 1 (off). |
| `layers` | number | Default 0 = auto; 0..64. |
| `crop` | number | Default 0 = auto-fit; else 128..8192 frames. |
| `cropMin` | number | Default 375; 128..8192. |
| `cropMax` | number | Default 0 = engine default cap; else 128..8192 and >= cropMin. |
| `cropAnchor` | `song` \| `zero` | Default 'song'. |
| `cropMode` | `structured` \| `random` | Default 'structured'. |
| `cropStartFrac` | number | Default 0.2; with cropEndFrac >= 0 and summing to <= 1. |
| `cropEndFrac` | number | Default 0.2. |
| `cropJitter` | boolean | Default false. |
| `learningRate` | number | Default 0.002 (lokr) / 0.0005 (lora); > 0 and <= 1. |
| `gradAccum` | number | Default 4; 1..64. |
| `gradClip` | number | Default 1.0; 0..100. |
| `warmupRatio` | number | Default 0.05; 0..0.5. |
| `weightDecay` | number | Default 0.001 (lokr) / 0.01 (lora); 0..1. |
| `lossWeighting` | `none` \| `flow_snr` | Default 'none' (lokr) / 'flow_snr' (lora). |
| `snrGamma` | number | Default 5; 1..100. |
| `tBias` | number | Default 0.5; 0..4. |
| `channelBalance` | boolean | Default true. |
| `timestepMu` | number | Default -0.4; -4..4. |
| `timestepSigma` | number | Default 1.0; > 0 and <= 4. |
| `tMin` | number | Default 0; 0..1, < tMax. |
| `tMax` | number | Default 1; 0..1. |
| `cfgRatio` | number | Default 0.15; 0..1. |
| `genreRatio` | number | Percent. Default 30; 0..100. |
| `seed` | number | Default 42. |
| `order` | `shuffle` \| `fixed` | Default 'shuffle'. |
| `milestoneStep` | number | Default 0.1; 0..5; 0 disables. |
| `milestoneKeep` | number | Default 6; 0..64. |
| `vramReserveMb` | number | Default 2048; 0..16384. |
| `batch` | number | Default 1; 1..16. |
| `ckptSegments` | number | Default 1 (auto); 0 = off; 2..32 fixed. |
| `optimizer` | `adamw` \| `muon` \| `prodigy` | Default 'prodigy'. |
| `muonLrScale` | number | Default 20; 0.001..1000. |
| `muonMomentum` | number | Default 0.95; 0..0.999. |
| `muonNsSteps` | number | Default 5; 1..20. |
| `muonMinDim` | number | Default 16; 1..4096. |
| `mirror` | `f32` \| `bf16` \| `bf16-f32` | Default 'bf16-f32'. |
| `bwd` | `outprod` \| `mm` | Default 'mm'. |
| `attnBackend` | `exact` \| `flash` \| `flash-f32` | Default 'exact'. |
| `stages` | `train` \| `export`[] | Default both. |
| `overwrite` | boolean | Default false. |
| `stopEngine` | boolean | Default true. |
| `calibrate` | boolean | Default false. |
| `calibrateRepoint` | boolean | Default true. |

#### `mm3Lm`

| Field | Type | Route default and notes |
|---|---|---|
| `preset` | `fast` \| `balanced` \| `thorough` | Sits under the explicit fields. fast/balanced/thorough = 300/600/900 steps (lr 8e-5, maxFrames 9000, prefixFrames 0, prefixChunk 256). Absent: the plain defaults, 500 steps. |
| `basePrecision` | string | An installed base id (train_runs readiness.mm3.bases). Default 'q8_0'; an id that is not installed also falls back to 'q8_0'. |
| `launder` | boolean | Train on the laundered codes cache. Default false. 400 if that cache is empty. |
| `rank` | number | Default 128; 1..512. |
| `alpha` | number | Default 128; 1..2048. |
| `lr` | number | Default 8e-5; 1e-7..1e-2. |
| `steps` | number | Default 500 (or the preset's); 1..100000. The cap in both stop modes. |
| `saveEvery` | number | Default 100; 0..100000. |
| `warmup` | number | Default 25; 0..100000. |
| `gradAccum` | number | Default 1; 1..64. |
| `seed` | number | Default 42. |
| `maxFrames` | number | Default 9000; 64..9000. |
| `longTracks` | `exclude` \| `crop` \| `excise` | Default 'exclude'. 400 if exclude would drop more than half the tracks. |
| `cropMode` | `random` \| `beginning` \| `structured` | Default 'structured'. |
| `cropStartFrac` | number | Default 0.2; 0..1. |
| `cropEndFrac` | number | Default 0.15; 0..1. |
| `cropStartTiles` | number | Default 1; 1..64. |
| `cropAnchor` | `song` \| `zero` | Default 'song'. |
| `endCropVary` | boolean | Default false. |
| `endCropMin` | number | Default 128; 1..9000. |
| `regScoreLast` | number | Default 0; 0..9000. |
| `scoreLast` | number | Default 0; 0..9000. |
| `scoreLastEndOnly` | boolean | Default false. |
| `keepResumeState` | boolean | Default false. |
| `verifyExport` | boolean | Default false. |
| `lyricsDropout` | number | Default 0; 0..1. |
| `trimTrailingSilence` | boolean | Default false. |
| `depthLossWeight` | number | Default 1.0; 0..10. |
| `depthLossFrames` | number | Default 128; 1..1024. |
| `optimizer` | `muon` \| `adamw` \| `prodigy` | Default 'adamw'. |
| `muonLrScale` | number | Default 64; 0.01..4096. |
| `holdout` | number | Default 0.15; 0..0.5. |
| `evalEvery` | number | Default 250; 0..100000. |
| `evalCrop` | number | Default 750; 8..9000, clamped to maxFrames. |
| `rankDropout` | number | Default 0.1; 0..0.9. |
| `adapterType` | `lora` \| `lokr` | Default 'lora'. |
| `lokrFactor` | number | Default 6; 1..64. |
| `lokrDim` | number | Default 512; 1..8192. |
| `lokrAlpha` | number | Default 512; 1..8192. |
| `attnBackend` | `exact` \| `flash` | Default 'flash'. Coerced to 'exact' on an engine with no fused attention-training kernel (Vulkan, Metal); the response echoes the value used. |
| `rslora` | boolean | Default false. |
| `dora` | boolean | Default false. More than one of dora/hira/loha/pissa/hra sent as true, or any of them with lokr, is a 400. |
| `hira` | boolean | Default false. |
| `loha` | boolean | Default false. |
| `pissa` | boolean | Default true (ignored under lokr). pissa: false (with hotPizza absent) also turns HOT-PiZZA off. |
| `hotPizza` | boolean | Default true. Implies pissa. |
| `pissaCache` | boolean | Default true. |
| `pissaFrozenF16` | boolean | Default true. |
| `hra` | boolean | Default false. hra + rslora is a 400. hra: true ALONE is dropped silently and the run trains HOT-PiZZA, because pissa defaults on and wins (mm3Train.ts:1333): send pissa: false with it. dora/hira/loha alone are fine. |
| `loraPlusRatio` | number | Default 1; 1..64. |
| `artistToken` | string | Default '' (off). |
| `artistTokenK` | number | Default 32; 1..256. |
| `artistTokenLr` | number | Default 0.005; 1e-6..1. |
| `prefixN` | number | Default 0; 0..64. With a regularisation corpus it is a 400. |
| `prefixFrames` | number | Default 0 (off); 0..9000. |
| `prefixChunk` | number | Default 256; 32..2048. |
| `prefixSelftest` | boolean | Default true. |
| `trigger` | string | Default: the dataset's custom tag. |
| `triggerPrepend` | boolean | Default true. |
| `lrEndFrac` | number | Default 0.005; 0..1. |
| `stopMode` | `steps` \| `loss` | Default 'steps'. |
| `targetLoss` | number | Default 1.0; 0..100. Used when stopMode is loss. |
| `targetLossMetric` | `train` \| `eval` | Default 'train'. 'eval' with stopMode loss needs holdout > 0 and evalEvery > 0 (400 otherwise). |
| `targetLossEpochs` | number | Default 5; 1..10000. |
| `regularisation` | object | Prior preservation. Default off. Keys: `datasetId`, `corpusDir`, `every`, `topK`. |
| `preview` | object | Mid-run previews. Default off. Keys: `everySteps`, `everyMinutes`, `seconds`, `seed`, `caption`, `lyrics`, `previewSongId`, `control`, `controlCaption`, `baseline`, `scaleMlp`, `scaleAttn`. |

#### `yue2Joint`

| Field | Type | Route default and notes |
|---|---|---|
| `steps` (required) | integer | Total steps. Required (400 without it). On a resume it must exceed resumeStep. |
| `saveEvery` | integer | Checkpoint cadence, 1..steps. Required on a fresh run (400 without it); on a resume it is inherited from the run, except that refine may set it and refinePlanner forces it to steps. |
| `resumeRunId` | string | Resume a run listed by train_runs (yue2-joint). Must be sent with resumeStep. Base, seed, optimizer, adapter shape, dataset, lyricTiming and cursorWeight come from that run; only the stop target, preview and refine fields can change. A base-matched run re-applies its recipe on resume, which clears refine/refinePlanner/freezePlannerNow and the loss/KL targets (read from the code, not exercised). |
| `resumeStep` | integer | Checkpoint step of resumeRunId that has an optimizer state. Must be sent with resumeRunId. |
| `autoPrepare` | boolean | Default false. true (fresh runs only) makes the job run joint preparation first with the readiness defaults, so `dataset` is not needed. Without it a fresh run needs `dataset`. A resume never auto-prepares. |
| `preparation` | object | Overrides for autoPrepare. Ignored without it. Keys: `models`, `legacyManifest`, `tokenizer`. |
| `dataset` | string | Prepared dataset.json (the `manifest` a train_prepare yue2 joint-prepare run returns). Required unless autoPrepare. |
| `base` | string | Base id (train_runs readiness.jointPrepare.bases). Default: the first runnable base. |
| `checkpoint` | string | Explicit base checkpoint file; overrides base. |
| `output` | string | Default: a new dir under the adapters root named after the trigger. Must not exist (or be empty with previews on). |
| `resume` | string | Low-level: an optimizer checkpoint file. Prefer resumeRunId/resumeStep, which fill this in. |
| `seed` | integer | Default 42; uint32. |
| `device` | string | ggml device, e.g. CUDA0, Vulkan0, MTL0, CPU. Default: from the engine build (CUDA0 on CUDA). |
| `method` | `tuned` \| `base-matched` | Default 'tuned'. 'base-matched' forces: stopMode steps (targetLoss/targetKl/targetKlMode cleared, narExtraSteps 0), lrSchedule cosine-floor with lrFloor 0.1, arTargets base, klWeight 0, captionDropout 0, plannerLrScale 1, narLrScale 1, cursorWeight 0 (lyric timing off whatever lyricTiming says), the spike/recon guards off, planCheck/refine/refinePlanner/freezePlannerNow cleared, autoRefine false. It fills blanks with optimizer adamw-lm, lr 1e-4, weightDecay 0.1, beta1 0.9, beta2 0.95, abcDropout 0.5, narCropFrames 1500, arLossWeight 0.25, gradAccum 4, text/lyric/bothDropout 0.1, warmup 3% of steps. An explicit optimizer 'adamw' with it is a 400 (cosine-floor needs another optimizer). |
| `calibrated` | boolean | Base-matched fresh runs only: scale steps and saveEvery to the album length. Default false. |
| `optimizer` | `adamw` \| `adamw-lm` \| `prodigy` \| `muon` | Default 'adamw'. |
| `cautious` | boolean | Default false. Needs an optimizer other than adamw. |
| `rank` | integer | Default 64; 1..65536. |
| `alpha` | number | Default 64; > 0. |
| `prodigyD0` | number | > 0. Default: engine. |
| `adapterType` | `lora` \| `lokr` | Default 'lora'. |
| `lokrDim` | integer | 1..65536. Default: engine. |
| `lokrFactor` | integer | 1..65536. Default: engine. |
| `muonLrScale` | number | > 0. Default: engine. |
| `muonNsSteps` | integer | 1..20. Default: engine. |
| `stopMode` | `steps` \| `loss` \| `kl` | Default 'steps'. |
| `targetLoss` | number | Required (>= 0) when stopMode is loss. |
| `targetKl` | number | Required (> 0, <= 100) when stopMode is kl. |
| `targetKlMode` | `mean` \| `trend` | kl only. Default 'mean'. |
| `narExtraSteps` | integer | kl only; >= 0. |
| `lr` | number | > 0, <= 1. Default: engine. |
| `weightDecay` | number | 0..10. Default: engine. |
| `klWeight` | number | 0..100. Default: engine. |
| `abcDropout` | number | 0..1. Default: engine. |
| `captionDropout` | number | 0..1. Default: engine. |
| `plannerLrScale` | number | > 0, <= 100. Default: engine. |
| `narLrScale` | number | > 0, <= 100. Default: engine. |
| `spikeFactor` | number | 0..1000. Default: off. |
| `spikeStop` | integer | 0..1000. Default: off. |
| `spikeStopWindow` | integer | 1..100000. |
| `reconStop` | number | 0..0.999. Default: off. |
| `reconStopWindow` | integer | 1..1000. |
| `reconTarget` | number | 0..10. Default: none. |
| `reconKeepDelta` | number | 0..1. |
| `lrSchedule` | `cosine` \| `cosine-floor` \| `constant` \| `linear` \| `wsd` \| `sgdr` | Default: the engine's cosine. Anything else needs an optimizer other than adamw. |
| `lrDecayShape` | `linear` \| `cosine` |  |
| `lrFloor` | number | 0..1. |
| `lrDecaySteps` | integer | 1..100000. |
| `lrCycleSteps` | integer | 1..100000. |
| `lrCycleMult` | number | 1..10. |
| `lrScale` | number | 0.001..10. |
| `klOvershootMargin` | number | 0..10. |
| `narCropFrames` | integer | 0 (whole song) or 1..12288. Default: engine. |
| `arCropFrames` | integer | 1..12288. 0 or absent = whole song. |
| `warmup` | integer | 0..steps (above steps the JOB fails, not the route). Default: engine. |
| `arLossWeight` | number | > 0, <= 10. |
| `beta1` | number | 0..0.999999. |
| `beta2` | number | 0..0.999999. |
| `gradAccum` | integer | 1..256. Default: engine. Above 1 needs lyric timing off (cursorWeight 0): with the default timing on, the route accepts it and the JOB fails. |
| `textDropout` | number | 0..1; the three dropouts sum to <= 1. Any of them above 0 needs lyric timing off and captionDropout 0, or the JOB fails (the route accepts it). |
| `lyricDropout` | number | 0..1. |
| `bothDropout` | number | 0..1. |
| `arTargets` | `tuned` \| `base` | Default 'tuned'. |
| `planCheck` | object | Plan-check planner stop. Only active with stopMode kl and narExtraSteps > 0 (otherwise dropped silently), and needs stopEngine: false or the JOB fails (the route does not check). Keys: `every`, `plans`, `margin`, `seed`, `caption`, `lyrics`. |
| `stopEngine` | boolean | Default true (stop ace-server for the run). |
| `lyricTiming` | boolean | Default true. Alias: alignmentEnabled (lyricTiming wins). |
| `alignmentEnabled` | boolean |  |
| `cursorWeight` | number | 0..10. Default 0.08 with lyric timing on, else 0. |
| `freezePlannerNow` | boolean | Resume only: freeze the planner. |
| `refine` | boolean | Resume only: decoder refinement. |
| `refinePlanner` | boolean | Resume only: planner refinement to targetKl in KL rungs. |
| `klCheckpointEvery` | number | refinePlanner: 0.01..1, default 0.1. |
| `refineLrScale` | number | refinePlanner: 0.05..1, default 0.3. |
| `autoRefine` | boolean | Fresh runs only: chain a planner refinement after the run. |
| `preview` | object | Checkpoint previews. Default off. Keys: `enabled`, `everySteps`, `seconds`, `takes`, `odeSteps`, `narCacheRatio`, `seed`, `previewMaxFrames`, `baseline`, `control`, `parallel`, `sharedSheet`, `caption`, `lyrics`, `lyricsSource`, `previewSongId`. |

### `train_runs`
`{ datasetId, backend }`, with `backend` as for `train_start`. Returns
`{ runs, readiness }`. Every call is a `GET`.

| backend | `runs` | `readiness` |
|---|---|---|
| `ace-lm` | `/datasets/:id/train-lm` (newest adapter state) | `preprocess`: `/preprocess` (tensor variants) |
| `ace-dit` | `/datasets/:id/train-dit` | `preprocess`: `/preprocess` |
| `mm3-lm` | `/datasets/:id/mm3-runs` | `mm3`: `/mm3` (codes, missing models, bases, regCandidates, defaults, presets) |
| `yue2-joint` | `/datasets/:id/yue2-joint-runs` (checkpoints, `resumeError`) | `yue2`: `/yue2` (latent cache); `jointPrepare`: `/yue2-joint-prepare` (bases, default base and device, missing models) |

`jointPrepare.ready` describes a fresh default output directory, so it reads
false even after a prepare has run. The prepare job's `manifest` is the path
to train with.

### `train_fields`
`{ backend, stage? }`, with `backend` any `train_prepare` or `train_start`
backend (`stage` for `yue2` prepare only). Returns `{ argument, route,
fields }`: the name of the field object to send, the route it posts to, and
one row per field with `name`, `type`, `required`, `default` and `notes`.
Nested objects are flattened as `parent.child`. The rows come from the same
zod schemas that validate the call, so they can't drift from it. `default` is
lifted from the description's "Default ..." clause and is `null` where the
route forwards nothing (the engine's default applies) or the description
states none.

## Design notes for maintainers

- `src/trainSchemas.ts` is a **client-side mirror** of what the six training
  start/prepare routes read from the body and what they default an absent
  field to. Its descriptions are the documentation: `train_fields` serves
  them, and the tables above were generated from them. When a route gains a
  field or changes a fallback, update the schema and regenerate the table.
  Until then the new field still works through `options`.
- The tool list is capped at 16 KB by a test in `test/client.test.ts`,
  because every MCP client loads the whole list into context each session.
  Keep tool descriptions to a clause or two and put the detail here.

- `src/http.ts` is the one place every tool's HTTP call goes through:
  auto-login on first use, one retry on a `401` after a fresh login (opt-out
  per call — `gen_submit` opts out), a 10s timeout per fetch, and the
  server's raw response text is always available on failure rather than a
  paraphrase.
- `src/tools.ts` holds the actual tool logic, deliberately separate from
  `src/server.ts`'s `server.tool(...)` registrations, so a test can call
  `genSubmit(...)` etc. directly against a loopback HTTP fixture without any
  MCP protocol machinery in the way (see `test/http.test.ts`).
  `test/client.test.ts` covers the one thing `tools.ts`-level tests can't: that
  `server.ts` actually wires everything up correctly end to end, over the
  SDK's in-memory transport.
- `BACKEND_OPERATIONS` in `src/tools.ts` is a **client-side mirror** of
  `ACE_OPERATIONS`/`MM3_OPERATIONS`/`YUE2_OPERATIONS` in
  `server/src/services/backends/{ace,minimax,yue2}/index.ts`. Nothing in the
  HTTP API exposes a backend's supported operations (the capabilities
  manifest has no such field), and this slice keeps generation strictly
  HTTP-only — so if a backend gains or loses an operation, update the mirror
  here too.
- `src/tools.ts` never imports `server/src/config.ts`, even though that would
  be the obvious way to get `audioDir`. That module bootstraps the app on
  import (creates `.env` from `.env.example` on first launch, logs to
  stdout), and this process's stdout *is* its MCP JSON-RPC wire once
  `index.ts` connects the stdio transport — a stray non-JSON line there
  corrupts the stream for the client. `AUDIO_DIR` is instead a small,
  deliberately duplicated mirror of `config.ts`'s `DATA_DIR` resolution; see
  the comment above its definition.

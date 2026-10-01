# HOT-Step MCP (generation + training)

An MCP server exposing HOT-Step CPP's **generation** and **training dataset/job**
APIs. Generation: submit, poll, cancel, inspect the queue, switch backends,
look up a saved song — all three backends (ACE-Step 1.5, MiniMax-Music3, YuE2)
go through the same eight tools, none of them backend-specific. Training:
list/create/rescan datasets, run the labeling pipeline's three stages, list
and control jobs — the nine `train_*` tools below, all thin wrappers over
`server/src/routes/training.ts`. Training a *backend* (LM/DiT/YuE2 adapters)
is a later slice; these nine cover the dataset lifecycle and job control only.

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

After editing this package's source, **reconnect the `hotstep` MCP server in
your client** — same rule as any MCP server: a connected session keeps its
old tool set until reconnected.

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
active; `label` and `caption` additionally refuse with `409`
`"MOSS and the /understand step need the engine, which a training job owns.
Wait for it, or caption with a cloud provider (Gemini) instead."` when the
engine is held by a training job.

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

## Design notes for maintainers

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

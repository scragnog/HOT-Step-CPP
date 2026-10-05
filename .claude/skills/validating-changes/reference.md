# Validating Changes — Reference

Deep detail backing [SKILL.md](SKILL.md). All paths/line refs verified against the
repo on 2026-07-02. Line numbers drift as code changes — treat them as anchors,
re-verify before relying on exact positions.

## 1. Health endpoints

`GET http://localhost:3000/api/health` (dev app — :3001 is the raw prod server) (`server/src/routes/health.ts:11-46`) returns:

```json
{
  "status": "ok",
  "aceServer": { "status": "ok|disconnected", "url": "...", "version": "..." },
  "server":    { "port": 3001, "uptime": 123.4 },
  "engine":    { "ready": true, "bootStatus": "..." }
}
```

Require `engine.ready === true` before submitting work. `aceServer.status:
"disconnected"` during an active generation may just mean the engine is busy —
ace-server is **single-threaded httplib** and cannot answer HTTP mid-DiT-step
(`server/src/services/aceClient.ts:6-19`; the client uses generous timeouts for
exactly this reason). Direct engine probe, bypassing Node:
`Invoke-RestMethod http://127.0.0.1:8085/health`.

## 2. Auth

All in `server/src/routes/auth.ts`:

- `GET /api/auth/auto` (lines 36-40) — get-or-create local user, returns
  `{ user, token }`. No password. Tokens are in-memory UUIDs (`tokens` Map),
  **reset on Node server restart**.
- Route guard: `getUserId(req)` (lines 98-102) reads `Authorization: Bearer <token>`.
- `POST /api/generate` requires it (`generate.ts:1365-1366` → 401 without).
- The status/queue endpoints do **not** require auth (they never call `getUserId`).

## 3. POST /api/generate — request semantics

Route at `server/src/routes/generate.ts:1355-1388`. `req.body` is stored verbatim
as `job.params`; `translateParams`
(`server/src/services/generation/translateParams.ts`) maps UI names → engine
(`AceRequest`) names:

- Caption source precedence: `params.prompt || params.songDescription ||
  params.caption || params.style` (translateParams.ts:13).
- `instrumental: true` ⇒ `lyrics: "[Instrumental]"` (lines 17-21).
- `randomSeed: true` ⇒ random seed; else `seed` passed through (lines 37-40).
- `duration` gets `durationBuffer` added only when `autoTrimEnabled` (lines 25-28).

Phase selection (`generate.ts:206-226`):

- `skipLm: true` ⇒ `needsLm = false`; defaults backfilled: `bpm 120`,
  **`duration 120` if unset** (always pass a short `duration` for smokes),
  `keyscale "C major"`, `timesignature "4"`.
- Cover-family task types (`cover`, `cover-nofsq`, `repaint`, `lego`, `extract`)
  also skip the LM, as does a request that already carries `audio_codes`.

Responses:

- `200 { jobId, status: "pending" }` (`generate.ts:1384-1387`).
- `503 { error: "Engine not ready: <bootStatus>" }` while bootstrapping
  (`generate.ts:1357-1363`).
- `401 { error: "Unauthorized" }` without a valid token.

## 4. Queueing, retries, watchdogs

- **Global serial queue** (`enqueueGeneration`, `generate.ts:1291-1352`): one job at
  a time — the engine is single-GPU, and progress-line subscription is a global
  pub/sub that would cross-contaminate concurrent jobs. Your smoke waits behind the
  user's job.
- **1 automatic retry** on transient failure with a randomized seed
  (`generate.ts:1294-1339`) — a smoke that "succeeded" may have succeeded on
  attempt 2 with a different seed; check the gen log for `[Retry]` lines before
  claiming seed-reproducibility.
- **Watchdogs** (`pollUntilDone`, `generate.ts:102-170`):
  - stall: 2 min (`STALE_TIMEOUT_MS = 120_000`) with no stage/progress change ⇒
    cancel + `Generation stalled — no progress for Ns`.
  - wall clock: default 45 min, clamped 5–120 via `generationTimeoutMinutes` param.
  - transient poll errors against :8085 are logged and retried, not fatal.
- **TRT first run:** stage becomes `Building TRT engine (first run only, ~5-10 min)...`
  (`generate.ts:750-762`) and is then mutated with elapsed seconds specifically to
  defeat the stall detector. Normal; do not cancel.

## 5. Endpoints summary (all in generate.ts)

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | auth required; returns `{ jobId, status }` |
| `GET /api/generate/status/:id` | `{ jobId, status, stage, progress, result, error, ace_job_id, ace_phase, ace_phase_progress }` (lines 1391-1406) |
| `GET /api/generate/queue` | queue depth / current job — check before smoking (line 1443) |
| `POST /api/generate/cancel/:id` | cancel one job (line 1409) |
| `POST /api/generate/cancel-all` | cancel everything of yours |
| `POST /api/generate/reset-queue` | force-drain — destructive to the user's queue, ask first (line 1469) |
| `GET /api/generate/stream/:id` | SSE progress stream |

Status values (`generate.ts:44`):
`pending | lm_running | synth_running | saving | succeeded | failed | cancelled`.

Success `result` shape (`generate.ts:55-65`): `{ audioUrls: ["/audio/<uuid>.wav"],
songIds, bpm?, duration?, keyScale?, timeSignature?, masteredAudioUrl?, timing?,
totalMs? }`.

`ace_phase` / `ace_phase_progress` surface the engine's fine-grained phase (e.g.
`adapter_precompute`, `step 12/50`) so you can tell "slow but alive" from "wedged".

## 6. Output files

**Live data dir = `server\data`** — `config.data.dir = path.resolve(__dirname, '..',
DATA_DIR='./data')` with `__dirname = server/src` under tsx
(`server/src/config.ts:191-199`). The repo-root `data\` directory exists but is
stale/legacy (last touched 2026-06-04 at verification time). Never verify outputs
against root `data\`.

Per successful track, written into `server\data\audio\` (`generate.ts:839-1034`):

| File | When |
|---|---|
| `<uuid>.wav` (or `.mp3`) | always — served at `http://localhost:3000/audio/<uuid>.wav` in dev |
| `<uuid>.lrc` | only if lyrics + engine alignment (skipped for instrumental) |
| `<uuid>.latent` | HSLAT post-DiT latent companion |
| `<uuid>_mastered.wav` | only if post-processing/mastering enabled |
| `<uuid>.lyrics.json` | only if Whisper transcription enabled |

DB: row inserted into `songs` in `server\data\hotstep.db`
(`generate.ts:1152-1161`); `generation_params` column holds the full request params
JSON (your reproduction recipe), `quality_scores` holds advisory evaluator telemetry.

## 7. Log anatomy

Session folders (`server/src/services/logger.ts:40-58`), name-sorted = time-sorted:

```
logs/YYYY-MM-DD_HH-MM-SS/
  ├── node_console.log                        # all Node stdout/stderr, mirrored
  ├── ace_engine.log                          # ace-server (C++ engine) stdout/stderr
  └── generations/gen_<jobId>_<taskType>.log  # per-generation, flushed ON COMPLETION
```

- The gen log starts with the full resolved engine params JSON
  (`logGenerationParams`, logger.ts:124-133), includes a
  `[Timing]` pipeline-breakdown table, and ends `GENERATION COMPLETED.`
  (logger.ts:145) or `GENERATION FAILED: <error>` (logger.ts:167).
- **Buffered:** the gen log lives in memory until finish/fail
  (`fs.writeFileSync` in `finishGenerationLog`/`failGenerationLog`). Mid-run,
  its lines are mirrored to the terminal SSE/console prefixed `[Gen:<jobId8>]`
  (logger.ts:118) — tail `node_console.log` or `ace_engine.log` instead.
- Failure triage order: matching `gen_*.log` → cross-ref `ace_engine.log` +
  `node_console.log`. Startup/crash issues: `node_console.log` + `ace_engine.log`.
- Clean engine start looks like (verified against a real session log,
  `logs/2026-07-01_12-21-25/ace_engine.log`): model registry scan lines
  (`[Server] Scanning models in ...`, `[Registry] ... -> DiT/LM/Text-Enc`), then
  `[Server] Listening on 127.0.0.1:8085`, with no exception in between.

## 8. Crash respawn mechanics (why dev-rebuild.bat exists)

`server/src/index.ts:284-308`: on ace-server exiting non-cleanly (not
SIGTERM/SIGINT/code 0), Node logs
`[ace-server] Restarting in 3 seconds... (crash N/MAX)` and respawns after 3 s.
A crash limiter counts crashes in a window; when exceeded it gives up and sets
bootStatus to `Engine crashed N times — check logs for missing DLLs`
(index.ts:296-302). This respawn is why you must never kill/rebuild ace-server
externally while the app runs: the respawned exe locks the file the linker needs.

`dev-rebuild.bat` (repo root, verified): POST `/api/shutdown`
(`server/src/routes/shutdown.ts` — kills ace-server by port, Vite, then itself),
poll `tasklist` for `ace-server.exe` up to 10 s, `taskkill /F` at 10 s, abort at
15 s, then `call engine\build.cmd`. Final message: *"Start the app with LAUNCH.bat
to pick up changes"* — it does **not** relaunch; relaunch with `dev.bat` (detached) and
check `http://localhost:3000`. `LAUNCH.bat` is the end-user prod path.

## 9. LM echo sideband gotcha (regression trap for param changes)

When the LM phase runs, ace-server echoes back a C++ `AceRequest` — but
**ServerFields-only "sideband" params** (fields the Node server adds that the C++
struct doesn't know: `adapter_runtime_quant`, alignment timing, rebase,
`plugin_params`, …) do not survive that round trip. The fix
(`generate.ts:312-328`) rebuilds each synth request from the **current** `aceReq`
spread plus only the LM-generated fields (`audio_codes`, `caption`, `lyrics`,
`bpm`, `duration`, `keyscale`, `timesignature`) from the echo.

Validation implication: any change adding a new server-side param must be smoke
tested **with the LM phase on** (omit `skipLm`) to prove the param reaches
`/synth` — and the mechanism is the spread reconstruction, never a whitelist.

## 10. Known feature states (don't misreport)

- `adapter_section_isolation` — wire plumbing still present
  (`aceClient.ts`, `translateParams.ts`) but the feature was **reverted**
  (commit `ee041e1`; it broke musical continuity). Inactive by design.
- `rebase_source` / `rebase_beta` (basin re-base, `translateParams.ts`) — exists,
  **unvalidated** for the cross-base LoKR problem. Never report it as working.
- Draft-LM speculative decoding — **deliberately disabled** by comment in
  `server/src/config.ts` (~line 117). Not a bug to fix.

## 11. Not verified in this document

- No live smoke generation was run while authoring this reference (app not
  confirmed running at the time). The request/poll flow is derived from code
  (`generate.ts`, `translateParams.ts`, `auth.ts`); the log and output shapes were
  confirmed against real 2026-07-01 session logs and actual `server\data\audio`
  contents. If a live smoke contradicts this document, trust the live run and
  update this file.

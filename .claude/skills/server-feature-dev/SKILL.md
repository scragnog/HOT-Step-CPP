---
name: server-feature-dev
description: Guides adding or modifying a feature in the HOT-Step CPP Node/TypeScript server (Express route + service pattern, better-sqlite3 schema changes, engine calls via aceClient, logging, tsx watch dev loop). Use when creating/editing files under server/src/, adding an /api/* endpoint, changing the SQLite schema, adding a config/env setting, wiring a server feature to the C++ engine, or debugging server-side failures (stalled/stuck generation jobs, engine HTTP timeouts, SQLite errors, engine crash-respawn loops).
---

# Node Server Feature Development

The Node server (`server/src/`) is the middle tier of HOT-Step CPP, a local AI music-generation app. It serves the React UI, owns the SQLite database, and orchestrates the C++ inference engine (`ace-server.exe`, spawned as a child process). This skill covers how to add or change a server feature the way this codebase does it.

**Glossary (no prior context assumed):**
- **Engine / ace-server** — the C++ binary doing actual AI inference, HTTP API on port **8085**. The Node server spawns it and talks to it over HTTP.
- **DiT** — Diffusion Transformer, the engine's main music-synthesis model. **LM** — the language model that generates "audio codes" before synthesis. **VAE** — decodes latents to audio. Pipeline: LM → DiT → VAE.
- **AceRequest** — the canonical JSON request shape the engine accepts, defined in `server/src/services/aceClient.ts:38-151`.
- **aceReq** — the AceRequest instance built for a generation job (from UI params via `translateParams`).
- **Adapter** — a LoRA/LoKr fine-tune applied to the DiT at generation time.

```
UI (fetch /api/*) → Express :3001 → routes/*.ts → services/*.ts
                                        ├→ getDb()  (better-sqlite3, synchronous)
                                        └→ aceClient (HTTP → ace-server :8085)
```

## When to use this skill

- Adding a new `/api/*` endpoint or a whole feature (route + service).
- Modifying SQLite schema or queries (`server/src/db/database.ts`).
- Adding a config value or a Settings-UI-exposed `.env` key (`server/src/config.ts`).
- Making the server call the engine (new AceRequest field, new engine endpoint).
- Debugging server-side failures (jobs stalling, engine timeouts, DB errors).

## Golden rules (hard constraints — each prevents expensive damage)

1. **Type-check with `npx tsc --noEmit` during dev; only `npm run build` before user testing.** (Institutional rule, verbatim.) `tsx watch` runs TypeScript directly, so a build is wasted work mid-dev — but tsx also tolerates errors that `tsc` rejects, so type-check before committing.
   ```powershell
   cd d:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit
   ```
2. **Node 18–22 only.** Node 24+ breaks dependencies; `server/package.json:5-7` enforces `>=18.0.0 <24.0.0`. better-sqlite3 is a native module — `postinstall` runs `npm rebuild better-sqlite3` (package.json:14).
3. **ESM everywhere — relative imports MUST end in `.js` even though sources are `.ts`.** `server/package.json:8` sets `"type": "module"`. `import { config } from '../config.js'` — omitting `.js` may run under tsx but fails `tsc`. There is no `__dirname`; the pattern is `const __dirname = path.dirname(fileURLToPath(import.meta.url))` (config.ts:7, index.ts:46).
4. **Never block a request on long work.** The engine is single-threaded; generations take minutes. Return a job id immediately and let the UI poll (see the worked example). The generation queue is deliberately serialized to one job at a time (generate.ts:1286-1292).
5. **Never call `getDb()` at module import time.** `initDb()` runs at index.ts:60; a query executed during module load throws `Database not initialized. Call initDb() first.` (database.ts:17-22). Query only inside handlers/functions.
6. **Declare static routes before `/:id` params.** Express matches in order — `GET /recent` after `GET /:id` returns a 404 for id "recent". See the explicit warning at songs.ts:47 and seeds.ts (`/favorites`, `/random` at lines 131/143 precede `/:name` at 156).
7. **Git: all work on `master`, stage explicit paths (never `git add -A`, never `git add -f` on gitignored paths — `data/`, `logs/`, `docs/plans/`, `checkpoints/` are gitignored on purpose), commit locally often, push only with explicit user approval.** Pushing any `v*` tag triggers a full multi-platform CI release build.
8. **PowerShell syntax: `;` not `&&`** for command chaining on this machine.
9. **Don't visually verify the UI with a browser agent** — ask the human user; they provide screenshots. (Hitting API endpoints programmatically is fine.)
10. **If your change touches C++ (`engine/src/`), rebuild via `dev-rebuild.bat` at repo root — never `engine/build.cmd` directly** (you cannot reliably tell whether the app is running; the Node server auto-respawns ace-server on crash, index.ts:284-309 → infinite respawn + file-lock loop). **Never `cmake --clean-first`** (20+ min CUDA recompile); for stale `.obj` issues delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
11. **Never delete generated audio or test outputs, even ones you predict are bad** — the human verifies results by ear. Leave all artifacts in place unless the user explicitly asks for cleanup.

## Dev loop

1. Start dev mode: `d:\Ace-Step-Latest\hot-step-cpp\dev.bat` — runs Vite (UI, :3000, HMR) plus `server\restart-loop.cmd`, which loops `npx tsx watch src/index.ts` (restart-loop.cmd:6). Prod mode is `LAUNCH.bat` (Node :3001 serving prebuilt `ui/dist/`).
2. Save any `server/src/**/*.ts` file → tsx watch restarts the **whole Node process**, which kills and respawns ace-server too (it's a child process). Expect a few seconds of engine downtime after every save; a generation in flight will die.
3. The restart loop re-enters if a `.restart-requested` marker file exists at repo root — this is how the in-app "restart server" works without killing Vite.
4. Type-check before committing: `cd d:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit`.
5. Logs land in `logs/<YYYY-MM-DD_HH-MM-SS>/` at repo root: `node_console.log` (server), `ace_engine.log` (engine), `generations/gen_<jobId>_<task>.log` (per generation). Newest folder = current session.

## Procedure: add a new feature (route + service)

1. **Create the route file** `server/src/routes/myFeature.ts`. Convention: open with a comment block stating the mount point and a route table (best example: seeds.ts:1-20).
   ```ts
   // myFeature.ts — one-line purpose
   // Mounts at: /api/my-feature
   // Routes:
   //   GET  /api/my-feature        — list things
   import { Router } from 'express';
   import { config } from '../config.js';   // note the .js extension

   const router = Router();

   router.get('/', (req, res) => {
     try {
       res.json({ ok: true });
     } catch (err: any) {
       console.error('[MyFeature] list failed:', err.message);
       res.status(500).json({ error: err.message });
     }
   });

   export default router;
   ```
2. **Register it in `server/src/index.ts`** — two edits: add the import to the block at index.ts:21-44 (`import myFeatureRoutes from './routes/myFeature.js';`) and mount it in the block at index.ts:72-95 (`app.use('/api/my-feature', myFeatureRoutes);`). URL segments are kebab-case (`/api/model-manager`, `/api/cover-art`, `/api/stem-studio`). One legacy exception: `songBuilder.ts` mounts at `/api/builder`.
3. **Put non-trivial logic in a service**: `server/src/services/myFeature.ts`. When a feature grows, use a sub-folder — `services/generation/` holds `translateParams.ts`, `lmCache.ts`, `postProcessing.ts`, `sourceAudio.ts`, etc., all extracted from the once-monolithic `generate.ts`.
4. **Auth-scope if the data is per-user**: `import { getUserId } from './auth.js';` then
   ```ts
   const userId = getUserId(req);
   if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
   ```
   (exact pattern: songs.ts:19-20). Auth is deliberately lightweight: an in-memory `Map<token, userId>` that resets on restart, with a single auto-created local user "Producer" (auth.ts:13-26). `getUserId` is auth.ts:98-102.
5. **Express 5 handler style**: end early with `res.status(...).json(...); return;` — do not `return res...` from typed handlers. The SPA fallback uses Express-5 wildcard syntax `app.get('/{*splat}', ...)` (index.ts:136) — Express 4 `'*'` patterns do not work.
6. **Log with a bracketed feature tag** — `console.log('[MyFeature] ...')`. Existing tags: `[Server]`, `[DB]`, `[Config]`, `[Generate]`, `[Settings]`, `[Logger]`, `[ace-server]`. This convention is what makes `node_console.log` greppable. To also surface lines in the UI's live log panel, `import { pushLog } from './logs.js'; pushLog('text', 'server');`.
7. **Type-check** (`npx tsc --noEmit`), test via the running dev server, then commit explicit paths:
   ```powershell
   cd d:\Ace-Step-Latest\hot-step-cpp; git add server/src/routes/myFeature.ts server/src/services/myFeature.ts server/src/index.ts; git commit -m "feat(server): my feature"
   ```

**Not every feature needs SQLite.** `routes/seeds.ts` stores JSON files under `server/data/seeds/` on purpose (ComfyUI-compatible, import/export-friendly) with a filename-sanitization regex at seeds.ts:44. Use seeds.ts as the template for small self-contained features; use generate.ts for engine-orchestrating ones.

## Procedure: SQLite schema change

All DB work lives in `server/src/db/database.ts`. One connection, **synchronous** better-sqlite3 API, module singleton (`initDb()` once at startup, `getDb()` everywhere else, `closeDb()` on shutdown at index.ts:564). Pragmas at open: WAL, `synchronous = NORMAL`, `foreign_keys = ON` (database.ts:32-34).

1. **New table**: add a `CREATE TABLE IF NOT EXISTS` block inside the big `db.exec(...)` at database.ts:37-209. Schema is re-run idempotently at every startup — there is **no migration framework and no version table**.
2. **New column on an existing table**: add an entry to the checked-migration list (the `songsMigrations` array pattern, database.ts:213-259):
   ```ts
   {
     check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='my_col'`,
     alter: `ALTER TABLE songs ADD COLUMN my_col TEXT DEFAULT ''`,
   },
   ```
   The loop runs `ALTER` only when the column is absent and logs `[DB] Migration: ...`. **Use this idiom for new columns.** (A second, older idiom — `try { db.exec("ALTER TABLE ...") } catch {}` — exists for the lireek tables at database.ts:262-278; don't add to it.)
3. **Column conventions**: TEXT primary keys are UUIDs (`uuidv4()`) for core tables; lireek (Lyric Studio) tables use `INTEGER PRIMARY KEY AUTOINCREMENT`; timestamps `TEXT DEFAULT (datetime('now'))`; booleans are INTEGER 0/1, re-hydrated as `!!row.is_public` (songs.ts:38); complex values are JSON-in-TEXT parsed with `JSON.parse(s.tags || '[]')`.
4. **Query style**: inline `getDb().prepare(sql).get/all/run(...params)` with `?` placeholders — no statement-caching layer. Dynamic filters append to the SQL string and a params array (songs.ts:24-32), including JSON queries like `json_extract(generation_params, '$.source') = ?`.
5. **Don't imitate the two-database pattern.** `db/lireekDb.ts` is a query-helper module over the *same* connection; its `initLireekDb`/`closeLireekDb` are deprecated no-ops (lireekDb.ts:12-23). The old separate `lireek.db` was consolidated into `server/data/hotstep.db` (one-time `ATTACH DATABASE` migration at database.ts:293-386, a good reference if you ever need bulk data migration).

## Procedure: add a config / .env setting

`server/src/config.ts` builds one exported `config` object (config.ts:96-257) grouping `aceServer` (engine exe autodetected from 4 candidate paths at config.ts:46-52; port default 8085), `server` (port 3001), `data` (getter `dbPath` → `server/data/hotstep.db`; `DATA_DIR` in `.env` resolves relative to `server/`, NOT the repo root — config.ts:192. A stale legacy `data/` dir may exist at repo root with an outdated hotstep.db; ignore it), `lireek` (LLM keys/endpoints), `vst`, `whisper`, `essentia`. `.env` lives at repo root and is auto-bootstrapped from `.env.example` on first launch (config.ts:22-30).

To expose a new env setting in the Settings UI, you must touch **three places** in config.ts or it silently won't work:

1. Add the key to the `EXPOSED_ENV_KEYS` whitelist (config.ts:265-286) — "nothing else leaks" to the UI.
2. If it only takes effect at process/engine spawn time, also add it to `RESTART_REQUIRED_KEYS` (config.ts:289-294) so the UI shows "restart required".
3. Add an `apply('MY_KEY', setter, getter)` line inside `reloadEnvConfig()` (config.ts:300-395) — **without this the value saves to `.env` but never hot-patches the live config**.

The Settings route (`POST /api/settings/env`, settings.ts:109-174 — note: the header comment says PUT, the code is POST; code wins) rewrites `.env` preserving comments and line endings, calls `reloadEnvConfig()`, and returns `restartRequired`.

## Procedure: call the engine (aceClient)

`server/src/services/aceClient.ts` is the typed wrapper over ace-server's HTTP API (`BASE = config.aceServer.url`, i.e. `http://127.0.0.1:8085` by default).

**Critical constraint** (aceClient.ts:6-8, verbatim comment): *"ace-server uses single-threaded httplib. During heavy compute (DiT generation, adapter merge, VAE decode) it cannot respond to HTTP requests."* Hence three timeout tiers via `AbortSignal.timeout` (aceClient.ts:17-19) — pick one deliberately for any new call:

| Tier | Value | For |
|---|---|---|
| `TIMEOUT_QUICK` | 15 s | health, props, job submit |
| `TIMEOUT_POLL` | 30 s | job polling — fail fast, let the watchdog decide |
| `TIMEOUT_RESULT` | 300 s | fetching large audio bodies |

- **Async-job pattern**: `submitLm` / `submitSynth` / `submitUnderstand` / `warm` POST and return a job id string; poll with `pollJob(id)` → `{ status: 'running'|'done'|'failed'|'cancelled', phase?, phase_step?, phase_total? }` (phases like `adapter_precompute`, `dit_inference` — aceClient.ts:165-180); fetch the result with `getJobResult(id)` (`GET /job?id=N&result=1`); cancel with `cancelJob(id)`. Sync endpoints (`/spectral-lifter`, `/pp-vae-reencode`) return processed WAV bytes directly.
- **A poll timeout is NOT a failure.** The engine may simply be busy mid-DiT-step. Catch transient poll errors and retry; only treat explicit `failed`/`cancelled` statuses as terminal (see generate.ts:158-166).
- **New engine parameter?** Add the field to the `AceRequest` interface first (aceClient.ts:38-151), then plumb it through `services/generation/translateParams.ts` (UI params → AceRequest). Known gotcha (institutional, from the LM-echo bug): server-only fields do **not** survive the engine's `/lm` round trip — synth requests are rebuilt from the original aceReq plus only the LM-generated fields (`audio_codes`, caption, lyrics, bpm, duration, keyscale, timesignature — generate.ts:237-246). Never assume a field you sent to `/lm` comes back.
- **File upload to the engine**: multipart is hand-rolled with a manual boundary because the engine's parser expects a `request` JSON part plus named binary parts (`audio`, `ref_audio`, `src_latents`, `ref_latents`, `seed_latents`). The multipart path takes a **single JSON object, not an array** (aceClient.ts:374-378). Copy `submitSynthMultipart` (aceClient.ts:347-424).
- **Error convention**: non-OK responses `throw new Error('ace-server POST /x failed (status): body')` (aceClient.ts:266-270); soft-fail helpers return `null`/`false` (`getJobLatent`, `isReachable`).
- **Readiness gate**: routes that hit the engine should return 503 while the engine is booting. `import { engineReady, engineBootStatus } from '../engineState.js';` — this module exists solely to break a circular import between index.ts and routes (engineState.ts:1-5). Pattern: generate.ts:1356-1363.

## Worked example (dissected): the generation feature

`routes/generate.ts` (~1600 lines) is the canonical "server orchestrates engine" feature. Copy its skeleton for anything long-running.

1. **In-memory job map** — `interface GenerationJob` (id = uuid, userId, status `pending|lm_running|synth_running|saving|succeeded|failed|cancelled`, `stage`/`progress` for the UI, `aceJobId`, `acePhase`, result, error, params) in `const jobs = new Map()` (generate.ts:41-79). A TTL sweeper prunes terminal jobs older than 1 h every 10 min, with `.unref()` so the interval never blocks process exit (generate.ts:83-94).
2. **POST `/api/generate`** (generate.ts:1355-1388): 503 if `!engineReady`; 401 if no user; build the job object from `req.body`; `jobs.set(...)`; `enqueueGeneration(job)`; respond `{ jobId, status }` **immediately**.
3. **Serialized queue + retry** (generate.ts:1286-1352): a plain closure array + `generationRunning` boolean — one generation at a time (the in-code comment explains why: `subscribeLines()` is a global log pub/sub with no job tagging, so concurrent jobs would cross-contaminate progress). One retry on transient failure with a fresh random seed; `Cancelled`/`Unauthorized` are non-retryable.
4. **Pipeline** (`runGeneration`, generate.ts:173 onward): `translateParams(job.params)` → aceReq; **write the resolved seed back into `job.params`** so the DB stores the actual seed used (reproducibility, generate.ts:193-197); decide whether to skip the LM phase (explicit skipLm, `audio_codes` already present, or a cover-type task — generate.ts:207-210); `startGenerationLog(job.id, taskType)` + `logGenerationParams(job.id, aceReq)`; check the LM cache (`services/generation/lmCache.ts`, keyed by seed+params); submit via `aceClient.submitSynth` or, when any source/reference audio or latents are present, `submitSynthMultipart` (generate.ts:814-827); stage timings via a small `timed()` helper.
5. **Watchdog polling** (`pollUntilDone`, generate.ts:102-170): 500 ms interval; honors an `AbortController` for cancellation; **stall detection** — no stage/progress change for 120 s → cancel the engine job and throw (generate.ts:126-134); wall-clock timeout clamped to 5–120 min, default 45 (generate.ts:104-106); transient poll errors are logged and retried.
6. **Persist**: on success, `INSERT INTO songs (...)` with a fresh uuid, `generation_params` = `JSON.stringify(job.params)` (generate.ts:1150-1161); audio files go under `config.data.audioDir` and are served statically at `/audio/*` (index.ts:98).
7. **Companion endpoints**: `GET /status/:id` returns `{ jobId, status, stage, progress, result, error, ace_job_id, ace_phase, ace_phase_progress }` (generate.ts:1391-1406); `POST /cancel/:id` sets status, cancels the engine job, fires the AbortController; plus `/cancel-all`, `/queue`, `/reset-queue`, and `GET /stream/:id` (SSE previews).

**Per-generation logging** (services/logger.ts): `startGenerationLog(jobId, taskType)` → `logGeneration(jobId, 'INFO'|'DEBUG'|'WARNING'|'ERROR', msg)` / `logGenerationParams(jobId, obj)` → `finishGenerationLog(jobId, taskType)` or `failGenerationLog(jobId, error, taskType)`. Lines are **buffered in memory and flushed to `logs/<session>/generations/gen_<jobId>_<task>.log` only on completion** (logger.ts:95-178) — a crash before finish loses the buffer, but every line is also streamed live via `pushLog`.

## Key files

| Path | Role |
|---|---|
| `server/src/index.ts` | App entry: route registration (imports :21-44, mounts :72-95), engine spawn/respawn (:158-316), graceful shutdown (:538-572) |
| `server/src/config.ts` | All configuration; `.env` hot-reload (`EXPOSED_ENV_KEYS` / `RESTART_REQUIRED_KEYS` / `reloadEnvConfig`) |
| `server/src/db/database.ts` | SQLite singleton, schema (`CREATE TABLE IF NOT EXISTS`), column migrations |
| `server/src/db/lireekDb.ts` | Lyric-Studio query helpers over the same connection (init/close are deprecated no-ops) |
| `server/src/services/aceClient.ts` | Typed HTTP client for the engine; `AceRequest`; timeout tiers; multipart |
| `server/src/engineState.ts` | `engineReady` / `engineBootStatus` flags (circular-import breaker) |
| `server/src/routes/generate.ts` | Worked example: job map, queue, watchdog, persistence |
| `server/src/services/generation/translateParams.ts` | UI params → AceRequest translation |
| `server/src/routes/seeds.ts` | Template for a small self-contained feature (filesystem storage, route-table header) |
| `server/src/routes/songs.ts` | Template for auth-scoped SQLite CRUD (incl. the static-before-`/:id` warning) |
| `server/src/routes/auth.ts` | Token-lite auth; `getUserId(req)` helper |
| `server/src/services/logger.ts` | Session log dirs, `logEngine`, per-generation buffered logs |
| `server/src/routes/logs.ts` | 2000-line ring buffer + SSE at `GET /api/logs`; `pushLog` / `subscribeLines` |
| `server/src/routes/settings.ts` | `.env` read/rewrite + hot-reload endpoint (`POST /api/settings/env`) |
| `server/restart-loop.cmd` | Dev-mode tsx watch wrapper with `.restart-requested` re-entry |
| `server/package.json` | Scripts (`dev`/`typecheck`), `"type": "module"`, Node `<24` engines pin |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| `Database not initialized. Call initDb() first.` | A module ran a DB query at import time, before index.ts:60 | Move queries inside handlers/functions |
| Engine requests time out during generation | Normal — single-threaded httplib engine can't respond mid-compute | Use the right timeout tier; retry transient poll errors; never treat a poll timeout as job failure |
| `Generation stalled — no progress for Ns` | The 120 s watchdog fired and cancelled a wedged engine job | Check `gen_*.log` + `ace_engine.log` in the newest `logs/` session; the queue is unblocked automatically |
| `[ace-server] Crashed 3 times within 30s — giving up` | Missing engine DLL / broken build; crash limiter halts respawn and sets engine not-ready (index.ts:296-301) | Fix the engine build (via `dev-rebuild.bat`); check `ace_engine.log` |
| New env setting saves but never takes effect | Key missing from the `apply(...)` list in `reloadEnvConfig()` or from `EXPOSED_ENV_KEYS` | Touch all three places in config.ts (see procedure above) |
| Setting saved, hot-reload reports OK, engine ignores it | Key is spawn-time (in `RESTART_REQUIRED_KEYS`) | Restart the app so the engine child respawns with new args |
| `GET /api/things/recent` → 404 with id "recent" | Static route declared after `/:id` | Reorder — static routes first (songs.ts:47) |
| better-sqlite3 native/ABI error after `npm install` or Node switch | Node 24+ (unsupported) or skipped native rebuild | Use Node 18–22; `npm rebuild better-sqlite3` |
| `tsc` errors on imports that run fine under tsx | Missing `.js` extension on a relative ESM import | Add `.js` to the import path |
| Spawned engine can't find DLLs despite a PATH edit | Spreading `process.env` on Windows creates a case-sensitive duplicate (`Path` vs `PATH`) that shadows | Find the real key case-insensitively before prepending — pattern at index.ts:221-244 |
| Multipart submit to `/synth` fails with a parse error | Sent a JSON **array** as the multipart `request` part | Engine multipart expects a single object (aceClient.ts:374-378) |

## Institutional knowledge

- **VALIDATED (rule, verbatim):** Type-check with `npx tsc --noEmit` during dev; only `npm run build` before user testing.
- **VALIDATED (code comment, aceClient.ts:6-8):** the engine is single-threaded httplib and cannot answer HTTP during heavy compute — every timeout, watchdog, and retry in the server exists because of this.
- **VALIDATED (bug fixed, per project memory + generate.ts:234-246):** LM echo sideband gotcha — server-only AceRequest fields don't survive the engine's `/lm` round trip. Rebuild synth requests from the original aceReq plus LM-generated fields only; never whitelist-copy from the LM response.
- **VALIDATED (in-code warning, index.ts:221-244):** Windows `process.env` is a case-insensitive proxy but spreading it makes plain case-sensitive keys — mutating `PATH` blindly creates a shadowing duplicate of `Path`.
- **VALIDATED:** the generation queue is intentionally single-file because `subscribeLines()` is global pub/sub with no job tagging (generate.ts:1286-1290); don't "fix" concurrency without solving log attribution first.
- **VALIDATED (config.ts:117-122):** draft-LM speculative decoding is DISABLED — GGML per-call overhead negates the speedup. `ACESTEPCPP_DRAFT_LM` env re-enables it; auto-detect is commented out (config.ts:160-173).
- **Field present, feature reverted:** `adapter_section_isolation` still exists in AceRequest (aceClient.ts:97) but the engine-side regional self-attn isolation was reverted for breaking musical continuity — treat the field as inert (unverified on the engine side — check engine code before relying on it).
- **VALIDATED:** `uncaughtException`/`unhandledRejection` are logged but do **not** exit the process (index.ts:576-581), and there is no global Express error middleware — a throw in an async handler will not produce a clean JSON 500, so catch locally in every handler.

More depth (full aceClient method catalog, engine spawn args, settings hot-reload internals, DB idiom code, SSE log stream details): [reference.md](reference.md).

## Deeper reading

- `CLAUDE.md` (repo root) — orientation map, build/git rules.
- `engine/docs/ARCHITECTURE.md` — engine internals, CLI, request JSON, generation modes (committed).
- `FEATURES.md` — full feature catalogue (committed).
- `docs/dev/plugins-authoring.md` — Lua plugin authoring; solvers/schedulers/guidance are plugins, not C++ (committed).
- `docs/plans/` — internal design/investigation docs. **Gitignored, local-only — may be absent on a fresh clone.**
- `server/src/data/assistant-knowledge.md` — in-app assistant knowledge base (committed).

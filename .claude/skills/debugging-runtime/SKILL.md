---
name: debugging-runtime
description: Diagnoses HOT-Step CPP generation failures, engine crashes, hangs, and startup problems from the logs/ session folders. Use when a music generation failed, ace-server crashed or keeps respawning, the app hangs with no progress, an API call returns 500/503, or you need to trace a gen_<uuid> log back to engine output.
---

# Debugging generation failures & crashes (log-driven playbook)

HOT-Step CPP has two runtime processes: the **Node server** (Express, port 3001) and its child **ace-server.exe** (the C++ inference engine, port 8085). The Node server orchestrates every generation: it optionally calls the engine's LM (language model that expands your caption/lyrics into audio codes), then submits synthesis (DiT — the Diffusion Transformer that generates audio latents — followed by VAE decode to a WAV), polls until done, then saves the result to SQLite. Almost every runtime problem is diagnosable from the per-session log folders under `logs/` at the repo root. This skill tells you which file to open first for each symptom, what the real failure strings mean, and how to correlate the three log files.

All `path:line` references verified against the code on 2026-07-02.

## When to use this skill

- A generation failed in the UI, or a job silently disappeared.
- ace-server crashed, keeps restarting, or "gave up" after repeated crashes.
- The app hangs — progress bar stuck, queue wedged.
- `POST /api/generate` returns 503 "Engine not ready", or any API route 500s.
- The app fails to start, or starts without the engine.
- You need to reproduce a failed generation exactly (seed, adapters, solver).

## Golden rules

1. **NEVER kill ace-server.exe externally (Task Manager / `taskkill` / `Stop-Process`) while the Node server is running.** The Node server auto-respawns the engine on non-zero exit ([server/src/index.ts:284-309](../../../server/src/index.ts)); if crashes are spaced more than 30 s apart, the crash limiter's window resets and the respawn loop continues indefinitely, holding file locks on the exe. To rebuild the engine, use `dev-rebuild.bat` at the repo root — it shuts the whole app down cleanly first, then builds. To just stop the engine, use `Invoke-RestMethod -Method Post http://localhost:3001/api/shutdown` (kills engine + server + Vite).
2. **Never rebuild the C++ engine via `engine/build.cmd` directly, under any circumstances** — you cannot reliably tell whether the app is running; same respawn/file-lock reason. `dev-rebuild.bat` wraps it safely (and is a harmless no-op shutdown when nothing runs).
3. **Never `cmake --build . --clean-first`** — CUDA kernel recompilation takes 20+ minutes. For stale `.obj` problems, delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
4. **Do not assume the engine is dead because HTTP to :8085 hangs.** ace-server uses single-threaded httplib; during DiT steps, adapter merges, or VAE decode it *cannot* answer any HTTP request ([server/src/services/aceClient.ts:6-19](../../../server/src/services/aceClient.ts)). Check the tail of `ace_engine.log` for advancing `[DiT] Step N/M` lines before declaring it hung.
5. **Do not delete or "clean up" log folders or generated test audio** — the user verifies experiments by ear and by log history. Logs are the only forensic record (gen logs especially: they contain the exact request JSON needed for repro).
6. **Do not visually verify the UI with a browser agent** — ask the user to check. Hitting API endpoints programmatically is fine.
7. **Windows PowerShell syntax**: separate commands with `;`, never `&&`.

## The log session layout

One folder per Node-server session, named so name-sort = time-sort. Created at server boot by `initLogger()` ([server/src/services/logger.ts:40](../../../server/src/services/logger.ts)):

```
logs/YYYY-MM-DD_HH-MM-SS/
  node_console.log            all Node stdout+stderr, mirrored transparently
  ace_engine.log              raw ace-server child stdout+stderr
  generations/
    gen_<jobId>_<taskType>.log   one per generation; jobId = server-side UUID
```

Facts you must know before reading them:

- **`node_console.log` contains every engine line EXCEPT the filtered GGML noise patterns** (which appear only in `ace_engine.log` — index.ts:262-272 gates console output on `isNoise()` but writes `ace_engine.log` unconditionally). Engine lines appear prefixed `[ace-server] `, interleaved with server lines in real arrival order — it is the **only** file where engine and server output are time-ordered relative to each other.
- **`ace_engine.log` and `node_console.log` have NO per-line timestamps.** Only `gen_*.log` lines carry ISO timestamps (`2026-07-01T10:28:33.514Z | INFO | ...`, logger.ts:113-119).
- **`gen_*.log` is buffered in RAM and only written to disk when the generation completes or fails** (`finishGenerationLog` / `failGenerationLog`, logger.ts:139-178 — both do a single `fs.writeFileSync`). Failed and cancelled generations DO get their log written. **If Node itself crashes or is hard-killed mid-generation, the gen log is never written.** A missing `gen_*.log` for a generation you know started = Node died mid-flight; fall back to `node_console.log`.
- `<taskType>` in the filename comes from the engine request's `task_type`: `text2music` (default), `cover`, `cover-nofsq`, `repaint`, `lego`, `extract` (generate.ts:208-213). The retry-exhausted final-failure path writes taskType `'unknown'` (generate.ts:1335), but the earlier per-attempt failure (generate.ts:1275/1281) already flushed and deleted the buffer, so `gen_<id>_unknown.log` is usually a silent no-op.
- Repetitive GGML noise (`CUDA graph warmup`, `CUDA Graph id`, `ggml_backend_cuda_graph_compute`) is dropped **at the engine source since 2026-07-17**: `acestep_ggml_log` (engine/src/backend.h) discards all GGML DEBUG-level messages (set `HOTSTEP_GGML_DEBUG=1` to pass them through) and digit-insensitively dedups consecutive near-identical lines, so these no longer reach ANY log file. The Node-side filters (index.ts `isNoise()`, [server/src/routes/logs.ts:27-31](../../../server/src/routes/logs.ts)) remain as belt-and-braces for older engine binaries. If you need CUDA-graph-layer logging, use the env var.
- Live tail without touching files: `GET http://localhost:3001/api/logs` is an SSE stream backed by a 2000-line ring buffer, each line tagged `source: 'engine' | 'server'` (logs.ts:21-52).

## Which log to open first, per symptom

| Symptom | Open first | Then | Looking for |
|---|---|---|---|
| Generation failed (UI error) | Newest `logs/<session>/generations/gen_<uuid>_*.log` — last line is `GENERATION FAILED: <reason>` | `node_console.log` around that job; `ace_engine.log` for C++ detail | The failure reason string (table below) |
| Engine crash | `node_console.log` | `ace_engine.log` tail | `[ace-server] Process exited with code N` + the FATAL/assert lines just before it |
| Server 500 / API error | `node_console.log` | — | Express stack traces; `[Server] Uncaught exception` / `Unhandled rejection` (logged and swallowed — process keeps running, index.ts:576-581) |
| Startup failure | `node --version` FIRST, then `node_console.log` | `ace_engine.log` | **Node 18–22 LTS only — Node 24+ breaks dependencies** (`engines` field in server/package.json enforces `<24`; switch Node versions, do NOT rebuild node_modules to work around native-module errors). Then: `[Server] ace-server not found at:`, CUDA-runtime download banner, `Crashed 3 times within 30s — giving up`, DB errors |
| Hang / no progress | `GET /api/generate/queue` (live), then `node_console.log` tail | `ace_engine.log` tail | Last engine line = the wedged phase. The 120 s stall watchdog usually converts hangs into a `Generation stalled` failure on its own |
| No gen log exists at all | `node_console.log` | — | Node died mid-generation (gen logs only flush at the end) |

No session folder at all → the server never reached `initLogger()`; run `npx tsx src/index.ts` from `server/` and read the terminal directly.

## Triage procedure

1. Find the newest session and tail the logs (repo root, PowerShell):
   ```powershell
   $s = (Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Get-Content "$s\node_console.log" -Tail 80
   Get-Content "$s\ace_engine.log" -Tail 60
   Get-ChildItem "$s\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
   ```
2. If a generation failed, read its gen log bottom-up: the last line is `GENERATION FAILED: <reason>`; the top of the file embeds the **full resolved engine request JSON** (seed, adapters, solver, scheduler — logger.ts:124-133). That JSON is ground truth for reproduction.
3. Match the reason against the failure-signature tables below.
4. If the reason is generic (`Generation failed on ace-server`), the real cause is engine-side only — grep the session:
   ```powershell
   Select-String -Path "$s\node_console.log" -Pattern "exited with code|Crashed|FATAL|CUDA error"
   ```
5. For live triage while the app runs:
   ```powershell
   Invoke-RestMethod http://localhost:3001/api/health | ConvertTo-Json -Depth 4
   Invoke-RestMethod http://localhost:3001/api/generate/queue
   ```
   `/api/health` reports `aceServer.status` (`ok`/`disconnected`) and `engine.{ready, bootStatus}` ([server/src/routes/health.ts:11-46](../../../server/src/routes/health.ts)). Remember rule 4: a `disconnected` aceServer during heavy compute can be a busy single-threaded engine, not a dead one.
6. To unwedge a stuck queue without restarting:
   ```powershell
   Invoke-RestMethod -Method Post http://localhost:3001/api/generate/reset-queue
   ```
   This cancels all non-terminal jobs (they fail with `Queue reset by user`) and drains the pending queue (generate.ts:1469-1500). Also available: `POST /api/generate/cancel/:id`, `POST /api/generate/cancel-all`.
7. To search all history for a pattern:
   ```powershell
   Select-String -Path "logs\*\generations\*.log" -Pattern "GENERATION FAILED" | Select-Object -Last 10
   ```

## Failure signatures — gen-log `GENERATION FAILED: <X>`

| Signature | Cause / next step |
|---|---|
| `Cancelled by user` | Benign — user hit cancel (generate.ts:1275). Most common "failure" in history. |
| `Generation stalled — no progress for Ns (last stage: "...")` | The 120 s stall watchdog fired (generate.ts:128-134). The quoted stage names the wedged phase (e.g. `"Loading CondEnc..."`); check `ace_engine.log` tail for what the engine was doing. Also fires when the engine died silently and polls just time out. |
| `Generation timed out (N min limit)` | Wall-clock watchdog: default 45 min, user-clamped 5–120 via `generationTimeoutMinutes` (generate.ts:105-106, 139). Huge duration/steps, or a first-run TensorRT engine build without a cached engine. |
| `Generation failed on ace-server` | Engine set job status `failed` without crashing (generate.ts:156). Deliberately generic — **the real reason is only in `ace_engine.log`**; grep it for `FATAL`, `[Server]`, `failed`. |
| `...POST /synth... failed (413)` | Engine rejected the upload: `audio exceeds max duration (10 min)` or `src_latents exceeds max frames` ([engine/tools/hot-step-server.cpp:2044, 2072](../../../engine/tools/hot-step-server.cpp)); global payload cap is 256 MB (hot-step-server.cpp:2700). Source audio for a cover/repaint is too long. |
| `Engine not ready: <status>` (HTTP 503, never reaches a gen log) | Bootstrap incomplete (CUDA DLL download still running) or the crash limiter gave up (generate.ts:1356-1363). Check `node_console.log`. |
| `Queue reset by user` | Someone hit `/api/generate/reset-queue`. Benign. |

## Failure signatures — engine side (`ace_engine.log` / `[ace-server]` lines)

| Signature | Cause |
|---|---|
| `[VAE] FATAL: tensor 'decoder.conv1.weight_v' not found in safetensors` then process exit | Wrong or corrupt VAE model file (real occurrence 2026-06-04). |
| `[Server] FATAL: LM load failed` / `[Server] FATAL: synth load failed` | Model file missing/corrupt or OOM during load (hot-step-server.cpp:831, 1232). Job goes to `failed`; the engine process usually survives. |
| `[Safetensors] Cannot open <path>/adapter_model.safetensors` | Usually **benign filename probing** — the adapter loader tries candidate filenames. Thousands appear in healthy sessions. Only meaningful if the adapter then actually fails to load. |
| `CUDA error: ...` (any) | GGML/CUDA fault; typically followed by process exit and respawn. |
| `Process exited with code 3221225786` (0xC000013A) | Console Ctrl+C / window closed — usually deliberate. |
| `Process exited with code 3221226505` (0xC0000409) | Fail-fast / abort / stack-buffer-overrun — a genuine C++ crash. Get the engine lines immediately preceding it. |
| `Process exited with code 1, signal null` **right after** `[Shutdown] Killed ace-server PID ...` or `[Server] Shutting down...` | **Benign** — `taskkill /F` yields exit code 1, not a signal. Only treat an exit line as a crash if it is NOT preceded by a shutdown line. |
| `[ace-server] Restarting in 3 seconds... (crash N/3)` | Routine supervised respawn — dozens exist in healthy log history. Investigate the crash cause, not the respawn itself. |
| `[ace-server] Crashed 3 times within 30s — giving up.` | Crash limiter tripped (index.ts:296-301). Usually missing DLLs (cuBLAS runtime on portable CUDA builds) — reconnect to the internet and restart, or re-extract the release zip. `engineReady` becomes false; `/api/generate` returns 503. |
| All solvers/schedulers/guidance behave identically or ignore settings, logs completely clean — especially after an upstream acestep.cpp sync | The `hot-step-sampler.h` hook in `pipeline-synth-ops.cpp` was silently overwritten (compiles fine, all plugins dead — **invisible in logs**). Run `engine/verify-hooks.ps1`; see the upstream-sync skill. |

## How the Node server supervises ace-server

- **Spawn**: `startAceServer()` (index.ts:158-316) launches `config.aceServer.exe` with `--models`, `--host`, `--port` (default **8085**, [server/src/config.ts:102](../../../server/src/config.ts)) plus optional `--adapters`, `--keep-loaded`, `--noise-profile`, `--draft-lm`, `--vae-chunk`, `--vae-overlap`, `--onnx-dir`. The exe is auto-detected among `engine/ace-server.exe`, `engine/build/Release/ace-server.exe`, `engine/build/ace-server.exe`, `engine/build/Debug/ace-server.exe` (config.ts:46-52); override with `ACESTEPCPP_EXE` in `.env`.
- **Respawn**: child exit with non-zero code (and signal not SIGTERM/SIGINT) → respawn after 3 s. **Crash limiter**: 3 crashes within a rolling 30 s window → give up, set `engineReady=false` with bootStatus `Engine crashed 3 times — check logs for missing DLLs` (index.ts:152-156, 284-309). Crashes spaced >30 s apart reset the window, so slow-cycle crashes (e.g. crash-on-first-request) still loop forever — hence Golden rule 1.
- **Clean shutdown**: Ctrl-C → `shutdown()` uses `taskkill /PID <child> /T /F` (index.ts:549). `POST /api/shutdown` kills ace-server by port 8085 via netstat ([server/src/routes/shutdown.ts:24-44](../../../server/src/routes/shutdown.ts)), then Vite (:3000), then its own process tree. `POST /api/restart` writes a `.restart-requested` marker (shutdown.ts:161); `LAUNCH.bat` loops and relaunches when it sees the marker.
- **`dev-rebuild.bat`** = POST `/api/shutdown` → wait up to 10 s for ace-server.exe to die (force-kill at 10 s, abort at 15 s) → `engine\build.cmd`. It does **not** restart the app — run `LAUNCH.bat` (or `dev.bat`) afterwards.
- **Bootstrap gate**: on portable Windows CUDA builds, the server first downloads cuBLAS DLLs from HuggingFace before the engine is usable; until then `POST /api/generate` returns 503 `Engine not ready: <bootStatus>`. No internet → CPU-only start with a banner in `node_console.log`.

## Generation orchestration facts that change your diagnosis

- **Single-slot FIFO queue** — only one generation runs at a time (`enqueueGeneration`, generate.ts:1291-1352), because engine log parsing is a global untagged pub/sub.
- **The auto-retry loop is dead code for generation failures**: `enqueueGeneration` has a retry-once-with-fresh-seed loop (generate.ts:1301-1339), but `runGeneration`'s top-level catch (generate.ts:1271-1283) swallows every generation error without rethrowing, so the retry never fires (the only escapable throw, `translateParams` at :191, happens before any engine submission). **One UI-visible failure = exactly ONE engine attempt**, and the seed in the gen log's single params block is directly trustworthy for repro. Zero `[Retry]` lines exist across 118 logged sessions. If retry is ever *intended* behavior, the swallowed rethrow is the server bug to fix.
- **Stall watchdog**: `pollUntilDone` (generate.ts:102-170) polls the engine every 500 ms; stage/progress unchanged for 120 s → cancel + `Generation stalled`. First-run TensorRT builds are special-cased (`[TRT-WARN]` lines mutate the stage string, generate.ts:756) so a 5–10 min TRT compile doesn't trip it.
- **Transient poll errors are normal**: `[Generate] Poll error ... (will retry)` (generate.ts:165) just means the busy single-threaded engine missed a 30 s poll window.
- **Fine-grained engine phase** is exposed at `GET /api/generate/status/:id` as `ace_phase` — one of `queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond, loading_dit, loading_adapter, adapter_precompute, dit_inference, loading_vae, vae_decode, encoding_output, done, failed, cancelled` (aceClient.ts:165-180) — plus `ace_phase_progress` (`"step N/M"`). These fields are optional on the wire for older engine builds; absence is not an error.
- **Job TTL**: the in-memory job map prunes terminal jobs after 1 h (generate.ts:83-94). A 404 from `/status/:id` on an old job is TTL cleanup, not data loss — the song row and gen log persist.
- **Post-processing failures are non-fatal**: mastering / Spectral Lifter / PP-VAE / Whisper / LRC / cover-art failures log a `WARNING` and the job still succeeds with raw audio. A "succeeded" job can still have WARNINGs worth reading in its gen log.

## Correlating a gen_<uuid> log with engine + node logs

There are **two job-ID namespaces**: the server UUID (the `gen_<uuid>` filename; SSE lines show only its first 8 chars as `[Gen:xxxxxxxx]`, logger.ts:118) and the engine's own job id (`ace_job_id` in `/status/:id`; returned by the engine's `/lm` and `/synth`). The engine id is NOT in the gen filename.

1. Open `gen_<uuid>_*.log`; note the ISO timestamps of the LM/synth submission lines and the final `GENERATION FAILED` line.
2. `ace_engine.log` has no timestamps, so pivot through **`node_console.log`**: search for `[Generate] Job <uuid>` — the submission line logs ditModel/synth_model/seed/source (generate.ts:199) and the failure logs `[Generate] Job <uuid> failed:` (generate.ts:1280). The block of `[ace-server] ...` lines between your job's submit and fail lines IS that generation's engine output.
3. If you need the exact region of `ace_engine.log` (e.g. to see noise-filtered lines), grep it for a distinctive engine line found in step 2 (the last `[DiT] Step N/M`, a model-load line) and read forward from there.
4. Repro comes from the params JSON at the top of the gen log — it is the fully resolved request actually sent to the engine.

Dev-mode note: under `dev.bat` (tsx watch), every source-change auto-restart creates a **new** session folder. A burst of near-identical `logs/` folders seconds apart means tsx was restarting, not that the app was crashing.

## Key files

| Path | Role |
|---|---|
| `server/src/services/logger.ts` | Session folders, console/engine mirroring, gen-log buffering & flush |
| `server/src/index.ts` | Engine spawn, respawn + crash limiter (:152-316), shutdown (:540), CUDA DLL bootstrap |
| `server/src/routes/generate.ts` | Orchestration: `pollUntilDone` :102, `runGeneration` :173, retry/queue :1291, status/queue/reset endpoints :1390+ |
| `server/src/services/aceClient.ts` | Typed engine HTTP client; timeouts; `AceJobPhase` list |
| `server/src/routes/logs.ts` | `GET /api/logs` SSE, 2000-line ring buffer, noise filter |
| `server/src/routes/health.ts` | `GET /api/health` |
| `server/src/routes/shutdown.ts` | `POST /api/shutdown` (port-based kill), `POST /api/restart` marker |
| `server/src/engineState.ts` | `engineReady` / `engineBootStatus` gate |
| `server/src/config.ts` | Engine exe auto-detect, port 8085, env-var overrides |
| `engine/tools/hot-step-server.cpp` | C++ engine HTTP server; FATAL emit sites; 413/payload limits |
| `dev-rebuild.bat` | The ONLY sanctioned way to rebuild the engine, always |
| `LAUNCH.bat` / `dev.bat` | Prod launcher (with restart loop) / dev mode (Vite HMR + tsx watch) |

## Institutional knowledge

- **VALIDATED**: Killing ace-server externally while Node runs causes a respawn/file-lock loop. The crash limiter (3 crashes / 30 s) caps *fast* crash loops, but slow-cycle crashes (>30 s apart) reset the window and loop indefinitely — CLAUDE.md's "infinite respawn" warning predates the limiter but the rule stands: rebuild only via `dev-rebuild.bat`.
- **VALIDATED**: gen logs are lost if Node dies mid-generation (flush-on-completion only). Failed/cancelled generations DO get their gen log.
- **VALIDATED**: `ace_engine.log` and `node_console.log` carry no per-line timestamps; only gen logs do. All time correlation goes through `node_console.log` line ordering.
- **VALIDATED**: `Process exited with code 1, signal null` after a shutdown line is a taskkill artifact, not a crash (dozens in history). Respawns themselves are routine (`Restarting in 3 seconds... (crash N/3)` appears across many healthy sessions).
- **VALIDATED**: `[Safetensors] Cannot open ...adapter_model.safetensors` is usually benign filename probing.
- **VALIDATED**: one UI failure = exactly one engine attempt. The retry loop in `enqueueGeneration` is unreachable for generation failures (runGeneration swallows its own errors, generate.ts:1271-1283); no `[Retry]` line has ever appeared in log history. The gen-log params block is the real, only attempt.
- **UNVALIDATED**: exact engine-side text accompanying `status:'failed'` for every failure class — `hot-step-server.cpp` has many FATAL/failed emit sites; the tables above list only those with real log occurrences. Linux/macOS shutdown paths exist in code (shutdown.ts:45+) but are unconfirmed by logs. `gen_<id>_unknown.log` files have a code path but none have been observed on disk.
- Draft-LM speculative decoding is **disabled by default** (config.ts:117-122) — `--draft-lm` is only passed if `ACESTEPCPP_DRAFT_LM` is set. Don't chase it as a crash suspect unless that env var is set.

## Deeper reading

- [reference.md](reference.md) (this folder) — copy-paste triage command pack, status JSON shapes, engine phase reference, watchdog timing table.
- [CLAUDE.md](../../../CLAUDE.md) — build/git rules that constrain what you may do while debugging.
- [engine/docs/ARCHITECTURE.md](../../../engine/docs/ARCHITECTURE.md) — engine internals, request JSON, generation modes.
- [docs/PLUGINS.md](../../../docs/PLUGINS.md) — Lua solver/scheduler plugins (a bad plugin can wedge DiT inference).
- `docs/plans/` — internal investigation docs. **Gitignored, local-only — may be absent on a fresh clone.**

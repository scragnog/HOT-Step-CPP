# debugging-runtime — reference

Companion to [SKILL.md](SKILL.md). Copy-paste command pack, wire formats, and timing constants. All line refs verified 2026-07-02.

## Triage command pack (PowerShell, repo root)

```powershell
# Newest session folder
$s = (Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1).FullName

# Tails
Get-Content "$s\node_console.log" -Tail 80
Get-Content "$s\ace_engine.log" -Tail 60

# Recent generation logs
Get-ChildItem "$s\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# All failures across history
Select-String -Path "logs\*\generations\*.log" -Pattern "GENERATION FAILED" | Select-Object -Last 10

# Crash forensics in current session
Select-String -Path "$s\node_console.log" -Pattern "exited with code|Crashed|FATAL|CUDA error"

# Trace one job end-to-end (paste the full UUID from the gen filename)
Select-String -Path "$s\node_console.log" -Pattern "Job <uuid>"

# Live health / queue (dev app: Vite :3000 proxies /api to the Node server on :3001)
Invoke-RestMethod http://localhost:3000/api/health | ConvertTo-Json -Depth 4
Invoke-RestMethod http://localhost:3000/api/generate/queue

# Engine direct (:8085) — WARNING: may hang mid-compute (single-threaded httplib); that is normal, not a crash
Invoke-RestMethod http://localhost:8085/health

# Unwedge
Invoke-RestMethod -Method Post http://localhost:3000/api/generate/cancel-all
Invoke-RestMethod -Method Post http://localhost:3000/api/generate/reset-queue

# Clean full shutdown (engine + Vite + Node)
Invoke-RestMethod -Method Post http://localhost:3000/api/shutdown
```

If the app never started, restart it detached with `dev.bat` and read the newest
`logs/<session>/node_console.log`. If there is no session folder at all, run the Node
server in the foreground for one diagnostic pass to see the error (this bypasses dev mode
— do not use it as the normal way to run the app):

```powershell
Set-Location server; npx tsx src/index.ts
```

## Watchdog & timing constants (server/src/routes/generate.ts, services/aceClient.ts)

| Constant | Value | Where | Effect |
|---|---|---|---|
| Poll interval | 500 ms | generate.ts:103 | Engine `GET /job?id=N` cadence |
| Stall timeout | 120 s no stage/progress change | generate.ts:107, 128-134 | → `Generation stalled — no progress for Ns (last stage: "...")` |
| Wall clock | 45 min default, clamped 5–120 (`generationTimeoutMinutes` in request) | generate.ts:105-106, 137-140 | → `Generation timed out (N min limit)` |
| Retry | UNREACHABLE dead code | generate.ts:1294-1339 | The loop exists but `runGeneration` swallows all generation errors (:1271-1283, no rethrow) — retries never fire; expect exactly one attempt |
| Job TTL | 1 h after terminal state, pruned every 10 min | generate.ts:83-94 | `/status/:id` → 404 afterwards |
| HTTP quick timeout | 15 s (health, props, submit) | aceClient.ts:17 | |
| HTTP poll timeout | 30 s | aceClient.ts:18 | Missed polls logged `Poll error ... (will retry)` — benign |
| HTTP result timeout | 300 s (large audio fetch) | aceClient.ts:19 | |
| Respawn delay | 3 s | index.ts:305-307 | |
| Crash limiter | 3 crashes / rolling 30 s window | index.ts:152-156, 288-302 | Then gives up, `engineReady=false` |
| Log ring buffer | 2000 lines | logs.ts:21 | Backs `GET /api/logs` SSE |
| Engine payload cap | 256 MB | hot-step-server.cpp:2700 | Multipart uploads (cover/repaint source) |
| Engine audio cap | 10 min source audio | hot-step-server.cpp:2044 | 413 `audio exceeds max duration (10 min)` |

## `GET /api/generate/status/:id` response shape (generate.ts:1395-1405)

```json
{
  "jobId": "<server uuid>",
  "status": "pending | lm_running | synth_running | saving | succeeded | failed | cancelled",
  "stage": "<human-readable stage string>",
  "progress": 0,
  "result": {},
  "error": "<message or undefined>",
  "ace_job_id": "<engine-side job id or null>",
  "ace_phase": "<AceJobPhase or null>",
  "ace_phase_progress": "step N/M"
}
```

`ace_phase` values (aceClient.ts:165-180, mirrors `job_phase_str()` in hot-step-server.cpp):

```
queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond,
loading_dit, loading_adapter, adapter_precompute, dit_inference,
loading_vae, vae_decode, encoding_output, done, failed, cancelled
```

`phase`/`phase_step`/`phase_total` are optional on the wire for older engine builds (aceClient.ts:154-161) — absence is not an error.

## `GET /api/generate/queue` response shape (generate.ts:1443-1466)

```json
{
  "depth": 1,
  "running": true,
  "current": { "id": "...", "status": "synth_running", "stage": "...", "progress": 42, "age": 130, "aceJobId": "..." },
  "pending": 0
}
```

`age` is seconds since job creation. `current: null` with `running: true` briefly during transitions is possible; persistent `running: true, current: null` after a Node error is what `reset-queue` fixes.

## `GET /api/health` response shape (health.ts:11-46)

```json
{
  "status": "ok",
  "aceServer": { "status": "ok | disconnected", "url": "http://127.0.0.1:8085", "version": "" },
  "server": { "port": 3001, "uptime": 1234.5 },
  "engine": { "ready": true, "bootStatus": "..." }
}
```

`aceServer.status: "disconnected"` while `engine.ready: true` during a generation usually means the single-threaded engine is busy computing, not dead.

## Engine spawn CLI (index.ts:158-214)

```
ace-server.exe --models <dir> --host 127.0.0.1 --port 8085
               [--adapters <dir>] [--keep-loaded] [--noise-profile <wav>]
               [--draft-lm <path>] [--vae-chunk N] [--vae-overlap N]
```

Exe candidates in order (config.ts:46-52): `engine/ace-server.exe` (portable), `engine/build/Release/ace-server.exe` (VS), `engine/build/ace-server.exe` (Ninja), `engine/build/Debug/ace-server.exe`. Env overrides in `.env`: `ACESTEPCPP_EXE`, `ACESTEPCPP_MODELS`, `ACESTEPCPP_PORT`, `ACESTEPCPP_HOST`, `ACESTEPCPP_ADAPTERS`, `ACESTEPCPP_KEEP_LOADED`, `ACESTEPCPP_DRAFT_LM`, `ACESTEPCPP_VAE_CHUNK`, `ACESTEPCPP_VAE_OVERLAP`, `CUDA_VISIBLE_DEVICES`.

## Windows exit codes seen in `Process exited with code N`

| Code (decimal) | Hex | Meaning |
|---|---|---|
| 1 | — | After a shutdown line: benign `taskkill /F` artifact. Otherwise: engine returned error from main (e.g. VAE FATAL). |
| 3221225786 | 0xC000013A | Console Ctrl+C / window closed — usually deliberate |
| 3221226505 | 0xC0000409 | Fail-fast / abort / stack-buffer-overrun — genuine C++ crash; read the preceding engine lines |

## Anatomy of a gen log

```
<ISO ts> | INFO    | ============================================================
<ISO ts> | INFO    | GENERATION STARTED: Job <uuid>
<ISO ts> | INFO    | Task Type: text2music
<ISO ts> | INFO    | Parameters:
<ISO ts> | INFO    | {  ...full resolved engine request JSON: seed, adapters, solver, scheduler...  }
<ISO ts> | INFO    | ============================================================
... phase lines, [Timing] table on success ...
<ISO ts> | INFO    | GENERATION COMPLETED.        ← success
<ISO ts> | ERROR   | GENERATION FAILED: <reason>  ← failure/cancel
```

Buffered in `generationBuffers` (logger.ts:24) and flushed in one `writeFileSync` by `finishGenerationLog`/`failGenerationLog` (logger.ts:139-178). Each gen log contains exactly one params block. (The retry loop in generate.ts is unreachable for generation failures — and even if it fired, `failGenerationLog` deletes the buffer before the `[Retry]` WARNING is logged and attempt 2 would overwrite the same filename, so a two-attempt log is structurally impossible.)

## Known-good real-world log excerpts (from `logs/` history)

- Crash + supervised respawn (2026-06-04, wrong VAE file):
  ```
  [ace-server] [VAE] FATAL: tensor 'decoder.conv1.weight_v' not found in safetensors
  [ace-server] Process exited with code 1, signal null
  [ace-server] Restarting in 3 seconds... (crash 1/3)
  ```
- Benign shutdown artifact:
  ```
  [Shutdown] Killed ace-server PID 1234
  [ace-server] Process exited with code 1, signal null
  ```

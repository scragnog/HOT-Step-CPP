---
name: engine-dev-loop
description: Safe edit-rebuild-test cycle for the HOT-Step C++ engine (dev-rebuild.bat, build script selection, stale-.obj recovery, smoke tests). Use when editing any file under engine/src/ or engine/tools/, when a build fails with linker/file-lock errors, or when ace-server won't restart cleanly.
---

# C++ Engine Development Loop

How to safely edit, rebuild, and smoke-test the C++ engine (`ace-server.exe` and friends)
without triggering the respawn/file-lock loop that this repo learned about the hard way.

**Context for readers with zero prior exposure:** the app has three tiers. A Node/Express
server (port 3001) spawns the C++ inference engine `ace-server.exe` (port 8085) as a managed
child process and serves the React UI. The engine runs the generation pipeline
LM → DiT → VAE (LM = language model that plans the song; DiT = Diffusion Transformer that
denoises latent audio; VAE = decoder that turns latents into 48 kHz stereo audio).
All commands below are Windows PowerShell (use `;` to chain, never `&&` in old
Windows PowerShell 5; this repo's convention is `;`).

## When to use this skill

- You edited (or are about to edit) anything under `engine/src/` or `engine/tools/`.
- A build fails with `LNK1104` / "cannot open ace-server.exe".
- ace-server keeps respawning every ~3 seconds, or the linker can't replace the exe.
- A header change seems to be ignored by the build (stale object files).
- A build after an upstream sync fails with the `hotstep_sampler_linked_` sentinel linker error. (For the full sync + hook-verification process itself, use the `upstream-sync` skill — it owns that procedure.)
- You need to confirm a rebuild actually worked and the engine is healthy.

## Golden rules (hard constraints — each prevented real damage)

1. **Recompile immediately after editing ANY `engine/src/` or `engine/tools/` file — do not
   batch up edits and wait.** (Verbatim instruction from the departing lead engineer.)
   WHY: uncompiled edits pile up, and when the eventual build breaks you can't tell which
   edit did it; the engine also silently keeps serving stale behavior in the meantime.

2. **Rebuild via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly.**
   WHY: you cannot reliably tell whether the app is running (a Node server or orphaned
   tsx-watch tree may be alive in another terminal). Node auto-respawns ace-server on
   abnormal exit (`server/src/index.ts:284-309`, retry after 3 s): if the linker
   deletes/replaces the exe while Node is alive, the respawned process re-locks
   `ace-server.exe` and the link fails with a file-lock error — an infinite respawn +
   file-lock loop. `dev-rebuild.bat` shuts everything down first, and is safe even when
   the app is already stopped (its shutdown request is a suppressed no-op,
   dev-rebuild.bat:10) — so there is no scenario where calling build.cmd directly is
   the right move.

3. **NEVER `cmake --build . --clean-first`.** WHY: CUDA kernel recompilation takes 20+
   minutes (documented in CLAUDE.md; not independently timed). Only justified if the
   GGML/CUDA layer itself changed. For stale-object problems use the surgical recovery
   in Procedure 2 instead.

4. **Don't switch between build scripts casually.** WHY: `buildcuda.cmd`, `buildvulkan.cmd`,
   and `buildall.cmd` all run `cmake ..` unconditionally against the shared `engine/build/`
   directory with *different* flags — reconfiguring the cache can invalidate large parts of
   the incremental build (CUDA-recompile territory). `build.cmd` only configures when
   `CMakeCache.txt` is absent (`engine/build.cmd:140-148`). Stick to `build.cmd` (via
   `dev-rebuild.bat`) unless you deliberately need a different backend set.

5. **`dev-rebuild.bat` does NOT check the build result and does NOT restart the app.**
   It prints "Done. Start the app with LAUNCH.bat" even if MSBuild failed. Read the build
   output yourself, then relaunch (`dev.bat` for dev, `LAUNCH.bat` for prod).

## Procedure 1 — Standard rebuild after a C++ edit

```powershell
# 1. Rebuild (shuts down app gracefully, waits for ace-server to die, builds):
#    ONLY works from an interactive console. From an agent tool call or any piped/
#    redirected shell it silently does nothing and exits 0 — see the warning below,
#    and run the three phases manually instead.
& "D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"

# 2. VERIFY the build yourself — dev-rebuild prints "Done" regardless of result, and the
#    wrapper ALWAYS exits 0 even when MSBuild failed (build.cmd's trailing `cd ..` resets
#    errorlevel) — $LASTEXITCODE is useless here. The ONLY valid checks are:
#    (a) no "error C..." or "LNK..." lines in the scrolled output, and
#    (b) a fresh LastWriteTime on the exe:
Get-Item "D:\Ace-Step-Latest\hot-step-cpp\engine\build\Release\ace-server.exe" | Select-Object LastWriteTime

# 3. Relaunch the app (dev-rebuild does NOT do this):
& "D:\Ace-Step-Latest\hot-step-cpp\dev.bat"       # dev mode (Vite :3000 HMR + tsx watch :3001)
# or LAUNCH.bat for prod mode
```

Note: dev-rebuild's shutdown also kills the Vite dev server on port 3000
(`server/src/routes/shutdown.ts:72-98`), so in dev mode you must restart with `dev.bat`,
not just relaunch the engine — despite the script's final message naming only LAUNCH.bat.

### What dev-rebuild.bat actually does (`dev-rebuild.bat:1-35`)

1. `curl -s -X POST http://localhost:3001/api/shutdown` — asks the **Node** server to shut
   everything down gracefully (errors suppressed, so it's safe when the app isn't running).
   The shutdown route (`server/src/routes/shutdown.ts:134-153`) kills, in order:
   ace-server (found via `netstat` on port 8085, then `taskkill /PID <pid> /T /F`), Vite
   (port 3000), and finally the Node server's own process tree via a detached killer —
   so nothing survives to respawn ace-server.
2. Polls `tasklist` for `ace-server.exe` once per second. At 10 retries it force-kills
   (`taskkill /F /IM ace-server.exe /T`); at 15 retries it aborts with
   `FATAL: could not stop ace-server` and exit code 1.
3. Once ace-server is gone: `call engine\build.cmd`.
4. Prints "Done." — **no build-result check, no restart.**

Why the graceful route matters: the kill is initiated by the server that *owns* the respawn
logic, and then that server kills itself. Killing ace-server externally while Node lives
triggers the respawn handler instead (`server/src/index.ts:284-309`; the crash limiter at
`index.ts:152-156` only gives up after 3 crashes within a 30-second window).

There is also `POST /api/restart` (`shutdown.ts:156-190`), which writes a
`.restart-requested` marker that `LAUNCH.bat` / `server/restart-loop.cmd` loop on. That is
for in-app restarts only — it does not rebuild anything.

### ⚠ dev-rebuild.bat does NOTHING from a non-interactive shell (agents: read this)

`dev-rebuild.bat` uses `timeout /t 1 /nobreak` in its wait loop (steps 2 above). `timeout`
requires a real console: with stdin redirected to the null device — which is every agent
tool call, every piped invocation, every CI-style run — it fails immediately with
`ERROR: Input redirection is not supported`. The loop then falls straight through, so the
script **skips the shutdown AND the build, prints nothing, and exits 0.**

This is maximally deceptive: exit 0, no error text, and (because it never shut anything
down) the app is still running afterwards, so nothing looks wrong. Observed 2026-08-21 —
the "build" took 2 seconds and `ace-server.exe` kept a five-hour-old timestamp.

Run the three phases yourself instead. This is the same sequence the script performs, and
it is safe for the reason Procedure 1 explains — Node initiates the kill, so no respawn:

```powershell
# 1. graceful shutdown (Node kills ace-server, Vite, then itself)
try { Invoke-RestMethod -Method Post -Uri "http://localhost:3001/api/shutdown" -TimeoutSec 10 | Out-Null } catch {}

# 2. wait for the binary lock to actually clear — never skip this, it is what
#    prevents LNK1104 and the respawn loop
$n=0; while ((Get-Process ace-server -ErrorAction SilentlyContinue) -and $n -lt 20) { Start-Sleep -Seconds 1; $n++ }
if (Get-Process ace-server -ErrorAction SilentlyContinue) { Stop-Process -Name ace-server -Force; Start-Sleep -Seconds 2 }

# 3. build, then VERIFY (see Procedure 1 step 2 — the exit code is meaningless)
& cmd.exe /c "engine\build.cmd"
Get-Item "engine\build\Release\ace-server.exe" | Select-Object LastWriteTime
```

Grep the build output for `ace-server.vcxproj ->` — that line is printed only on a
successful link, and is a positive signal rather than the absence of a negative one.

### Errors in a header you never edited = broken literal in the file included *before* it

A single unterminated string literal or comment makes the parser run on into the next
header, so MSVC reports the damage at the wrong file entirely. Symptoms are surreal:
`(double) some_func() / 1048576.0` failing with *"cannot convert from `size_t (__cdecl *)(void)`
to `double`"* plus *"Context does not allow for disambiguation of overloaded function"* — a
call the compiler cannot see the `()` on, in code that is provably correct and unmodified.

Do not debug the reported file. Check the previous one in the include chain, at your edit.

The usual cause when an agent made the edit: a `\n` inside a C string written as a **real
newline**. Shell heredocs and some scripting layers collapse `\\n` → `\n` → an actual line
break, splitting the literal across two lines. Verify after any scripted C++ edit:

```bash
sed -n '<start>,<end>p' <file> | cat -A     # look for a line ending mid-string
```

Prefer the Edit tool over heredocs for C++ containing escapes. (Same class of bug bit a
Windows path in a Markdown file the same day: `\\2026` → octal escape → mangled text.)

### How long builds take

- Incremental `build.cmd` after a single .cpp/.h edit: minutes-scale — configure is skipped
  when `engine/build/CMakeCache.txt` exists (unverified exact duration; no timing data in repo).
- Full clean CUDA build / `--clean-first`: **20+ minutes** (CLAUDE.md claim). Avoid.
- First generation after a rebuild is slow: LoKr adapter precompute takes ~17 s per
  DiT+adapter combo unless `--keep-loaded` is set (`server/src/index.ts:177-180`). That
  slowness is normal, not a regression.

## Procedure 2 — Stale-.obj recovery (surgical, NOT clean)

Use when a header change isn't picked up, or you see weird ODR/linker behavior. This
recompiles only the core lib, leaving the expensive GGML/CUDA objects intact:

```powershell
Remove-Item -Recurse -Force "D:\Ace-Step-Latest\hot-step-cpp\engine\build\acestep-core.dir"
Remove-Item -Force "D:\Ace-Step-Latest\hot-step-cpp\engine\build\Release\acestep-core.lib"
& "D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"
```

(Both paths verified to exist in the current tree.)

## Procedure 3 — After an upstream sync: verify fork hooks

The engine is a patched fork of acestep.cpp. Three upstream files carry HOT-Step `#include`
hooks that break if a sync overwrites them:

| Upstream file | Hook | If lost |
|---|---|---|
| `engine/src/pipeline-synth-ops.cpp` | `hot-step-sampler.h` (replaces `dit-sampler.h`) | **SILENT** — compiles, but all solvers/guidance/schedulers go dead |
| `engine/src/model-store.h` | `hot-step-params.h` | compile error |
| `engine/src/dit.h` | `adapter-merge.h` + `adapter-runtime.h` | compile error |

```powershell
powershell -File "D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1"   # exit 0 = all hooks intact
```

It checks 5 things (`engine/verify-hooks.ps1:15-68`): the three hooks above,
hot-step-server.cpp → hot-step-params.h, and the linker sentinel below.

**Linker sentinel (silent-hook tripwire):** `engine/src/hot-step-sampler.h:1311-1315`
defines `hotstep_sampler_linked_` (selectany/weak, external linkage);
`engine/tools/hot-step-server.cpp:51-52` references it via `extern`. If a sync reverts
pipeline-synth-ops.cpp to `dit-sampler.h`, the symbol vanishes and the link fails with
`unresolved external symbol hotstep_sampler_linked_`. **That error means "re-hook
hot-step-sampler.h", not a real linker problem.** It converts the otherwise-silent hook
loss into a build failure.

## Procedure 4 — Smoke-testing after a rebuild

Relaunch the app, then probe the engine directly. **ace-server takes a few seconds to come
up after `dev.bat` returns** (the script just `start`s detached windows) — retry `/health`
for up to ~30 s; connection refused immediately after launch is normal, not a build failure.
(Use `curl.exe` to be safe — in legacy Windows PowerShell 5.1 bare `curl` is an alias for
`Invoke-WebRequest`; in PowerShell 7 it resolves to the real curl.exe.)

```powershell
# 1. Engine alive? Returns {"status":"ok"}  (engine/tools/hot-step-server.cpp:2709-2711)
curl.exe -s http://localhost:8085/health

# 2. Models discovered + config sane?
curl.exe -s http://localhost:8085/props

# 3. Lua plugin registry loaded? (solvers/schedulers/guidance — the thing silent hook loss kills)
curl.exe -s http://localhost:8085/plugins

# 4. CUDA context up? Returns used/total/free VRAM MB; HTTP 500 if CUDA is broken
#    (hot-step-server.cpp:2722-2741)
curl.exe -s http://localhost:8085/vram
```

Full endpoint surface, if you need deeper probes (`hot-step-server.cpp:2704-2852`):
`POST /lm /synth /understand /vae /warm /job /models/unload /pp-vae-reencode`,
`GET /health /props /logs /jobs /job /plugins /vram /models/loaded`, plus
`/supersep/*` and `/spectral-lifter` routes.

**Real smoke test:** kick a short generation from the UI — **ask the human user to do this
and report back; do NOT use a browser agent for visual verification** (project rule).
Watch the newest session folder under `logs/` (folders are named
`YYYY-MM-DD_HH-MM-SS`, so name-sorted = time-sorted): `ace_engine.log` (C++
stdout/stderr), `node_console.log`, and `generations/gen_<uuid>_<task>.log`.
**Never delete generated test audio or other generation outputs — even ones you predict
are bad.** The user verifies results by ear; leave all artifacts in place.

**Standalone engine (without the Node server):** `engine/server.cmd` runs
`ace-server.exe --host 0.0.0.0 --port 8085 --models .\models --adapters .\adapters --max-batch 1`.
Caveat: its `--models .\models` is relative to `engine/`, but the real model directory is at
repo root (`models/` — the Node server passes the absolute path at
`server/src/index.ts:166-170`). Treat server.cmd as a template, not a turnkey smoke test.

## The build scripts — when each applies

All in `engine/`, all build into the shared `engine/build/` with
`--config Release -j %NUMBER_OF_PROCESSORS%`, all auto-locate `vcvars64.bat` via vswhere
and skip re-sourcing when `VSCMD_VER` is already set (prevents PATH overflow on repeated runs).

| Script | CMake config | Use when |
|---|---|---|
| `engine/build.cmd` | Configures **only if `CMakeCache.txt` absent** (`build.cmd:140-148`). Defaults: `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120a" -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON`. Honors `HOT_STEP_CMAKE_FLAGS` (set by `update.bat:271-285` for auto-detected backends). Also bootstraps ONNX Runtime GPU 1.25.1 into `engine/deps/onnxruntime` and cuDNN 9 DLLs via pip (`build.cmd:42-132`). | **The default dev build.** This is what dev-rebuild calls. Incremental. |
| `engine/buildcuda.cmd` | `cmake .. -DGGML_CUDA=ON` unconditionally (`buildcuda.cmd:36`). No ORT/cuDNN bootstrap, no CPU variants. | Rare — `build.cmd` supersedes it for day-to-day dev. |
| `engine/buildvulkan.cmd` | `cmake .. -DGGML_VULKAN=ON` unconditionally (`buildvulkan.cmd:36`). | Testing the Vulkan backend (AMD/Intel) or Vulkan-specific fixes. |
| `engine/buildall.cmd` | `cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON %RELEASE_CMAKE_EXTRA%` (`buildall.cmd:97`) + ORT bootstrap. | Release-style all-backends build (what CI resembles). Slowest. |

Linux/mac equivalents (`buildcuda.sh`, `buildvulkan.sh`, `buildcpu.sh`, `buildall.sh`,
`build-mac.sh`) exist but are not part of the Windows dev loop.

### Build outputs & which exe the server picks

Binaries land in `engine/build/Release/`: `ace-server.exe`, `ace-lm.exe`, `ace-synth.exe`,
`ace-understand.exe`, `neural-codec.exe`, `mp3-codec.exe`, `quantize.exe`, plus
`mastering.exe`, `vst-host.exe`, `vulkan-shaders-gen.exe` (10 total — the "seven binaries"
in `engine/docs/ARCHITECTURE.md` is stale).

The Node server auto-discovers the exe in priority order (`server/src/config.ts:46-52`):
`engine/ace-server.exe` (portable, flat) → `engine/build/Release/ace-server.exe` (VS
multi-config — **the dev path**) → `engine/build/ace-server.exe` (Ninja) →
`engine/build/Debug/ace-server.exe`. Override with `ACESTEPCPP_EXE` in `.env`.
**Gotcha:** a stale flat `engine/ace-server.exe` (e.g. from unpacking a portable release
into the repo) wins over your fresh Release build. None exists in the tree today, but
check if the server seems to run old code.

## What does NOT need a C++ rebuild

- **Solvers, schedulers, and guidance modes are hot-loadable Lua plugins** in
  `engine/plugins/` (`solvers/`, `schedulers/`, `guidance/` subdirs). Drop a `.lua` in the
  right subdir; it appears in the UI next launch. Native bridge is `apg()`; advanced
  plugins use `post_step()` for extra forward passes. Editing `dit-sampler.h` to add a
  solver is obsolete practice — routing goes through `hot-step-sampler.h`. Guide:
  `docs/PLUGINS.md`.
- **Node/TypeScript changes:** `npx tsc --noEmit` to type-check; `tsx watch` auto-restarts
  the server in dev mode. Don't `npm run build` during dev.

## Key files

| Path | Role |
|---|---|
| `dev-rebuild.bat` (repo root) | The safe rebuild entry point: graceful shutdown → wait/kill → `engine/build.cmd` |
| `engine/build.cmd` | Default incremental CUDA dev build + ORT/cuDNN bootstrap |
| `engine/buildcuda.cmd` / `buildvulkan.cmd` / `buildall.cmd` | Alternate backend builds (reconfigure unconditionally — see Golden rule 4) |
| `engine/verify-hooks.ps1` | Post-upstream-sync hook checker (exit 0 = intact) |
| `engine/src/hot-step-sampler.h` | Fork sampler (solvers/schedulers/guidance routing) + linker sentinel at :1311-1315 |
| `engine/tools/hot-step-server.cpp` | ace-server main; sentinel extern at :51-52; HTTP routes at :2704+ |
| `engine/build/Release/` | Build output dir (the exe Node runs in dev) |
| `engine/build/acestep-core.dir/` + `engine/build/Release/acestep-core.lib` | The ONLY things to delete for stale-.obj recovery |
| `server/src/index.ts` | Spawns ace-server (:158+); respawn-on-crash handler (:284-309); crash limiter (:152-156) |
| `server/src/routes/shutdown.ts` | `/api/shutdown` (kills ace-server, Vite, self) and `/api/restart` |
| `server/src/config.ts` | Exe discovery order (:46-52), `ACESTEPCPP_EXE` override |
| `engine/server.cmd` | Standalone ace-server launcher (template — model paths need adjustment) |
| `logs/<session>/ace_engine.log` | C++ engine output for the newest run |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| `LNK1104` / cannot open `ace-server.exe` during link | ace-server still running (respawn loop or manual start) | Run `dev-rebuild.bat` (its shutdown phase), or check `tasklist /FI "IMAGENAME eq ace-server.exe"` |
| `unresolved external symbol hotstep_sampler_linked_` | Upstream sync clobbered the `hot-step-sampler.h` include in `pipeline-synth-ops.cpp` | Re-add the include; run `verify-hooks.ps1` |
| Builds fine, but solver/scheduler/guidance selections have no effect | Silent hook loss predating the sentinel, or sentinel removed | Run `verify-hooks.ps1`; check `GET :8085/plugins` |
| Compile error: missing `hot-step-params.h` / `adapter-merge.h` / `adapter-runtime.h` | Sync clobbered `model-store.h` / `dit.h` hooks (these fail loudly) | Re-add the includes; run `verify-hooks.ps1` |
| `[ace-server] Crashed 3 times within 30s — giving up` in node_console.log | Missing DLL next to the exe (cuBLAS/cuDNN/ORT) or startup crash | Check `ace_engine.log` in the newest `logs/` session |
| Engine respawns endlessly every ~3 s | Something external kills ace-server while Node lives (respawn handler fires); crashes spaced >30 s apart reset the limiter | Stop killing it externally; use `dev-rebuild.bat` / `/api/shutdown` |
| Node runs an old binary despite a successful build | Stale flat `engine/ace-server.exe` shadowing `build/Release/` (config.ts:46-52), or `ACESTEPCPP_EXE` set in `.env` | Delete the flat exe / unset the override |
| Header edit seemingly ignored | Stale .obj | Procedure 2 (surgical delete of `acestep-core.dir` + `acestep-core.lib`) |
| `dev-rebuild.bat` says "Done" but nothing changed | It never checks the build result | Scroll up and read the MSBuild output; check the exe timestamp |
| `dev-rebuild.bat` exits 0 in ~2 s, prints nothing, app still running | Non-interactive shell: its `timeout /t` needs a console, so the whole script falls through — no shutdown, no build | Run the 3 phases manually (Procedure 1 → "does NOTHING from a non-interactive shell") |
| Compile errors in a header you never touched | Unterminated string/comment in the file included *before* it — usually a `\n` written as a real newline by a heredoc | `sed -n 'A,Bp' <the file you edited> \| cat -A`; use the Edit tool for C++ with escapes |
| Vite dead after rebuild in dev mode | dev-rebuild's shutdown kills port 3000 too (`shutdown.ts:72-98`) | Restart with `dev.bat`, not LAUNCH.bat |
| Node server won't start / weird npm dep errors after an otherwise-good build | Wrong Node version — **Node 18–22 LTS only; Node 24+ breaks dependencies** (`engines` enforces `<24`) | `node --version`; switch Node, don't touch the engine or build cache |
| Connection refused on :8085 right after relaunch | Engine still starting — dev.bat returns before services listen | Retry `/health` up to ~30 s, then check `ace_engine.log` |
| PATH weirdness after many rebuilds in one shell | Shouldn't happen — scripts skip vcvars when `VSCMD_VER` is set | Open a fresh shell |

## Institutional knowledge

- **VALIDATED (lead-engineer rule, verbatim in meaning):** Recompile immediately after
  editing any `engine/src/` or `engine/tools/` file — do not batch up edits and wait.
- **VALIDATED (code-verified):** the respawn/file-lock loop mechanism — respawn handler at
  `server/src/index.ts:284-309`, 3 s delay, crash limiter `MAX_CRASHES = 3` within
  `CRASH_WINDOW_MS = 30_000`.
- **VALIDATED (code-verified):** the linker sentinel exists specifically to turn the silent
  `pipeline-synth-ops.cpp` hook loss into a hard link error.
- **VALIDATED (learned the hard way, per CLAUDE.md):** `--clean-first` costs 20+ minutes of
  CUDA kernel recompilation. Duration is a doc claim, not independently timed.
- **UNVERIFIED:** exact incremental build duration ("minutes-scale") — no timing data in repo.
- **UNVERIFIED:** whether `engine/server.cmd` finds models when run standalone from
  `engine/` — its `--models .\models` path likely needs adjusting to the repo-root `models/`.

## Deeper reading

- `engine/docs/ARCHITECTURE.md` — engine internals, CLI, request JSON, generation modes
  (note: its binary count and raw manual-cmake instructions are stale; the wrapper scripts
  + dev-rebuild are current practice).
- `docs/PLUGINS.md` — Lua plugin authoring (the no-rebuild path for solvers/schedulers/guidance).
- `docs/RELEASING.md` — release builds; pushing any `v*` tag triggers a full multi-platform CI build.
- `CLAUDE.md` — repo-wide rules (git discipline, environment, log layout).
- **Full sync + hook-repair process: the `upstream-sync` skill** (`.claude/skills/upstream-sync/`,
  always present) — it carries the concrete re-hook repair steps this skill's failure table
  abbreviates. `docs/plans/upstream-sync-workflow.md` is the local long-form doc, **gitignored**
  — may be absent on a fresh clone.

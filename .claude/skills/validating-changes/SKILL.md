---
name: validating-changes
description: Defines the verification bar every HOT-Step CPP change must clear before being called done — per-tier checks (TypeScript, C++ engine, UI, audio, end-to-end smoke generation) and honest result reporting. Use when finishing any code change, deciding whether work is "done", running a smoke/e2e test, or reporting test/validation results.
---

# Validating Changes to the Required Standard

HOT-Step CPP is a 3-tier app: a C++17/CUDA/GGML **engine** (`engine/`, spawned as
`ace-server.exe` on port 8085), a Node/TypeScript/Express **server** (`server/src/`,
port 3001), and a React 19 **UI** (`ui/src/`, Vite dev server on port 3000). A change
is NOT done until it clears the bar for **every tier it touches**. This skill defines
those bars, how to run a headless end-to-end smoke generation, and how to report
results honestly.

All commands are Windows PowerShell (`;` as separator, never `&&`).
Deep API/log detail lives in [reference.md](reference.md) in this folder.

## When to use this skill

- You just finished a code change and need to decide whether it is "done".
- You need to verify a C++ engine change, a server/UI TypeScript change, or a
  Lua plugin change actually works.
- You need to run an end-to-end generation without the UI (headless smoke test).
- You are about to report validation results to the user.

## Golden rules (hard constraints — each prevents expensive damage)

1. **ONLY the human can judge audio quality, by ear.** Never claim audio "sounds
   fine" or "is broken" from metrics, file size, duration, RMS, spectrograms, log
   silence, or the pipeline's quality-evaluator scores (stored in the `quality_scores`
   DB column — advisory telemetry only). Say: "generation completed mechanically;
   audio at `<path>` awaits your ear." WHY: metrics have repeatedly failed to predict
   what the user hears; false verdicts destroy trust and mislead debugging.

2. **PRESERVE EXPERIMENT ARTIFACTS.** Never delete generated test audio, latents,
   or other outputs because *you* predict they are bad. The human verifies by ear and
   has lost artifacts to premature cleanup before. WHY: a generation costs GPU time and
   a seed; a deleted artifact may be unreproducible evidence.

3. **Report outcomes faithfully.** Failed tests are reported as FAILED with the actual
   output/error pasted. Skipped steps are reported as SKIPPED with the reason. Never
   write hedged "should work" / "probably fine". Distinguish what you **ran and saw**
   vs what you **inferred** vs what you **did not verify**. WHY: the user makes
   release and debugging decisions off your report.

4. **Rebuild C++ only via `dev-rebuild.bat` at repo root — never `engine\build.cmd`
   directly, under any circumstances.** WHY: you cannot reliably tell whether the app
   is running; the Node server auto-respawns ace-server 3 s after a non-clean exit
   (`server/src/index.ts:284-308`), and the respawned exe file-locks the linker
   output — infinite loop. `dev-rebuild.bat` is safe even when the app is stopped
   (its shutdown request is a suppressed no-op).

5. **Never push a `v*` tag to "test" a build.** WHY: any pushed `v*` tag triggers the
   full multi-platform Release workflow and drafts a GitHub Release. Throwaway compile
   checks use a `-CI-Test` suffix; any push still requires explicit user approval.

6. **Never `cmake --build . --clean-first`** unless the GGML/CUDA layer itself
   changed. WHY: CUDA kernel recompilation takes 20+ minutes. For stale-`.obj`
   symptoms delete only `engine/build/acestep-core.dir/` and
   `engine/build/Release/acestep-core.lib`.

7. **No `npm run build` during dev.** Type-check with `tsc` only (§ below). WHY:
   builds are slow and unnecessary until the user tests a prod (`LAUNCH.bat`) flow.

8. **Never visually verify UI with a browser agent** — ask the human; they provide
   screenshots. Browser/`Invoke-RestMethod` IS fine for non-visual API checks.
   WHY: the browser agent is too slow/unreliable in this environment.

9. **A feature is not done until a USER can run it.** Anything the app resolves
   at runtime — model weights, a data file, a binary — must be reachable by
   someone who downloaded a release, not just present on this machine. Run
   `node server/scripts/check-release-prereqs.mjs` before any push (§ Tier 6).
   WHY: this has shipped broken twice. MM3 training asked for two GGUFs that
   existed only on the dev box (#137), and the MM3 caption corpus was committed
   but never copied into the archives (#139). Both passed tsc, both built green,
   both were dead on arrival for every user.

10. **Don't clobber the user's generation queue.** Jobs run one at a time on a single
   GPU. Check `GET /api/generate/queue` before submitting a smoke generation, and
   never call `POST /api/generate/reset-queue` (force-drains everything) without
   asking. WHY: it cancels the user's in-flight work.

## Tier 1 — TypeScript (server and UI)

Bar: type-check clean. Zero errors.

> **Node 18–22 LTS only** — Node 24+ breaks dependencies (`engines: >=18.0.0 <24.0.0`
> in `server/package.json`). If tsc/tsx/npm fails oddly, check `node --version`
> BEFORE blaming the change — a wrong Node version produces false validation verdicts.

```powershell
cd D:\Ace-Step-Latest\hot-step-cpp\server; npx tsc --noEmit
cd D:\Ace-Step-Latest\hot-step-cpp\ui; npx tsc -b
```

- `server/package.json` defines `"typecheck": "tsc --noEmit"`. The UI uses project
  references (`ui/tsconfig.json` → `tsconfig.app.json` + `tsconfig.node.json`, both
  `noEmit`-safe), so `npx tsc -b` type-checks without emitting JS.
- In dev mode (`dev.bat`) the server runs under `tsx watch` (`server/package.json`
  `"dev": "tsx watch src/index.ts"`) — it auto-restarts on save. Server-side bar =
  passing tsc **plus** a clean restart visible in the newest
  `logs/<session>/node_console.log` (no stack trace after the restart).

## Tier 2 — C++ engine

Bar: rebuild compiles AND the engine starts clean after relaunch (log check).

1. Recompile **immediately** after editing any `engine/src/` or `engine/tools/`
   file — don't wait to be asked:

   ```powershell
   D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat
   ```

   What it does (verified in the script): POSTs `http://localhost:3001/api/shutdown`
   (kills ace-server, Vite, and the Node server cleanly), waits up to 10 s for
   `ace-server.exe` to exit, force-kills at 10 s, aborts at 15 s, then runs
   `engine\build.cmd`.

2. **The rebuild does NOT restart the app.** Relaunch (`dev.bat` for dev,
   `LAUNCH.bat` for prod) before any runtime verification.

3. Clean-start check — read the newest `logs/<session>/ace_engine.log` and confirm:
   - model registry scan ran (`[Server] Scanning models in ...` / `[Registry] ...` lines),
   - `[Server] Listening on 127.0.0.1:8085` appears,
   - no crash/exception before the Listening line,
   - `node_console.log` has **no** `[ace-server] Restarting in 3 seconds...`
     (that exact string = crash respawn, `server/src/index.ts:304`).

   ```powershell
   $sess = (Get-ChildItem D:\Ace-Step-Latest\hot-step-cpp\logs | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Select-String -Path "$sess\ace_engine.log" -Pattern 'Listening on'
   Select-String -Path "$sess\node_console.log" -Pattern 'Restarting in 3 seconds'
   ```

4. After any **upstream sync** (the engine is a patched fork of acestep.cpp):

   ```powershell
   powershell -File D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1
   ```

   Exit 0 = good. Note: CLAUDE.md says "3 hook files" but the script checks **5**
   things (the script is the truth): `pipeline-synth-ops.cpp → hot-step-sampler.h`
   (loss is **SILENT** — compiles fine but all solvers/guidance/schedulers go dead),
   `model-store.h → hot-step-params.h`, `dit.h → adapter-merge.h` +
   `adapter-runtime.h`, `tools/hot-step-server.cpp → hot-step-params.h`, and a
   linker sentinel `hotstep_sampler_linked_` in `hot-step-sampler.h`.

5. Lua plugins (`engine/plugins/`) need **no** rebuild, but they load at engine
   start — relaunch the app for a new/edited plugin to appear.

## Tier 3 — UI visuals

Bar: `npx tsc -b` clean + Vite HMR applies the change without console errors +
**the human confirms the visuals**. You cannot close this tier yourself. Ask the
user to look and wait for their screenshot/feedback.

## Tier 4 — Audio quality

Bar: **the human's ear.** There is no mechanical substitute — see Golden rules 1–2.
Your job ends at: generation succeeded mechanically, here is the file path
(`server/data/audio/<uuid>.wav` or `http://localhost:3001/audio/<uuid>.wav`), here
are the seed and params for reproduction. Then wait.

## Tier 5 — End-to-end headless smoke generation

Bar: job reaches `status: "succeeded"`, a WAV exists on disk, and the per-generation
log ends with `GENERATION COMPLETED.` (Mechanical success only — quality is Tier 4.)

Full request/response/log detail: [reference.md](reference.md).

1. **Pre-flight health** — require `engine.ready -eq $true`:

   ```powershell
   Invoke-RestMethod http://localhost:3001/api/health
   ```

2. **Check the queue is free** (don't queue behind/ahead of the user silently):

   ```powershell
   Invoke-RestMethod http://localhost:3001/api/generate/queue
   ```

3. **Get an auth token** (`POST /api/generate` returns 401 without a Bearer token;
   tokens are in-memory and reset on server restart):

   ```powershell
   $auth = Invoke-RestMethod http://localhost:3001/api/auth/auto
   $hdr = @{ Authorization = "Bearer $($auth.token)" }
   ```

4. **Submit a fast smoke request** — `skipLm: $true` skips the language-model phase
   (the "LM" that writes audio codes/metadata before the DiT diffusion transformer
   runs). Always pass `duration` explicitly: with skipLm the server defaults an
   unset duration to **120 s** (`generate.ts:222`).

   ```powershell
   $body = @{
     prompt         = 'minimal ambient techno, warm pads'
     instrumental   = $true
     skipLm         = $true
     duration       = 30
     inferenceSteps = 8
     seed           = 42      # fixed seed = reproducible; or randomSeed = $true
   } | ConvertTo-Json
   $job = Invoke-RestMethod -Method Post -Uri http://localhost:3001/api/generate -Headers $hdr -ContentType 'application/json' -Body $body
   $job.jobId
   ```

   Omit `skipLm` if your change touches the LM phase — then the smoke must
   exercise it.

5. **Poll to completion** (status values: `pending | lm_running | synth_running |
   saving | succeeded | failed | cancelled`):

   ```powershell
   do { Start-Sleep 3; $s = Invoke-RestMethod "http://localhost:3001/api/generate/status/$($job.jobId)";
        "{0} {1}% {2}" -f $s.status, $s.progress, $s.stage
   } while ($s.status -notin 'succeeded','failed','cancelled')
   $s | ConvertTo-Json -Depth 5
   ```

   Built-in watchdogs (`pollUntilDone`, `generate.ts:102-170`): 2 min with no
   stage/progress change ⇒ "Generation stalled"; wall-clock default 45 min. A
   first-ever TensorRT run legitimately shows
   `Building TRT engine ... (first run only, ~5-10 min)` — do not panic-cancel it.
   There is 1 automatic retry with a randomized seed on transient failure.

6. **Confirm outputs.** On success `$s.result.audioUrls` is like
   `["/audio/<uuid>.wav"]`. **The live data dir is `server\data`, NOT the repo-root
   `data\`** (verified: `config.data.dir` resolves relative to `server/src`,
   `server/src/config.ts:191-199`; root `data/` is stale/legacy — never "verify"
   outputs there). Files land in `server\data\audio\`; DB row in the `songs` table
   of `server\data\hotstep.db` with the full request in `generation_params`.

7. **Confirm the log.** Per-generation logs are buffered in memory and flushed
   **only at finish/fail** — mid-run, tail `node_console.log` (lines prefixed
   `[Gen:<jobId8>]`) or `ace_engine.log` instead.

   ```powershell
   $sess = (Get-ChildItem D:\Ace-Step-Latest\hot-step-cpp\logs | Sort-Object Name -Descending | Select-Object -First 1).FullName
   Get-Content (Get-ChildItem "$sess\generations" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 40
   ```

   Success ends `GENERATION COMPLETED.`; failure ends `GENERATION FAILED: <error>`
   (`server/src/services/logger.ts:145,167`).

8. **Hand off to the human** (Tier 4): file path + seed + params. Do not delete
   the artifacts.

## Tier 6 — Distributability (does it work for someone who is not you?)

Bar: `node server/scripts/check-release-prereqs.mjs` exits 0.

Applies to **any change that adds a file the app looks for at runtime** — a new
model, a new data file, a new registry entry, a new packaged binary. This tier is
invisible to every other bar: the path is resolved at runtime, so tsc passes, the
build is green, and the feature is dead in every download.

```powershell
node D:\Ace-Step-Latest\hot-step-cpp\server\scripts\check-release-prereqs.mjs
# --offline to skip the Hugging Face half
```

What it checks:

1. **Weights** — every entry in `server/src/data/model-registry.json` exists in
   the Hugging Face repo it names, at the size the registry claims, in a repo
   that is public. Companion files (LICENSE) too.
2. **Packs** — no pack references a file id that does not exist (that is how the
   Model Manager blanked in #120).
3. **Runtime data** — everything in `server/src/data/` is packaged by
   `.github/workflows/release.yml`, which copies the directory wholesale for all
   three platforms. Put new runtime data there and it ships automatically; put it
   anywhere else and it will not.

What it CANNOT check, so do it by hand:

- A model resolved by prefix scan rather than by registry id. `resolveMm3TrainModels`
  in `server/src/services/training/mm3Train.ts` takes the newest
  `mm3-rvq-*.gguf` on disk — the registry has to publish something that matches
  the prefix, and only a human knows that. **`fs.existsSync` or a directory scan
  against a models dir is the smell**; grep for those when adding a feature.
- Anything gated behind an HF token or a licence the user has to accept.
- Weights you converted locally. Uploading is a deliberate act: `huggingface_hub`
  is installed and the token is in `~/.cache/huggingface/token`, but ask before
  pushing anything to a public repo, and credit the original author in the model
  card if the weights are not ours.

## Key files

| Path | Role |
|---|---|
| `dev-rebuild.bat` | The ONLY sanctioned C++ rebuild entry point (shutdown → build) |
| `engine/verify-hooks.ps1` | Post-upstream-sync check of the 5 fork hooks |
| `server/src/routes/generate.ts` | Generation queue, POST `/api/generate`, status/queue/cancel endpoints, watchdogs |
| `server/src/routes/health.ts` | `GET /api/health` — engine readiness |
| `server/src/routes/auth.ts` | `GET /api/auth/auto` — free Bearer token; `getUserId` guard |
| `server/src/services/generation/translateParams.ts` | UI param names → engine `AceRequest` names |
| `server/src/services/aceClient.ts` | HTTP client to ace-server :8085 (single-threaded httplib caveat) |
| `server/src/services/logger.ts` | Session log folders + buffered per-generation logs |
| `server/src/config.ts` | Ports, paths; `config.data.dir` = `server/data` |
| `logs/YYYY-MM-DD_HH-MM-SS/` | Per-session logs: `ace_engine.log`, `node_console.log`, `generations/gen_<jobId>_<task>.log` |
| `server/data/audio/` | Where generated audio actually lands |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| 503 `Engine not ready: ...` on POST /generate | ace-server still bootstrapping — wait; check `/api/health` `engine.bootStatus` |
| 401 Unauthorized | missing/stale Bearer token (in-memory, resets on Node restart) — re-hit `/api/auth/auto` |
| `Generation stalled — no progress for Ns` | 2-min watchdog fired; engine wedged or emitting unrecognized progress — check `ace_engine.log` tail |
| `Generation failed on ace-server` | engine-side failure — the real error is in `ace_engine.log`, not the Node response |
| `[ace-server] Restarting in 3 seconds...` in node_console.log | engine crashed; repeated ⇒ missing DLL or a bad C++ change |
| `/api/health` bootStatus `Engine crashed N times — check logs for missing DLLs` | crash limiter gave up (`index.ts:296-302`) — fix the crash, relaunch |
| Solvers/schedulers/guidance silently ignored after upstream sync (no error!) | lost `hot-step-sampler.h` hook — run `engine/verify-hooks.ps1` |
| Adapter/sideband params vanish between LM and synth phases | LM echo drops ServerFields-only params; synth requests are rebuilt from the current request + LM fields only (`generate.ts:312-328`) — new sideband params must flow through that reconstruction, never a whitelist |
| First run stuck on `Building TRT engine` 5–10 min | normal one-time TensorRT compile, not a hang |
| HTTP timeouts against :8085 during a generation | normal — ace-server is single-threaded httplib and can't answer mid-DiT-step (`aceClient.ts:6-8`) |
| Output file "missing" from repo-root `data\` | wrong directory — live outputs are in `server\data\audio\` |

## Institutional knowledge

- **VALIDATED (ground truth):** PRESERVE EXPERIMENT ARTIFACTS — never delete
  generated test audio/outputs on your own prediction that they are bad; the human
  verifies by ear and has lost artifacts to premature cleanup before.
- **VALIDATED (ground truth):** report outcomes faithfully — failed = FAILED with
  output pasted; skipped = SKIPPED with reason; no hedged "should work".
- **VALIDATED:** audio quality is human-ear-only; `quality_scores` telemetry is
  advisory, never a verdict.
- **VALIDATED:** live data dir is `server\data`, root `data\` is stale
  (`config.ts:191-199` + runtime log confirm).
- **VALIDATED:** `verify-hooks.ps1` checks 5 hooks, not the 3 CLAUDE.md lists —
  the script is the truth.
- **VALIDATED (code):** `adapter_section_isolation` still exists on the wire
  (`aceClient.ts`, `translateParams.ts`) but the feature was reverted (commit
  `ee041e1`, broke musical continuity) — don't "fix" it back in or count it as active.
- **UNVALIDATED:** basin re-base params `rebase_source`/`rebase_beta` exist in
  `translateParams.ts` but the cross-base fix is unvalidated — never report it working.
- **DELIBERATE, not a bug:** draft-LM speculative decoding is disabled by a code
  comment in `server/src/config.ts` (~line 117).
- **VALIDATED (twice, in production):** local presence is not distribution.
  MM3 training required `mm3-rvq-*.gguf` + `mm3-enc-*.gguf` that had never been
  uploaded (#137, fixed 96d442fb); `mm3-corpus.json` was committed but not copied
  into release archives (#139). Tier 6 exists because of these.
- **VALIDATED:** smoke generations queue globally one-at-a-time behind the user's
  jobs — check `GET /api/generate/queue` first; `reset-queue` is destructive.

## Deeper reading

- [reference.md](reference.md) — same folder: full API shapes, output-file table,
  log anatomy, all generate endpoints.
- `CLAUDE.md` (repo root) — orientation map, build/git rules.
- `engine/docs/ARCHITECTURE.md` — engine internals, request JSON, generation modes.
- `docs/PLUGINS.md` — Lua plugin authoring (plugins hot-load, no rebuild).
- `docs/RELEASING.md` — release process (any pushed `v*` tag triggers a full
  multi-platform CI build — never push casual `v*` tags).
- `docs/plans/` — internal design/investigation docs. **Gitignored, local-only —
  may be absent on a fresh clone.**

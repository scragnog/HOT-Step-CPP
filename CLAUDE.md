# CLAUDE.md — HOT-Step CPP

Orientation map for agents. Keep this short and navigational — point at the deep docs, don't duplicate them.

## What this is

A desktop app for **local AI music generation** — a heavily-extended superset of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (a C++/GGML port of ACE-Step 1.5). Caption + lyrics in → stereo 48 kHz audio out, fully local. Ships as portable releases (Windows CUDA/Vulkan/CPU, Linux, macOS Metal). GitHub: `scragnog/HOT-Step-CPP`.

## Architecture (3 tiers)

| Tier | Stack | Location | Role |
|------|-------|----------|------|
| **Engine** | C++17 / CUDA / GGML | [engine/](engine/) | Inference binaries: `ace-lm`, `ace-synth`, `ace-server`, `ace-understand`, `neural-codec`, `mp3-codec`, `quantize`. Pipeline: LM → DiT → VAE. Also hosts the **MiniMax-Music3 backend** ([engine/src/minimax/](engine/src/minimax/), `/mm3/*` endpoints) |
| **Server** | Node / TypeScript / Express / better-sqlite3 | [server/src/](server/src/) | Orchestrates the engine, manages songs/jobs/SQLite, serves UI. Per-feature [routes/](server/src/routes/) + [services/](server/src/services/) |
| **UI** | React 19 / Vite / Zustand / Tailwind | [ui/src/](ui/src/) | Browser frontend. Component folder per "studio" |

```
LAUNCH.bat → Node server (Express :3001)
  ├── serves React frontend (prebuilt ui/dist/)
  ├── /api/* → SQLite
  └── spawns child: ace-server.exe (C++ engine) on :8085
```

| Service | Port |
|---------|------|
| Node server | 3001 (prod) |
| Vite dev server | 3000 (dev, HMR) |
| ace-server (C++ engine) | 8085 (default, `config.ts`) |

## Environment

- **Windows 11 + PowerShell.** This repo's primary dev environment is Windows. The Claude Code harness also gives you a Bash (POSIX) tool — each takes its own syntax. In PowerShell use `;` not `&&`.
- **Node 18–22 LTS only.** Node 24+ breaks dependencies (`engines` field enforces `<24`).

## Build & run rules (IMPORTANT — learned the hard way)

- **C++ engine changes → `dev-rebuild.bat`, NEVER `engine/build.cmd` directly.** The Node server auto-respawns ace-server on crash; killing it without clean shutdown causes an infinite respawn + file-lock loop. `dev-rebuild.bat` handles clean shutdown + rebuild — it does **not** relaunch; start the app again yourself with `dev.bat`/`LAUNCH.bat`.
  - Recompile **immediately** after editing any `engine/src/` or `engine/tools/` file — don't wait to be asked.
- **NEVER `cmake --build . --clean-first`** unless the GGML/CUDA layer itself changed — CUDA kernel recompilation is **20+ min**. For stale `.obj` issues, delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
- **Don't `npm run build` during dev.** Only build before user testing. Type-check with:
  - `server/` → `npx tsc --noEmit`
  - `ui/` → **`npx tsc --noEmit -p tsconfig.app.json`** (or `npx tsc -b`). A bare `npx tsc --noEmit` in `ui/` **silently checks nothing and exits 0** — `ui/tsconfig.json` is `{"files": [], "references": [...]}`, so the root project has no inputs. It is not a passing check, it is no check.
- **`dev.bat`** = dev mode (Vite :3000 HMR + Node :3001, tsx watch auto-restart). **`LAUNCH.bat`** = prod. Use `dev.bat` for development.

## Git rules

- **All work on `master`. No feature branches, ever.**
- **Never `git add -A`** (re-adds gitignored dirs: `.agents/`, `checkpoints/`, `node_modules/`, etc.). **Never `git add -f`** on gitignored paths. Stage explicit paths.
- **Push requires explicit user approval — always ask first.**
- Commit to local git **often** (data has been lost before to uncommitted files).
- **Releases:** push a `vX.Y.Z` tag → the `Release` workflow builds all platforms and drafts a GitHub Release. **Any pushed `v*` tag triggers a build** — use a `-CI-Test` suffix for throwaway compile checks, and don't push local feature tags matching `v*`. Full process + gotchas: [docs/RELEASING.md](docs/RELEASING.md).
- Use `gh` CLI for GitHub ops (authenticated as `scragnog`).

## Shipping to users (works here ≠ works for them)

This is a public app. A feature can pass every local check because **this machine**
holds a file that was never part of the distribution — model weights sitting in
`models/`, a data file that CI never copies into the archive. Nothing catches it:
paths are resolved at runtime, so tsc is clean and the build is green, and the
feature is simply dead for everyone who downloads it. It has shipped twice
(MM3 training encoders, #137; the MM3 caption corpus, #139).

- **Any new file the app resolves at runtime must be reachable by a user** —
  weights uploaded to Hugging Face **and** listed in
  [`server/src/data/model-registry.json`](server/src/data/model-registry.json);
  runtime data files packaged by [`release.yml`](.github/workflows/release.yml)
  (it copies `server/src/data/` wholesale, so put them there).
- **Before pushing anything to `master`, and always before a release tag:**

  ```
  node server/scripts/check-release-prereqs.mjs
  ```

  It verifies every registry entry exists on HF at the claimed size, that packs
  reference real files, and that every runtime data file gets packaged. Exit 1 =
  do not ship. Details: [.claude/skills/validating-changes/SKILL.md](.claude/skills/validating-changes/SKILL.md) (Tier 6).

## Upstream sync (fork hooks that break silently)

The C++ engine is a patched fork of acestep.cpp. Three upstream files carry HOT-Step `#include` hooks that break if overwritten during a sync:

| Upstream file | Hook | If lost |
|---|---|---|
| `pipeline-synth-ops.cpp` | `hot-step-sampler.h` (replaces `dit-sampler.h`) | **SILENT** — compiles, but all solvers/guidance/schedulers go dead |
| `model-store.h` | `hot-step-params.h` | compile error |
| `dit.h` | `adapter-merge.h` + `adapter-runtime.h` | compile error |

After any sync: run `engine/verify-hooks.ps1`. Full process: `docs/plans/upstream-sync-workflow.md` *(local, gitignored)*.

## UI / browser verification

- **Don't use the built-in browser agent to visually verify UI** — too slow/unreliable here. **Ask the user to check**; they provide screenshots/feedback. Browser agent is fine for non-visual tasks (hitting API endpoints).

## Debugging — logs

App writes per-session logs to `logs/` at repo root:

```
logs/YYYY-MM-DD_HH-MM-SS/        ← one folder per session (name-sorted = time-sorted)
  ├── ace_engine.log              ← C++ engine output
  ├── node_console.log            ← Node server output
  └── generations/gen_<uuid>_<task>.log
```

Start with the newest session folder. Generation failures → matching `gen_*.log` first, then cross-ref `ace_engine.log` + `node_console.log`. Startup/crash → `node_console.log` + `ace_engine.log`.

## Plugin system

Solvers (17), schedulers (9), guidance modes, and postprocess are **hot-loadable Lua plugins** in [engine/plugins/](engine/plugins/) — drop a `.lua` in the right subdir, appears in the UI next launch, no C++ rebuild. Each plugin can declare its own UI params. Native C++ bridge via `apg()`; advanced plugins use `post_step()` for extra forward passes. **Adding a solver/scheduler/guidance = write a `.lua` plugin** (the old approach of editing `dit-sampler.h` is obsolete — the engine now routes through `hot-step-sampler.h`). Authoring guide: [docs/PLUGINS.md](docs/PLUGINS.md).

## Discord transcripts

The MM3 working group lives in Discord, and a lot of project-relevant decisions
happen there. [tools/discord-claude/](tools/discord-claude/) bridges that thread to
Claude *and* logs every message to `logs/discord/<channelId>.jsonl` (gitignored —
it is other people's chat). Read it with:

```
node tools/discord-claude/read-log.mjs --list                     # channels + message counts
node tools/discord-claude/read-log.mjs --since 12h                # busiest channel, recent
node tools/discord-claude/read-log.mjs --channel all-for-one --last 200
node tools/discord-claude/read-log.mjs --all --grep "encoder|NVFP4" --context 3
```

`backfill.mjs` pulls history from the Discord API (safe to re-run; dedupes by message id).
**Do not** reconstruct the thread by scraping `~/.claude/projects/*.jsonl` — those sessions
only ever saw a rolling window and are lossy.

## Read-Y-for-X index

| For… | Read |
|------|------|
| **Any maintenance task — start here** (per-domain procedures, gotchas, distilled institutional knowledge) | [.claude/skills/README.md](.claude/skills/README.md) — 16 skills (13 fact-checked + 3 MM3) |
| **MiniMax-Music3 backend** (second generation backend: engine port, /mm3 endpoints, backend registry/toggle, trap list) | [.claude/skills/mm3-backend/SKILL.md](.claude/skills/mm3-backend/SKILL.md) |
| MM3 caption/prompt format (genre adherence) | [.claude/skills/mm3-captioning/SKILL.md](.claude/skills/mm3-captioning/SKILL.md) |
| **Training an MM3 LM adapter** (album/artist clone: rank, steps, which checkpoint to ship, likeness-vs-coherence) | [.claude/skills/mm3-lm-adapter-training/SKILL.md](.claude/skills/mm3-lm-adapter-training/SKILL.md) |
| **What the Discord working group said** (MM3 group: bghira, Serveurperso, testerf, Shaz…) — searchable transcripts of every channel | `node tools/discord-claude/read-log.mjs --list` — see [Discord transcripts](#discord-transcripts) |
| **Writing anything a human reads** (issue replies, commits, PR bodies, release notes, docs) | [docs/WRITING-STYLE.md](docs/WRITING-STYLE.md) — no emojis, no AI tells, honest confidence |
| Full feature catalogue (100+) | [FEATURES.md](FEATURES.md) |
| Engine internals, CLI, request JSON, generation modes | [engine/docs/ARCHITECTURE.md](engine/docs/ARCHITECTURE.md) |
| **Training system** (dataset→preprocess→LM/DiT training→audition; ace-train, FSQ, ggml training gotchas) | [docs/TRAINING.md](docs/TRAINING.md) |
| Writing a Lua plugin | [docs/PLUGINS.md](docs/PLUGINS.md) |
| Build / install / releases | [README.md](README.md) |
| Cutting & publishing a release (agent runbook) | [docs/RELEASING.md](docs/RELEASING.md) |
| Internal design/investigation docs (perf, adapters, upstream sync, feature designs) | `docs/plans/` *(gitignored, local-only)* |
| In-app assistant behaviour/KB | [server/src/data/assistant-knowledge.md](server/src/data/assistant-knowledge.md) |

> **Doc convention:** committed contributor-facing docs = `README.md`, `FEATURES.md`, `docs/PLUGINS.md`, `engine/docs/ARCHITECTURE.md`. Internal planning/investigation docs live in `docs/plans/`, which is **gitignored** (local only). This file (`CLAUDE.md`) is committed.

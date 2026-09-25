---
name: generation-request-flow
description: Traces the full life of a music generation request (UI form → Node job queue → engine LM → synth → SQLite song row) and every place params can silently drop, especially the LM echo sideband gotcha. Use when adding/plumbing a generation parameter, debugging "param set in UI but engine uses default", touching /api/generate, translateParams, aceClient, hot-step-server.cpp request parsing, the LM/synth request rebuild, or understanding task modes (text2music/cover/repaint/lego/extract/complete), the serial job queue, and stall/timeout watchdogs.
---

# Generation Request Lifecycle & Param Plumbing

This skill maps the end-to-end path of a generation request in HOT-Step CPP and catalogs every point where a parameter can be silently lost. Most historical bugs in this area are **param-plumbing bugs**: a value set in the UI never reaches the C++ sampler, and nothing errors.

Terms used below:
- **DiT** — the diffusion transformer that turns audio codes + text embeddings into audio latents (the "synth" phase). Pipeline is LM → DiT → VAE.
- **LM** — the language model phase that generates metadata (bpm, duration, key), a chain-of-thought (CoT) caption, lyrics, and `audio_codes` for the DiT.
- **aceReq** — the Node-side request object (interface `AceRequest`, [server/src/services/aceClient.ts:38](../../../server/src/services/aceClient.ts)) sent to the C++ engine. Snake_case, produced from camelCase UI params by `translateParams()`.
- **Sideband** — HOT-Step-only fields (solver, scheduler, guidance, adapter knobs, …) that travel *alongside* the upstream C++ `AceRequest` struct in the same JSON body but are parsed separately into `ServerFields` (engine) because they don't exist in the upstream struct.

## When to use this skill

- Adding a new generation parameter (UI knob → engine behavior).
- Debugging: a param is set in the UI but the engine logs/behavior show the default.
- Modifying `server/src/routes/generate.ts`, `translateParams.ts`, `aceClient.ts`, or `engine/tools/hot-step-server.cpp` request handling.
- Understanding generation modes (`text2music`, `cover`, `repaint`, `lego`, …) or the job queue/watchdogs.

## Golden rules

1. **NEVER rebuild the post-LM synth request by copying the LM echo or by whitelisting fields to carry across.** The engine's `/lm` response only contains fields in the C++ `AceRequest` struct — every sideband field vanishes. Always rebuild as `{ ...aceReq, <the 7 LM output fields> }`. WHY: every field-copy whitelist ever written here went stale and silently dropped params; commits `8ea519b` and `168dcb5` fixed real bugs of exactly this class. Full detail in the next section.
2. **When adding a float field to engine JSON parsing, use the int-safe `get_num` lambda, not bare `yyjson_get_real()`.** WHY: `yyjson_get_real()` returns **0 for JSON integers**, and JavaScript serializes `2.0` as `2` — so a whole-number value silently becomes `0.0`. See [hot-step-server.cpp:700-705](../../../engine/tools/hot-step-server.cpp).
3. **C++ changes rebuild ONLY via `.\dev-rebuild.bat` at repo root** — never `engine\build.cmd` directly, under any circumstances (you cannot reliably tell whether the app is running; Node auto-respawns ace-server on crash → infinite respawn + file-lock loop). Never `cmake --build . --clean-first` (20+ min CUDA recompile).
4. **Type-check TypeScript with `npx tsc --noEmit`; do not `npm run build` during dev.**
5. **After any upstream sync, run `powershell -File engine\verify-hooks.ps1`.** WHY: `pipeline-synth-ops.cpp` losing its `hot-step-sampler.h` include is **SILENT** — everything compiles but all solvers/schedulers/guidance/sideband go dead (they read `g_hotstep_params`, which nothing consumes anymore).
6. **Git: all work on `master`, stage explicit paths (never `git add -A`), push only with explicit user approval.** Any pushed `v*` tag triggers a full multi-platform CI release build.
7. **Params consumed as GPU tensors inside `hot-step-sampler.h` (masks, encoder states, constants) must be re-uploaded EVERY step** — the GGML scheduler aliases input tensor buffers as scratch space; uploading once leaves them garbage from step 2 onward ([hot-step-sampler.h:591-601](../../../engine/src/hot-step-sampler.h), re-upload sites :936, :1091). Never hoist an upload out of the step loop as an "optimization".
8. **Never delete generated audio artifacts (`server/data/audio/*`, test outputs) based on your own judgment of quality** — the user verifies results by ear.
9. **Node 18–22 LTS only** — Node 24+ breaks dependencies (`engines` field enforces `<24`).

## THE LM ECHO SIDEBAND GOTCHA (read this before touching generate.ts)

**Ground truth from the departing lead engineer, verified line-by-line in code:**

Server-side params (`ServerFields`) do **not** survive the `/lm` round trip — the engine echoes back only what it knows. When building the synth request after the LM step, rebuild it from the **original `aceReq` plus the LM-returned fields ONLY**. Never maintain a whitelist of fields to copy across — every whitelist has gone stale and silently dropped params. Commits `8ea519b` and `168dcb5` fixed bugs of exactly this class.

**Mechanics of why:**
- `/lm`'s result body is `request_to_json()` serialized from the **C++ `AceRequest` struct** ([hot-step-server.cpp:869](../../../engine/tools/hot-step-server.cpp), struct at [engine/src/request.h:29](../../../engine/src/request.h)).
- Fields that exist only in `ServerFields` ([hot-step-server.cpp:557-604](../../../engine/tools/hot-step-server.cpp)) — `vae_model`, `emb_model`, `scheduler`, `guidance_mode`, `adapter_group_scales`, `adapter_mode`, `adapter_runtime_quant`, `rebase_source`/`rebase_beta`, `apg_*`, `stork_substeps`, `denoise_*`, `plugin_params`, `seed_strength`, `evict_lm`, `vae_chunk`, `batch_cfg`, and more — are **not in the C++ struct**, so the LM echo cannot contain them. Copy the echo forward and they all silently drop.

**The correct pattern — it exists in exactly two places, keep them identical:**

```ts
// generate.ts:319-328 (fresh LM) and generate.ts:237-246 (LM-cache hit)
lmResults = lmResults.map(lmOut => ({
  ...aceReq,                       // CURRENT request = every sideband field survives
  audio_codes: lmOut.audio_codes,  // then ONLY the 7 LM-generated fields:
  caption: lmOut.caption,
  lyrics: lmOut.lyrics,
  bpm: lmOut.bpm,
  duration: lmOut.duration,
  keyscale: lmOut.keyscale,
  timesignature: lmOut.timesignature,
}));
```

The **only** whitelist allowed is the LM-output list (those 7 fields). The base must always be a spread of the live `aceReq`. If you add a new server-side param you do **not** touch this rebuild code — that is the whole point of the pattern.

## The pipeline at a glance

```
UI form (CreatePanel + Zustand globalParamsStore)
  → GenerationParams JSON, camelCase          [ui/src/types.ts:80]
  → POST /api/generate (Node :3001)           [server/src/routes/generate.ts:1355]
  → in-memory job Map + strictly serial queue [generate.ts:44 statuses]
  → translateParams(): camelCase → snake_case [server/src/services/generation/translateParams.ts:11]
  → POST http://:8085/lm (unless skipped)     [aceClient.ts:319; engine handle_lm hot-step-server.cpp:888]
  → REBUILD synth req: {...aceReq, 7 LM fields} [generate.ts:319-328]  ← THE SIDEBAND GOTCHA
  → POST http://:8085/synth (JSON or multipart) [aceClient.ts:332 / :347; engine handle_synth :1442]
  → engine: parse_server_fields() + request_parse_json() on the SAME body [hot-step-server.cpp:606]
  → synth_worker: ServerFields → g_hotstep_params global [hot-step-server.cpp:949, copy at :1132-1194]
  → synth_batch_run: TextEnc → DiT (hot-step-sampler.h reads g_hotstep_params) → VAE → WAV/MP3
  → Node saves data/audio/<uuid>.wav|mp3      [generate.ts:840]
  → auto-trim → post-processing chain → INSERT INTO songs [generate.ts:1057-1094, :1152]
```

Engine port 8085 is set in [server/src/config.ts:102](../../../server/src/config.ts) (`ACESTEPCPP_PORT`). Node job statuses: `pending | lm_running | synth_running | saving | succeeded | failed | cancelled` (generate.ts:44).

## Procedure: adding a new generation parameter end-to-end

A new param must be threaded through **five** layers. Missing any one = silent default.

1. **UI state:** add to `ui/src/stores/globalParamsStore.ts` (persisted per-field to localStorage under an `hs-*` key) and include it in `getGlobalParams()` (line 324). Watch the conditional gating there — many params are sent only when their mode is active (e.g. `apgMomentum` only when `guidanceMode === 'apg'`, line 396). Add the field to `GenerationParams` in `ui/src/types.ts:80`. The visible controls for engine knobs live in `ui/src/components/global-bar/` (`GlobalParamBar.tsx`, `GenerationDropdown.tsx`) — grep an existing knob's setter to find the right spot.
2. **Node translate:** map camelCase → snake_case in `server/src/services/generation/translateParams.ts`. Use `!== undefined` guards for fields where `0`/`false` is meaningful — a truthy guard (`if (params.foo)`) silently drops legitimate zeros.
3. **Node wire type:** add the snake_case field to the `AceRequest` interface in `server/src/services/aceClient.ts:38` (this interface is the Node-side canonical superset: C++ struct fields + sideband).
4. **Engine parse:** add to `ServerFields` (hot-step-server.cpp:557) and `parse_server_fields()` (:606). For floats, use the `get_num` lambda pattern — Golden rule 2. **Note: `get_num` is declared mid-function at :703** — add your parse AFTER that line (e.g. next to the denoise fields at :763), or inline the `is_real ? get_real : get_int` ternary like `rebase_beta` does (:663); a parse placed above :703 fails to compile (use before declaration).
5. **Engine global:** copy into `g_hotstep_params` in the block at hot-step-server.cpp:1132-1194 (note: this copy happens **before** `ace_synth_load()` because adapter merge reads group scales at merge time), and add the field to `HotStepParams` in `engine/src/hot-step-params.h`. Consume it in `hot-step-sampler.h` (or wherever).
6. **Do NOT touch** the LM-echo rebuild in generate.ts — the `...aceReq` spread carries your field automatically.
7. Rebuild: `.\dev-rebuild.bat` (C++), `npx tsc --noEmit` (TS). Verify arrival — **but beware the `[DIAG]` trap**: the `[DIAG] /synth body (first 300 chars)` line (hot-step-server.cpp:1568) truncates at 300 chars, and the body starts with caption then lyrics, so with any realistic caption your sideband field will NOT appear in the visible prefix. **A field absent from the DIAG line is NOT evidence it was dropped.** Reliable checks: add a temporary `fprintf` of your parsed value in `parse_server_fields` (or next to the `[Server] HOT-Step params:` summary at hot-step-server.cpp:1197), or log the full `aceReq` server-side before `submitSynth`. For any new UI control, ask the user to visually confirm it — do not use the browser agent for visual verification (API checks are fine).

## Key files

| Path | Role |
|---|---|
| `ui/src/stores/globalParamsStore.ts` | Zustand store for all engine knobs; `getGlobalParams()` (:324) is the single assembly point with mode-conditional gating |
| `ui/src/types.ts:80` | `GenerationParams` — canonical camelCase UI shape (~120 fields, incl. post-processing-only fields that never reach the engine) |
| `ui/src/App.tsx:502` | `handleGenerate` — merges global params + Create-tab content, enqueues via `enqueueSimpleGen` |
| `ui/src/services/api.ts:148-153` | `generateApi` — POST `/generate`, status polling, cancel |
| `server/src/routes/generate.ts` | The whole Node lifecycle: queue, LM phase, sideband rebuild, trigger re-injection, synth loop, watchdogs, DB insert |
| `server/src/services/generation/translateParams.ts` | camelCase → `AceRequest` snake_case; pure function |
| `server/src/services/generation/lmCache.ts` | LM output cache — stores only the 7 LM fields, keyed by LM-affecting params only (sha256/16, LRU max 20) |
| `server/src/services/aceClient.ts` | Wire types + HTTP client to engine :8085; timeouts 15 s submit / 30 s poll / 300 s result (:17-19) |
| `engine/tools/hot-step-server.cpp` | Engine HTTP server: `handle_lm` (:888), `handle_synth` (:1442), `ServerFields` (:557), `parse_server_fields` (:606), `synth_worker` (:949) |
| `engine/src/hot-step-params.h` | `HotStepParams` sideband global (`g_hotstep_params`) read by `hot-step-sampler.h` inside the DiT; also `AdapterGroupScales`/`AdapterSpec`/`AdapterSection` |
| `engine/src/request.h:29` | Upstream C++ `AceRequest` struct — fields here DO survive the `/lm` echo |
| `server/src/config.ts:102` | Engine port (8085 default) |

## Every place params can silently drop (checklist)

1. **The LM echo** (see gotcha section above) — THE classic.
2. **yyjson int-vs-float trap** — fields parsed with bare `yyjson_get_real()` (e.g. `apg_momentum` :667, `apg_norm_threshold` :670, `seed_strength` :674, `beat_stability`, `dcw_scaler`, `latent_shift`, `cfg_cutoff_ratio`, `cache_ratio`) turn whole-number JSON values into `0.0`. Only `adapter_group_scales`, `rebase_beta`, and a few others use the int-safe `get_num` lambda (:703).
3. **UI-side gating** in `getGlobalParams()` (globalParamsStore.ts:324-458) — params conditionally `undefined` based on mode/toggles. A stale `advancedAdapters=false` suppresses a persisted adapter stack (:334). If a param "never arrives," check the gate here first.
4. **translateParams falsy guards** — `if (params.bpm)` / `if (params.duration)` / `if (params.inferenceSteps)` intentionally drop 0 ("LM fills it in"), but a new field copied with a truthy guard drops legitimate `0`/`false`. Use `!== undefined` where 0 matters.
5. **Trigger word destroyed by the LM's CoT caption** — `translateParams` injects it (:136-144), the LM's chain-of-thought caption replaces the whole caption, and generate.ts re-injects it (:366-403) **only when** `triggerPlacement` AND `loraPath` are both set; otherwise it logs a `[Trigger]` WARNING and skips (:399-403). Debugging "adapter trigger missing from caption" starts here.
6. **LM cache staleness** — only the 7 LM fields are cached/restored, keyed by `computeLmCacheKey` (lmCache.ts:23-46). If you add an LM-affecting param, it MUST go into that key or stale audio codes get served for new settings. DiT/adapter params are deliberately excluded from key and value.
7. **Multipart array truncation** — `submitSynthMultipart` sends only `request[0]` if given an array (aceClient.ts:376).
8. **Engine registry fallbacks** — unknown `vae_model`/`emb_model` fall back to the default with only a stderr warning, not an error; unknown `rebase_source` is skipped with a warning. Unknown adapter name (after absolute-path fallback) DOES fail the job.
9. **Duplicated fields** — `dcw_*`, `latent_shift`/`latent_rescale`, `custom_timesteps`, `infer_method` exist in BOTH the C++ `AceRequest` struct (request.h) and `ServerFields`. The server sampler reads the **`g_hotstep_params` (ServerFields) copy**; keep both parse paths in sync or CLI-tool vs server behavior diverges. Also note naming: the C++ struct has `vae` (request.h:142, used by CLI/`/vae`) while the server sideband uses `vae_model`.
10. **Upstream sync hook loss** — silent solver/guidance/sideband death; run `engine\verify-hooks.ps1`.
11. **LM-skip paths** (Song Builder, cover tasks, `skipLm`) bypass the LM entirely — anything depending on LM-filled metadata must handle 0/empty. When `skipLm` on non-cover: defaults filled (bpm 120, duration 120, 'C major', '4') at generate.ts:221-224.

## Generation modes (code truth)

Engine `task_type` (request.h:109, docs in engine/docs/ARCHITECTURE.md): `text2music` (default), `cover` (FSQ roundtrip, free reinterpretation), `cover-nofsq` (clean latents, faithful remix), `repaint` (region regenerate via `repainting_start`/`repainting_end`; negative start = outpaint before), `lego` (new layered track; base model only), `extract` (stem isolation; base only), `complete` (fill mix around a stem; base only). LM modes: `generate` / `inspire` / `format`.

**Discrepancy — code wins:** ARCHITECTURE.md:360 says `lego` uses the LM, but Node's `isCoverTask` list `['cover','cover-nofsq','repaint','lego','extract']` (generate.ts:208) means the Node pipeline **never runs the LM for lego**. `complete` is absent from that list, so it *would* run the LM if submitted via `/api/generate`. That list is Node-side policy, not the engine's task registry.

## Failure signatures

| Symptom | Cause / where to look |
|---|---|
| Param set in UI, engine behaves as default | Sideband dropped post-LM (gotcha), UI gate (checklist #3), or whole-number float → 0 (checklist #2). Do NOT trust the 300-char `[DIAG] /synth body` prefix as proof of absence (procedure step 7) — add a temporary parse-site fprintf instead |
| "Generation stalled — no progress for Ns" | 2-min no-progress watchdog (generate.ts:107,:128); engine wedged |
| "Generation timed out (N min limit)" | Wall-clock watchdog, `generationTimeoutMinutes` clamped [5,120], default 45 (generate.ts:139) |
| "Generation failed on ace-server" | Engine job status 2 — read `logs\<session>\generations\gen_<uuid>_*.log`, then `ace_engine.log` for `[Server] Adapter not found` / synth load failure |
| All solvers/schedulers behave like euler after an upstream sync | Lost `hot-step-sampler.h` hook in `pipeline-synth-ops.cpp` — SILENT; run `engine\verify-hooks.ps1` |
| Trigger word absent from final caption | CoT caption replaced it and re-injection gate unmet — grep gen log for `[Trigger]` WARNING |
| Stale LM output despite changed DiT/adapter params | Correct behavior (LM cache keys only LM params); set `cacheLmCodes: false` to force a fresh LM run |
| 503 "Engine not ready" on submit | Engine bootstrap unfinished (generate.ts:1355 handler checks `engineReady`) |
| Song Builder tracks get quieter each append | wav16 peak-renormalization — fixed by forced `wav32` for builder/PP (generate.ts:549-566) |

## Institutional knowledge

- **VALIDATED — LM echo sideband gotcha:** described above; the rebuild pattern at generate.ts:237-246 and :319-328 is the fix for real shipped bugs (commits `8ea519b`, `168dcb5`).
- **VALIDATED — yyjson int trap:** documented in-code at hot-step-server.cpp:700-705; use `get_num`.
- **VALIDATED — copy-before-load ordering:** `g_hotstep_params` must be populated **before** `ace_synth_load()` because adapter merge reads group scales at merge time (comment near hot-step-server.cpp:1128).
- **VALIDATED — zombie plumbing:** regional self-attn isolation was reverted (`0f3bf6d` → `ee041e1`, "broke musical continuity"), but `adapter_section_isolation` is still parsed (:656) and copied to `g_hotstep_params` (:1145) while nothing in `engine/src/` consumes it. It is a **dead knob** — don't mistake its plumbing for a working feature, and do NOT re-implement the consumer: per-section self-attn masking breaks musical continuity across section boundaries; any retry needs a cross-section-coherence design first.
- **VALIDATED — auto-shift fork behavior:** when `shift == -1`, this fork computes adaptive shift from duration + steps (hot-step-server.cpp:1244-1250). ARCHITECTURE.md's "shift 0 = auto" text describes upstream, not this fork.
- **VALIDATED — DiT batch clamp:** `synth_batch_size` expands per-seed variants; total clamped to DiT max **9** (hot-step-server.cpp:964-973). Node avoids engine `multipart/mixed` results by submitting one track per `/synth` call.
- **UNVALIDATED / HYPOTHESIS — basin re-base:** the β·(S−T) nudge for cross-base LoKR adapters is plumbed end-to-end (globalParamsStore:381 → translateParams:126 → ServerFields:575 → g_hotstep_params:1168) but its effectiveness is unproven. **Invariant (VALIDATED): the nudge applies to the FIRST adapter of a stack only** (`dit.h:565-579` — `rb_src = (si == 0) ? rebase_source : nullptr`); one nudge per stack, never per adapter. Any plumbing change that makes rebase per-adapter is a bug (issue #72).

## Useful commands (PowerShell)

```powershell
npx tsc --noEmit                                    # type-check server/UI (never npm run build in dev)
.\dev-rebuild.bat                                   # the ONLY sanctioned way to rebuild C++, always
powershell -File engine\verify-hooks.ps1            # after any upstream sync
Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1   # newest session log folder
Invoke-RestMethod http://localhost:3001/api/generate/queue                   # queue health
Invoke-RestMethod -Method Post http://localhost:3001/api/generate/reset-queue # unwedge the queue
```

## Deeper reading

- [reference.md](reference.md) (same folder) — full node-tier walkthrough, ServerFields field list, endpoints/watchdogs, LM cache internals, translateParams special cases.
- [engine/docs/ARCHITECTURE.md](../../../engine/docs/ARCHITECTURE.md) — engine internals, task types, request JSON. **Caveat:** its request-JSON reference predates several sideband fields; the code (`aceClient.ts` `AceRequest` interface + `ServerFields`) is the authoritative schema.
- [docs/dev/plugins-authoring.md](../../../docs/dev/plugins-authoring.md) — Lua solver/scheduler/guidance plugins (how `plugin_params` are consumed).
- `docs/plans/` — internal design docs (per-section masking, upstream sync workflow). **Gitignored, local-only — may be absent on a fresh clone.**
- [CLAUDE.md](../../../CLAUDE.md) — build/git rules and log layout.

---
name: lua-plugin-authoring
description: Explains how to write, test, and debug HOT-Step Lua plugins (solvers, schedulers, guidance modes, postprocess) including the apg()/post_step() API and UI param flow. Use when adding or modifying a sampling solver, noise scheduler, CFG/guidance mode, or VAE-decode postprocess plugin, or when a plugin fails to load, has no effect, or produces noise.
---

# Writing Lua plugins for HOT-Step CPP

Solvers, schedulers, guidance modes, and postprocess (VAE-decode replacement) are **Lua 5.4 plugins** loaded by the C++ engine at startup. Adding one = drop a `.lua` file in the right directory and restart the app. **No C++ rebuild.**

Glossary (used throughout):
- **DiT** — the diffusion transformer that denoises audio latents over N steps. A **solver** decides how the latent `xt` advances each step given the model's predicted velocity `vt`. A **scheduler** decides the timestep values. A **guidance** mode combines the conditional and unconditional model predictions (classifier-free guidance, CFG). A **postprocess** plugin replaces the built-in tiled VAE (variational autoencoder) latent-to-audio decode.
- **FloatArray** — a zero-copy Lua userdata view over a raw C++ `float*`. It is **0-indexed** (`xt[0] … xt[n-1]`), unlike normal Lua tables. `#xt` returns its length. Defined in `engine/src/lua-plugin.h:32-87`.
- **`params`** — a Lua global table injected before every plugin call, holding the values the user set in the UI for this plugin's declared parameters.

## Golden rules

1. **The old approach of editing `engine/src/dit-sampler.h` is OBSOLETE.** All sampling routes through `engine/src/hot-step-sampler.h` (included by upstream `pipeline-synth-ops.cpp`). Adding a solver/scheduler/guidance = write a `.lua` plugin. WHY: C++ edits to the old sampler are dead code and waste a rebuild cycle; losing the `hot-step-sampler.h` include during an upstream sync kills every plugin (a linker sentinel `hotstep_sampler_linked_` at `hot-step-sampler.h:1309-1314` turns that into a link error; `engine/verify-hooks.ps1` also checks it).
2. **"Hot-loadable" means no rebuild, NOT live reload.** Plugins are scanned once at ace-server startup (`engine/tools/hot-step-server.cpp:2676`). After editing a `.lua` file you must restart the app. `POST /api/plugins/reload` only clears the Node server's 60-second cache of the plugin list (`server/src/routes/plugins.ts:31-35`) — it does not re-read files.
3. **Never kill `ace-server.exe` directly.** The Node server auto-respawns it on crash, causing an infinite respawn + file-lock loop. The working restart is `Invoke-RestMethod -Method Post http://localhost:3001/api/shutdown/restart` (the shutdown router is mounted at `/api/shutdown` — `server/src/index.ts:77` — so plain `/api/restart` 404s; the loop wrapper relaunches Node, which respawns ace-server and rescans plugins). Alternatively `dev-rebuild.bat` (clean shutdown; its compile is a no-op for Lua-only changes) **then** start again with `dev.bat` — note `dev-rebuild.bat` does NOT relaunch, and re-running `dev.bat` while the app is still up just spawns port-conflicting duplicates. If plugin work escalates into editing any `engine/src/` C++ file (e.g. a new bridge function in `lua-plugin.h`), rebuild via `dev-rebuild.bat` immediately — never `engine/build.cmd` directly, and never `cmake --clean-first` (20+ min CUDA recompile; for stale `.obj` issues delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`).
4. **FloatArrays are 0-indexed; Lua tables are 1-indexed.** An out-of-range index raises a Lua error which aborts the call — the engine prints the error and continues, so `xt` never advances and the "successful" generation is pure noise. WHY: this is the single most common plugin bug and it fails silently from the user's perspective.
5. **Put your plugins in repo-root `plugins/<type>/`, not `engine/plugins/<type>/`.** Both are scanned (engine tier first, then repo root — `engine/src/lua-plugin-registry.h:38-50`); the repo-root tier keeps built-ins clean. Duplicate `name`s: **first loaded wins** (so you cannot shadow a built-in).
6. **In guidance plugins, always route the base combine through `apg()`.** Raw `uncond + w*(cond-uncond)` produces audible artifacts; the native `apg()` bridge adds momentum smoothing, perpendicular projection, and norm thresholding. Customize the *scale* or post-process `result` instead (see `engine/plugins/guidance/cfg_pp.lua`).
7. **Do not visually verify the UI with a browser agent** — ask the user. API checks (`/api/plugins`) are fine.
8. **Never delete generated test audio, even output you believe is noise or broken** — the user verifies plugin results by ear and compares control vs. plugin runs. Leave every test generation in place and ask the user to listen.

## Where plugins live and how they load

Scanned at startup, in order (`lua-plugin-registry.h:35-55`):

1. `engine/plugins/{solvers,schedulers,guidance,postprocess}/` — built-ins
2. `<repo-root>/plugins/{solvers,schedulers,guidance,postprocess}/` — user/community drop-ins

Missing directories are skipped silently (e.g. `engine/plugins/postprocess/` does not exist — the only shipped postprocess plugin is `plugins/postprocess/md_audio_tiled.lua`).

Loader rules (`lua-plugin-registry.h:157-226`):
- Only `.lua` files; sorted for deterministic order.
- **Companion-file exclusion**: filename stems containing `_constants`, `_math`, or `_data` are never loaded as plugins — they are `require()` targets (e.g. `beta_math.lua`, `stork4_constants.lua`). Stems ending `_core` are skipped only if a sibling without `_core` exists in the same dir (`md_audio_tiled_core` is skipped; `storm_sampler_core` loads as a plugin because no `storm_sampler.lua` exists).
- Plugin **type is detected by which global table the file defines**: `solver`, `scheduler`, `guidance`, or `postprocess` (`lua-plugin.h:348-386`). Wrong table for the directory → "declares wrong type, skipping". Empty/missing `name` → skipped. Lua syntax error → `[Plugins] ERROR loading <path>: <message>` on stderr, plugin absent.
- Startup log (in `logs/<session>/ace_engine.log`): one line per plugin, then `[Plugins] Loaded N solvers, N schedulers, N guidance, N postprocess`.

Sandbox (`lua-plugin.h:180-195, 316-343`): `math`, `string`, `table`, `print`, `pairs`, `ipairs`, `tonumber`, `tostring`, `require` are available. `os`, `io`, `debug`, `dofile`, `loadfile` are removed. `require()` searches ONLY the plugin's own directory; C modules are blocked. Each plugin gets its **own `lua_State` that lives for the whole process** — file-level `local` variables persist across steps AND across generations (that is how stateful solvers work, and why they must self-reset; see Golden rule of state below).

## Procedure: write and test a plugin

1. Create `plugins\<type>\my_thing.lua` (repo root). Start by copying the closest template (see the table in "Worked templates" below).
2. Restart the app. If it is running: `Invoke-RestMethod -Method Post http://localhost:3001/api/restart` (relaunches Node → respawns ace-server → rescans plugins). If it is fully stopped: start with `dev.bat`. Do NOT re-run `dev.bat` over a running app (port conflicts; the old ace-server keeps serving the stale plugin list), and never kill `ace-server.exe` yourself.
3. Confirm it loaded:
   ```powershell
   $s = Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\]'
   ```
   Expect a per-plugin line naming yours. A load failure prints `[Plugins] ERROR loading <path>: <lua error>` here.
4. Confirm the API sees it:
   ```powershell
   Invoke-RestMethod http://localhost:3001/api/plugins | ConvertTo-Json -Depth 6
   ```
   (or engine-direct `http://localhost:8085/plugins`). If you edited metadata and the list looks stale, bust the Node cache: `Invoke-RestMethod -Method Post http://localhost:3001/api/plugins/reload` — cache only; file changes still need an app restart.
5. Generate once with known-good settings (solver `euler` + scheduler `linear`) as a control, then swap in your plugin. A generation selects plugins via request JSON fields `infer_method` (solver — note the non-obvious name), `scheduler`, `guidance_mode`, and `postprocess_plugin` (UI params `inferMethod` / `scheduler` / `guidanceMode` / `postprocessPlugin`, mapped in `translateParams.ts:58-60`). Runtime confirmation lines in `ace_engine.log`:
   - `[DiT] Solver: <display> (<name>, N NFE/step, order K)` (`hot-step-sampler.h:503`)
   - `[DiT] Guidance: <display> (<name>) [native APG] [post_step]` (`hot-step-sampler.h:525`)
   - `[DiT] Custom schedule: <display> (<name>), shift=X` (`engine/src/sampler-schedule.h:151`)
   - `[Postprocess] Using plugin '<name>' for VAE decode` (`pipeline-synth-ops.cpp:1764`)
6. Debug with `print()` — it goes to engine stdout, captured in `ace_engine.log` and the matching `logs/<session>/generations/gen_*.log`. Runtime Lua errors surface as `[Plugins] ERROR in solver '<name>' step(): <traceback>` (also `schedule()`/`guide()`/`post_step()`/`process()` variants):
   ```powershell
   Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\] ERROR'
   ```

## The four plugin types — contracts

All contracts below are verified against `engine/src/lua-plugin.h` (the single source of truth). Full detail including full-loop solvers, `post_step()`, and postprocess internals: [reference.md](reference.md).

### 1. Solver — `plugins/solvers/`

```lua
solver = {
    name = "my_solver", display = "My Solver", description = "...",
    nfe = 1, order = 1,          -- informational (shown in logs/UI)
    needs_model = false,          -- true => step() gets model_fn + vt_buf
    stateful = false, stochastic = false,  -- informational
    -- owns_loop = true          -- advanced: define sample() instead of step(); see reference.md
    params = { ... },             -- optional; see "Declared UI params"
}

-- single-eval (needs_model = false): 5 args
function step(xt, vt, t_curr, t_prev, n)
    -- xt: mutable FloatArray (modify IN PLACE); vt: READ-ONLY FloatArray
    -- t_curr = current t; t_prev = the NEXT (lower) t despite the name
    local dt = t_curr - t_prev   -- positive
    for i = 0, n - 1 do xt[i] = xt[i] - vt[i] * dt end
end
```

Multi-eval (`needs_model = true`) gets 7 args: `step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)`. `model_fn(xt_arr, t_val)` runs a full CFG'd forward pass and writes guided velocity into `vt_buf` (a live-memory FloatArray). The `vt` arg is a **snapshot** taken before your step — read the original velocity from `vt`, read fresh model results from `vt_buf` (`hot-step-sampler.h:1179-1195`; this separation fixed the historical "Heun silently becomes Euler" bug).

Globals injected per call: `step_index` (0-based), `batch_n`, `n_per` (elements per batch item), `params` (`lua-plugin.h:451-457`). `n` = whole flattened batch (`batch_n * n_per`).

Engine invariants: the **final step never calls `step()`** — the engine computes `output = xt - vt * t_curr` itself (`hot-step-sampler.h:1171-1175`). DCW correction and repaint injection are applied after your step by the engine — do not reimplement them.

**Stateful solvers must self-reset**: `lua_State` persists across generations, so reset file-locals when `step_index == 0` (see `engine/plugins/solvers/unipc.lua:153-158`). Checking only "did `n` change" is insufficient — two same-length generations back-to-back will bleed state (`plugins/solvers/md_pingpong_simple.lua:242-258` documents the explosion this causes).

### 2. Scheduler — `plugins/schedulers/`

```lua
scheduler = { name = "my_sched", display = "My Sched", description = "...", params = {...} }

function schedule(output, num_steps, shift)
    -- write num_steps DESCENDING t values (1.0 -> ~0.0) into output (FloatArray)
    -- do NOT append a trailing 0 — the engine handles the final x0 step
    for i = 0, num_steps - 1 do output[i] = 1.0 - i / num_steps end
    -- apply the standard shift warp yourself (every shipped scheduler does):
    if shift ~= 1.0 then
        for i = 0, num_steps - 1 do
            local t = output[i]
            output[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end
end
```

`shift` is NOT taken from the UI — it is **back-calculated** from the upstream schedule's second timestep and clamped (`<0.5 → 1.0`, `>10 → 3.0`) at `sampler-schedule.h:61-74`. Your plugin only runs when the request names a scheduler; empty = upstream default shifted-linear. A `custom_timesteps` CSV in the request overrides all schedulers (`hot-step-sampler.h:91-103`). Composite syntax `composite:A+B:crossover:split` blends two scheduler plugins engine-side (`sampler-schedule.h:79-133`). Name aliases: `karras` → `sgm_uniform`; `power:4.00` falls back to prefix `power` (`lua-plugin-registry.h:72-91`).

### 3. Guidance — `plugins/guidance/`

```lua
guidance = { name = "my_guide", display = "My Guide", description = "...", params = {...} }

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- pred_cond / pred_uncond: READ-ONLY FloatArrays [Oc*T]; result: mutable [Oc*T]
    -- customize the scale, then ALWAYS combine via apg():
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
```

Called per batch element per model eval. Globals: `step_idx` (note: guidance uses `step_idx`; solvers use `step_index`), `total_steps`, `dt`, `t_curr`, `params`.

`apg()` is a C bridge over native `apg_forward()` (`lua-plugin.h:643-671`), registered lazily on the first `guide()` call (`lua-plugin.h:697-704`). Calling it at file top level therefore fails at load with Lua's `attempt to call a nil value (global 'apg')` and the plugin is skipped. Calling it from `post_step()` does NOT error — the `_apg_mbuf` global set before each `guide()` call (`lua-plugin.h:688-689`) is never cleared, so it silently reuses the momentum buffer of whichever batch element was guided last (wrong/stale state). Either way: only call `apg()` inside `guide()`.

**Native bypass gotcha**: when the selected guidance is named exactly `"apg"`, the engine takes the native C++ path and never calls Lua `guide()` (`use_apg_native`, `hot-step-sampler.h:524`). Editing `apg.lua`'s body does nothing; it is a documented fallback (`apg.lua:4-5`). To experiment, copy it under a new `name`.

Advanced: define a global `post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)` and the engine calls it after every solver step except the last, only while CFG is active. Each `eval_cond`/`eval_uncond` call is a **full model forward pass** — expensive. Details + gating conditions: [reference.md](reference.md). Real user: `engine/plugins/guidance/cfg_mp.lua:47`.

### 4. Postprocess — `plugins/postprocess/` (replaces tiled VAE decode)

```lua
postprocess = { name = "my_pp", display = "My PP", params = {...} }

function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    -- latents: plain Lua TABLE (1-indexed, NOT a FloatArray), channel-major [C_lat=64, W]
    -- B is always 1 (engine iterates batch items); upscale_factor = 1920 samples/latent frame
    -- vae_decode_fn(latent_table, T_latent) -> audio_table, T_audio
    -- MUST return: audio_table (1-indexed, [2 * T_audio]), T_audio
end
```

Selected per request via JSON field `postprocess_plugin` (`engine/src/request.cpp:177-178`; UI param `postprocessPlugin` → `server/src/services/generation/translateParams.ts:181-183`). Unknown name or `T_audio <= 0` → warning + automatic fallback to the built-in tiled decoder (`pipeline-synth-ops.cpp:1730-1734, 1792-1804`). This path uses Lua tables with transposes on both sides — deliberately not zero-copy. Reference implementation: `plugins/postprocess/md_audio_tiled.lua` + `md_audio_tiled_core.lua` (the `require()`-a-`_core`-module pattern). Full contract: [reference.md](reference.md).

## Declared UI params and how values reach your plugin

Schema (extracted at `lua-plugin.h:230-291`) — four types: `slider` (`default`/`min`/`max`/`step`), `select` (`default` + `options` as `{value=,label=}` tables or bare strings), `toggle` (bool `default`), `text` (string `default`). Common fields: `key`, `label`, `hint`, `visible_when = { key = "...", equals = "..." }` (string-compared against a sibling param's current value). Field reference + JSON shape: [reference.md](reference.md).

Flow: engine `GET /plugins` → Node proxy `GET /api/plugins` (60 s cache, empty lists on engine failure) → UI `PluginControls` (`ui/src/components/global-bar/PluginControls.tsx`) stores flat strings in `{ "pluginName:paramKey": "value" }`, persisted under localStorage key `hs-pluginParams` (`ui/src/stores/globalParamsStore.ts:149`) → request field `plugin_params` (`translateParams.ts:176-178`) → engine parses (`[DIAG] Parsed N plugin_params` in the log) → before every Lua call, `lua_inject_params` (`lua-plugin.h:410-436`) filters by `"<plugin.name>:"` prefix and sets a fresh `params` global. Coercion: numeric string → number, `"true"`/`"false"` → boolean, else string.

Param traps:
- **`params` and any key may be nil** (untouched params are absent from the map). Always default: `local x = (params and params.x) or 0.5`.
- **The `or` idiom is WRONG for toggles defaulting true**: `(params and params.rms_servo) or true` is `true` even when the user turned it OFF (`false or true == true`). `md_pingpong_simple.lua:267` (`rms_servo_on`) carries exactly this latent bug — do not copy it. Correct:
  ```lua
  local v = true
  if params and params.rms_servo ~= nil then v = params.rms_servo end
  ```
- Keys are namespaced by plugin **`name`**, not filename. Renaming the plugin silently orphans users' stored values.
- **Do not rely on the `transform` schema field** — it is extracted and serialized but `PluginControls.tsx` never applies it; values are sent verbatim. `docs/dev/plugins-authoring.md`'s claim that the UI transforms values is not implemented.
- `accent` (UI colorway) must be one of: amber, cyan (default), blue, teal, green, emerald, purple, indigo, orange, pink, rose, sky, violet (`PluginControls.tsx:20-35`).
- Fields like `stork_substeps`, `beat_stability`, `apg_momentum` are a **separate legacy sideband** channel (`hot-step-params.h:97-100`, `translateParams.ts:147-155`), not `plugin_params`. New plugins must use declared `params` only.

## Worked templates (all real, in-repo)

| Want to write… | Copy from |
|---|---|
| Solver, single-eval | `engine/plugins/solvers/euler.lua` (21 lines, canonical) |
| Solver, multi-eval / stateful | `engine/plugins/solvers/unipc.lua` (`needs_model=true`, history reset on `step_index==0`) |
| Solver, full-loop / stochastic / heavy params | `plugins/solvers/md_pingpong_simple.lua` (owns_loop, pure-Lua RNG, hoisted scratch buffers; but see the toggle-bug note above) |
| Scheduler, simple | `engine/plugins/schedulers/linear.lua` |
| Scheduler with companion `require()` | `engine/plugins/schedulers/beta57.lua` + `beta_math.lua` |
| Guidance, scale-modifying | `engine/plugins/guidance/cfg_pp.lua` (21 lines) |
| Guidance with `post_step()` | `engine/plugins/guidance/cfg_mp.lua` |
| Postprocess | `plugins/postprocess/md_audio_tiled.lua` + `_core` |

## Key files

| Path | Role |
|---|---|
| `engine/src/lua-plugin.h` | The plugin API source of truth: FloatArray, sandbox, all call contracts, param extraction |
| `engine/src/lua-plugin-registry.h` | Scan dirs, companion exclusion, name lookup + aliases, JSON for `GET /plugins` |
| `engine/src/hot-step-sampler.h` | Sampling loop: solver/guidance dispatch, final-step x0, vt snapshot, post_step gating, linker sentinel |
| `engine/src/sampler-schedule.h` | Scheduler dispatch, shift back-calculation, composite schedulers |
| `engine/src/pipeline-synth-ops.cpp` | Postprocess plugin caller + fallback (`ops_vae_decode_postprocess`, ~line 1721) |
| `engine/src/hot-step-params.h` | `plugin_params` map + legacy sideband params |
| `engine/tools/hot-step-server.cpp` | Registry init (~2676), `GET /plugins` (~2716), `plugin_params` JSON parse (~771) |
| `server/src/routes/plugins.ts` | Node proxy `/api/plugins` + 60 s cache + `/reload` |
| `server/src/services/generation/translateParams.ts` | UI params → engine request JSON (`plugin_params`, `postprocess_plugin`) |
| `ui/src/components/global-bar/PluginControls.tsx` | Renders declared params; accent map |
| `ui/src/stores/globalParamsStore.ts` | `hs-pluginParams` localStorage persistence |
| `engine/plugins/` + `plugins/` | Built-in and user plugin tiers |
| `docs/dev/plugins-authoring.md` | Committed authoring guide (mostly accurate; see caveats in reference.md) |
| `plugins/README.md` | Community plugins folder README: points at the authoring guide; its full-loop solver section is accurate |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| Plugin absent from UI dropdown | Lua syntax error at load (grep `[Plugins] ERROR loading` in `ace_engine.log`); wrong global table for its directory; empty `name`; duplicate `name` (first wins); or filename matched companion exclusion (`_constants`/`_math`/`_data`, or `_core` with a non-core sibling) |
| All dropdowns empty / fallback lists | Node couldn't reach the engine — `/api/plugins` returned empty lists (`plugins.ts:26-28`). Check engine is up on :8085 |
| Output pure noise, generation "succeeds" | `step()` raised (index out of range, nil arithmetic) — error printed each step, `xt` never advanced. Grep `[Plugins] ERROR in` |
| `FloatArray is read-only` | Wrote to `vt` / `pred_cond` / `pred_uncond`. Write to `xt` / `result` / scratch tables |
| `attempt to call a nil value (global 'apg')` at load | Called `apg()` at file top level — it is only registered lazily inside `guide()` dispatch |
| `apg()` from `post_step()` behaves oddly (no error) | Silently reuses the stale momentum buffer from the last `guide()` call — never call `apg()` outside `guide()` |
| Multi-eval solver quietly acts like Euler | Expected the original velocity after calling `model_fn` — read original from the snapshot arg `vt`, fresh results from `vt_buf` |
| First step of 2nd generation explodes | Stateful solver didn't reset file-locals on `step_index == 0` (an n-change check alone misses same-length runs) |
| Toggle "can't be turned off" | `(params and params.key) or true` idiom — see Param traps |
| Param changes do nothing | Key mismatch vs schema `key`; edited `apg.lua` (native bypass); or values persisted under an old plugin `name` |
| Guidance `post_step` never fires | `guidance_scale <= 1` (no CFG), final step, or past the `cfg_cutoff_ratio` step (CFG turned off) |
| Every solver/scheduler/guidance dead after upstream sync | `pipeline-synth-ops.cpp` lost the `hot-step-sampler.h` include — now a link error via the `hotstep_sampler_linked_` sentinel. Run `engine/verify-hooks.ps1` |
| Generation very slow with custom guidance | Each `eval_cond`/`eval_uncond` in `post_step` is a full forward pass — budget them |

## Three samplers run these plugins, not one

The plugin layer never had an ACE dependency — every `lua_call_*` takes raw
`float*`, element counts and a param map. What was ACE-specific was the
*sampler*. Three now dispatch into it, and a plugin change affects all three:

| Sampler | Call sites | Notes |
|---|---|---|
| ACE-Step DiT | `hot-step-sampler.h` | the original; time-major `[T][Oc]`, `t` descends 1→0 |
| MiniMax-Music3 flow DiT | `minimax/mm3-plugins.h` | sigma ascends, latents channel-major — the bridge flips both |
| StableStep / SA3 refine | `sa3-refine.h` | `t` already descends like ACE; latents channel-major |

**The layout trap, twice learned:** `apg_project()` (`guidance/apg-core.h`)
normalises per channel over time and indexes `[t*Oc + c]` — it is hard-wired to
ACE's **time-major** memory. MM3 and SA3 both store latents **channel-major**,
so both bridges transpose into the ACE view before any guidance plugin sees a
buffer, and back after. Skip that and APG silently normalises scrambled
mixtures rather than channels — it compiles, it runs, it sounds subtly wrong.
Solvers are exempt: `lua_call_solver_step` is handed only a flat `n`, and
`lua_call_solver_loop` uses `T`/`Oc` solely to publish `n_per`, so no solver
can index across channels.

Per-backend caveats worth knowing before you assume a plugin "works everywhere":
- **`owns_loop` solvers**: fine on ACE and SA3, **refused on MM3** — a full-loop
  solver bypasses MM3's per-step window-overlap blend and breaks every seam.
- **Guidance on SA3 is near-decorative.** SA3 was trained at cfg=1 and has no
  unconditional branch, so the engine passes the cond velocity as *both*
  predictions. APG-family modes see `diff = 0` and pass through unchanged; only
  plugins doing cond-side work have any effect.
- **Schedulers on SA3 get rescaled.** Schedulers emit `sigma_max = 1.0` for full
  denoising; the SA3 refine is SDEdit and starts at `strength` (~0.3), so the
  returned curve is linearly rescaled onto `[strength → 0]` — spacing preserved,
  starting point not. Note the override array is `steps` long while the loop
  indexes `sigmas[i+1]`, so the terminal `0.0` must be appended (getting this
  wrong is an out-of-bounds read, not a compile error).
- **`sampler_build_scheduler_override()` reads process-global
  `g_hotstep_params`**, not its arguments. `/sa3-refine` therefore saves and
  restores `solver_name`/`scheduler`/`plugin_params` around the call — without
  that, a refine leaks its picks into the next ACE generation on the same worker.

## Institutional knowledge

- **VALIDATED**: `hot-step-sampler.h` replaced `dit-sampler.h` as the sampling path; the include lives in upstream `pipeline-synth-ops.cpp` and its loss during a sync used to be silent (everything compiled, all plugins dead). The linker sentinel now makes it a link error. Always run `engine/verify-hooks.ps1` after touching upstream files.
- **VALIDATED**: the engine snapshots `vt` before multi-eval solver steps (`hot-step-sampler.h:1179-1195`) because sharing one buffer between "original velocity" and "model_fn output" silently degraded Heun to Euler.
- **VALIDATED**: stateful plugins must reset on `step_index == 0`; persisting `lua_State`s bleed state across generations (documented in-code at `md_pingpong_simple.lua:242-246` — "explosive velocity on step 1").
- **VALIDATED**: `philox_randn` is NOT exposed to Lua. `sde.lua:3-4` mentions it as a required C helper, but it is not registered; the SDE stochastic path is handled C++-side for that specific plugin. Pure-Lua stochastic plugins must roll their own RNG (see the LCG + Box-Muller in `md_pingpong_simple.lua`).
- The outdated module-return examples that used to live in `plugins/README.md` were removed on 2026-09-25; the README now points at `docs/dev/plugins-authoring.md`. Trust that guide and `engine/src/lua-plugin.h`.
- **UNVERIFIED**: whether the TensorRT sampler variant (`hot-step-sampler-trt.h`) covers solver/guidance plugins identically — it calls the same scheduler override, but its plugin dispatch was not audited. Check before relying on plugins under the TensorRT backend.

## Deeper reading

- [reference.md](reference.md) (this folder) — full-loop solver contract, `post_step()` details, postprocess internals, param schema JSON shape, docs-vs-code discrepancy list.
- `docs/dev/plugins-authoring.md` — committed authoring guide. Known inaccuracies: says `shift` comes "from UI" (actually back-calculated); documents `transform` as applied by the UI (it is not); omits `owns_loop`/`sample()` and the postprocess type; uses `step_idx` naming loosely (solvers get `step_index`, guidance gets `step_idx`).
- `engine/docs/ARCHITECTURE.md` — engine internals, request JSON.
- `docs/plans/` — internal design docs, **gitignored and local-only** (may be absent on a fresh clone).

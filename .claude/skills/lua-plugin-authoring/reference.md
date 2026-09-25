# Lua plugin authoring — deep reference

Companion to [SKILL.md](SKILL.md). All line references verified 2026-07-02 against `d:\Ace-Step-Latest\hot-step-cpp`. Where any doc disagrees with `engine/src/lua-plugin.h`, the code wins.

## 1. Full-loop solvers (`owns_loop = true`)

For solvers that need to control the entire sampling iteration (adaptive dispatch, velocity caching, SDE restarts, ping-pong ancestral sampling). Set `owns_loop = true` in the `solver` table and define `sample()` instead of `step()`. Called via `lua_call_solver_loop` (`engine/src/lua-plugin.h:530-604`) from `hot-step-sampler.h:775-955`.

```lua
function sample(xt, vt_buf, schedule, n, model_fn)
  -- xt:       FloatArray [n], mutable. Contains noise initially.
  -- vt_buf:   FloatArray [n], mutable. model_fn writes velocity here.
  -- schedule: plain Lua table, 1-INDEXED, num_steps descending t values.
  -- n:        total element count (batch_n * n_per).
  -- model_fn(xt_array, t_val): full CFG'd forward pass; result lands in vt_buf.
end
```

Globals injected: `num_steps`, `batch_n`, `n_per`, `params`, and the function `on_step(step_idx, t_curr, t_next) -> bool` (true = cancelled, you should return immediately).

Contract:
1. Call `model_fn(xt, t)` to evaluate the model at any timestep; read the velocity from `vt_buf` afterwards.
2. After completing each step's `xt` update, call `on_step(step_idx, t_curr, t_next)`. **`on_step` is where the engine applies**: cancel checks, cover-switch, DCW correction, guidance `post_step`, repaint injection, CFG cutoff, and progress logging (`hot-step-sampler.h:798-951`). Skip it and all of those silently stop working for your solver.
3. On the final step leave the x0 prediction in `xt` (`xt[i] = xt[i] - vt_buf[i] * t_curr`); there is no engine-side final step for full-loop solvers.

Worked example with full commentary: `plugins/README.md` (its "Full-Loop Solvers" section). Real production plugin: `plugins/solvers/md_pingpong_simple.lua`.

Known engine-side TODO: DCW for full-loop multi-eval solvers uses the live `vt` rather than a pre-solver snapshot (`hot-step-sampler.h:844` TODO comment) — if your full-loop solver leaves stale velocities in `vt_buf` when calling `on_step`, DCW sees those.

## 2. Guidance `post_step()` — exact semantics

If a guidance plugin defines a global `post_step` function, this is detected at load time (`lua-plugin.h:374-379`, sets `has_post_step`, shown as `[post_step]` in the `[DiT] Guidance:` log line).

The engine calls it after each solver step under ALL of these conditions (`hot-step-sampler.h:1215-1230`):
- `has_post_step` is true, AND
- `do_cfg` is true (guidance_scale > 1 and CFG not yet cut off by `cfg_cutoff_ratio`), AND
- `step < num_steps - 1` (never on the final step).

Signature (7 args, `lua-plugin.h:736-803`):

```lua
function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
  -- xt: mutable FloatArray, the latent AFTER the solver step
  -- t:  the timestep just stepped TO
  -- eval_cond(xt_arr, t_val) / eval_uncond(xt_arr, t_val): each is a FULL model
  --   forward pass (expensive). Results land in vt_cond / vt_uncond (FloatArrays).
end
```

Globals: same as `guide()` — `step_idx`, `total_steps`, `dt`, `t_curr`, `params`. `apg()` must NOT be used here: it does not raise — the `_apg_mbuf` global left over from the last `guide()` call (`lua-plugin.h:688-689`, never cleared) makes it silently run with the momentum buffer of whichever batch element was guided last. Wrong state, no error.

Real user: `engine/plugins/guidance/cfg_mp.lua` (manifold-projection CFG), `post_step` at line 47.

## 3. Postprocess plugin — full contract

Caller: `ops_vae_decode_postprocess` in `engine/src/pipeline-synth-ops.cpp:1721-1838`; bridge: `lua_call_postprocess` in `lua-plugin.h:820-964`.

```lua
postprocess = { name = "...", display = "...", params = {...} }

function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
  -- latents:        plain Lua TABLE (1-indexed), CHANNEL-major [C_lat, W]
  --                 (C++ holds time-major [T,64]; the bridge transposes for you)
  -- B:              always 1 — the engine iterates batch items and calls per item
  -- C_lat:          latent channels (64)
  -- W:              latent frame count (T_latent)
  -- C_aud:          audio channels (2)
  -- final_samples:  W * upscale_factor (expected audio length per channel)
  -- upscale_factor: 1920 samples per latent frame (hardcoded, lua-plugin.h:845)
  -- vae_decode_fn(latent_table, T_latent) -> audio_table, T_audio
  --                 latent_table must be 1-indexed channel-major [64, T_latent];
  --                 returned audio_table is 1-indexed [2 * T_audio]
  -- MUST return: audio_table, T_audio
end
```

Failure handling: unknown plugin name → `[Postprocess] WARNING: plugin '<name>' not found, falling back to built-in decoder`; `process()` erroring or returning `T_audio <= 0` → `[Postprocess Batch<b>] ERROR: plugin decode failed, falling back to built-in`. So a broken postprocess plugin degrades gracefully — check the log to know which path actually ran (`[Postprocess] Using plugin '<name>' for VAE decode (T_latent=N)`).

Performance note: this path marshals through Lua tables with a transpose on each side of every `vae_decode_fn` call — it is the deliberately-slow flexible path, not zero-copy like the solver/guidance FloatArrays.

Selection plumbing: UI `postprocessPlugin` → `req.postprocess_plugin` (`server/src/services/generation/translateParams.ts:181-183`) → parsed in `engine/src/request.cpp:177-178`, logged as `[Request] postprocess_plugin: <name>` (`request.cpp:713-714`).

Reference implementation: `plugins/postprocess/md_audio_tiled.lua` which `require()`s `md_audio_tiled_core.lua` (excluded from the plugin scan by the `_core`-with-sibling rule).

## 4. Param schema — full field reference

Extraction: `lua_extract_param` (`lua-plugin.h:230-291`). JSON serialization for `GET /plugins`: `param_to_json` (`lua-plugin-registry.h:243-288`).

```lua
params = {
  { key = "strength",              -- REQUIRED; how you read it: params.strength
    type = "slider",               -- "slider" (default) | "select" | "toggle" | "text"
    label = "Strength",            -- UI label (defaults to key)
    hint = "How hard to push",     -- tooltip
    default = 0.5, min = 0, max = 1, step = 0.01,   -- slider fields
    -- select fields:
    -- default = "a", options = { {value="a", label="Option A"}, "b" },  -- bare strings ok
    -- toggle: default = true|false
    -- text:   default = "some string"
    visible_when = { key = "mode", equals = "advanced" },  -- show only when sibling
                                    -- param "mode" currently equals "advanced" (string compare)
    -- transform = "value * 0.05"  -- DO NOT USE: extracted & serialized but the UI
                                    -- never applies it (PluginControls.tsx); values sent verbatim
  },
}
```

Value round trip:
1. UI stores flat strings: `{ "<plugin.name>:<key>": "value" }` in Zustand + localStorage `hs-pluginParams` (`ui/src/stores/globalParamsStore.ts:149, 307-321`).
2. Sent as `plugin_params` on the generation request (`translateParams.ts:176-178`); parsed engine-side (`engine/tools/hot-step-server.cpp:771-796`, log `[DIAG] Parsed N plugin_params`); copied to `g_hotstep_params.plugin_params` (`hot-step-server.cpp:1190`; declared `engine/src/hot-step-params.h:185-186`).
3. `lua_inject_params` (`lua-plugin.h:410-436`) filters by `"<plugin.name>:"` prefix, strips it, sets a fresh global `params` table before EVERY Lua call. Coercion: fully-numeric string → Lua number; `"true"`/`"false"` → boolean; anything else → string.

Consequences worth internalizing:
- A param the user never touched is simply **absent** — `params.key` is nil, and `params` itself can be an empty table. Defaults in the schema are UI-side only; your Lua code must re-default.
- Because values transit as strings and are re-coerced, a `text` param containing `"2.5"` arrives as a *number*. If you need a literal numeric-looking string, that is not expressible.
- The 60 s Node cache (`server/src/routes/plugins.ts:14-15`) only caches the *schema list*, not values — `POST /api/plugins/reload` is only needed after metadata changes, and an engine restart is still required for the engine to see file edits.

## 5. Name lookup, aliases, defaults

`lua-plugin-registry.h:59-127`:
- Solver: alias `"ode"` → `euler`. Empty name → first entry of an `unordered_map` (nondeterministic — the sampler always passes a concrete name in practice). Unknown at generation time → `[DiT] ERROR: unknown solver '<name>', falling back to euler` (`hot-step-sampler.h:498-501`).
- Scheduler: alias `"karras"` → `sgm_uniform`; parameterized names fall back on the prefix before `:` (`"power:4.00"` → `power`). Empty scheduler never reaches lookup — the sampler uses the upstream shifted-linear default (`hot-step-params.h:88`).
- Guidance: unknown → fallback to `apg` with an ERROR log (`hot-step-sampler.h:519-522`).
- Duplicate `name` across files: first loaded wins, `[Plugins] WARNING: duplicate plugin '<name>' from <path> (keeping first)`. Load order is engine tier before repo-root tier, alphabetical within a directory.

## 6. Docs-vs-code discrepancies (code wins)

1. **Old module-return examples** (`return { ..., step = function(...) }`, `x:get(i)` / `x:set(i,v)`, a `schedule()` returning a table) still circulate in community files and were in `plugins/README.md` until 2026-09-25. The real API is a global `solver = {...}` table plus a global `function step(xt, vt, t_curr, t_prev, n)` using `xt[i]` indexing. Every shipped plugin uses the real style.
2. **`transform` is dead in the UI**: extracted (`lua-plugin.h:235`), serialized (`lua-plugin-registry.h:254`), typed (`ui/src/types/pluginTypes.ts`), never applied by `PluginControls.tsx`. `docs/dev/plugins-authoring.md` (~line 289) claims otherwise.
3. **`docs/dev/plugins-authoring.md` scheduler section** says `shift` is "from UI" — actually back-calculated from the upstream schedule's second timestep and clamped (`<0.5 → 1.0`, `>10 → 3.0`) in `engine/src/sampler-schedule.h:61-74`.
4. **`docs/dev/plugins-authoring.md` omits** `owns_loop`/`sample()` and the entire postprocess plugin type. Postprocess is documented nowhere except the code (and now this reference).
5. **Global naming split**: solvers get `step_index`; guidance (`guide()` and `post_step()`) gets `step_idx`. Two names, same concept — do not mix them up across plugin types.

## 7. Handy debug snippets (PowerShell)

```powershell
# Newest session's plugin lines
$s = Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\]'

# Runtime errors from any plugin type
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[Plugins\] ERROR'

# Which solver/scheduler/guidance actually ran in the last generation
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern '\[DiT\] (Solver|Guidance|Custom schedule)|\[Postprocess\]'

# What params the engine actually received
Select-String -Path "$($s.FullName)\ace_engine.log" -Pattern 'Parsed \d+ plugin_params'

# Dump your plugin's schema as the UI sees it
(Invoke-RestMethod http://localhost:3001/api/plugins).solvers |
  Where-Object { $_.name -eq 'my_solver' } | ConvertTo-Json -Depth 6
```

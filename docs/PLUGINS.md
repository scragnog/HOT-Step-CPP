# Plugin Authoring Guide

How to create custom solvers, schedulers, and guidance modes for the HOT-Step CPP engine.

---

## Quick Start

1. Create a `.lua` file in the appropriate directory:
   - `engine/plugins/solvers/` — ODE/SDE solvers
   - `engine/plugins/schedulers/` — noise schedules
   - `engine/plugins/guidance/` — CFG guidance modes
2. Declare a metadata table (`solver`, `scheduler`, or `guidance`)
3. Implement the required function (`step`, `schedule`, or `guide`)
4. Restart the app — your plugin appears in the UI automatically

No C++ rebuild required. The engine hot-loads all `.lua` files at startup.

---

## Plugin Types

### Solver

Solvers advance the latent state `xt` by one step along the ODE/SDE trajectory.

**Metadata table:**

```lua
solver = {
    name        = "my_solver",           -- unique internal ID (lowercase, underscores)
    display     = "My Solver (2 NFE)",   -- name shown in UI dropdown
    description = "A custom solver",     -- tooltip text
    nfe         = 2,                     -- number of function evaluations per step
    order       = 2,                     -- solver order (informational)
    needs_model = true,                  -- true if step() needs model_fn callback
    stateful    = false,                 -- true if solver carries state across steps
    stochastic  = false,                 -- true if solver uses randomness (SDE)
}
```

**Required function — single-eval solver:**

```lua
function step(xt, vt, t_curr, t_prev, n)
    -- xt:     mutable FloatArray — current latent state (modify in-place)
    -- vt:     read-only FloatArray — velocity at (xt, t_curr)
    -- t_curr: float — current timestep
    -- t_prev: float — next timestep (we step FROM t_curr TO t_prev)
    -- n:      int — total elements in xt/vt

    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end
end
```

**Required function — multi-eval solver** (when `needs_model = true`):

```lua
function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    -- Additional args when needs_model = true:
    -- model_fn(xt_array, t_val): evaluates the model at (xt_array, t_val),
    --                            writes velocity output to vt_buf
    -- vt_buf:  mutable FloatArray — receives model_fn output

    local dt = t_curr - t_prev
    local t_mid = t_curr - 0.5 * dt

    -- Save state
    local k1 = {}
    local xt_orig = {}
    for i = 0, n - 1 do
        k1[i] = vt[i]
        xt_orig[i] = xt[i]
    end

    -- Evaluate at midpoint
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - 0.5 * k1[i] * dt
    end
    model_fn(xt, t_mid)  -- result appears in vt_buf

    -- Final update using midpoint velocity
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - vt_buf[i] * dt
    end
end
```

---

### Scheduler

Schedulers produce a timestep sequence for the denoising trajectory.

**Metadata table:**

```lua
scheduler = {
    name        = "my_schedule",
    display     = "My Schedule",
    description = "Custom noise schedule",
}
```

**Required function:**

```lua
function schedule(output, num_steps, shift)
    -- output:    mutable FloatArray — write num_steps timestep values
    -- num_steps: int — number of timesteps to generate
    -- shift:     float — noise shift parameter from UI

    for i = 0, num_steps - 1 do
        output[i] = 1.0 - i / num_steps
    end
    apply_shift(output, num_steps, shift)
end
```

Timesteps go from `1.0` (pure noise) to `~0.0` (clean signal). The engine appends a final `0.0` step automatically — your schedule should produce `num_steps` values, not `num_steps + 1`.

**Common helper — shift warp:**

```lua
function apply_shift(ts, n, shift)
    if shift == 1.0 then return end
    for i = 0, n - 1 do
        local t = ts[i]
        ts[i] = shift * t / (1.0 + (shift - 1.0) * t)
    end
end
```

---

### Guidance

Guidance modes control how the conditional and unconditional model predictions are combined.

**Metadata table:**

```lua
guidance = {
    name        = "my_guidance",
    display     = "My Guidance",
    description = "Custom guidance mode",
}
```

**Required function:**

```lua
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- pred_cond:      read-only FloatArray — conditional velocity prediction
    -- pred_uncond:    read-only FloatArray — unconditional velocity prediction
    -- guidance_scale: float — the guidance scale (w) from the UI
    -- result:         mutable FloatArray — write the guided velocity here
    -- Oc:             int — output channels per timestep frame
    -- T:              int — number of timestep frames (n = Oc * T)
    -- norm_threshold: float — APG norm threshold from the UI

    -- Route through APG for stability (STRONGLY RECOMMENDED)
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
```

> **⚠️ Important:** Always route through `apg()` for the base guidance computation. Raw linear interpolation (`result = uncond + w * (cond - uncond)`) produces severe audio artifacts (static, underwater sound, frequency distortion). The `apg()` function provides momentum smoothing, perpendicular projection, and norm thresholding that are essential for stable audio output. If your guidance mode needs custom math, apply it as a correction on top of the APG result.

**Available globals in guidance plugins:**

| Global | Type | Description |
|--------|------|-------------|
| `step_idx` | int | Current step index (0-based) |
| `total_steps` | int | Total number of denoising steps |
| `dt` | float | Current timestep delta (t_curr - t_next) |
| `t_curr` | float | Current timestep value |
| `params` | table | User-configured parameter values |

---

## The `apg()` Bridge

The `apg()` function is the native C++ APG (Analytical Perpendicular Guidance) implementation, exposed to Lua guidance plugins. It handles:

1. **Perpendicular projection** — removes the component of `(cond - uncond)` parallel to `uncond`, keeping only the steering signal
2. **Momentum smoothing** — exponential moving average across steps to prevent jitter
3. **Norm thresholding** — caps per-channel magnitudes to prevent blowup

**Signature:**

```lua
apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
```

All guidance plugins have access to this function. It is registered automatically on first use.

---

## Advanced: The `post_step()` Hook

For guidance modes that need to run extra model evaluations *after* the solver step (e.g., manifold projection), guidance plugins can declare a `post_step()` function. The engine detects this at load time and calls it after each solver step.

**When to use this:**
- Your guidance technique requires iterative refinement of the latent state
- You need to evaluate the model at positions different from the main solver trajectory
- The technique calls the model with conditioning and unconditioning separately

**Performance warning:** Each call to `eval_cond` or `eval_uncond` runs a full model forward pass. This is expensive — use sparingly.

```lua
function post_step(xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond)
    -- xt:          mutable FloatArray — latent state after solver step (modify in-place)
    -- t:           float — timestep we just stepped TO (t_next)
    -- n:           int — total elements in xt
    -- eval_cond:   function(xt_arr, t) — runs model with conditioning, writes to vt_cond
    -- eval_uncond: function(xt_arr, t) — runs model without conditioning, writes to vt_uncond
    -- vt_cond:     mutable FloatArray — output buffer for conditional velocity
    -- vt_uncond:   mutable FloatArray — output buffer for unconditional velocity

    -- Example: one iteration of manifold projection
    local a = math.abs(dt) / 2.0

    eval_uncond(xt, t)                        -- fills vt_uncond
    for i = 0, n - 1 do
        xt[i] = xt[i] - a * vt_uncond[i]     -- push away from uncond manifold
    end

    eval_cond(xt, t)                          -- fills vt_cond
    for i = 0, n - 1 do
        xt[i] = xt[i] + a * vt_cond[i]       -- pull toward cond manifold
    end
end
```

The `post_step` hook has access to the same globals as `guide()` (`step_idx`, `total_steps`, `dt`, `t_curr`, `params`).

The hook is **not called on the final step** (the latent is about to be decoded, so further projection is pointless).

---

## Parameter Schema

Plugins can declare user-facing parameters that appear in the UI. Parameters are defined in the `params` array of the metadata table.

### Slider

```lua
{ key = "strength", type = "slider", label = "Strength",
  default = 0.5, min = 0.0, max = 1.0, step = 0.01,
  hint = "Controls the effect intensity" }
```

### Select (Dropdown)

```lua
{ key = "mode", type = "select", label = "Mode",
  default = "fast",
  options = {
      { value = "fast",    label = "Fast" },
      { value = "quality", label = "Quality" },
  },
  hint = "Choose between speed and quality" }
```

### Toggle

```lua
{ key = "enabled", type = "toggle", label = "Enable Feature",
  default = false,
  hint = "Turn this feature on or off" }
```

### Conditional Visibility

Parameters can be shown/hidden based on another parameter's value:

```lua
{ key = "sub_param", type = "slider", label = "Sub-Parameter",
  default = 1.0, min = 0.0, max = 5.0, step = 0.1,
  visible_when = { key = "mode", equals = "quality" },
  hint = "Only visible when Mode is set to Quality" }
```

### Transform Expressions

The `transform` field allows the UI to apply a mathematical transformation to the displayed value before sending it to the plugin. This is useful when the internal value differs from what the user sees:

```lua
{ key = "sigma", type = "slider", label = "Noise σ",
  default = 5, min = 0, max = 100, step = 1,
  transform = "value * 0.05",
  hint = "Displayed as 0-100, sent to plugin as 0-5" }
```

### Reading Parameters

Parameters are available via the `params` global table, keyed by their `key` field:

```lua
local strength = (params and params.strength) or 0.5
local mode     = (params and params.mode) or "fast"
local enabled  = (params and params.enabled) or false
```

Always provide a fallback default with `or` — `params` may be `nil` if no parameters have been set.

---

## FloatArray

All array data passes between C++ and Lua via the `FloatArray` userdata type. This is a **zero-copy** bridge — Lua reads and writes the same memory that the C++ engine uses.

**Indexing:** 0-based (matching C++ convention, not Lua's typical 1-based).

```lua
-- Read
local val = xt[i]

-- Write (only on mutable arrays)
xt[i] = val

-- Length
local n = #xt
```

Read-only arrays (like `pred_cond` and `pred_uncond` in guidance) will raise an error if you try to write to them.

---

## Available Globals

### Solver globals

| Global | Type | Description |
|--------|------|-------------|
| `step_index` | int | Current step index |
| `batch_n` | int | Number of batch elements |
| `n_per` | int | Elements per batch element |
| `params` | table | Plugin parameters |

### Guidance globals

| Global | Type | Description |
|--------|------|-------------|
| `step_idx` | int | Current step index (0-based) |
| `total_steps` | int | Total denoising steps |
| `dt` | float | Timestep delta |
| `t_curr` | float | Current timestep |
| `params` | table | Plugin parameters |

---

## Sandbox

Each plugin runs in an isolated Lua 5.4 VM with:

**Available:** `math`, `string`, `table`, `print`, `type`, `pairs`, `ipairs`, `tonumber`, `tostring`, `require` (for companion data files)

**Blocked:** `os`, `io`, `debug`, `dofile`, `loadfile` — no filesystem access, no shell commands, no process control.

The `require()` function works for loading companion Lua data files (e.g., precomputed constants in a separate `.lua` file in the same directory), but cannot load C modules.

---

## Complete Examples

### Minimal Solver

```lua
-- my_solver.lua
solver = {
    name        = "my_solver",
    display     = "My Solver",
    description = "Simple Euler variant",
    nfe         = 1,
    order       = 1,
    needs_model = false,
}

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end
end
```

### Scheduler with Custom Curve

```lua
-- my_schedule.lua
scheduler = {
    name        = "my_schedule",
    display     = "Quadratic",
    description = "Quadratic timestep spacing",
    params      = {
        { key = "power", type = "slider", label = "Power",
          default = 2.0, min = 1.0, max = 4.0, step = 0.1 },
    },
}

function schedule(output, num_steps, shift)
    local p = (params and params.power) or 2.0
    for i = 0, num_steps - 1 do
        local frac = i / num_steps
        output[i] = (1.0 - frac) ^ p
    end
    -- Apply shift warp
    if shift ~= 1.0 then
        for i = 0, num_steps - 1 do
            local t = output[i]
            output[i] = shift * t / (1.0 + (shift - 1.0) * t)
        end
    end
end
```

### Guidance with APG + Custom Logic

```lua
-- my_guidance.lua
guidance = {
    name        = "my_guidance",
    display     = "My Guidance",
    description = "Warm-up guidance with linear ramp",
    params      = {
        { key = "warmup_steps", type = "slider", label = "Warm-Up Steps",
          default = 3, min = 0, max = 10, step = 1 },
    },
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local warmup = (params and params.warmup_steps) or 3
    local progress = math.min((step_idx or 0) / math.max(warmup, 1), 1.0)
    local effective_scale = 1.0 + (guidance_scale - 1.0) * progress

    apg(pred_cond, pred_uncond, effective_scale, result, Oc, T, norm_threshold)
end
```

---

## Tips

- **Test with Euler + Linear first.** The simplest solver/scheduler combination isolates your plugin's behaviour.
- **Use `print()` for debugging.** Output goes to the terminal panel in the app.
- **Lua tables for scratch space.** If you need temporary arrays, use Lua tables: `local tmp = {}; for i = 0, n-1 do tmp[i] = 0 end`. They're slower than FloatArrays but work for intermediate calculations.
- **Be careful with the loop range.** FloatArrays are 0-indexed: `for i = 0, n - 1 do ... end`.
- **Guidance plugins: always use `apg()`.** Raw math without APG produces audio artifacts. Apply your custom logic as a delta on top.
- **Stateful plugins** can use file-level `local` variables to carry state across steps (e.g., previous velocity buffers, error accumulators). These reset when the plugin is reloaded.

# HOT-Step Community Plugins

Drop custom Lua plugin files here to extend the engine without rebuilding.

## Directory Structure

```
plugins/
├── solvers/        ← Custom ODE/SDE solvers
├── schedulers/     ← Custom noise schedules
└── guidance/       ← Custom guidance modes
```

## How It Works

1. Place `.lua` files in the appropriate subdirectory
2. Restart the engine (or the app)
3. Your plugin appears in the UI dropdown automatically

The engine scans `engine/plugins/` (built-in) first, then this `plugins/` directory.
Duplicate names are skipped with a console warning.

## Writing a Plugin

Every plugin is a single `.lua` file that returns a table with metadata and a `step()` function.

### Solver Example

```lua
return {
  name    = "my_solver",
  display = "My Custom Solver",
  type    = "solver",
  nfe     = 1,
  accent  = "pink",

  -- Optional user-facing parameters
  params = {
    { key = "strength", type = "slider", label = "Strength",
      default = 0.5, min = 0, max = 1, step = 0.01 },
  },

  step = function(x, v, t, t_next, dt, params)
    -- x: current latent (FloatArray)
    -- v: velocity prediction (FloatArray)
    -- t, t_next, dt: timestep scalars
    -- params: table of user values { strength = "0.5", ... }
    for i = 0, x:size() - 1 do
      x:set(i, x:get(i) + dt * v:get(i))
    end
  end,
}
```

### Scheduler Example

```lua
return {
  name    = "my_schedule",
  display = "My Schedule",
  type    = "scheduler",

  schedule = function(n_steps, params)
    -- Return a table of n_steps+1 descending floats from 1.0 to 0.0
    local ts = {}
    for i = 0, n_steps do
      ts[i + 1] = 1.0 - i / n_steps
    end
    return ts
  end,
}
```

### Guidance Example

```lua
return {
  name    = "my_guidance",
  display = "My Guidance",
  type    = "guidance",

  guide = function(cond, uncond, scale, t, params)
    -- cond/uncond: FloatArray (conditional/unconditional predictions)
    -- scale: guidance scale (number)
    -- t: current timestep (0→1)
    -- Return guided prediction in cond (modified in-place)
    for i = 0, cond:size() - 1 do
      local c = cond:get(i)
      local u = uncond:get(i)
      cond:set(i, u + scale * (c - u))
    end
  end,
}
```

## Full-Loop Solvers

For advanced solvers that need to control the entire sampling iteration
(e.g., adaptive dispatch, velocity caching, SDE restarts), set
`owns_loop = true` and define a `sample()` function instead of `step()`.

### Full-Loop Solver Example

```lua
solver = {
  name       = "my_sampler",
  display    = "My Sampler",
  nfe        = 0,         -- varies per step
  order      = 1,
  stateful   = true,
  owns_loop  = true,      -- takes over the sampling loop

  params = {
    { key = "my_param", type = "slider", label = "My Param",
      default = 0.5, min = 0, max = 1, step = 0.01 },
  },
}

function sample(xt, vt_buf, schedule, n, model_fn)
  -- xt:        FloatArray [n], mutable. Contains noise initially.
  -- vt_buf:    FloatArray [n], mutable. model_fn writes velocity here.
  -- schedule:  Lua table {t_1, t_2, ..., t_N}  (1-indexed, N = num_steps)
  -- n:         total element count
  -- model_fn:  function(xt_array, t_val) → writes velocity to vt_buf
  --
  -- Globals: on_step, num_steps, batch_n, n_per, params
  --
  -- Contract:
  --   1. Call model_fn(xt, t) to evaluate the model at any timestep
  --   2. Read velocity from vt_buf after model_fn returns
  --   3. After each step: call on_step(step_idx, t_curr, t_next) → bool
  --      Returns true if generation was cancelled (you should return)
  --   4. When done, xt must contain the denoised output (x0)

  local ns = #schedule
  for i = 1, ns do
    local t_curr = schedule[i]

    -- Evaluate model
    model_fn(xt, t_curr)

    if i < ns then
      -- Euler step (replace with your solver logic)
      local t_next = schedule[i + 1]
      local dt = t_curr - t_next
      for j = 0, n - 1 do
        xt[j] = xt[j] - vt_buf[j] * dt
      end
      -- Report step completion (engine applies DCW, repaint, etc.)
      if on_step(i - 1, t_curr, t_next) then return end
    else
      -- Final step: predict x0
      for j = 0, n - 1 do
        xt[j] = xt[j] - vt_buf[j] * t_curr
      end
    end
  end
end
```

> **Note:** `on_step()` applies engine corrections (DCW, repaint, guidance
> post-step) automatically. You don't need to handle these yourself.

## Parameter Types

| Type     | Fields                                           |
|----------|--------------------------------------------------|
| `slider` | `key`, `label`, `default`, `min`, `max`, `step`  |
| `select` | `key`, `label`, `default`, `options`              |
| `toggle` | `key`, `label`, `default`                        |
| `text`   | `key`, `label`, `default`, `hint`                |

## Safety

Plugins run in a sandboxed Lua environment:
- ❌ No `os`, `io`, `debug`, `dofile`, `loadfile`
- ✅ `math`, `string`, `table`, `require` (for companion data files)
- ✅ Full `FloatArray` API for zero-copy memory access

## Sharing Plugins

Share your `.lua` files with other HOT-Step users! Just drop them in the right folder.

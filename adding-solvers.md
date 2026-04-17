# Adding RK4 Solver to acestep.cpp

## Background

The user wants to prototype adding custom solvers (starting with RK4) to the `acestep.cpp` C++ backend. This is a foundational test for later adding **guidance modes** and **schedulers** too.

## Research Findings

### How HOT-Step 9000 (Python) Does It

The Python solver system in [solvers.py](file:///D:/Ace-Step-Latest/hot-step-9000/acestep/core/generation/solvers.py) uses a clean callback pattern:

```python
def rk4_step(xt, vt, t_curr, t_prev, state, model_fn=None):
    k1 = vt                          # Already computed
    k2 = model_fn(xt - 0.5*dt*k1, t_mid)   # Midpoint eval
    k3 = model_fn(xt - 0.5*dt*k2, t_mid)   # Midpoint eval  
    k4 = model_fn(xt - dt*k3, t_prev)       # Endpoint eval
    xt_next = xt - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

Key: `model_fn(xt, t) -> vt` is a **callback** that runs the full DiT forward pass + CFG at an arbitrary `(xt, t)` point. RK4 calls it **3 extra times** per step (4 total).

### How acestep.cpp Does It Now

The C++ sampler in [dit-sampler.h](file:///D:/Ace-Step-Latest/acestepcpp/acestep.cpp/src/dit-sampler.h) is a ~700-line monolithic function `dit_ggml_generate()`. The step loop (lines 393-647) is tightly coupled:

```
For each step:
    1. Set timestep on GPU tensors (L436-442)
    2. Re-upload constants (L444-449) — scheduler reuses input buffers
    3. Pack xt into input tensor (L451-464)
    4. Forward pass: ggml_backend_sched_graph_compute() (L467)
    5. Read output, apply CFG/APG (L514-576) → produces vt
    6. Step update: Euler or SDE (L584-611)
```

There's **no model callback abstraction**. Steps 1-5 are ~180 lines of low-level tensor manipulation and GPU dispatch. RK4 needs to call steps 1-5 at different `(xt, t)` values.

### The Constraint

This is compiled C++. We **cannot** hook into it at runtime. The only option is to **modify the source and recompile**.

## Proposed Changes

> [!IMPORTANT]
> This requires modifying `dit-sampler.h` in the acestep.cpp repository. The modification is self-contained to this one file. We'll maintain it as a patch that can be applied before building.

### [MODIFY] [dit-sampler.h](file:///D:/Ace-Step-Latest/acestepcpp/acestep.cpp/src/dit-sampler.h)

Two changes:

#### 1. Extract a `evaluate_velocity()` helper (refactor, not behavior change)

Factor out lines 436-576 (the forward pass + CFG/APG + timestep setup) into a reusable lambda/function:

```cpp
// callable: evaluate_velocity(xt_data, t_val) -> writes into vt
auto evaluate_velocity = [&](const float* xt_in, float t_val) {
    // Set timestep
    ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float));
    ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float));
    
    // Re-upload constants
    ggml_backend_tensor_set(t_enc, enc_buf.data(), ...);
    ggml_backend_tensor_set(t_pos, pos_data.data(), ...);
    // ... masks ...
    
    // Pack xt into input
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[...], &xt_in[...], Oc * sizeof(float));
        }
        if (batch_cfg) { /* uncond duplicate */ }
    }
    ggml_backend_tensor_set(t_input, input_buf.data(), ...);
    
    // Forward pass
    ggml_backend_sched_graph_compute(model->sched, gf);
    
    // Read output + CFG/APG → vt
    // (batched/2-pass/no-cfg branches)
};
```

#### 2. Add RK4 step update (new code path)

In the step update section (lines 584-611), add a third branch alongside Euler and SDE:

```cpp
if (step == num_steps - 1) {
    // final step: predict x0
    for (int i = 0; i < n_total; i++)
        output[i] = xt[i] - vt[i] * t_curr;
} else {
    float t_next = schedule[step + 1];
    
    if (use_rk4) {
        // RK4: 4th-order Runge-Kutta (3 extra model evaluations)
        float dt = t_curr - t_next;
        float t_mid = (t_curr + t_next) / 2.0f;
        
        // k1 = vt (already computed)
        std::vector<float> k1(vt);
        
        // k2 = evaluate(xt - 0.5*dt*k1, t_mid)
        std::vector<float> xt_tmp(n_total);
        for (int i = 0; i < n_total; i++)
            xt_tmp[i] = xt[i] - 0.5f * dt * k1[i];
        evaluate_velocity(xt_tmp.data(), t_mid);
        std::vector<float> k2(vt);
        
        // k3 = evaluate(xt - 0.5*dt*k2, t_mid)
        for (int i = 0; i < n_total; i++)
            xt_tmp[i] = xt[i] - 0.5f * dt * k2[i];
        evaluate_velocity(xt_tmp.data(), t_mid);
        std::vector<float> k3(vt);
        
        // k4 = evaluate(xt - dt*k3, t_next)
        for (int i = 0; i < n_total; i++)
            xt_tmp[i] = xt[i] - dt * k3[i];
        evaluate_velocity(xt_tmp.data(), t_next);
        std::vector<float> k4(vt);
        
        // Weighted average: xt_next = xt - (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        float w = dt / 6.0f;
        for (int i = 0; i < n_total; i++)
            xt[i] -= w * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    } else if (use_sde && seeds) {
        // ... existing SDE code ...
    } else {
        // ODE Euler: x_{t+1} = x_t - v_t * dt
        float dt = t_curr - t_next;
        for (int i = 0; i < n_total; i++)
            xt[i] -= vt[i] * dt;
    }
}
```

### [MODIFY] [request.h](file:///D:/Ace-Step-Latest/acestepcpp/acestep.cpp/src/request.h)

No change needed — `infer_method` is already a `std::string` that accepts any value.

### [MODIFY] [pipeline-synth.cpp](file:///D:/Ace-Step-Latest/acestepcpp/acestep.cpp/src/pipeline-synth.cpp)

Add `use_rk4` flag parsing alongside `use_sde`:

```cpp
s.use_sde = (s.rr.infer_method == "sde");
s.use_rk4 = (s.rr.infer_method == "rk4");  // NEW
```

Then pass `use_rk4` into `dit_ggml_generate()`.

### Frontend: No changes needed

The UI already sends `infer_method` as a string parameter. We just need to add "rk4" as an option in the dropdown.

## Performance Impact

| Solver | NFE/step | 8 steps total | Quality |
|--------|----------|---------------|---------|
| Euler  | 1        | 8 evals       | Baseline |
| RK4    | 4        | 32 evals      | Higher order accuracy |

> [!WARNING]
> RK4 at 8 steps = 32 DiT forward passes (vs Euler's 8). This means ~4x longer synthesis time. However, you can often use **fewer steps** with RK4 for equivalent quality. 2 RK4 steps (8 NFE) matches or beats 8 Euler steps (8 NFE).

## Verification Plan

### Build
```powershell
cd D:\Ace-Step-Latest\acestepcpp\acestep.cpp
cmake --build build --config Release
```

### Test
1. Generate with `infer_method: "ode"` (Euler) — confirm no regression
2. Generate with `infer_method: "rk4"` — confirm it completes and produces audio
3. Compare audio quality at equivalent NFE budgets (2 RK4 steps vs 8 Euler steps)

## Open Questions

> [!IMPORTANT]
> **Modification policy**: This requires editing `dit-sampler.h` and `pipeline-synth.cpp` in the acestep.cpp repository. Are you OK with that? We can maintain the changes as a git patch that gets applied before building, keeping our modifications separate from upstream.

> [!IMPORTANT]
> **Build system**: Do you have a working CMake build for acestep.cpp on this machine? I need to verify we can compile after the change.

#pragma once
// solver-sde.h: Stochastic Differential Equation Euler solver (1 NFE + Philox)
//
// SDE Euler: predict x0, then re-noise with fresh Philox random numbers.
//   x0   = xt - vt * t_curr
//   xt+1 = t_prev * noise + (1 - t_prev) * x0
//
// Requires per-batch seeds in SolverState for reproducible re-noising.
// Extracted from the original dit-sampler.h SDE branch.

#include "solver-interface.h"
#include "../philox.h"

static void solver_sde_step(float *       xt,
                            const float * vt,
                            float         t_curr,
                            float         t_prev,
                            int           n,
                            SolverState & state,
                            SolverModelFn /*model_fn*/,
                            float *       /*vt_buf*/) {
    if (!state.seeds || state.batch_n <= 0 || state.n_per <= 0) {
        // Fallback to ODE Euler if no seed info
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    int batch_n = state.batch_n;
    int n_per   = state.n_per;

    for (int b = 0; b < batch_n; b++) {
        // Generate fresh noise per batch item
        std::vector<float> fresh(n_per);
        philox_randn(state.seeds[b] + state.step_index + 1, fresh.data(), n_per, /*bf16_round=*/true);

        for (int i = 0; i < n_per; i++) {
            int   idx = b * n_per + i;
            float x0  = xt[idx] - vt[idx] * t_curr;
            xt[idx]   = t_prev * fresh[i] + (1.0f - t_prev) * x0;
        }
    }
}

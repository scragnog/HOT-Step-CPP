#pragma once
// solver-dpm.h: DPM++ multistep solvers (all 1 NFE per step, stateful)
//
// Three Adams-Bashforth-family solvers that use velocity history from previous
// steps for higher-order corrections without extra model evaluations:
//
//   DPM++ 2M:          Fixed AB2 coefficients (3/2, -1/2)
//   DPM++ 3M:          Fixed AB3 coefficients (23/12, -16/12, 5/12)
//   DPM++ 2M Adaptive: Step-ratio-corrected AB2 for non-uniform schedules
//
// Matches Python solvers.py dpm_pp_2m_step(), dpm_pp_3m_step(), dpm_pp_2m_ada_step()

#include "solver-interface.h"

// ── DPM++ 2M ────────────────────────────────────────────────────────────────
// 2nd order, uses previous step's velocity. First step = Euler.
static void solver_dpm2m_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn /*model_fn*/,
                              float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty()) {
        // Second-order: Adams-Bashforth 2
        // v_eff = 1.5 * v_t - 0.5 * v_{t-1}
        for (int i = 0; i < n; i++) {
            float v_eff = 1.5f * vt[i] - 0.5f * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        // First step: plain Euler
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    // Save current velocity for next step
    state.prev_vt.assign(vt, vt + n);
}


// ── DPM++ 3M ────────────────────────────────────────────────────────────────
// 3rd order, uses two previous velocities. Falls back to AB2 on step 2, Euler on step 1.
static void solver_dpm3m_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn /*model_fn*/,
                              float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty() && !state.prev_prev_vt.empty()) {
        // Third-order: Adams-Bashforth 3
        // v_eff = (23/12)*v_n - (16/12)*v_{n-1} + (5/12)*v_{n-2}
        for (int i = 0; i < n; i++) {
            float v_eff = (23.0f / 12.0f) * vt[i]
                        - (16.0f / 12.0f) * state.prev_vt[i]
                        + ( 5.0f / 12.0f) * state.prev_prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else if (!state.prev_vt.empty()) {
        // Second-order: AB2 (same as DPM++ 2M)
        for (int i = 0; i < n; i++) {
            float v_eff = 1.5f * vt[i] - 0.5f * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        // First step: plain Euler
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    // Shift history: prev -> prev_prev, current -> prev
    state.prev_prev_vt = std::move(state.prev_vt);
    state.prev_vt.assign(vt, vt + n);
}


// ── DPM++ 2M Adaptive ───────────────────────────────────────────────────────
// 2nd order with step-ratio-corrected AB2 coefficients for non-uniform schedules.
// For uniform steps (r=1) this reduces to standard DPM++ 2M.
static void solver_dpm2m_ada_step(float *       xt,
                                  const float * vt,
                                  float         t_curr,
                                  float         t_prev,
                                  int           n,
                                  SolverState & state,
                                  SolverModelFn /*model_fn*/,
                                  float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty() && state.prev_dt > 0.0f) {
        // Step-ratio-corrected AB2
        float r  = dt / state.prev_dt;
        float c1 = 1.0f + r / 2.0f;   // weight for current velocity
        float c0 = r / 2.0f;           // weight for previous velocity (subtracted)
        for (int i = 0; i < n; i++) {
            float v_eff = c1 * vt[i] - c0 * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        // First step: plain Euler
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    state.prev_vt.assign(vt, vt + n);
    state.prev_dt = dt;
}

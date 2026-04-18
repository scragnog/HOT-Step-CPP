#pragma once
// solver-euler.h: Euler method (1st order, 1 NFE per step)
//
// The simplest ODE solver: x_{t+1} = x_t - v_t * dt
// Matches Python solvers.py euler_step()

#include "solver-interface.h"

static void solver_euler_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & /*state*/,
                              SolverModelFn /*model_fn*/,
                              float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;
    for (int i = 0; i < n; i++) {
        xt[i] -= vt[i] * dt;
    }
}

#pragma once
// solver-registry.h: Compile-time registry mapping solver names to step functions
//
// Usage:
//   const SolverInfo* info = solver_lookup("heun");
//   if (info) info->step_fn(xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf);

#include "solver-interface.h"
#include "solver-euler.h"
#include "solver-heun.h"
#include "solver-dpm.h"
#include "solver-rk4.h"
#include "solver-dopri.h"
#include "solver-jkass.h"
#include "solver-stork.h"
#include "solver-sde.h"

#include <cstring>

struct SolverInfo {
    const char *  name;           // internal identifier (lowercase)
    const char *  display_name;   // human-readable name for UI
    SolverStepFn  step_fn;        // step function pointer
    int           nfe;            // model evaluations per step (0 = variable)
    int           order;          // ODE integration order
    bool          needs_model_fn; // true if solver calls model_fn for extra evaluations
    bool          is_stateful;    // true if solver maintains velocity history
    bool          is_stochastic;  // true if solver uses random noise per step
};

// All registered solvers — update this array when adding new solvers.
static const SolverInfo SOLVER_REGISTRY[] = {
    // ── Single Evaluation (1 NFE) ──
    {"euler",       "Euler (ODE)",        solver_euler_step,          1,  1, false, false, false},
    {"dpm2m",       "DPM++ 2M",           solver_dpm2m_step,          1,  2, false, true,  false},
    {"dpm3m",       "DPM++ 3M",           solver_dpm3m_step,          1,  3, false, true,  false},
    {"dpm2m_ada",   "DPM++ 2M Adaptive",  solver_dpm2m_ada_step,      1,  2, false, true,  false},
    {"jkass_fast",  "JKASS Fast",         solver_jkass_fast_step,     1,  1, false, true,  false},
    {"stork2",      "STORK 2",            solver_stork2_step,         1,  2, false, true,  false},
    {"stork4",      "STORK 4",            solver_stork4_step,         1,  4, false, true,  false},
    {"sde",         "SDE (Stochastic)",   solver_sde_step,            1,  1, false, false, true },

    // ── Multi Evaluation ──
    {"heun",           "Heun (2 NFE)",          solver_heun_step,           2,  2, true,  false, false},
    {"jkass_quality",  "JKASS Quality (2 NFE)", solver_jkass_quality_step,  2,  2, true,  false, false},
    {"rk4",            "RK4 (4 NFE)",           solver_rk4_step,            4,  4, true,  false, false},
    {"rk5",            "RK5 (6 NFE)",           solver_rk5_step,            6,  5, true,  false, false},
    {"dopri5",         "DOPRI5 (7+ NFE)",       solver_dopri5_step,         0,  5, true,  false, false},
    {"dop853",         "DOP853 (13 NFE)",       solver_dop853_step,        13,  8, true,  false, false},
};

static const int SOLVER_REGISTRY_SIZE = (int) (sizeof(SOLVER_REGISTRY) / sizeof(SOLVER_REGISTRY[0]));

// Look up a solver by name. Returns nullptr if not found.
// Accepts aliases: "ode" -> "euler"
static const SolverInfo * solver_lookup(const char * name) {
    if (!name || !name[0]) return &SOLVER_REGISTRY[0]; // default: euler

    // Aliases
    const char * resolved = name;
    if (strcmp(name, "ode") == 0) resolved = "euler";

    for (int i = 0; i < SOLVER_REGISTRY_SIZE; i++) {
        if (strcmp(SOLVER_REGISTRY[i].name, resolved) == 0) {
            return &SOLVER_REGISTRY[i];
        }
    }
    return nullptr;
}

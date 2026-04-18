#pragma once
// solver-stork.h: STORK solvers for ACE-Step flow matching
//
// STORK = Stabilized Taylor Orthogonal Runge-Kutta
// From: Tan et al., 2025, arXiv:2505.24210
//
// Two variants (both 1 NFE per step):
//   - STORK2: 2nd-order Runge-Kutta-Gegenbauer (RKG2) sub-stepping
//   - STORK4: 4th-order ROCK4 sub-stepping with precomputed Chebyshev coefficients
//
// Key idea: velocity derivatives are approximated from history (finite
// differences), then cheap arithmetic sub-steps handle stiffness without
// extra model evaluations.
//
// Adaptive: if sub-stepping produces NaN/Inf, sub-steps are halved.
// Falls back to Euler as a last resort.
//
// Matches Python stork_solver.py stork2_step(), stork4_step()

#include "solver-interface.h"
#include "solver-stork4-constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Check if any element in array is NaN or Inf
static bool _stork_has_nan_inf(const float * data, int n) {
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(data[i])) return true;
    }
    return false;
}

// Compute RMS norm of an array
static double _stork_rms(const float * data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double) data[i] * (double) data[i];
    }
    return sqrt(sum / (double) n);
}

// Taylor expansion: v(t + diff) ≈ v + diff*dv + 0.5*diff²*d2v
// Writes result into out[n]. v, dv, d2v are all [n].
static void _stork_taylor_approx(float *       out,
                                 int           order,
                                 float         diff_val,
                                 const float * v,
                                 const float * dv,
                                 const float * d2v,
                                 int           n) {
    if (order >= 1 && dv) {
        if (order >= 2 && d2v) {
            float half_diff2 = 0.5f * diff_val * diff_val;
            for (int i = 0; i < n; i++) {
                out[i] = v[i] + diff_val * dv[i] + half_diff2 * d2v[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                out[i] = v[i] + diff_val * dv[i];
            }
        }
    } else {
        memcpy(out, v, n * sizeof(float));
    }
}

// Compute velocity derivatives from history using finite differences.
// Returns derivative order used (0, 1, or 2).
// dv and d2v are allocated by caller [n] each; filled if order >= 1 or 2.
static int _stork_compute_derivatives(const float *   vt,
                                      int             n,
                                      const SolverState & state,
                                      float *         dv,
                                      float *         d2v) {
    const auto & hist = state.velocity_history;
    if (hist.empty()) return 0;

    // 1st derivative: forward difference
    const float * v_prev = hist.back().vt.data();
    float         h1     = hist.back().dt;
    if (h1 == 0.0f) return 0;

    for (int i = 0; i < n; i++) {
        dv[i] = (v_prev[i] - vt[i]) / h1;
    }

    // Safety: clamp if derivative is noise-dominated
    double vt_rms = _stork_rms(vt, n);
    double dv_rms = _stork_rms(dv, n);
    if (vt_rms > 0.0 && dv_rms * fabs(h1) > 5.0 * vt_rms) {
        return 0;
    }

    if ((int) hist.size() < 2) return 1;

    // 2nd derivative: three-point formula
    const float * v_prev2 = hist[hist.size() - 2].vt.data();
    float         h2      = hist[hist.size() - 2].dt;
    if (h2 == 0.0f) return 1;

    double h1d = (double) h1;
    double h2d = (double) h2;
    double denom = h1d * h2d * (h1d + h2d);
    if (fabs(denom) < 1e-30) return 1;

    double coeff = 2.0 / denom;
    for (int i = 0; i < n; i++) {
        d2v[i] = (float) (coeff * (
            (double) v_prev2[i] * h1d -
            (double) v_prev[i] * (h1d + h2d) +
            (double) vt[i] * h2d
        ));
    }

    // Safety: clamp 2nd derivative
    double d2v_rms = _stork_rms(d2v, n);
    if (vt_rms > 0.0 && d2v_rms * h1 * h1 > 5.0 * vt_rms) {
        return 1;
    }

    return 2;
}

// Update velocity history in state (keep last 3)
static void _stork_update_history(SolverState & state, const float * vt, int n, float dt) {
    SolverState::VelocityRecord rec;
    rec.vt.assign(vt, vt + n);
    rec.dt = dt;
    state.velocity_history.push_back(std::move(rec));
    if ((int) state.velocity_history.size() > 3) {
        state.velocity_history.erase(state.velocity_history.begin());
    }
}


// ---------------------------------------------------------------------------
// STORK2: RKG2 Chebyshev sub-stepping
// ---------------------------------------------------------------------------

// RKG2 recurrence coefficient b_j
static double _rkg2_b_coeff(int j) {
    if (j <= 0) return 1.0;
    if (j == 1) return 1.0 / 3.0;
    return 4.0 * (j - 1) * (j + 4) / (3.0 * j * (j + 1) * (j + 2) * (j + 3));
}

// Run the RKG2 sub-stepping loop. Returns true if result is finite.
static bool _rkg2_substep(float *       xt_out,
                          const float * xt,
                          const float * vt,
                          int           s,
                          float         t_curr,
                          float         t_prev,
                          int           deriv_order,
                          const float * dv,
                          const float * d2v,
                          int           n) {
    float dt = t_curr - t_prev;

    // Y_j_2, Y_j_1, Y_j — working arrays
    std::vector<float> Y_j_2(xt, xt + n);
    std::vector<float> Y_j_1(xt, xt + n);
    std::vector<float> Y_j(xt, xt + n);
    std::vector<float> vel_approx(n);

    float s2ps = (float) (s * s + s - 2);

    for (int j = 1; j <= s; j++) {
        if (j == 1) {
            float mu_tilde = 6.0f / (float) ((s + 4) * (s - 1));
            for (int i = 0; i < n; i++) {
                Y_j[i] = Y_j_1[i] - dt * mu_tilde * vt[i];
            }
        } else {
            float fraction;
            if (j == 2) {
                fraction = 4.0f / (3.0f * s2ps);
            } else {
                fraction = (float) ((j - 1) * (j - 1) + (j - 1) - 2) / s2ps;
            }

            double bj   = _rkg2_b_coeff(j);
            double bj_1 = _rkg2_b_coeff(j - 1);
            double bj_2 = _rkg2_b_coeff(j - 2);

            float mu         = (float) ((2 * j + 1) * bj / (j * bj_1));
            float nu         = (float) (-(j + 1) * bj / (j * bj_2));
            float mu_tilde_j = mu * 6.0f / (float) ((s + 4) * (s - 1));
            float gamma_tilde = -mu_tilde_j * (float) (1.0 - j * (j + 1) * bj_1 / 2.0);

            float diff_val = -fraction * dt;
            _stork_taylor_approx(vel_approx.data(), deriv_order, diff_val, vt, dv, d2v, n);

            for (int i = 0; i < n; i++) {
                Y_j[i] = mu * Y_j_1[i] + nu * Y_j_2[i] + (1.0f - mu - nu) * xt[i]
                        - dt * mu_tilde_j * vel_approx[i]
                        - dt * gamma_tilde * vt[i];
            }
        }

        // Shift: Y_j_2 = Y_j_1, Y_j_1 = Y_j
        memcpy(Y_j_2.data(), Y_j_1.data(), n * sizeof(float));
        if (j >= 1) {
            memcpy(Y_j_1.data(), Y_j.data(), n * sizeof(float));
        }
    }

    memcpy(xt_out, Y_j.data(), n * sizeof(float));
    return !_stork_has_nan_inf(xt_out, n);
}


static void solver_stork2_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & state,
                               SolverModelFn /*model_fn*/,
                               float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;

    // Bootstrap: step 0 uses plain Euler
    if (state.step_index == 0) {
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        _stork_update_history(state, vt, n, dt);
        state.step_index = 1;
        return;
    }

    // Compute velocity derivatives from history
    std::vector<float> dv(n), d2v(n);
    int deriv_order = _stork_compute_derivatives(vt, n, state, dv.data(), d2v.data());

    // Adaptive sub-stepping: try s, halve on NaN until s=2
    int s = std::max(state.stork_substeps, 2);
    std::vector<float> xt_next(n);
    bool success = false;

    while (s >= 2) {
        if (_rkg2_substep(xt_next.data(), xt, vt, s, t_curr, t_prev,
                          deriv_order, dv.data(), d2v.data(), n)) {
            success = true;
            break;
        }
        s /= 2;
    }

    if (success) {
        memcpy(xt, xt_next.data(), n * sizeof(float));
    } else {
        // Euler fallback
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        fprintf(stderr, "[STORK2] step %d t=%.4f: all sub-steps NaN -> Euler fallback\n",
                state.step_index, t_curr);
    }

    _stork_update_history(state, vt, n, dt);
    state.step_index++;
}


// ---------------------------------------------------------------------------
// STORK4: ROCK4 Chebyshev sub-stepping
// ---------------------------------------------------------------------------

// Find optimal degree in the precomputed MS table
static void _rock4_mdegr(int s, int & mdeg, int & mz, int & mr) {
    int mp1 = 1;
    mdeg = s;
    mz = 0;

    for (int i = 0; i < STORK4_MS_LEN; i++) {
        if (STORK4_MS[i] >= s) {
            mdeg = STORK4_MS[i];
            mz = i;
            mr = mp1 - 1;
            return;
        }
        mp1 += STORK4_MS[i] * 2 - 1;
    }
    // Didn't find — use last entry
    mz = STORK4_MS_LEN - 1;
    mdeg = STORK4_MS[mz];
    mr = mp1 - 1;
}

// Run the ROCK4 sub-stepping loop. Returns true if result is finite.
static bool _rock4_substep(float *       xt_out,
                           const float * xt,
                           const float * vt,
                           int           s,
                           float         t_curr,
                           float         t_prev,
                           int           deriv_order,
                           const float * dv,
                           const float * d2v,
                           int           n) {
    // Clamp s to max supported ROCK4 degree
    int max_rock4 = STORK4_MS[STORK4_MS_LEN - 1];
    if (s > max_rock4) s = max_rock4;

    int mdeg, mz, mr;
    _rock4_mdegr(s, mdeg, mz, mr);

    float dt = t_curr - t_prev;

    std::vector<float> Y_j_2(xt, xt + n);
    std::vector<float> Y_j_1(xt, xt + n);
    std::vector<float> Y_j(n);
    std::vector<float> vel_approx(n);

    // ci1, ci2, ci3 track the "time" at each sub-step (scalar)
    float ci1 = t_curr, ci2 = t_curr, ci3 = t_curr;

    // Part 1: ROCK4 Chebyshev sub-stepping
    for (int j = 1; j <= mdeg; j++) {
        if (j == 1) {
            float temp1_val = -dt * (float) STORK4_RECF[mr];
            ci1 = t_curr + temp1_val;
            ci2 = ci1;
            for (int i = 0; i < n; i++) {
                Y_j_1[i] = xt[i] + temp1_val * vt[i];
            }
        } else {
            float diff_val = ci1 - t_curr;
            _stork_taylor_approx(vel_approx.data(), deriv_order, diff_val, vt, dv, d2v, n);

            float temp1_val =  -dt * (float) STORK4_RECF[mr + 2 * (j - 2) + 1];
            float temp3_val = -(float) STORK4_RECF[mr + 2 * (j - 2) + 2];
            float temp2_val = 1.0f - temp3_val;

            float ci1_new = temp1_val + temp2_val * ci2 + temp3_val * ci3;

            for (int i = 0; i < n; i++) {
                Y_j[i] = temp1_val * vel_approx[i] + temp2_val * Y_j_1[i] + temp3_val * Y_j_2[i];
            }

            memcpy(Y_j_2.data(), Y_j_1.data(), n * sizeof(float));
            memcpy(Y_j_1.data(), Y_j.data(), n * sizeof(float));
            ci3 = ci2;
            ci2 = ci1_new;
            ci1 = ci1_new;
        }
    }

    // After the loop, Y_j_1 holds the sub-stepped result, ci1 holds its time position

    // Part 2: ROCK4 finishing four-step procedure
    // The Python code reads Y_j as the sub-step output, which is Y_j_1 here (last iteration).
    // Let's use Y_j_1 as the base (it was updated to Y_j in the last iteration of the loop).
    float * Y_base = Y_j_1.data();

    // F1, F2, F3, F4: velocity evaluations at finishing stages
    std::vector<float> F1(n), F2(n), F3(n), F4(n);
    std::vector<float> Y_finish(n);

    // Step 1
    float fpa0 = -dt * (float) STORK4_FPA[mz][0];
    float diff1 = ci1 - t_curr;
    _stork_taylor_approx(F1.data(), deriv_order, diff1, vt, dv, d2v, n);
    for (int i = 0; i < n; i++) {
        Y_finish[i] = Y_base[i] + fpa0 * F1[i];
    }

    // Step 2
    float ci2_f = ci1 + fpa0;
    float fpa1 = -dt * (float) STORK4_FPA[mz][1];
    float fpa2 = -dt * (float) STORK4_FPA[mz][2];
    float diff2 = ci2_f - t_curr;
    _stork_taylor_approx(F2.data(), deriv_order, diff2, vt, dv, d2v, n);
    for (int i = 0; i < n; i++) {
        Y_finish[i] = Y_base[i] + fpa1 * F1[i] + fpa2 * F2[i];
    }

    // Step 3
    ci2_f = ci1 + fpa1 + fpa2;
    float fpa3 = -dt * (float) STORK4_FPA[mz][3];
    float fpa4 = -dt * (float) STORK4_FPA[mz][4];
    float fpa5 = -dt * (float) STORK4_FPA[mz][5];
    float diff3 = ci2_f - t_curr;
    _stork_taylor_approx(F3.data(), deriv_order, diff3, vt, dv, d2v, n);

    // Step 4 (final)
    ci2_f = ci1 + fpa3 + fpa4 + fpa5;
    float fpb0 = -dt * (float) STORK4_FPB[mz][0];
    float fpb1 = -dt * (float) STORK4_FPB[mz][1];
    float fpb2 = -dt * (float) STORK4_FPB[mz][2];
    float fpb3 = -dt * (float) STORK4_FPB[mz][3];
    float diff4 = ci2_f - t_curr;
    _stork_taylor_approx(F4.data(), deriv_order, diff4, vt, dv, d2v, n);

    for (int i = 0; i < n; i++) {
        xt_out[i] = Y_base[i] + fpb0 * F1[i] + fpb1 * F2[i] + fpb2 * F3[i] + fpb3 * F4[i];
    }

    return !_stork_has_nan_inf(xt_out, n);
}


static void solver_stork4_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & state,
                               SolverModelFn /*model_fn*/,
                               float *       /*vt_buf*/) {
    float dt = t_curr - t_prev;

    // Bootstrap: step 0 uses plain Euler
    if (state.step_index == 0) {
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        _stork_update_history(state, vt, n, dt);
        state.step_index = 1;
        return;
    }

    // Compute velocity derivatives from history
    std::vector<float> dv_vec(n), d2v_vec(n);
    int deriv_order = _stork_compute_derivatives(vt, n, state, dv_vec.data(), d2v_vec.data());

    // Adaptive sub-stepping: try s, halve on NaN until s=2
    int s = std::max(state.stork_substeps, 2);
    std::vector<float> xt_next(n);
    bool success = false;

    while (s >= 2) {
        if (_rock4_substep(xt_next.data(), xt, vt, s, t_curr, t_prev,
                           deriv_order, dv_vec.data(), d2v_vec.data(), n)) {
            success = true;
            break;
        }
        s /= 2;
    }

    if (success) {
        memcpy(xt, xt_next.data(), n * sizeof(float));
    } else {
        // Euler fallback
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        fprintf(stderr, "[STORK4] step %d t=%.4f: all sub-steps NaN -> Euler fallback\n",
                state.step_index, t_curr);
    }

    _stork_update_history(state, vt, n, dt);
    state.step_index++;
}

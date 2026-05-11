#pragma once
// sampler-schedule.h — Schedule building helpers for hot-step-sampler.h
//
// Extracts the custom timestep parsing and scheduler override dispatch
// from the monolithic sampler into focused helpers.
// These are included BY hot-step-sampler.h — not independently compiled.

#include "hot-step-params.h"
#include "lua-plugin-registry.h"
#include "schedulers/scheduler-registry.h" // kept for scheduler_clamp()

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Parse g_hotstep_params.custom_timesteps CSV into a schedule vector.
// Returns true if custom timesteps are active (overriding everything else).
// On success, out_schedule is populated and out_num_steps is updated.
static bool sampler_parse_custom_timesteps(
    std::vector<float> & out_schedule,
    int &                out_num_steps
) {
    if (g_hotstep_params.custom_timesteps.empty()) return false;

    const char * p = g_hotstep_params.custom_timesteps.c_str();
    out_schedule.clear();
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char * end = nullptr;
        float v = strtof(p, &end);
        if (end == p) break;  // no parse progress
        out_schedule.push_back(v);
        p = end;
    }
    // Drop trailing 0 (x0 endpoint) — sampler handles final step separately
    if (!out_schedule.empty() && out_schedule.back() == 0.0f) {
        out_schedule.pop_back();
    }
    if (!out_schedule.empty()) {
        out_num_steps = (int) out_schedule.size();
        fprintf(stderr, "[DiT] Custom timesteps: %d steps (overrides scheduler)\n", out_num_steps);
        return true;
    }
    return false;
}

// Build a custom schedule from the configured scheduler name.
// Handles both standard schedulers and composite (A+B:crossover:split) syntax.
// shift_val is back-calculated from the existing schedule if possible.
// On return, out_schedule contains num_steps values.
static void sampler_build_scheduler_override(
    std::vector<float> & out_schedule,
    int                  num_steps,
    const float *        existing_schedule
) {
    out_schedule.resize(num_steps);

    // Back-calculate shift from the existing schedule
    float shift_val = 1.0f;
    if (num_steps >= 2 && existing_schedule[0] > 0.0f) {
        float u = 1.0f - 1.0f / (float) num_steps;
        float t1 = existing_schedule[1];
        if (t1 > 0.0f && t1 < 1.0f && u > 0.0f) {
            float denom = u * (1.0f - t1);
            if (denom > 1e-8f) {
                shift_val = t1 * (1.0f - u) / denom;
                if (shift_val < 0.5f) shift_val = 1.0f;
                if (shift_val > 10.0f) shift_val = 3.0f;
            }
        }
    }

    const std::string & ss = g_hotstep_params.scheduler;

    // Composite scheduler dispatch
    if (ss.rfind("composite:", 0) == 0) {
        const char * body = ss.c_str() + 10;
        const char * plus = strchr(body, '+');
        if (plus) {
            std::string name_a(body, plus - body);
            const char * after_plus = plus + 1;
            const char * colon1 = strchr(after_plus, ':');
            std::string name_b;
            float crossover = 0.0f, split = 0.5f;
            if (colon1) {
                name_b = std::string(after_plus, colon1 - after_plus);
                crossover = (float) atof(colon1 + 1);
                const char * colon2 = strchr(colon1 + 1, ':');
                if (colon2) split = (float) atof(colon2 + 1);
            } else {
                name_b = std::string(after_plus);
            }
            if (crossover < 0.0f) crossover = 0.0f;
            if (crossover > 1.0f) crossover = 1.0f;
            if (split < 0.0f) split = 0.0f;
            if (split > 1.0f) split = 1.0f;

            auto & reg = PluginRegistry::instance();
            LuaPlugin * sa = reg.scheduler_lookup(name_a.c_str());
            LuaPlugin * sb = reg.scheduler_lookup(name_b.c_str());
            if (!sa) sa = reg.scheduler_lookup("linear");
            if (!sb) sb = reg.scheduler_lookup("linear");

            std::vector<float> va(num_steps), vb(num_steps);
            auto & pp = g_hotstep_params.plugin_params;
            lua_call_scheduler(*sa, va.data(), num_steps, shift_val, pp);
            lua_call_scheduler(*sb, vb.data(), num_steps, shift_val, pp);

            float zone_lo = split - crossover * 0.5f;
            float zone_hi = split + crossover * 0.5f;
            for (int i = 0; i < num_steps; i++) {
                float frac = (float) i / (float) num_steps;
                float w;
                if (crossover < 1e-6f || frac <= zone_lo) {
                    w = (frac < split) ? 0.0f : 1.0f;
                } else if (frac >= zone_hi) {
                    w = 1.0f;
                } else {
                    w = (frac - zone_lo) / (zone_hi - zone_lo);
                }
                out_schedule[i] = (1.0f - w) * va[i] + w * vb[i];
            }
            for (int i = 1; i < num_steps; i++) {
                if (out_schedule[i] > out_schedule[i-1])
                    out_schedule[i] = out_schedule[i-1];
            }
            scheduler_clamp(out_schedule.data(), num_steps);
            fprintf(stderr, "[DiT] Custom schedule: composite %s+%s (cross=%.2f, split=%.2f), shift=%.2f\n",
                    sa->display_name.c_str(), sb->display_name.c_str(), crossover, split, shift_val);
        }
    } else {
        // Standard scheduler lookup
        auto & reg = PluginRegistry::instance();
        LuaPlugin * sched = reg.scheduler_lookup(ss.c_str());
        if (!sched) {
            fprintf(stderr, "[DiT] WARNING: unknown scheduler '%s', using linear\n", ss.c_str());
            sched = reg.scheduler_lookup("linear");
        }
        if (!sched) {
            // No plugins loaded at all — hardcoded linear fallback to avoid nullptr crash
            fprintf(stderr, "[DiT] ERROR: no scheduler plugins loaded, using hardcoded linear\n");
            for (int i = 0; i < num_steps; i++) {
                out_schedule[i] = 1.0f - (float) i / (float) num_steps;
            }
        } else {
            auto & pp = g_hotstep_params.plugin_params;
            lua_call_scheduler(*sched, out_schedule.data(), num_steps, shift_val, pp);
            fprintf(stderr, "[DiT] Custom schedule: %s (%s), shift=%.2f\n",
                    sched->display_name.c_str(), sched->name.c_str(), shift_val);
        }
    }
}

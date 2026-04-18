#pragma once
// scheduler-registry.h: Compile-time registry mapping scheduler names to functions
//
// Usage:
//   const SchedulerInfo* info = scheduler_lookup("beta57");
//   if (info) info->fn(output, num_steps, shift);

#include "scheduler-interface.h"
#include "scheduler-implementations.h"

#include <cstring>

struct SchedulerInfo {
    const char *  name;          // internal identifier (lowercase)
    const char *  display_name;  // human-readable name for UI
    SchedulerFn   fn;            // schedule function pointer
    const char *  description;   // short description
};

// All registered schedulers — update this array when adding new schedulers.
static const SchedulerInfo SCHEDULER_REGISTRY[] = {
    {"linear",           "Linear",              scheduler_linear,           "Uniform spacing (default)"},
    {"ddim_uniform",     "DDIM Uniform",        scheduler_ddim_uniform,     "Log-SNR uniform (S-shaped)"},
    {"sgm_uniform",      "SGM-Uniform (Karras)",scheduler_sgm_uniform,     "Karras σ-ramp (ρ=7), front-loads structural steps"},
    {"bong_tangent",     "Tangent",             scheduler_bong_tangent,     "Front-loaded (structural focus)"},
    {"linear_quadratic", "Linear-Quadratic",    scheduler_linear_quadratic, "Linear start, quadratic finish"},
    {"cosine",           "Cosine",              scheduler_cosine,           "Cosine annealing — balanced S-curve"},
    {"power",            "Power (p=2)",         scheduler_power,            "Power-law t^p, front-loaded"},
    {"beta57",           "Beta 57",             scheduler_beta57,           "Beta(0.5,0.7) — smooth S-curve from RES4LYF"},
};

static const int SCHEDULER_REGISTRY_SIZE = (int) (sizeof(SCHEDULER_REGISTRY) / sizeof(SCHEDULER_REGISTRY[0]));

// Look up a scheduler by name. Returns nullptr if not found.
// Aliases: "karras" -> "sgm_uniform"
static const SchedulerInfo * scheduler_lookup(const char * name) {
    if (!name || !name[0]) return &SCHEDULER_REGISTRY[0]; // default: linear

    // Aliases
    const char * resolved = name;
    if (strcmp(name, "karras") == 0) resolved = "sgm_uniform";

    for (int i = 0; i < SCHEDULER_REGISTRY_SIZE; i++) {
        if (strcmp(SCHEDULER_REGISTRY[i].name, resolved) == 0) {
            return &SCHEDULER_REGISTRY[i];
        }
    }
    return nullptr;
}

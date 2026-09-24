// Self-check for yue2_aitk_runtime::lr_schedule_multiplier. Header-only, no
// CUDA: cl /std:c++17 /EHsc /I..\src\train test-yue2-lr-schedule.cpp
#include "yue2-aitk-runtime.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

using yue2_aitk_runtime::Config;
using yue2_aitk_runtime::lr_schedule_multiplier;

static bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

int main() {
    Config c; c.steps = 500; c.warmup = 10;
    // The default must be the expression the trainer always used.
    for (int step : {0, 5, 9, 10, 140, 499, 500}) {
        double old = 1.0;
        if (c.warmup > 0 && step < c.warmup) old *= (double)(step + 1) / (double)c.warmup;
        else old *= 0.5 * (1.0 + std::cos(3.14159265358979 * std::min(1.0, (double)(step - c.warmup) / (double)std::max<int32_t>(1, c.steps - c.warmup))));
        assert(lr_schedule_multiplier(c, step, -1) == old);
    }
    assert(near(lr_schedule_multiplier(c, 4, -1), 0.5));           // warmup wins for every shape
    c.lr_schedule = "constant"; assert(lr_schedule_multiplier(c, 400, -1) == 1.0);
    c.lr_schedule = "linear"; assert(near(lr_schedule_multiplier(c, 255, -1), 0.5));
    c.lr_schedule = "cosine-floor"; c.lr_floor = 0.1f;
    assert(near(lr_schedule_multiplier(c, 500, -1), 0.1f));
    c.lr_schedule = "wsd"; c.lr_decay_steps = 40;
    assert(lr_schedule_multiplier(c, 300, -1) == 1.0);             // plateau
    assert(lr_schedule_multiplier(c, 300, 300) == 1.0);            // tail's first step
    assert(near(lr_schedule_multiplier(c, 320, 300), 0.5));        // linear midpoint
    assert(near(lr_schedule_multiplier(c, 340, 300), 0.0));        // tail spent
    c.lr_decay_shape = "cosine"; assert(near(lr_schedule_multiplier(c, 320, 300), 0.5));
    c.lr_schedule = "sgdr"; c.lr_cycle_steps = 100; c.lr_cycle_mult = 2.0f;
    assert(near(lr_schedule_multiplier(c, 10, -1), 1.0));          // cycle 1 start
    assert(near(lr_schedule_multiplier(c, 110, -1), 1.0));         // cycle 2 (200 long) start
    assert(near(lr_schedule_multiplier(c, 210, -1), 0.5));         // halfway through cycle 2
    std::puts("ok");
}

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

// Read-only check, shared by synthesis backends before audio encoding.
static inline bool synth_finite(const float * data, size_t n, const char * stage,
                                std::string * err = nullptr) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) count += !std::isfinite(data[i]);
    if (!count) return true;
    const std::string message = std::string(stage) + ": " + std::to_string(count) +
                                " non-finite values out of " + std::to_string(n);
    fprintf(stderr, "[Synth-Finite] %s\n", message.c_str());
    if (err) *err = message;
    return false;
}

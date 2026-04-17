#pragma once
// timer.h: simple wall-clock timer (steady_clock)

#include <chrono>

struct Timer {
    std::chrono::steady_clock::time_point t;

    Timer() : t(std::chrono::steady_clock::now()) {}

    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }

    void reset() { t = std::chrono::steady_clock::now(); }
};

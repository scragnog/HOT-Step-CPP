#pragma once
// adapter-cancel.h: cooperative cancel hook for the LoKr/LoRA delta precompute.
//
// The precompute loops in adapter-runtime.h read `g_adapter_cancel` between
// every delta. The ace-server worker (engine/tools/ace-server.cpp) sets it
// to point at the active job's atomic<bool> cancel flag right before calling
// ace_synth_load and clears it after. Without this hook a wrapper-side
// cancel during the 17 s cold-start adapter precompute has no effect — the
// load runs to completion before the engine can react.
//
// Split into its own header so ace-server.cpp does not need to pull in all
// of adapter-runtime.h's heavy ggml/safetensors transitive dependencies.

#include <atomic>

// Set by the worker around ace_synth_load (and similar load paths). Reads
// from inside the precompute loops are cheap (one acquire-load + one
// relaxed-load on the pointee). C++17 `inline` keeps a single instance
// across translation units.
inline std::atomic<const std::atomic<bool> *> g_adapter_cancel{ nullptr };

inline bool adapter_cancel_requested() {
    const std::atomic<bool> * flag = g_adapter_cancel.load(std::memory_order_acquire);
    return flag != nullptr && flag->load(std::memory_order_relaxed);
}

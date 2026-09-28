#pragma once
// gpu-mem.h — honest device VRAM, via NVML when it is available.
//
// WHY THIS EXISTS. Every VRAM number the trainer prints used to come from
// ggml_backend_dev_memory(), which on CUDA is cudaMemGetInfo(). Under Windows'
// WDDM driver model that number is a lie for our purposes: memory another
// process has committed but is not actively touching is EVICTABLE, and
// cudaMemGetInfo counts it as free. Measured on this machine: 31 GB "free"
// while nvidia-smi reported 22.4 GB free at the same instant. The auto-fit then
// sizes a run against ~9 GB that does not exist, the allocation "succeeds"
// because Windows silently backs it with shared system memory, and the run
// trains at a crawl instead of failing honestly.
//
// NVML reports what nvidia-smi reports — the real per-device totals. It is
// loaded DYNAMICALLY (nvml.dll ships with the driver, not with us) so a build
// without it, a non-NVIDIA card, or a stripped driver install all degrade to the
// existing ggml numbers rather than failing to start.
//
// DiT-side only: lm-vram.h and lm-common.h's probes are untouched, so the LM
// trainer's numbers stay bit-for-bit what they were.

#include "train/lm-common.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

struct DitGpuMem {
    size_t total = 0;  // bytes
    size_t used  = 0;
    size_t free  = 0;
    bool   nvml  = false;  // true = NVML figures, false = ggml_backend_dev_memory

    size_t total_mb() const { return total / (1024 * 1024); }
    size_t used_mb() const { return used / (1024 * 1024); }
    size_t free_mb() const { return free / (1024 * 1024); }
    const char * source() const { return nvml ? "nvml" : "ggml"; }
};

#ifdef _WIN32

// Minimal local mirror of the NVML ABI — we never link against nvml.lib, and
// pulling in the CUDA toolkit's nvml.h would make the build depend on it.
// Field ORDER matters: nvmlMemory_t is {total, free, used}.
struct DitNvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef int (*dit_nvml_init_fn)(void);
typedef int (*dit_nvml_shutdown_fn)(void);
typedef int (*dit_nvml_handle_fn)(unsigned int, void **);
typedef int (*dit_nvml_meminfo_fn)(void *, DitNvmlMemory *);

// Queries device 0 (the trainer is single-GPU by construction — it takes
// ggml_backend_init_best()). Returns false on any failure, leaving *out alone.
static bool dit_gpu_mem_nvml(DitGpuMem * out) {
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (!lib) {
        // Not on PATH on some driver installs; the SDK location is stable.
        lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    }
    if (!lib) {
        return false;
    }
    const dit_nvml_init_fn     f_init  = (dit_nvml_init_fn) (void *) GetProcAddress(lib, "nvmlInit_v2");
    const dit_nvml_shutdown_fn f_down  = (dit_nvml_shutdown_fn) (void *) GetProcAddress(lib, "nvmlShutdown");
    const dit_nvml_handle_fn   f_hand  = (dit_nvml_handle_fn) (void *) GetProcAddress(lib,
                                                                                      "nvmlDeviceGetHandleByIndex_v2");
    const dit_nvml_meminfo_fn  f_mem   = (dit_nvml_meminfo_fn) (void *) GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
    bool                       ok      = false;
    if (f_init && f_hand && f_mem && f_init() == 0) {
        void * dev = nullptr;
        if (f_hand(0, &dev) == 0 && dev) {
            DitNvmlMemory m = { 0, 0, 0 };
            if (f_mem(dev, &m) == 0 && m.total > 0) {
                out->total = (size_t) m.total;
                out->free  = (size_t) m.free;
                out->used  = (size_t) m.used;
                out->nvml  = true;
                ok         = true;
            }
        }
        if (f_down) {
            f_down();
        }
    }
    FreeLibrary(lib);
    return ok;
}

#else

static bool dit_gpu_mem_nvml(DitGpuMem * out) {
    (void) out;
    return false;  // NVML is loaded dynamically on Windows only (D-WDDM)
}

#endif  // _WIN32

// NVML when it answers, ggml_backend_dev_memory() otherwise. `source()` tells
// the caller which it got, so the JSONL can be honest about it.
static DitGpuMem dit_gpu_mem_query(ggml_backend_t backend) {
    DitGpuMem g;
    if (dit_gpu_mem_nvml(&g)) {
        return g;
    }
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    g.total = t;
    g.free  = f;
    g.used  = (t > f) ? (t - f) : 0;
    g.nvml  = false;
    return g;
}

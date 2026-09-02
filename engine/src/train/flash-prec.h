#pragma once
// flash-prec.h — the fused-attention capability probe and the RESOLVED-precision
// readback, shared by every trainer that carries `--attn exact|flash|flash-f32`.
//
// These three functions were written for the DiT trainer and lived in
// dit-train-graph.h until the AS1.5 LM port (R2, docs/plans/2026-09-02-lm-flash-attn.md
// D5/D6) needed exactly the same two answers. They are MOVED here verbatim,
// names and bodies unchanged, so dit-train-graph.h keeps emitting the same graph
// and calling the same symbols — a second copy under a second name is the one
// thing the adoption contract forbids, because the two would drift and the probe
// is the check that stops a silent CPU split.
//
// Deliberately dependency-free (ggml only): train/lm-common.h is pulled in by
// train/dit-adapter.h, so a header that both lm-graph.h and dit-train-graph.h
// include must not reach back into either tree or the include graph cycles.
//
// .claude/skills/flash-attn-training/SKILL.md §2 points 2 and 6.

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

// ─── spec §9.8: the supports_op probe ───────────────────────────────
//
// A `false` from ggml_backend_supports_op is not a failure the trainer would
// ever see: backend_sched_new registers the CPU backend alongside the GPU one,
// so the scheduler would simply SPLIT the graph — Q/K/V and the F16 mask copied
// across PCIe for every layer of every step. Correct, unusably slow, and with
// LOW VRAM and a silent NVML tripwire, i.e. indistinguishable from a pass on
// every number the run reports. So flash mode asks the question explicitly, at
// init, before the first graph, and refuses to start rather than fall back.
//
// The probe uses the run's real D / Nh / Nkv / S / S_kv / B because a capability
// check is allowed to be shape-dependent, and ours is (D must be 64 or 128).
// Contiguous probe tensors are faithful enough: the only stride the CUDA check
// looks at is nb[0], which is sizeof(float) for the real permuted views too.
static bool dit_flash_probe(ggml_backend_t backend, int D, int Nh, int Nkv, int S, int S_kv, int B, float scale,
                            bool * fwd_ok, bool * bwd_ok) {
    *fwd_ok = false;
    *bwd_ok = false;
    ggml_context * probe;
    {
        ggml_init_params p = { 64 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        probe              = ggml_init(p);
    }
    if (!probe) {
        return false;
    }
    ggml_tensor * pq = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S, Nh, B);
    ggml_tensor * pk = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * pv = ggml_new_tensor_4d(probe, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * pm = ggml_new_tensor_2d(probe, GGML_TYPE_F16, S_kv, S);
    ggml_tensor * pf = ggml_flash_attn_train(probe, pq, pk, pv, pm, scale);
    *fwd_ok          = ggml_backend_supports_op(backend, pf);
    // The back node needs a forward-shaped `fwd` and a same-sized `dfwd`; the
    // forward node itself is exactly the former.
    ggml_tensor * pd = ggml_new_tensor_1d(probe, GGML_TYPE_F32, ggml_flash_attn_train_nelements(pq));
    ggml_tensor * pb = ggml_flash_attn_train_back(probe, pq, pk, pv, pm, pf, pd, scale);
    *bwd_ok          = ggml_backend_supports_op(backend, pb);
    ggml_free(probe);
    return *fwd_ok && *bwd_ok;
}

// HOT-Step patch: flash-attn-train
//
// Which arithmetic the backend's LAST fused-attention launch actually used
// (dir 0 = forward, 1 = backward): "tf32", or "f32 (<reason>)" when the request
// was overridden, or "n/a" when the backend has no such kernels or none has run.
//
// This is not a restatement of --attn. The request is a request: the CUDA
// dispatch drops to v1's scalar kernels on pre-Ampere devices, at D != 128, and
// on an 8-byte-unaligned view (tf32 design §3.3), so two runs whose logs both
// say "flash" can differ in arithmetic depending on the GPU they landed on.
// Resolved through the backend registry rather than linked, because ggml-cuda is
// a loadable module and the trainer must build without it.
static const char * dit_flash_last_prec(ggml_backend_t backend, int dir) {
    typedef const char * (*fa_last_prec_fn)(int);
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return "n/a";
    }
    fa_last_prec_fn fn =
        (fa_last_prec_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fattn_train_last_prec");
    return fn ? fn(dir) : "n/a";
}

// The two directions collapsed into one log field. They resolve through the same
// helper and normally agree; when they do not, say so rather than pick one.
static std::string dit_flash_prec_label(ggml_backend_t backend) {
    const std::string f = dit_flash_last_prec(backend, 0);
    const std::string b = dit_flash_last_prec(backend, 1);
    return (f == b) ? f : ("fwd " + f + " / bwd " + b);
}

#pragma once
// yue2/yue2-imatrix.h — activation-importance ("imatrix") collection for the
// YuE2 LM (AR + NAR share one GGUF, one tmap_lm), and the GGUF file it writes
// for engine/tools/quantize.cpp.
//
// HOT-Step file (does not exist upstream, no acestep.cpp analog). Copied from
// minimax/mm3-imatrix.h (renamed symbols only — do not include or modify
// anything under minimax/, that is MM3's private tree) and simplified for
// YuE2's single tmap: MM3 collects across two GGUFs (tmap_lm + tmap_synth)
// with non-colliding name prefixes; YuE2's AR blocks (blk.N.attn_*/ffn_*) and
// NAR twins (nar_blk.N.nar_attn_*/nar_ffn_*, per docs/plans/yue2/
// 05-gguf-layout.md §3) already live in the one tmap_lm, so there is only
// ever one allow-set to build. Included by yue2-lm-graph.h (arms the AR
// prefill/decode hooks) and yue2-nar-graph.h (arms the velocity-step hook),
// and by yue2-server.h, which owns POST /yue2/imatrix.
//
// See minimax/mm3-imatrix.h's own header comment for the full rationale (why
// this beats round-to-nearest, why IQ2_XXS/IQ2_XS/IQ1_S refuse to run
// without one, how the eval-callback mechanism works, why it must run on the
// unquantized LM, and the exact GGUF layout quantize.cpp's --imatrix loader
// expects). Reproduced only where the two diverge below.
//
// ── Divergence from MM3 worth flagging ───────────────────────────────────────
//
//   1. THE ADAPTER WARNING IS LIVE AGAIN. This note used to say YuE2 had no
//      adapters at all, so MM3's "an LM adapter is merged in" warning had
//      nothing to check. Phase 3 of docs/plans/yue2/08-nar-lora-trainer.md
//      expired that reasoning: yue2-adapter.h merges NAR LoRAs into the
//      resident LM at load time, and collecting an imatrix through one
//      measures the ADAPTED model while quantize.cpp would apply the result
//      to the base GGUF. yue2-server.h's arm path warns on
//      g_yue2.lm_adapter_desc, the way mm3-server.h:2138-2141 does. There is
//      still no RUNTIME (unmerged) adapter path, so there is nothing else to
//      check.
//
//   2. ONE hook per call site, armed on THREE schedulers instead of MM3's two
//      (prefill + decode) plus a depth-graph twin: yue2_ar_prefill's sched,
//      yue2_ar_decode_step's sched (yue2-lm-graph.h), and
//      yue2_nar_velocity's sched (yue2-nar-graph.h) — one prefill per chunk,
//      one decode step per AR token, two velocity evals per ODE step. All
//      three are freshly created per call (YuE2 does not keep a persistent
//      per-slot scheduler the way MM3's g->prefill.sched does), so the hook
//      call sits right after backend_sched_new()/sched_alloc_graph() and
//      before compute at each site, same as MM3's persistent-slot pattern
//      just re-armed every time instead of once.
//
//   3. yue2_ar_forward() (the one-shot teacher-forced full forward used only
//      by yue2-probe.cpp's --ar-parity check) is deliberately NOT hooked —
//      it never runs during a real /yue2/synth job, so calibrating through
//      it would not describe production activations any better than the
//      real prefill/decode path does, for one more schedulers to keep in
//      sync.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct Yue2ImatrixEntry {
    std::vector<double> sum2;      // [n_in] summed squares down each input column
    int64_t             rows = 0;  // activation rows accepted into sum2
    int64_t             bad  = 0;  // rows rejected for holding a non-finite value
};

struct Yue2ImatrixState {
    bool                                    armed = false;
    std::set<std::string>                   allow;    // collectable weight names
    std::map<std::string, Yue2ImatrixEntry> ent;
    std::vector<std::string>                sources;  // labels of the runs so far
    int64_t                                 runs = 0;

    // Diagnostics. A silent zero here is the failure mode that would otherwise
    // produce a plausible-looking but useless imatrix, so they are reported.
    int64_t nodes          = 0;  // matmuls collected
    int64_t skipped_type   = 0;  // src[1] was neither F32 nor F16
    int64_t skipped_stride = 0;  // src[1] rows were not contiguous in ne[0]
    int64_t skipped_shape  = 0;  // src[1] ne[0] disagreed with an earlier call
    int64_t bad_rows       = 0;  // rows dropped for holding a non-finite value

    // Scratch, kept across calls so a 28-layer forward does not churn the heap.
    std::vector<uint8_t> scratch;
    std::vector<float>   conv;
};

static Yue2ImatrixState g_yue2_imatrix;

// Follow a view chain back to the tensor that actually owns the data.
static const ggml_tensor * yue2_imatrix_base(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool yue2_imatrix_wanted(const ggml_tensor * t, std::string * name) {
    if (!t || t->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = yue2_imatrix_base(t->src[0]);
    if (!w || w->name[0] == '\0') {
        return false;
    }
    if (g_yue2_imatrix.allow.find(w->name) == g_yue2_imatrix.allow.end()) {
        return false;
    }
    if (name) {
        name->assign(w->name);
    }
    return true;
}

// One activation row into the accumulator, or dropped whole (not
// element-wise — see minimax/mm3-imatrix.h's mm3_imatrix_accumulate comment
// for why a partially-finite row is not trustworthy either).
static bool yue2_imatrix_accumulate(Yue2ImatrixEntry & e, const float * f, int64_t n_in) {
    for (int64_t j = 0; j < n_in; j++) {
        if (!std::isfinite(f[j])) {
            e.bad++;
            g_yue2_imatrix.bad_rows++;
            return false;
        }
    }
    for (int64_t j = 0; j < n_in; j++) {
        const double v = (double) f[j];
        e.sum2[(size_t) j] += v * v;
    }
    e.rows++;
    return true;
}

static bool yue2_imatrix_eval_cb(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;

    std::string wname;
    if (!yue2_imatrix_wanted(t, &wname)) {
        return false;
    }
    if (ask) {
        return true;
    }

    const ggml_tensor * x = t->src[1];
    if (!x) {
        return true;
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) {
        g_yue2_imatrix.skipped_type++;
        return true;
    }
    const size_t ts = ggml_type_size(x->type);
    if (x->nb[0] != ts) {
        // A transposed src[1] would make "one row" non-contiguous; the column
        // walk below would read the wrong numbers rather than fail, so refuse.
        g_yue2_imatrix.skipped_stride++;
        return true;
    }

    const int64_t n_in = x->ne[0];

    Yue2ImatrixEntry & e = g_yue2_imatrix.ent[wname];
    if (e.sum2.empty()) {
        e.sum2.assign((size_t) n_in, 0.0);
    } else if ((int64_t) e.sum2.size() != n_in) {
        g_yue2_imatrix.skipped_shape++;
        return true;
    }

    g_yue2_imatrix.conv.resize((size_t) n_in);

    if (ggml_is_contiguous(x)) {
        // One device->host copy for the whole activation block rather than
        // one memcpy per row — prefill can be a few hundred rows per matmul.
        const size_t nb = ggml_nbytes(x);
        g_yue2_imatrix.scratch.resize(nb);
        ggml_backend_tensor_get(x, g_yue2_imatrix.scratch.data(), 0, nb);

        const int64_t nrows = ggml_nelements(x) / n_in;
        for (int64_t r = 0; r < nrows; r++) {
            const uint8_t * p = g_yue2_imatrix.scratch.data() + (size_t) r * (size_t) n_in * ts;
            if (x->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, g_yue2_imatrix.conv.data(), n_in);
                yue2_imatrix_accumulate(e, g_yue2_imatrix.conv.data(), n_in);
            } else {
                yue2_imatrix_accumulate(e, (const float *) p, n_in);
            }
        }
    } else {
        g_yue2_imatrix.scratch.resize((size_t) n_in * ts);
        for (int64_t i3 = 0; i3 < x->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < x->ne[1]; i1++) {
                    const size_t off = (size_t) i1 * x->nb[1] + (size_t) i2 * x->nb[2] + (size_t) i3 * x->nb[3];
                    ggml_backend_tensor_get(x, g_yue2_imatrix.scratch.data(), off, (size_t) n_in * ts);
                    if (x->type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) g_yue2_imatrix.scratch.data(),
                                              g_yue2_imatrix.conv.data(), n_in);
                        yue2_imatrix_accumulate(e, g_yue2_imatrix.conv.data(), n_in);
                    } else {
                        yue2_imatrix_accumulate(e, (const float *) g_yue2_imatrix.scratch.data(), n_in);
                    }
                }
            }
        }
    }

    g_yue2_imatrix.nodes++;
    return true;
}

// Attach or detach the hook on one scheduler. Called immediately before every
// graph compute rather than once at scheduler creation: all three YuE2 call
// sites (yue2_ar_prefill, yue2_ar_decode_step, yue2_nar_velocity) build a
// fresh ggml_backend_sched_t per call, so there is no persistent slot to arm
// once and forget the way MM3's g->prefill.sched is.
static void yue2_imatrix_hook(ggml_backend_sched_t sched) {
    if (!sched) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, g_yue2_imatrix.armed ? yue2_imatrix_eval_cb : nullptr,
                                          &g_yue2_imatrix);
}

static void yue2_imatrix_reset() {
    g_yue2_imatrix.ent.clear();
    g_yue2_imatrix.sources.clear();
    g_yue2_imatrix.runs           = 0;
    g_yue2_imatrix.nodes          = 0;
    g_yue2_imatrix.skipped_type   = 0;
    g_yue2_imatrix.skipped_stride = 0;
    g_yue2_imatrix.skipped_shape  = 0;
    g_yue2_imatrix.bad_rows       = 0;
}

// Tensors that have seen at least one finite row. The gap between this and
// ent.size() is the number that would be written as all-zeros, so it is the
// number that matters when judging whether a collection run is usable.
static size_t yue2_imatrix_usable() {
    size_t n = 0;
    for (const auto & kv : g_yue2_imatrix.ent) {
        if (kv.second.rows > 0) {
            n++;
        }
    }
    return n;
}

// Total activation rows seen, summed over every collected tensor. Reporting
// only: a session that ends with 0 here collected nothing at all.
static int64_t yue2_imatrix_total_rows() {
    int64_t n = 0;
    for (const auto & kv : g_yue2_imatrix.ent) {
        n += kv.second.rows;
    }
    return n;
}

static bool yue2_imatrix_save(const std::string & path, std::string * err) {
    if (g_yue2_imatrix.ent.empty()) {
        if (err) {
            *err = "nothing collected — arm it, run at least one /yue2/synth, then save";
        }
        return false;
    }

    // An entry with no accepted rows is all zeros. Writing it would hand the
    // quantizer an importance vector that says "every column is worthless",
    // which is strictly worse than handing it nothing -- so it is omitted, and
    // quantize.cpp falls back to round-to-nearest for that tensor.
    size_t                                                          n_nonfinite_sums = 0;
    std::vector<const std::pair<const std::string, Yue2ImatrixEntry> *> keep;
    for (const auto & kv : g_yue2_imatrix.ent) {
        if (kv.second.rows <= 0) {
            continue;
        }
        bool ok = true;
        for (double v : kv.second.sum2) {
            if (!std::isfinite(v) || v < 0.0) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            n_nonfinite_sums++;
            fprintf(stderr, "[YUE2-IMAT] '%s' accumulated a non-finite sum over %lld rows - omitted\n",
                    kv.first.c_str(), (long long) kv.second.rows);
            continue;
        }
        keep.push_back(&kv);
    }
    if (n_nonfinite_sums) {
        fprintf(stderr, "[YUE2-IMAT] %zu tensors had non-finite accumulators\n", n_nonfinite_sums);
    }
    if (keep.empty()) {
        if (err) {
            *err = "every collected tensor had only non-finite rows — nothing worth writing";
        }
        return false;
    }
    if (keep.size() < g_yue2_imatrix.ent.size()) {
        fprintf(stderr, "[YUE2-IMAT] %zu of %zu tensors had no finite rows and were omitted\n",
                g_yue2_imatrix.ent.size() - keep.size(), g_yue2_imatrix.ent.size());
    }

    size_t n_floats = 0;
    for (const auto * kv : keep) {
        n_floats += kv->second.sum2.size() + 1;  // in_sum2 + counts
    }

    const size_t n_tensors = keep.size() * 2;
    const size_t mem       = ggml_tensor_overhead() * (n_tensors + 16) + n_floats * sizeof(float) +
                       n_tensors * GGML_MEM_ALIGN * 2 + (1u << 20);

    ggml_init_params ip  = { mem, nullptr, /*no_alloc*/ false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        if (err) {
            *err = "could not allocate the imatrix write context";
        }
        return false;
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_val_str(out, "general.type", "imatrix");
    gguf_set_val_str(out, "general.architecture", "yue2");
    gguf_set_val_str(out, "general.name", "YuE2 LM imatrix");
    gguf_set_val_u32(out, "imatrix.chunk_count", (uint32_t) g_yue2_imatrix.runs);
    gguf_set_val_u32(out, "imatrix.chunk_size", 0);
    if (!g_yue2_imatrix.sources.empty()) {
        std::vector<const char *> ds;
        ds.reserve(g_yue2_imatrix.sources.size());
        for (const auto & s : g_yue2_imatrix.sources) {
            ds.push_back(s.c_str());
        }
        gguf_set_arr_str(out, "imatrix.datasets", ds.data(), (int) ds.size());
    }

    for (const auto * kvp : keep) {
        const std::string &      name = kvp->first;
        const Yue2ImatrixEntry & e    = kvp->second;

        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) e.sum2.size());
        ggml_set_name(s, (name + ".in_sum2").c_str());
        float * sd = (float *) s->data;
        for (size_t j = 0; j < e.sum2.size(); j++) {
            sd[j] = (float) e.sum2[j];
        }
        gguf_add_tensor(out, s);

        ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(c, (name + ".counts").c_str());
        ((float *) c->data)[0] = (float) e.rows;
        gguf_add_tensor(out, c);
    }

    const bool ok = gguf_write_to_file(out, path.c_str(), /*only_meta*/ false);
    gguf_free(out);
    ggml_free(ctx);

    if (!ok && err) {
        *err = "could not write " + path;
    }
    return ok;
}

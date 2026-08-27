#pragma once
// minimax/mm3-imatrix.h — activation-importance ("imatrix") collection for the
// MiniMax-Music3 LM, and the GGUF file it writes for engine/tools/quantize.cpp.
//
// HOT-Step file (does not exist upstream). Included by minimax/mm3-lm-graph.h,
// which arms the hook, and by minimax/mm3-server.h, which owns POST /mm3/imatrix.
//
// ── What this is for ─────────────────────────────────────────────────────────
//
// Plain round-to-nearest quantization minimises the error in the WEIGHTS. That
// is the wrong objective: what matters is the error in the OUTPUT, and a weight
// column multiplying a near-dead activation can be mangled for free while a
// column multiplying a hot one cannot. An importance matrix is the missing
// term — one non-negative number per input column of every weight matrix,
// measured by actually running the model, that the k-quant and i-quant solvers
// use to weight their per-block error.
//
// It matters most exactly where our Q2_K hurts. At 8 bits the rounding error is
// below the noise floor either way; at 2-3 bits the solver is making real
// choices about what to throw away, and without an imatrix it makes them blind.
//
// GGML's IQ2_XXS / IQ2_XS / IQ1_S do not merely benefit from this, they REFUSE
// to run without it (ggml_quantize_requires_imatrix, ggml.c). So this file is
// the prerequisite for the whole IQ family, not just a Q2_K improvement.
//
// ── How it is collected ──────────────────────────────────────────────────────
//
// ggml_backend_sched already offers the hook llama.cpp's own imatrix tool uses:
// an eval callback invoked once per node with ask=true ("do you want this?"),
// then again with ask=false after the node has been computed AND synchronised.
// At that second call `t->src[1]` is the live activation tensor that just went
// into the matmul, so we read it back and accumulate the sum of squares down
// each of its ne[0] columns.
//
// Two consequences worth knowing before reading a number off this thing:
//
//   1. ARMING IT MAKES THE LM MUCH SLOWER. With a callback set, the scheduler
//      stops submitting whole splits and instead computes node-range by
//      node-range with a full device synchronise between each. That is inherent
//      to the mechanism, not a bug here. Calibration is a one-off; never leave
//      it armed for a real render.
//
//   2. IT MUST RUN ON THE F16 (or BF16) LM. The statistics describe the model
//      being quantized. Collecting them through an already-quantized checkpoint
//      measures that checkpoint's damage and bakes it in.
//
// The `allow` set is the safety rail: only tensors named in it are collected,
// and mm3-server.h fills it from MM3Model::tmap_lm — the exact set of names
// that came out of the LM GGUF. Anything else in the graph (LoRA A/B factors,
// KV views, the depth decoder's own weights) is ignored by construction rather
// than by a name pattern that could drift. Calibrate with NO adapter loaded:
// an adapter changes the activations, so the resulting imatrix would describe a
// model that the base GGUF is not.
//
// ── The file it writes ───────────────────────────────────────────────────────
//
// GGUF, following llama.cpp's imatrix layout so the files stay interchangeable:
//
//   general.type          = "imatrix"
//   imatrix.chunk_count   = number of calibration runs accumulated
//   imatrix.datasets      = the prompts (or their labels) that produced them
//   <weight>.in_sum2      F32 [n_in]  summed squares, one per input column
//   <weight>.counts       F32 [1]     rows summed into it
//
// The sums are stored raw and divided by counts at load, so two files could be
// merged by adding both tensors. quantize.cpp --imatrix reads exactly this.

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

struct MM3ImatrixEntry {
    std::vector<double> sum2;      // [n_in] summed squares down each input column
    int64_t             rows = 0;  // activation rows accepted into sum2
    int64_t             bad  = 0;  // rows rejected for holding a non-finite value
};

struct MM3ImatrixState {
    bool                                   armed = false;
    std::set<std::string>                  allow;    // collectable weight names
    std::map<std::string, MM3ImatrixEntry> ent;
    std::vector<std::string>               sources;  // labels of the runs so far
    int64_t                                runs = 0;

    // Diagnostics. A silent zero here is the failure mode that would otherwise
    // produce a plausible-looking but useless imatrix, so they are reported.
    int64_t nodes          = 0;  // matmuls collected
    int64_t skipped_type   = 0;  // src[1] was neither F32 nor F16
    int64_t skipped_stride = 0;  // src[1] rows were not contiguous in ne[0]
    int64_t skipped_shape  = 0;  // src[1] ne[0] disagreed with an earlier call
    int64_t bad_rows       = 0;  // rows dropped for holding a non-finite value

    // Scratch, kept across calls so a 36-layer forward does not churn the heap.
    std::vector<uint8_t> scratch;
    std::vector<float>   conv;
};

static MM3ImatrixState g_mm3_imatrix;

// Follow a view chain back to the tensor that actually owns the data. The LM
// head is a ggml_view_2d over output.weight whenever head_slice is on
// (mm3-lm-graph.h), and a view carries its own name, not the weight's.
static const ggml_tensor * mm3_imatrix_base(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool mm3_imatrix_wanted(const ggml_tensor * t, std::string * name) {
    if (!t || t->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm3_imatrix_base(t->src[0]);
    if (!w || w->name[0] == '\0') {
        return false;
    }
    if (g_mm3_imatrix.allow.find(w->name) == g_mm3_imatrix.allow.end()) {
        return false;
    }
    if (name) {
        name->assign(w->name);
    }
    return true;
}

// One activation row into the accumulator, or dropped whole.
//
// DROPPED WHOLE, not element-wise. The MM3 AR loop already reports non-finite
// candidate logits on this checkpoint (it clamps them to -inf and carries on --
// see the mm3-ar-loop.h warning), so non-finite activations are a condition that
// really does occur here rather than a hypothetical. A single NaN poisons a
// column sum permanently, and a row that contains one is not trustworthy in its
// finite half either: it came from the same garbage forward. Taking the finite
// elements would quietly weight those columns against a different sample count
// than their neighbours, which is exactly the kind of wrongness an imatrix
// cannot show on its face.
static bool mm3_imatrix_accumulate(MM3ImatrixEntry & e, const float * f, int64_t n_in) {
    for (int64_t j = 0; j < n_in; j++) {
        if (!std::isfinite(f[j])) {
            e.bad++;
            g_mm3_imatrix.bad_rows++;
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

static bool mm3_imatrix_eval_cb(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;

    std::string wname;
    if (!mm3_imatrix_wanted(t, &wname)) {
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
        g_mm3_imatrix.skipped_type++;
        return true;
    }
    const size_t ts = ggml_type_size(x->type);
    if (x->nb[0] != ts) {
        // A transposed src[1] would make "one row" non-contiguous; the column
        // walk below would read the wrong numbers rather than fail, so refuse.
        g_mm3_imatrix.skipped_stride++;
        return true;
    }

    const int64_t n_in = x->ne[0];

    MM3ImatrixEntry & e = g_mm3_imatrix.ent[wname];
    if (e.sum2.empty()) {
        e.sum2.assign((size_t) n_in, 0.0);
    } else if ((int64_t) e.sum2.size() != n_in) {
        g_mm3_imatrix.skipped_shape++;
        return true;
    }

    g_mm3_imatrix.conv.resize((size_t) n_in);

    if (ggml_is_contiguous(x)) {
        // One device->host copy for the whole activation block. The strided
        // path below is correct too, but on CUDA it is one memcpy per token,
        // and prefill can be a couple of thousand of them per matmul.
        const size_t nb = ggml_nbytes(x);
        g_mm3_imatrix.scratch.resize(nb);
        ggml_backend_tensor_get(x, g_mm3_imatrix.scratch.data(), 0, nb);

        const int64_t nrows = ggml_nelements(x) / n_in;
        for (int64_t r = 0; r < nrows; r++) {
            const uint8_t * p = g_mm3_imatrix.scratch.data() + (size_t) r * (size_t) n_in * ts;
            if (x->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, g_mm3_imatrix.conv.data(), n_in);
                mm3_imatrix_accumulate(e, g_mm3_imatrix.conv.data(), n_in);
            } else {
                mm3_imatrix_accumulate(e, (const float *) p, n_in);
            }
        }
    } else {
        g_mm3_imatrix.scratch.resize((size_t) n_in * ts);
        for (int64_t i3 = 0; i3 < x->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < x->ne[1]; i1++) {
                    const size_t off = (size_t) i1 * x->nb[1] + (size_t) i2 * x->nb[2] + (size_t) i3 * x->nb[3];
                    ggml_backend_tensor_get(x, g_mm3_imatrix.scratch.data(), off, (size_t) n_in * ts);
                    if (x->type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) g_mm3_imatrix.scratch.data(),
                                              g_mm3_imatrix.conv.data(), n_in);
                        mm3_imatrix_accumulate(e, g_mm3_imatrix.conv.data(), n_in);
                    } else {
                        mm3_imatrix_accumulate(e, (const float *) g_mm3_imatrix.scratch.data(), n_in);
                    }
                }
            }
        }
    }

    g_mm3_imatrix.nodes++;
    return true;
}

// Attach or detach the hook on one scheduler. Called immediately before every
// graph compute rather than once at scheduler creation: mm3-lm-graph.h frees
// and rebuilds a slot's scheduler whenever the KV bucket grows, so a
// set-it-once approach would quietly stop collecting partway through a run.
static void mm3_imatrix_hook(ggml_backend_sched_t sched) {
    if (!sched) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, g_mm3_imatrix.armed ? mm3_imatrix_eval_cb : nullptr, &g_mm3_imatrix);
}

static void mm3_imatrix_reset() {
    g_mm3_imatrix.ent.clear();
    g_mm3_imatrix.sources.clear();
    g_mm3_imatrix.runs           = 0;
    g_mm3_imatrix.nodes          = 0;
    g_mm3_imatrix.skipped_type   = 0;
    g_mm3_imatrix.skipped_stride = 0;
    g_mm3_imatrix.skipped_shape  = 0;
    g_mm3_imatrix.bad_rows       = 0;
}

// Tensors that have seen at least one finite row. The gap between this and
// ent.size() is the number that would be written as all-zeros, so it is the
// number that matters when judging whether a collection run is usable.
static size_t mm3_imatrix_usable() {
    size_t n = 0;
    for (const auto & kv : g_mm3_imatrix.ent) {
        if (kv.second.rows > 0) {
            n++;
        }
    }
    return n;
}

// Total activation rows seen, summed over every collected tensor. Reporting
// only: a session that ends with 0 here collected nothing at all.
static int64_t mm3_imatrix_total_rows() {
    int64_t n = 0;
    for (const auto & kv : g_mm3_imatrix.ent) {
        n += kv.second.rows;
    }
    return n;
}

static bool mm3_imatrix_save(const std::string & path, std::string * err) {
    if (g_mm3_imatrix.ent.empty()) {
        if (err) {
            *err = "nothing collected — arm it, run at least one /mm3/lm-plan, then save";
        }
        return false;
    }

    // An entry with no accepted rows is all zeros. Writing it would hand the
    // quantizer an importance vector that says "every column is worthless",
    // which is strictly worse than handing it nothing -- so it is omitted, and
    // quantize.cpp falls back to round-to-nearest for that tensor.
    // The second filter is a belt-and-braces scan of the accumulator itself.
    // Rejecting non-finite ROWS on the way in should make this impossible, but a
    // silently NaN importance vector is the one defect that survives every
    // downstream check and only shows up as a bad-sounding model, so it gets
    // caught here as well as there.
    size_t n_nonfinite_sums = 0;
    std::vector<const std::pair<const std::string, MM3ImatrixEntry> *> keep;
    for (const auto & kv : g_mm3_imatrix.ent) {
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
            fprintf(stderr, "[MM3-IMAT] '%s' accumulated a non-finite sum over %lld rows - omitted\n",
                    kv.first.c_str(), (long long) kv.second.rows);
            continue;
        }
        keep.push_back(&kv);
    }
    if (n_nonfinite_sums) {
        fprintf(stderr, "[MM3-IMAT] %zu tensors had non-finite accumulators\n", n_nonfinite_sums);
    }
    if (keep.empty()) {
        if (err) {
            *err = "every collected tensor had only non-finite rows — nothing worth writing";
        }
        return false;
    }
    if (keep.size() < g_mm3_imatrix.ent.size()) {
        fprintf(stderr, "[MM3-IMAT] %zu of %zu tensors had no finite rows and were omitted\n",
                g_mm3_imatrix.ent.size() - keep.size(), g_mm3_imatrix.ent.size());
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
    gguf_set_val_str(out, "general.architecture", "qwen3");
    gguf_set_val_str(out, "general.name", "MiniMax-Music3 LM imatrix");
    gguf_set_val_u32(out, "imatrix.chunk_count", (uint32_t) g_mm3_imatrix.runs);
    gguf_set_val_u32(out, "imatrix.chunk_size", 0);
    if (!g_mm3_imatrix.sources.empty()) {
        std::vector<const char *> ds;
        ds.reserve(g_mm3_imatrix.sources.size());
        for (const auto & s : g_mm3_imatrix.sources) {
            ds.push_back(s.c_str());
        }
        gguf_set_arr_str(out, "imatrix.datasets", ds.data(), (int) ds.size());
    }

    for (const auto * kvp : keep) {
        const std::string &     name = kvp->first;
        const MM3ImatrixEntry & e    = kvp->second;

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

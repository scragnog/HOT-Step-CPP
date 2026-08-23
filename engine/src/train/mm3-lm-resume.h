// train/mm3-lm-resume.h — pause / resume for `ace-train mm3-lm-train`.
//
// WHY THIS EXISTS: mid-run audio previews.
//
// The trainer owns the card — 22.6 GB of 32.6 at the q8_0 recipe, 31 at f16 —
// and a warm MM3 render stack measures 11.8 GB. There is no configuration in
// which both are resident, so "render a sample every N steps" cannot be a
// sidecar process and cannot be an in-process render either (the render path
// needs cond+dit+voc and a KV-cache AR loop this program does not build).
//
// So the loop is: train -> pause -> exit -> the server renders -> resume. That
// is only honest if resuming is EXACT, which means persisting more than the
// weights:
//
//   * LoRA A/B                — obviously
//   * optimizer momentum      — Muon carries one buffer, AdamW two. Dropping it
//                               restarts momentum at zero every sample point,
//                               which at momentum 0.95 is a ~20-step transient
//                               each time. Sampling every 100 steps would then
//                               spend a fifth of the run in a transient the
//                               un-sampled run never sees, and the two runs
//                               would no longer be comparable.
//   * opt_step / opt_iter     — the LR schedule and AdamW bias correction
//   * the RNG                 — crop offsets
//   * the epoch order + cursor— uniform exposure is the whole point of the
//                               shuffled pass (see the sampler note in
//                               mm3-lm-train-run.h); restarting the pass at a
//                               sample point would re-bias exposure exactly the
//                               way the pass was introduced to prevent
//   * best-held-out tracking  — so "best step" spans the whole run, not the
//                               last segment
//
// The file is single-machine, host-endian and NOT a distribution format: it is
// written and read by the same binary within minutes. It is fingerprinted
// against the run's shape and refuses to load into a different one, because a
// silently-mismatched resume would look like training and be noise.
//
// Size: ~5.6 GB at rank 256 (2.8 GB of LoRA + 2.8 GB of Muon momentum). On
// NVMe that is a few seconds each way, against ~16 s for the render it exists
// to enable.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "hot-step-fsutf8.h"
#include "train/lm-common.h"
#include "train/lm-graph.h"
#include "train/lm-optim.h"

static const char     MM3_RESUME_MAGIC[8] = { 'M', 'M', '3', 'R', 'E', 'S', 'U', 'M' };
static const uint32_t MM3_RESUME_VERSION  = 1;

/** Everything about a paused run that is not a tensor. Tensors travel
 *  separately, keyed by ggml name, so a shape or ordering change in LmLora
 *  cannot silently mis-restore them. */
struct MM3LmResumeState {
    // Fingerprint — a resume into a different run shape is refused, not fudged.
    int32_t  rank = 0, alpha = 0, seed = 0;
    int32_t  n_params = 0, n_samples = 0, n_holdout = 0;
    int32_t  optimizer_muon = 0;

    // Position in the run.
    int32_t  steps_done = 0;      // completed optimizer steps
    int32_t  n_micro = 0;
    double   running = 0.0;       // summed micro-step loss, for the mean

    // Epoch bookkeeping.
    int32_t          epoch = 0, epoch_n = 0;
    double           epoch_loss_sum = 0.0;
    std::vector<int32_t> order;
    int32_t          order_pos = 0;

    // Held-out tracking, so "best step" is a whole-run statement.
    double   best_eval = -1.0;
    int32_t  best_eval_step = 0;

    // Optimizer scalars.
    int32_t  opt_step = 0, opt_iter = 0;

    // Crop RNG.
    uint64_t rng[4] = { 0, 0, 0, 0 };
};

// ── tiny binary IO ──────────────────────────────────────────────────────────

template <typename T>
static bool mm3_rs_w(FILE * f, const T & v) {
    return fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
static bool mm3_rs_r(FILE * f, T * v) {
    return fread(v, sizeof(T), 1, f) == 1;
}

static bool mm3_rs_w_str(FILE * f, const std::string & s) {
    const uint32_t n = (uint32_t) s.size();
    return mm3_rs_w(f, n) && (n == 0 || fwrite(s.data(), 1, n, f) == n);
}

static bool mm3_rs_r_str(FILE * f, std::string * s) {
    uint32_t n = 0;
    if (!mm3_rs_r(f, &n) || n > 4096) {
        return false;
    }
    s->assign(n, '\0');
    return n == 0 || fread(&(*s)[0], 1, n, f) == n;
}

/** One named F32 tensor, read back from the backend and written straight out.
 *  The largest LoRA tensor at rank 256 is ~12 MB, so a whole-tensor host
 *  staging buffer is not worth chunking. */
static bool mm3_rs_write_tensor(FILE * f, ggml_tensor * t, std::vector<uint8_t> * scratch) {
    const size_t bytes = ggml_nbytes(t);
    scratch->resize(bytes);
    ggml_backend_tensor_get(t, scratch->data(), 0, bytes);
    const uint64_t n64 = (uint64_t) bytes;
    return mm3_rs_w_str(f, std::string(ggml_get_name(t)))
        && mm3_rs_w(f, n64)
        && fwrite(scratch->data(), 1, bytes, f) == bytes;
}

// ── save ────────────────────────────────────────────────────────────────────

/** Write the whole pause state to `path` (via a .tmp + rename, so a crash
 *  mid-write cannot leave a truncated file that a resume would half-believe). */
static bool mm3_lm_resume_save(const std::string & path, const MM3LmResumeState & st,
                               const LmLora & lora, const LmOptim & opt, std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp, "wb");
    if (!f) {
        *err = "cannot open " + tmp + " for writing";
        return false;
    }
    std::vector<uint8_t> scratch;
    bool                 ok = true;

    ok = ok && fwrite(MM3_RESUME_MAGIC, 1, sizeof(MM3_RESUME_MAGIC), f) == sizeof(MM3_RESUME_MAGIC);
    ok = ok && mm3_rs_w(f, MM3_RESUME_VERSION);
    ok = ok && mm3_rs_w(f, st.rank) && mm3_rs_w(f, st.alpha) && mm3_rs_w(f, st.seed);
    ok = ok && mm3_rs_w(f, st.n_params) && mm3_rs_w(f, st.n_samples) && mm3_rs_w(f, st.n_holdout);
    ok = ok && mm3_rs_w(f, st.optimizer_muon);
    ok = ok && mm3_rs_w(f, st.steps_done) && mm3_rs_w(f, st.n_micro) && mm3_rs_w(f, st.running);
    ok = ok && mm3_rs_w(f, st.epoch) && mm3_rs_w(f, st.epoch_n) && mm3_rs_w(f, st.epoch_loss_sum);
    ok = ok && mm3_rs_w(f, st.order_pos);
    ok = ok && mm3_rs_w(f, st.best_eval) && mm3_rs_w(f, st.best_eval_step);
    ok = ok && mm3_rs_w(f, st.opt_step) && mm3_rs_w(f, st.opt_iter);
    for (int i = 0; i < 4; i++) ok = ok && mm3_rs_w(f, st.rng[i]);
    {
        const uint32_t n = (uint32_t) st.order.size();
        ok = ok && mm3_rs_w(f, n);
        ok = ok && (n == 0 || fwrite(st.order.data(), sizeof(int32_t), n, f) == n);
    }

    // Tensors: LoRA parameters, then momentum, each block prefixed by its count
    // so a reader knows what to expect without trusting the fingerprint alone.
    {
        const uint32_t n = (uint32_t) lora.params.size();
        ok = ok && mm3_rs_w(f, n);
        for (uint32_t j = 0; ok && j < n; j++) {
            ok = mm3_rs_write_tensor(f, lora.params[j], &scratch);
        }
    }
    {
        uint32_t n = 0;
        for (size_t j = 0; j < opt.mom_m.size(); j++) {
            if (opt.mom_m[j]) n++;
            if (j < opt.mom_v.size() && opt.mom_v[j]) n++;
        }
        ok = ok && mm3_rs_w(f, n);
        for (size_t j = 0; ok && j < opt.mom_m.size(); j++) {
            if (opt.mom_m[j]) ok = ok && mm3_rs_write_tensor(f, opt.mom_m[j], &scratch);
            if (ok && j < opt.mom_v.size() && opt.mom_v[j]) {
                ok = mm3_rs_write_tensor(f, opt.mom_v[j], &scratch);
            }
        }
    }
    const uint32_t eof_marker = 0xD09EF00Du;   // "done", so a truncated tail is loud
    ok = ok && mm3_rs_w(f, eof_marker);

    if (fclose(f) != 0) ok = false;
    if (!ok) {
        *err = "write failed (disk full?) — " + tmp;
        hs_remove(tmp);
        return false;
    }
    hs_remove(path);
    if (hs_rename(tmp, path) != 0) {
        *err = "cannot rename " + tmp + " into place";
        return false;
    }
    return true;
}

// ── load ────────────────────────────────────────────────────────────────────

/** Restore a paused run. The tensors are matched BY NAME, and a name present in
 *  the file but not in the live run (or vice versa) is an error rather than a
 *  shrug: a partial restore is indistinguishable from training on garbage. */
static bool mm3_lm_resume_load(const std::string & path, MM3LmResumeState * st,
                               LmLora & lora, LmOptim & opt, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open " + path;
        return false;
    }
    auto fail = [&](const std::string & why) {
        *err = why;
        fclose(f);
        return false;
    };

    char magic[sizeof(MM3_RESUME_MAGIC)] = { 0 };
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic)
        || memcmp(magic, MM3_RESUME_MAGIC, sizeof(magic)) != 0) {
        return fail(path + " is not an mm3-lm-train resume file");
    }
    uint32_t ver = 0;
    if (!mm3_rs_r(f, &ver) || ver != MM3_RESUME_VERSION) {
        return fail("resume file version " + std::to_string(ver) + ", this build writes "
                    + std::to_string(MM3_RESUME_VERSION));
    }
    MM3LmResumeState in;
    bool             ok = true;
    ok = ok && mm3_rs_r(f, &in.rank) && mm3_rs_r(f, &in.alpha) && mm3_rs_r(f, &in.seed);
    ok = ok && mm3_rs_r(f, &in.n_params) && mm3_rs_r(f, &in.n_samples) && mm3_rs_r(f, &in.n_holdout);
    ok = ok && mm3_rs_r(f, &in.optimizer_muon);
    ok = ok && mm3_rs_r(f, &in.steps_done) && mm3_rs_r(f, &in.n_micro) && mm3_rs_r(f, &in.running);
    ok = ok && mm3_rs_r(f, &in.epoch) && mm3_rs_r(f, &in.epoch_n) && mm3_rs_r(f, &in.epoch_loss_sum);
    ok = ok && mm3_rs_r(f, &in.order_pos);
    ok = ok && mm3_rs_r(f, &in.best_eval) && mm3_rs_r(f, &in.best_eval_step);
    ok = ok && mm3_rs_r(f, &in.opt_step) && mm3_rs_r(f, &in.opt_iter);
    for (int i = 0; i < 4; i++) ok = ok && mm3_rs_r(f, &in.rng[i]);
    if (!ok) return fail("truncated resume header");
    {
        uint32_t n = 0;
        if (!mm3_rs_r(f, &n) || n > (1u << 24)) return fail("bad epoch-order length");
        in.order.resize(n);
        if (n && fread(in.order.data(), sizeof(int32_t), n, f) != n) return fail("truncated epoch order");
    }

    // Fingerprint. Every one of these changes what the stored tensors MEAN.
    if (in.rank != st->rank || in.alpha != st->alpha || in.n_params != st->n_params
        || in.optimizer_muon != st->optimizer_muon) {
        return fail("resume file is from a different run shape (rank " + std::to_string(in.rank)
                    + "/alpha " + std::to_string(in.alpha) + "/" + std::to_string(in.n_params)
                    + " tensors, this run is rank " + std::to_string(st->rank) + "/alpha "
                    + std::to_string(st->alpha) + "/" + std::to_string(st->n_params) + ")");
    }
    if (in.n_samples != st->n_samples || in.n_holdout != st->n_holdout) {
        return fail("resume file was written over " + std::to_string(in.n_samples) + " training songs (+"
                    + std::to_string(in.n_holdout) + " held out), this run has "
                    + std::to_string(st->n_samples) + " (+" + std::to_string(st->n_holdout)
                    + ") — the dataset changed under the run");
    }

    // Tensors.
    std::unordered_map<std::string, ggml_tensor *> live;
    for (ggml_tensor * p : lora.params) live[ggml_get_name(p)] = p;
    for (size_t j = 0; j < opt.mom_m.size(); j++) {
        if (opt.mom_m[j]) live[ggml_get_name(opt.mom_m[j])] = opt.mom_m[j];
        if (j < opt.mom_v.size() && opt.mom_v[j]) live[ggml_get_name(opt.mom_v[j])] = opt.mom_v[j];
    }
    std::vector<uint8_t> scratch;
    size_t               restored = 0;
    for (int block = 0; block < 2; block++) {
        uint32_t n = 0;
        if (!mm3_rs_r(f, &n)) return fail("truncated tensor block header");
        for (uint32_t i = 0; i < n; i++) {
            std::string name;
            uint64_t    bytes = 0;
            if (!mm3_rs_r_str(f, &name) || !mm3_rs_r(f, &bytes)) return fail("truncated tensor record");
            auto it = live.find(name);
            if (it == live.end()) return fail("resume file has tensor \"" + name + "\", this run does not");
            if (ggml_nbytes(it->second) != (size_t) bytes) {
                return fail("tensor \"" + name + "\" is " + std::to_string(bytes)
                            + " bytes in the resume file and " + std::to_string(ggml_nbytes(it->second))
                            + " here");
            }
            scratch.resize((size_t) bytes);
            if (fread(scratch.data(), 1, (size_t) bytes, f) != (size_t) bytes) {
                return fail("truncated tensor data for \"" + name + "\"");
            }
            ggml_backend_tensor_set(it->second, scratch.data(), 0, (size_t) bytes);
            restored++;
        }
    }
    {
        uint32_t eof_marker = 0;
        if (!mm3_rs_r(f, &eof_marker) || eof_marker != 0xD09EF00Du) {
            return fail("resume file is truncated — it was written by a run that did not finish saving");
        }
    }
    if (restored != live.size()) {
        return fail("restored " + std::to_string(restored) + " of " + std::to_string(live.size())
                    + " tensors — a partial resume is not a resume");
    }
    fclose(f);

    // Only now, with every tensor in place, does the caller's state change.
    const int32_t keep_rank = st->rank, keep_alpha = st->alpha, keep_seed = st->seed;
    const int32_t keep_np = st->n_params, keep_ns = st->n_samples, keep_nh = st->n_holdout;
    const int32_t keep_muon = st->optimizer_muon;
    *st = in;
    st->rank = keep_rank; st->alpha = keep_alpha; st->seed = keep_seed;
    st->n_params = keep_np; st->n_samples = keep_ns; st->n_holdout = keep_nh;
    st->optimizer_muon = keep_muon;
    return true;
}

// ── the pause sentinel ──────────────────────────────────────────────────────
//
// A FILE, not a signal or a pipe. The server writes `<out>/PAUSE` and this
// program notices at the next step boundary. Deliberately the dumbest
// mechanism that works: it is cross-platform without a shim, it cannot block
// the training loop, it survives the server restarting under a running trainer,
// and "touch a file to make it stop" is debuggable from a shell at 2 a.m.
//
// Cost is one stat() per optimizer step — against a 3.9 s step.

static inline std::string mm3_lm_pause_path(const std::string & out_dir) {
    return out_dir + "/PAUSE";
}

static inline bool mm3_lm_pause_requested(const std::string & pause_file) {
    if (pause_file.empty()) {
        return false;
    }
    FILE * f = hs_fopen(pause_file, "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

/** Remove the sentinel. Called by the trainer as it pauses, so the server never
 *  has to clean up after a trainer that exited for another reason. */
static inline void mm3_lm_pause_clear(const std::string & pause_file) {
    if (!pause_file.empty()) {
        hs_remove(pause_file);
    }
}

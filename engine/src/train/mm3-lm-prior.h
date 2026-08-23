// train/mm3-lm-prior.h — prior preservation for the MM3 LM LoRA trainer.
//
// THE PROBLEM IT SOLVES
//
// A trigger-word LoRA over one album has no way of knowing which of its changes
// are "the artist" and which are "how music works". Trained only on 11 System Of
// A Down tracks, it will happily rewrite the planner's general competence
// because nothing in the objective punishes that — the training loss falls
// either way. What you hear afterwards is a trace of the right voice over a
// backing track that has quietly gone simple and cheap, and a loss curve that
// looked fine the whole time.
//
// bghira's fix, from the regularised variant of his fiona-crapple LoRA: train
// on a SECOND set of unrelated songs, and on those batches do not ask the model
// to predict the ground-truth codes. Ask it to predict WHAT IT USED TO PREDICT.
// The frozen base's own next-token distribution becomes the target, so the
// adapter is explicitly penalised for changing its mind about material that has
// nothing to do with the artist. He reports it "sharply reduces style bleed";
// his unregularised run summoned the style even from generic captions.
//
// HOW IT WORKS HERE
//
// The base distribution is captured ONCE, before training, and cached to disk:
//
//   * At init the LoRA's B factor is zero, so the adapter contributes exactly
//     nothing and a forward pass IS the frozen base. That window closes after
//     the first optimizer step, which is why the cache is mandatory on resume
//     rather than something we can regenerate.
//   * Only the top-K of each position is kept (K=64 by default, 16,385 classes).
//     Storage is s_tr * K * 8 bytes — about 2 MB for a 4096-frame song.
//   * Softmax is taken over the full width BEFORE truncation, so the stored
//     probabilities are the base's real confidence. The dropped tail leaves each
//     row summing to slightly under 1, which cross-entropy reads as "no opinion"
//     about the rest — the honest thing, since we did not record one.
//
// A cached file is only valid for one (base model, song, crop) triple, so the
// filename carries the LM's stem and K, and the header carries the position
// count. Anything that does not line up is regenerated rather than trusted.
//
// The loss itself needed no new code: ggml_cross_entropy_loss takes a dense
// [V, n] label tensor and its backward is (softmax - labels), both already
// correct for a distribution. See LmChunkLabelGuard in train/lm-data.h.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hot-step-fsutf8.h"

static const char     MM3_PRIOR_MAGIC[8] = { 'M', 'M', '3', 'P', 'R', 'I', 'O', 'R' };
static const uint32_t MM3_PRIOR_VERSION  = 1;

struct MM3PriorCache {
    int32_t              k = 0;
    int32_t              n_pos = 0;
    int32_t              width = 0;   // scored width the capture was taken over
    std::vector<int32_t> idx;         // n_pos * k
    std::vector<float>   p;           // n_pos * k
};

/** `<dir>/<id>.<lm stem>.k<K>.prior` — the base model and K are in the NAME
 *  because a cache from a different base is not merely stale, it is a different
 *  teacher, and silently training against it would be undetectable. */
static inline std::string mm3_prior_path(const std::string & dir, const std::string & id,
                                         const std::string & lm_path, int k) {
    std::string stem = lm_path;
    const size_t slash = stem.find_last_of("/\\");
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    const size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    return dir + "/" + id + "." + stem + ".k" + std::to_string(k) + ".prior";
}

static inline bool mm3_prior_save(const std::string & path, const MM3PriorCache & c,
                                  std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp, "wb");
    if (!f) {
        *err = "cannot write " + tmp;
        return false;
    }
    bool ok = fwrite(MM3_PRIOR_MAGIC, 1, sizeof(MM3_PRIOR_MAGIC), f) == sizeof(MM3_PRIOR_MAGIC);
    ok = ok && fwrite(&MM3_PRIOR_VERSION, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fwrite(&c.k, sizeof(int32_t), 1, f) == 1;
    ok = ok && fwrite(&c.n_pos, sizeof(int32_t), 1, f) == 1;
    ok = ok && fwrite(&c.width, sizeof(int32_t), 1, f) == 1;
    const size_t n = (size_t) c.n_pos * (size_t) c.k;
    ok = ok && c.idx.size() == n && c.p.size() == n;
    ok = ok && fwrite(c.idx.data(), sizeof(int32_t), n, f) == n;
    ok = ok && fwrite(c.p.data(), sizeof(float), n, f) == n;
    if (fclose(f) != 0) ok = false;
    if (!ok) {
        *err = "short write on " + tmp + " (disk full?)";
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

/** Load and validate. Returns false (with `err`) for anything that does not
 *  match what the caller is about to train on — a mismatched cache is silently
 *  wrong, so it is always regenerated rather than adapted. */
static inline bool mm3_prior_load(const std::string & path, int want_k, int want_pos,
                                  int want_width, MM3PriorCache * out, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "not cached";
        return false;
    }
    auto fail = [&](const std::string & why) { *err = why; fclose(f); return false; };

    char     magic[sizeof(MM3_PRIOR_MAGIC)] = { 0 };
    uint32_t ver = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic)
        || memcmp(magic, MM3_PRIOR_MAGIC, sizeof(magic)) != 0) {
        return fail("not a prior cache");
    }
    if (fread(&ver, sizeof(uint32_t), 1, f) != 1 || ver != MM3_PRIOR_VERSION) {
        return fail("prior cache version " + std::to_string(ver));
    }
    if (fread(&out->k, sizeof(int32_t), 1, f) != 1
        || fread(&out->n_pos, sizeof(int32_t), 1, f) != 1
        || fread(&out->width, sizeof(int32_t), 1, f) != 1) {
        return fail("truncated prior header");
    }
    if (out->k != want_k || out->n_pos != want_pos || out->width != want_width) {
        return fail("prior cache is for k=" + std::to_string(out->k) + "/pos="
                    + std::to_string(out->n_pos) + "/width=" + std::to_string(out->width)
                    + ", this run needs k=" + std::to_string(want_k) + "/pos="
                    + std::to_string(want_pos) + "/width=" + std::to_string(want_width));
    }
    const size_t n = (size_t) out->n_pos * (size_t) out->k;
    out->idx.resize(n);
    out->p.resize(n);
    if (fread(out->idx.data(), sizeof(int32_t), n, f) != n
        || fread(out->p.data(), sizeof(float), n, f) != n) {
        return fail("truncated prior payload");
    }
    fclose(f);
    return true;
}

/** Mean captured probability mass per position, i.e. how much of the base's
 *  distribution the top-K actually covers.
 *
 *  Reported rather than assumed: K=64 over 16,385 codes usually captures well
 *  over 99%, but a base that is genuinely uncertain somewhere would be poorly
 *  represented, and the fix (raise K) is only obvious if the number is visible. */
static inline double mm3_prior_coverage(const MM3PriorCache & c) {
    if (c.n_pos <= 0 || c.k <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < c.n_pos; i++) {
        double row = 0.0;
        for (int k = 0; k < c.k; k++) {
            row += (double) c.p[(size_t) i * (size_t) c.k + (size_t) k];
        }
        sum += row;
    }
    return sum / (double) c.n_pos;
}

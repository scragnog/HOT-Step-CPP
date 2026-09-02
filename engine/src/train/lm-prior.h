#pragma once
// lm-prior.h — prior preservation for the ACE-Step 1.5 planner-LM trainer
// (`ace-train train-lm --reg-every N --reg-codes ...`), 2026-09-02.
//
// THE PROBLEM IT SOLVES
//
// A trigger-word adapter over one album has no way of knowing which of its
// changes are "the artist" and which are "how music works". Trained only on
// seven PRESIDENT tracks it will happily rewrite the planner's general
// competence, because nothing in the objective punishes that — the training
// loss falls either way. The 2026-09-02 attribute study is the receipt: every
// measurable thing an ACE LM adapter does is done by CE ~2.0, and training
// deeper past that mostly buys looping plans. Something has to hold the base
// still while the style is learned.
//
// bghira's fix, ported to MM3 first (train/mm3-lm-prior.h) and now to ACE:
// train on a SECOND set of unrelated songs, and on those steps do not ask the
// model to predict the ground-truth codes. Ask it to predict WHAT IT USED TO
// PREDICT. The frozen base's own next-token distribution becomes the target, so
// the adapter is explicitly penalised for changing its mind about material that
// has nothing to do with the artist.
//
// HOW IT WORKS HERE
//
// The base distribution is captured ONCE, before the first optimizer step, and
// cached to disk:
//
//   * At init the LoRA's B factor (or the LoKr w2/w2_b) is exactly zero, so the
//     adapter contributes nothing and a forward pass IS the frozen base. That
//     window closes after the first optimizer step, which is why the cache is
//     mandatory on resume rather than something we can regenerate.
//   * Only the top-K of each supervised position is kept (K = 64 by default).
//     Storage is n_pos * K * 8 bytes — under a megabyte for a 600 s song.
//   * Softmax is taken over the FULL scored width before truncation, so the
//     stored probabilities are the base's real confidence and each row sums to
//     the CAPTURED MASS, not to 1. That row sum is the coverage number the run
//     reports. It is a capture format: before the rows are used as
//     cross-entropy labels the trainer brings each one to 1 by spreading the
//     missing mass flat over the audio-code tokens (LmSample::soft_tail,
//     train/lm-train-run.h), because ggml's cross-entropy backward is only the
//     true gradient for labels summing to 1 — see train/lm-data.h. It does NOT
//     rescale the captured K: at ACE's 18-24% coverage that would make the
//     target far sharper than the base, and the trainer warns about coverage.
//
// WHY THIS IS NOT JUST `#include "train/mm3-lm-prior.h"`
//
// The on-disk layout is deliberately identical (same header fields, same order,
// same payload) because there is no reason to invent a second one. The MAGIC is
// not: an MM3 cache is 16,385 classes of MM3 semantic slice, an ACE cache is
// ~216,000 classes of Qwen3 vocabulary, and the two files are the same size for
// wildly different reasons. A wrong-model cache that merely fails a field check
// is one refactor away from being accepted; a wrong-model cache with a
// different magic can never be read at all. The loader also validates K, the
// position count and the scored width, so a stale cache is regenerated rather
// than trusted.
//
// The loss itself needed no new code: ggml_cross_entropy_loss takes a dense
// [V, n] label tensor and its backward is (softmax - labels), which is correct
// for a distribution PROVIDED the label row sums to 1 — hence the renormalise
// on the way in. See LmChunkLabelGuard in train/lm-data.h.

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hot-step-fsutf8.h"

static const char     LM_PRIOR_MAGIC[8] = { 'A', 'C', 'E', 'P', 'R', 'I', 'O', 'R' };
static const uint32_t LM_PRIOR_VERSION  = 1;

struct LmPriorCache {
    int32_t              k     = 0;
    int32_t              n_pos = 0;
    int32_t              width = 0;  // scored width the capture was taken over
    std::vector<int32_t> idx;        // n_pos * k
    std::vector<float>   p;          // n_pos * k
};

/** Filesystem-safe identifier for one regularisation ROW.
 *
 *  Two different artists' lm_codes.jsonl can hold rows with the same `_id`
 *  (an 8-hex content hash is unique per file, not across files), so the codes
 *  file's own stem is part of the name. Without it a second corpus would read
 *  the first one's cached teacher and nothing would say so. */
static inline std::string lm_prior_row_id(const std::string & codes_path, const std::string & row_file,
                                          const std::string & row_id) {
    auto stem = [](const std::string & p) {
        std::string s     = p;
        const size_t slash = s.find_last_of("/\\");
        if (slash != std::string::npos) {
            s = s.substr(slash + 1);
        }
        const size_t dot = s.find_last_of('.');
        if (dot != std::string::npos && dot > 0) {
            s = s.substr(0, dot);
        }
        return s;
    };
    // The codes file is nearly always literally "lm_codes.jsonl" sitting in
    // `<artist slug>/<model variant>/`, so the readable part of the name is the
    // GRANDPARENT (the slug) — the parent is the same string for every artist
    // and would name nothing. A layout that has no grandparent falls back to the
    // parent, and a 32-bit FNV-1a of the full path is appended regardless: two
    // corpora can hold rows with the same `_id` (it is a per-file content hash,
    // not a global one), and a cache silently read across corpora would be a
    // different teacher with no way to notice.
    std::string corpus;
    {
        std::string dir    = codes_path;
        const size_t slash = dir.find_last_of("/\\");
        if (slash != std::string::npos) {
            dir             = dir.substr(0, slash);
            const size_t s2 = dir.find_last_of("/\\");
            const std::string parent = (s2 == std::string::npos) ? dir : dir.substr(s2 + 1);
            std::string grand;
            if (s2 != std::string::npos) {
                const std::string up = dir.substr(0, s2);
                const size_t      s3 = up.find_last_of("/\\");
                grand                = (s3 == std::string::npos) ? up : up.substr(s3 + 1);
            }
            corpus = grand.empty() ? parent : grand;
        }
        if (corpus.empty()) {
            corpus = stem(codes_path);
        }
    }
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < codes_path.size(); i++) {
        h ^= (uint32_t) (unsigned char) codes_path[i];
        h *= 16777619u;
    }
    char hx[16];
    snprintf(hx, sizeof(hx), "%08x", h);
    std::string id = corpus + "_" + hx + "_" + (row_id.empty() ? stem(row_file) : row_id);
    for (size_t i = 0; i < id.size(); i++) {
        const unsigned char ch = (unsigned char) id[i];
        if (!isalnum(ch) && ch != '-' && ch != '_' && ch != '.') {
            id[i] = '_';
        }
    }
    if (id.size() > 120) {
        id.resize(120);
    }
    return id;
}

/** `<dir>/<row id>.<lm stem>.k<K>.prior` — the base model and K are in the NAME
 *  because a cache from a different base is not merely stale, it is a different
 *  teacher, and silently training against it would be undetectable. */
static inline std::string lm_prior_path(const std::string & dir, const std::string & id,
                                        const std::string & lm_path, int k) {
    std::string  stem  = lm_path;
    const size_t slash = stem.find_last_of("/\\");
    if (slash != std::string::npos) {
        stem = stem.substr(slash + 1);
    }
    const size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) {
        stem = stem.substr(0, dot);
    }
    return dir + "/" + id + "." + stem + ".k" + std::to_string(k) + ".prior";
}

static inline bool lm_prior_save(const std::string & path, const LmPriorCache & c, std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp, "wb");
    if (!f) {
        *err = "cannot write " + tmp;
        return false;
    }
    bool ok = fwrite(LM_PRIOR_MAGIC, 1, sizeof(LM_PRIOR_MAGIC), f) == sizeof(LM_PRIOR_MAGIC);
    ok      = ok && fwrite(&LM_PRIOR_VERSION, sizeof(uint32_t), 1, f) == 1;
    ok      = ok && fwrite(&c.k, sizeof(int32_t), 1, f) == 1;
    ok      = ok && fwrite(&c.n_pos, sizeof(int32_t), 1, f) == 1;
    ok      = ok && fwrite(&c.width, sizeof(int32_t), 1, f) == 1;
    const size_t n = (size_t) c.n_pos * (size_t) c.k;
    ok = ok && c.idx.size() == n && c.p.size() == n;
    ok = ok && fwrite(c.idx.data(), sizeof(int32_t), n, f) == n;
    ok = ok && fwrite(c.p.data(), sizeof(float), n, f) == n;
    if (fclose(f) != 0) {
        ok = false;
    }
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
static inline bool lm_prior_load(const std::string & path, int want_k, int want_pos, int want_width,
                                 LmPriorCache * out, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "not cached";
        return false;
    }
    auto fail = [&](const std::string & why) {
        *err = why;
        fclose(f);
        return false;
    };

    char     magic[sizeof(LM_PRIOR_MAGIC)] = { 0 };
    uint32_t ver                           = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) || memcmp(magic, LM_PRIOR_MAGIC, sizeof(magic)) != 0) {
        return fail("not an ACE prior cache");
    }
    if (fread(&ver, sizeof(uint32_t), 1, f) != 1 || ver != LM_PRIOR_VERSION) {
        return fail("prior cache version " + std::to_string(ver));
    }
    if (fread(&out->k, sizeof(int32_t), 1, f) != 1 || fread(&out->n_pos, sizeof(int32_t), 1, f) != 1 ||
        fread(&out->width, sizeof(int32_t), 1, f) != 1) {
        return fail("truncated prior header");
    }
    if (out->k != want_k || out->n_pos != want_pos || out->width != want_width) {
        return fail("prior cache is for k=" + std::to_string(out->k) + "/pos=" + std::to_string(out->n_pos) +
                    "/width=" + std::to_string(out->width) + ", this run needs k=" + std::to_string(want_k) +
                    "/pos=" + std::to_string(want_pos) + "/width=" + std::to_string(want_width));
    }
    const size_t n = (size_t) out->n_pos * (size_t) out->k;
    out->idx.resize(n);
    out->p.resize(n);
    if (fread(out->idx.data(), sizeof(int32_t), n, f) != n || fread(out->p.data(), sizeof(float), n, f) != n) {
        return fail("truncated prior payload");
    }
    fclose(f);
    return true;
}

/** Mean captured probability mass per position, i.e. how much of the base's
 *  distribution the top-K actually covers. Runs on the RAW capture — the
 *  trainer adds a uniform tail for the loss and leaves this one alone, because
 *  a completed row sums to 1 by construction and would report nothing.
 *
 *  Reported rather than assumed. On MM3's 16,385 classes K=64 covers well over
 *  99%; on ACE's 217,204 it measured 18.5% (and K=256 only 24.0%) against
 *  another artist's codes. The lower it is, the more of the target is the flat
 *  floor rather than the base's own shape, and that is only visible if the
 *  number is. */
static inline double lm_prior_coverage(const LmPriorCache & c) {
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

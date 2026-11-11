#pragma once
// pissa-residual.h — the frozen half of a PiSSA / HOT-PiZZA LM adapter, shipped
// ONCE per base model instead of inside every adapter file.
//
// HOT-Step file, 2026-09-09.
//
// A PiSSA adapter is y = Wx + s(BA - B0A0)x, where A0/B0 are the base weight's
// own top-r singular directions (lm-pissa.h). Until today every export wrote
// [B - B0, B0][A; A - A0] as a rank-2r LoRA: 2.79 GB at r128 in F32, of which
// exactly half (B0, A0) is the SAME BYTES in every adapter trained on the same
// base. This header defines the file that carries that half on its own:
//
//   <models>/mm3/<base stem>.pissa-r<rank>.safetensors        (~0.7 GB, F16)
//
//   pissa.meta                       F32 [10] {version=1, rank, oversample q,
//                                             iters, layer_lo, layer_hi,
//                                             base_size_hi, base_size_lo,
//                                             energy_mean, energy_min}
//   pissa.L<l>.<peft site>.A0        F16 [rank, in]   torch row-major
//   pissa.L<l>.<peft site>.B0        F16 [out, rank]
//
// The factors are stored in CANONICAL form (scale 1: B0 A0 is W's rank-r
// truncation). A trainer at scale s divides both by sqrt(s) on load; the delta
// loader multiplies the B0 half by sqrt(s) when it rebuilds the pair. At the
// shipped recipe (alpha == rank) s is 1 and nothing is scaled.
//
// base_size is the base GGUF's byte length, split into two exactly-representable
// floats (hi = size / 65536, lo = size % 65536) because the engine's safetensors
// reader skips __metadata__ and a 9.1e9 does not survive a single F32. It is
// what ties a residual to its base: a re-quantized file of the same name has a
// different size and is refused rather than silently paired.
//
// An adapter exported against a residual (train/lm-export.h's delta form) holds
// only A - A0 and s(B - B0) at rank r, marks hot_step.param_method = 4, names the
// residual file in adapter_config.json ("hot_step_pissa_residual") and carries a
// hot_step.pissa.meta tensor {rank, scale, layer_lo, layer_hi, base_size_hi,
// base_size_lo}. minimax/mm3-lm-adapter.h reads both files and rebuilds the
// same rank-2r pair the old export carried, so everything downstream of the
// loader (runtime graph, merge, group scales) is unchanged.
//
// New delta exports also carry hot_step.pissa.fingerprint, F32[5]: version 1
// followed by four low-to-high 16-bit words of XXH64(seed=0) over EVERY byte
// of the residual file, including its header. This is an accidental-mismatch
// check, not a security signature. Repacking a residual changes its identity.
// Capture it at init, before scaling the factors, and keep it for every export.
// Files without this tensor retain the legacy rank/base-size checks. Never
// trust a digest stored in the residual or a sidecar instead of hashing bytes.
// Hash the existing mapping, with no second file read or persistent cache.
//
// Reader only. The writer lives with the trainer (train/lm-pissa.h) because it
// is the trainer that has the factors and the st-write.h dependency.

#include "hot-step-fsutf8.h"
#include "safetensors.h"

#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define PISSA_RESIDUAL_VERSION 1

struct PissaResidualMeta {
    int       version  = 0;
    int       rank     = 0;
    int       q        = 0;   // rank + oversample, as the SVD ran
    int       iters    = 0;
    int       layer_lo = 0;
    int       layer_hi = 0;
    long long base_size = 0;
    double    energy_mean = 0.0, energy_min = 0.0;  // captured / ||W||_F^2, informational
    std::vector<float> fingerprint;
};

// XXH64, seed 0. Explicit little-endian reads keep the on-disk identity portable.
static inline uint64_t pissa_xxh64(const void * data, size_t size) {
    constexpr uint64_t p1 = 11400714785074694791ull, p2 = 14029467366897019727ull;
    constexpr uint64_t p3 = 1609587929392839161ull, p4 = 9650029242287828579ull;
    constexpr uint64_t p5 = 2870177450012600261ull;
    auto rot = [](uint64_t x, int n) { return (x << n) | (x >> (64 - n)); };
    auto read = [](const uint8_t * p, int n) {
        uint64_t x = 0;
        for (int i = 0; i < n; ++i) x |= uint64_t(p[i]) << (8 * i);
        return x;
    };
    auto round = [&](uint64_t a, uint64_t x) { return rot(a + x * p2, 31) * p1; };
    const auto * p = static_cast<const uint8_t *>(data);
    size_t left = size;
    uint64_t h = p5;
    if (left >= 32) {
        uint64_t a = p1 + p2, b = p2, c = 0, d = 0 - p1;
        do {
            a = round(a, read(p, 8)); b = round(b, read(p + 8, 8));
            c = round(c, read(p + 16, 8)); d = round(d, read(p + 24, 8));
            p += 32; left -= 32;
        } while (left >= 32);
        h = rot(a, 1) + rot(b, 7) + rot(c, 12) + rot(d, 18);
        for (uint64_t v : {a, b, c, d}) h = (h ^ round(0, v)) * p1 + p4;
    }
    h += size;
    while (left >= 8) {
        h = rot(h ^ round(0, read(p, 8)), 27) * p1 + p4;
        p += 8; left -= 8;
    }
    if (left >= 4) {
        h = rot(h ^ (read(p, 4) * p1), 23) * p2 + p3;
        p += 4; left -= 4;
    }
    while (left--) h = rot(h ^ (uint64_t(*p++) * p5), 11) * p1;
    h ^= h >> 33; h *= p2; h ^= h >> 29; h *= p3; h ^= h >> 32;
    return h;
}

static inline std::vector<float> pissa_residual_fingerprint(const STFile & st) {
    const uint64_t h = pissa_xxh64(st.mapping, st.file_size);
    return {1.0f, float(h & 65535), float((h >> 16) & 65535),
            float((h >> 32) & 65535), float(h >> 48)};
}

static inline bool pissa_residual_check_fingerprint(const STFile & adapter, const std::string & adapter_path,
                                                    const std::string & residual_path,
                                                    const std::vector<float> & actual, std::string * err) {
    const STEntry * e = st_find(adapter, "hot_step.pissa.fingerprint");
    if (!e) return true;  // Legacy adapter.
    bool match = e->dtype == "F32" && e->n_dims == 1 && e->shape[0] == 5 &&
                 e->data_end - e->data_start == 5 * sizeof(float) && actual.size() == 5;
    if (match) {
        const float * expected = static_cast<const float *>(st_data(adapter, *e));
        for (size_t i = 0; i < actual.size(); ++i) match = match && expected[i] == actual[i];
    }
    if (!match && err) {
        *err = "adapter " + adapter_path + " does not match PiSSA residual " + residual_path +
               " (hot_step.pissa.fingerprint). Restore the exact residual used for training, or retrain "
               "against the installed residual. Keep a backup before replacing a locally generated residual.";
    }
    return match;
}

struct PissaResidualSite {
    int                l = 0;
    std::string        site;   // "self_attn.q_proj" ... "mlp.down_proj"
    int                in = 0, out = 0;
    std::vector<float> A0;     // ggml [in, rank]: element (i, k) at k*in + i
    std::vector<float> B0;     // ggml [rank, out]: element (k, j) at j*rank + k
};

static inline void pissa_size_split(long long size, float * hi, float * lo) {
    *hi = (float) (size / 65536);
    *lo = (float) (size % 65536);
}
static inline long long pissa_size_join(float hi, float lo) {
    return (long long) hi * 65536ll + (long long) lo;
}

/** Byte length of a file, 0 when it cannot be stat'ed. */
static inline long long pissa_file_size(const std::string & path) {
    HS_STAT_T sb {};
    if (path.empty() || hs_stat(path, &sb) != 0) {
        return 0;
    }
    return (long long) sb.st_size;
}

/** `<stem>.pissa-r<rank>.safetensors` for a base at `lm_path` (stem = basename
 *  without its last extension). */
static inline std::string pissa_residual_name(const std::string & lm_path, int rank) {
    std::string base = lm_path;
    const size_t cut = base.find_last_of("/\\");
    if (cut != std::string::npos) {
        base = base.substr(cut + 1);
    }
    const size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && dot > 0) {
        base = base.substr(0, dot);
    }
    return base + ".pissa-r" + std::to_string(rank) + ".safetensors";
}

static inline std::string pissa_dirname(const std::string & path) {
    const size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? std::string(".") : path.substr(0, cut);
}

/** Full path of the residual that belongs beside `lm_path`. */
static inline std::string pissa_residual_path(const std::string & lm_path, int rank) {
    if (lm_path.empty()) {
        return "";
    }
    return pissa_dirname(lm_path) + "/" + pissa_residual_name(lm_path, rank);
}

/** Reads only pissa.meta. False when the file is absent or is not a residual. */
static inline bool pissa_residual_read_meta(const std::string & path, PissaResidualMeta * meta, std::string * err) {
    STFile st;
    if (!st_open(&st, path.c_str())) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    const STEntry * m = st_find(st, "pissa.meta");
    if (!m || m->dtype != "F32" || m->n_dims != 1 || m->shape[0] < 8) {
        st_close(&st);
        if (err) {
            *err = path + " is not a PiSSA residual (no pissa.meta)";
        }
        return false;
    }
    const float * v  = (const float *) st_data(st, *m);
    meta->version    = (int) v[0];
    meta->rank       = (int) v[1];
    meta->q          = (int) v[2];
    meta->iters      = (int) v[3];
    meta->layer_lo   = (int) v[4];
    meta->layer_hi   = (int) v[5];
    meta->base_size  = pissa_size_join(v[6], v[7]);
    if (m->shape[0] >= 10) {
        meta->energy_mean = (double) v[8];
        meta->energy_min  = (double) v[9];
    }
    st_close(&st);
    if (meta->version != PISSA_RESIDUAL_VERSION) {
        if (err) {
            *err = path + " is residual format v" + std::to_string(meta->version) + "; this build reads v" +
                   std::to_string(PISSA_RESIDUAL_VERSION);
        }
        return false;
    }
    return true;
}

/** Reads the whole file into host F32 (from F16 exactly). `sites` comes back
 *  in file order; look sites up by (l, site) rather than by index. */
static inline bool pissa_residual_read(const std::string & path, PissaResidualMeta * meta,
                                       std::vector<PissaResidualSite> * sites, std::string * err,
                                       bool fingerprint = false) {
    if (!pissa_residual_read_meta(path, meta, err)) {
        return false;
    }
    STFile st;
    if (!st_open(&st, path.c_str())) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    sites->clear();
    meta->fingerprint = fingerprint ? pissa_residual_fingerprint(st) : std::vector<float>();
    auto to_f32 = [&](const STEntry & e, std::vector<float> * out) -> bool {
        int64_t n = 1;
        for (int d = 0; d < e.n_dims; d++) {
            n *= e.shape[d];
        }
        out->assign((size_t) n, 0.0f);
        const void * src = st_data(st, e);
        if (e.dtype == "F16") {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, out->data(), n);
        } else if (e.dtype == "F32") {
            memcpy(out->data(), src, (size_t) n * sizeof(float));
        } else if (e.dtype == "BF16") {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, out->data(), n);
        } else {
            return false;
        }
        return true;
    };
    // Every A0 pulls its B0 by name, so a file with an orphan of either fails.
    for (const STEntry & e : st.entries) {
        if (e.name.rfind("pissa.L", 0) != 0 || e.name.size() < 4 || e.name.compare(e.name.size() - 3, 3, ".A0") != 0) {
            continue;
        }
        // pissa.L<l>.<site>.A0
        const std::string body = e.name.substr(7, e.name.size() - 7 - 3);  // "<l>.<site>"
        const size_t      dot  = body.find('.');
        if (dot == std::string::npos) {
            continue;
        }
        PissaResidualSite s;
        s.l    = atoi(body.substr(0, dot).c_str());
        s.site = body.substr(dot + 1);
        const std::string bname = "pissa.L" + body + ".B0";
        const STEntry *   b     = st_find(st, bname.c_str());
        if (!b || e.n_dims != 2 || b->n_dims != 2 || e.shape[0] != b->shape[1] || e.shape[0] != meta->rank) {
            st_close(&st);
            sites->clear();
            if (err) {
                *err = path + ": malformed site " + e.name;
            }
            return false;
        }
        s.in  = (int) e.shape[1];   // torch [rank, in]
        s.out = (int) b->shape[0];  // torch [out, rank]
        if (!to_f32(e, &s.A0) || !to_f32(*b, &s.B0)) {
            st_close(&st);
            sites->clear();
            if (err) {
                *err = path + ": unsupported dtype on " + e.name;
            }
            return false;
        }
        sites->push_back(std::move(s));
    }
    st_close(&st);
    if (sites->empty()) {
        if (err) {
            *err = path + " holds no pissa.L*.A0 tensors";
        }
        return false;
    }
    return true;
}

static inline const PissaResidualSite * pissa_residual_find(const std::vector<PissaResidualSite> & sites, int l,
                                                            const char * site) {
    for (const PissaResidualSite & s : sites) {
        if (s.l == l && s.site == site) {
            return &s;
        }
    }
    return nullptr;
}

#pragma once

#include "lm-optim.h"

// Shared settings for the YuE2 AR and NAR trainers. Gradient-check contexts
// still use AdamW; only real training passes these settings to context init.
struct Yue2OptimConfig {
    std::string optimizer = "prodigy";
    float prodigy_d0 = 1e-6f;
    LmMuonCfg muon;
};

static bool yue2_optim_valid(const Yue2OptimConfig & c) {
    if (c.optimizer != "adamw" && c.optimizer != "prodigy" && c.optimizer != "muon") {
        fprintf(stderr, "ace-train: --optimizer must be adamw, prodigy or muon\n");
        return false;
    }
    if (!std::isfinite(c.prodigy_d0) || c.prodigy_d0 <= 0 ||
        !std::isfinite(c.muon.lr_scale) || c.muon.lr_scale <= 0 ||
        !std::isfinite(c.muon.momentum) || c.muon.momentum < 0 || c.muon.momentum >= 1 ||
        c.muon.ns_steps < 1 || c.muon.ns_steps > 20 || c.muon.min_dim < 1 ||
        c.muon.bucket < 1 || c.muon.bucket > 256) {
        fprintf(stderr, "ace-train: invalid Prodigy/Muon settings (d0/scale > 0, momentum [0,1), "
                        "ns-steps 1..20, min-dim >= 1, bucket 1..256)\n");
        return false;
    }
    return true;
}

static void yue2_optim_configure(LmOptim * o, const Yue2OptimConfig * c) {
    o->optimizer = c ? c->optimizer : "adamw";
    if (c) {
        o->prodigy_d0 = c->prodigy_d0;
        o->muon = c->muon;
    }
}

static void yue2_optim_report(const LmOptim & o) {
    fprintf(stderr, "[yue2-optim] optimizer %s, %d/%zu tensors on Muon\n",
            o.optimizer.c_str(), o.n_muon, o.params.size());
    if (o.optimizer == "prodigy") {
        fprintf(stderr, "[yue2-optim] Prodigy d0 %.6g; --lr is ignored, schedule multiplier starts at 1.0; "
                        "step logs report effective lr\n", (double) o.prodigy_d0);
    } else if (o.optimizer == "muon" && o.n_muon == 0) {
        fprintf(stderr, "[yue2-optim] WARNING: no tensor qualifies for Muon; all use AdamW. "
                        "Check rank and --muon-min-dim.\n");
    }
}

static std::vector<ggml_tensor *> yue2_optim_state_tensors(const LmOptim & o) {
    std::vector<ggml_tensor *> out;
    for (size_t j = 0; j < o.mom_m.size(); ++j) {
        if (o.mom_m[j]) out.push_back(o.mom_m[j]);
        if (j < o.mom_v.size() && o.mom_v[j]) out.push_back(o.mom_v[j]);
    }
    for (auto * t : o.pg_s) if (t) out.push_back(t);
    for (auto * t : o.pg_x0) if (t) out.push_back(t);
    return out;
}

template<typename T> static bool yue2_optim_write(FILE * f, const T & v) {
    return fwrite(&v, sizeof(v), 1, f) == 1;
}
template<typename T> static bool yue2_optim_read(FILE * f, T & v) {
    return fread(&v, sizeof(v), 1, f) == 1;
}

// Version 3 adds optimizer identity/settings and Prodigy's adaptive scalars
// before the existing tensor blocks. s and x0 join the moments block.
static bool yue2_optim_save(FILE * f, const LmOptim & o) {
    const int32_t code = o.optimizer == "prodigy" ? 2 : o.optimizer == "muon" ? 1 : 0;
    const int32_t ns = o.muon.ns_steps, dim = o.muon.min_dim, bucket = o.muon.bucket;
    const int32_t nesterov = o.muon.nesterov ? 1 : 0;
    return yue2_optim_write(f, code) && yue2_optim_write(f, o.prodigy_d0) &&
        yue2_optim_write(f, o.muon.lr_scale) && yue2_optim_write(f, o.muon.momentum) &&
        yue2_optim_write(f, ns) && yue2_optim_write(f, dim) &&
        yue2_optim_write(f, bucket) && yue2_optim_write(f, nesterov) &&
        yue2_optim_write(f, o.prodigy_d) && yue2_optim_write(f, o.prodigy_r);
}

static bool yue2_optim_load(FILE * f, uint32_t version, LmOptim * o, std::string * err) {
    if (version == 2) {
        if (o->optimizer == "adamw") return true;
        *err = "legacy checkpoint uses AdamW; resume with --optimizer adamw";
        return false;
    }
    int32_t code, ns, dim, bucket, nesterov;
    float d0, scale, momentum;
    double d, r;
    if (!(yue2_optim_read(f, code) && yue2_optim_read(f, d0) &&
          yue2_optim_read(f, scale) && yue2_optim_read(f, momentum) &&
          yue2_optim_read(f, ns) && yue2_optim_read(f, dim) &&
          yue2_optim_read(f, bucket) && yue2_optim_read(f, nesterov) &&
          yue2_optim_read(f, d) && yue2_optim_read(f, r))) {
        *err = "truncated optimizer header";
        return false;
    }
    const int32_t want = o->optimizer == "prodigy" ? 2 : o->optimizer == "muon" ? 1 : 0;
    if (code != want || (code == 2 && d0 != o->prodigy_d0) ||
        (code == 1 && (scale != o->muon.lr_scale || momentum != o->muon.momentum ||
         ns != o->muon.ns_steps || dim != o->muon.min_dim || bucket != o->muon.bucket ||
         nesterov != (o->muon.nesterov ? 1 : 0)))) {
        *err = "optimizer or optimizer settings differ from the checkpoint; restore them or start a new run";
        return false;
    }
    if (code == 2) {
        if (!std::isfinite(d) || d <= 0 || !std::isfinite(r)) {
            *err = "invalid Prodigy state";
            return false;
        }
        o->prodigy_d = d;
        o->prodigy_r = r;
    }
    return true;
}

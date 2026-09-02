#pragma once
// dit-node-profile.h — env-gated per-NODE timing and site attribution for
// `ace-train train-dit`.
//
// WHY, and why it is not `--profile-ops`.
// The existing DitOpProf (dit-train-run.h) already times one micro-step node by
// node through the scheduler's eval callback and buckets it by (op, shape).
// That answers "which OP costs what". It cannot answer "which SITE costs what":
// a MUL_MAT of [128,625] is an adapter contraction in one place and something
// else in another, an OUT_PROD of [1877,625,32] is cross-attention's backward
// but nothing in the key says so, and in flash mode the SCALE/ACC pair that
// builds the packed tensor's gradient is indistinguishable from the 700-odd
// other SCALEs in the graph. Attributing the exact-vs-flash step-time delta
// needs the site, so this file adds one.
//
// HOW the site is known — no guessing where it can be avoided:
//  1. FORWARD nodes are tagged AT CONSTRUCTION. `DnpScope` records the ggml
//     context's last tensor on entry and, on exit, tags every tensor created in
//     between. The graph builder therefore declares its own sites (the three
//     attention regions in dit-train-graph.h — windowed self, full self, cross —
//     and every DitAdapter::apply call via DitAdapter::applyP), and nothing is
//     inferred from shapes.
//  2. BACKWARD nodes inherit from a tagged FORWARD src, depth 1. This is the
//     rule that does not bleed: a fused attention backward takes the forward
//     packed tensor as a src, soft_max_back takes the forward softmax, and the
//     attention mul_mat backwards take the forward q / vt — while the *outputs*
//     of the attention backward (dQ/dK/dV) are themselves backward nodes, so
//     rope_back and the QK-norm backward downstream of them inherit NOTHING and
//     land in `other` rather than being miscounted as attention.
//  3. A shape fallback catches the leftovers that are unmistakably attention by
//     geometry (ne2 == Nh with ne0 in {D, S, enc_S}; ne0 == D && ne1 == Nh; or
//     the GQA REPEAT_BACK shape [D, S_kv, Nkv, B]). It is applied ONLY to nodes
//     rules 1 and 2 left untagged, it cannot tell a windowed layer from a full
//     one (hence `attn.self.unk`), and every node it claims is counted separately
//     in the JSON (`shapeFallbackNodes`) so its size can be checked, not trusted.
//  4. Inside an `apply.*` site, `adapter` vs `base-proj` is decided by whether
//     the node touches a trainable factor: any src in DitAdapter::params(), or
//     any src already marked adapter. The one deliberate over-count is the
//     `y = Wx + delta` ADD, which has an adapter src and so bills to `adapter`.
//
// COST WHEN OFF: `g_dnp` is null, so DnpScope's constructor returns after one
// pointer test and its destructor after one more. Nothing else in this file
// runs, and the eval callback is never installed — the un-profiled graph is
// emitted, allocated and computed exactly as before.
//
// THE ONE THING THESE NUMBERS ARE NOT: absolute per-step cost. ggml's eval
// callback computes the graph one node at a time and calls
// ggml_backend_synchronize between them (ggml-backend.cpp:1681-1712), which
// kills CUDA-graph capture and exposes every kernel launch. The serialised
// total therefore over-states the real compute bucket, and it over-states it in
// proportion to NODE COUNT, which differs between arms. The JSON records the
// un-serialised compute bucket of every OTHER micro-step in the same run
// (`realComputeMsPerStep`) beside the serialised total, so the inflation is a
// measured quantity rather than a caveat — it lands at ~10 us/node in both arms.
//
// Env: DIT_PROFILE_NODES=1 (from micro-step 15) or =N (from micro-step N);
//      DIT_PROFILE_NODES_STEPS=K averages K consecutive micro-steps (default 10,
//      because one serialised step moved a category by 19 % between re-runs).
// Output: <out_dir>/dit_node_profile_<attn_mode>.json

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct DnpAgg {
    long long us = 0;
    int       n  = 0;
};

struct DitNodeProfile {
    // ── configuration ───────────────────────────────────────────────────
    bool        enabled = false;  // DIT_PROFILE_NODES was set
    int         at_step = 15;     // first micro-step to profile
    int         n_steps = 10;     // how many consecutive micro-steps to profile
    int         done    = 0;      // how many have been profiled
    std::string out_path;

    // ── build-time tagging (live only while the profiled graph is built) ──
    bool                                         tagging = false;
    std::unordered_map<const ggml_tensor *, int> tag;
    std::vector<std::string>                     sites;
    std::unordered_map<std::string, int>         site_id;
    std::unordered_set<const ggml_tensor *>      params;  // trainable adapter factors
    std::unordered_set<const ggml_tensor *>      fwd;     // forward graph nodes

    // ── measurement ─────────────────────────────────────────────────────
    long long                                     t0 = 0;
    std::unordered_map<const ggml_tensor *, char>        kind;   // 'a' adapter, 'b' base
    std::unordered_map<const ggml_tensor *, std::string> carry;  // class handed to an untagged consumer
    std::map<std::string, DnpAgg>                 by_class, by_site, by_op, by_key, by_class_op;
    long long                                     total_us  = 0;
    int                                           n_nodes   = 0;
    int                                           n_fwd     = 0;
    int                                           n_bwd     = 0;
    int                                           n_fallback = 0;
    long long                                     fallback_us = 0;
    bool                                          captured  = false;

    // ── geometry, for the shape fallback and for the JSON header ────────
    int S = 0, enc_S = 0, D = 0, Nh = 0, Nkv = 0, B = 1, layers = 0, crop = 0;
    // ── run context ─────────────────────────────────────────────────────
    std::string attn_mode = "exact", attn_prec = "-", adapter = "-", dataset = "-";
    int         graph_nodes = 0, sched_splits = 0;
    long long   step_compute_us = 0;  // real compute bucket of the SERIALISED step
    double      real_compute_ms = 0;  // mean compute bucket of the un-serialised steps
    long long   real_steps      = 0;

    int siteIndex(const char * s) {
        std::string                                    k(s);
        std::unordered_map<std::string, int>::iterator it = site_id.find(k);
        if (it != site_id.end()) {
            return it->second;
        }
        const int id = (int) sites.size();
        sites.push_back(k);
        site_id[k] = id;
        return id;
    }
    const char * siteName(int id) const {
        return (id >= 0 && id < (int) sites.size()) ? sites[(size_t) id].c_str() : "other";
    }
    void reset() {
        tag.clear();
        fwd.clear();
        kind.clear();
        carry.clear();
        sites.clear();
        site_id.clear();
    }
};

// Armed only for the one profiled micro-step; null every other step and in every
// run that did not set the env var.
static DitNodeProfile * g_dnp = nullptr;

// ─── construction-time site scopes ──────────────────────────────────────────

static inline ggml_tensor * dnp_last_tensor(ggml_context * ctx) {
    ggml_tensor * last = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        last = t;
    }
    return last;
}

struct DnpScope {
    ggml_context * ctx  = nullptr;
    int            sid  = -1;
    ggml_tensor *  mark = nullptr;

    DnpScope(ggml_context * c, const char * site) {
        if (!g_dnp || !g_dnp->tagging) {
            return;
        }
        ctx  = c;
        sid  = g_dnp->siteIndex(site);
        mark = dnp_last_tensor(c);
    }
    ~DnpScope() {
        if (!ctx) {
            return;
        }
        ggml_tensor * t = mark ? ggml_get_next_tensor(ctx, mark) : ggml_get_first_tensor(ctx);
        for (; t; t = ggml_get_next_tensor(ctx, t)) {
            g_dnp->tag[t] = sid;
        }
    }
    DnpScope(const DnpScope &)             = delete;
    DnpScope & operator=(const DnpScope &) = delete;
};

// ─── classification ─────────────────────────────────────────────────────────

// "self" / "cross" from a tensor that carries the KV length somewhere in its
// shape, or from its sources. enc_S != S is checked by the caller before the
// fallback is trusted (dit_node_prof_arm refuses to run otherwise).
static inline bool dnp_is_cross_shape(const DitNodeProfile & P, const ggml_tensor * t) {
    for (int i = 0; i < 4; i++) {
        if (t->ne[i] == (int64_t) P.enc_S) {
            return true;
        }
    }
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        if (!t->src[s]) {
            continue;
        }
        for (int i = 0; i < 4; i++) {
            if (t->src[s]->ne[i] == (int64_t) P.enc_S) {
                return true;
            }
        }
    }
    return false;
}

static inline bool dnp_looks_like_attn(const DitNodeProfile & P, const ggml_tensor * t) {
    const int64_t D = P.D, Nh = P.Nh, S = P.S, eS = P.enc_S;
    if (t->ne[2] == Nh && (t->ne[0] == D || t->ne[0] == S || t->ne[0] == eS)) {
        return true;
    }
    if (t->ne[0] == D && t->ne[1] == Nh) {
        return true;
    }
    // GQA leftovers: exact mode's attention mul_mat backward folds the broadcast
    // K/V heads with REPEAT_BACK, whose output is [D, S_kv, Nkv, B]. 128 such
    // nodes per step were being billed to `other` on the exact side only, which
    // flattered flash by the same amount.
    if (t->ne[2] == P.Nkv && t->ne[0] == D && (t->ne[1] == S || t->ne[1] == eS)) {
        return true;
    }
    return false;
}

// The class a node is billed to. `site` is what rules 1-2 produced (-1 = none).
static inline std::string dnp_class_of(DitNodeProfile & P, const ggml_tensor * t, int site, bool is_fwd,
                                       bool * used_fallback) {
    *used_fallback  = false;
    const char * sn = (site >= 0) ? P.siteName(site) : "";

    // Rules 1-2 put every attention node in one of the three named regions
    // (attn.self.win / attn.self.full / attn.cross), so only the direction and
    // the packed-gradient split are left to decide.
    if (strncmp(sn, "attn.", 5) == 0) {
        const std::string base(sn);
        if (t->op == GGML_OP_FLASH_ATTN_TRAIN) {
            return base + ".fwd";
        }
        if (t->op == GGML_OP_FLASH_ATTN_TRAIN_BACK) {
            return base + ".bwd";
        }
        // The packed tensor's gradient: ggml builds it as scale(packed, 0)
        // followed by acc(dO). Both are backward nodes whose only tagged src is
        // the forward packed tensor, so they land here and nowhere else.
        if (!is_fwd && (t->op == GGML_OP_SCALE || t->op == GGML_OP_ACC)) {
            const std::string c = base + ".gradpack";
            P.carry[t]          = c;  // the ACC that follows has no tagged src of its own
            return c;
        }
        return base + (is_fwd ? ".fwd" : ".bwd");
    }

    // A fused op that lost its tag would be a scope bug, not a reason to
    // mislabel it: src[1] is K, whose ne[1] is the KV length.
    if (t->op == GGML_OP_FLASH_ATTN_TRAIN || t->op == GGML_OP_FLASH_ATTN_TRAIN_BACK) {
        const bool cross = t->src[1] && t->src[1]->ne[1] == (int64_t) P.enc_S;
        *used_fallback   = true;
        return std::string("attn.") + (cross ? "cross" : "self.unk") +
               (t->op == GGML_OP_FLASH_ATTN_TRAIN ? ".fwd" : ".bwd");
    }

    // A node that reads a TRAINABLE FACTOR is adapter work wherever it sits. The
    // LoKR/LoRA backward contractions live outside every apply scope — their srcs
    // are the factor (which is not a graph tensor, so no scope ever tagged it)
    // and a grad node — so without this rule the adapter's whole backward would
    // be billed to `other` and its share would read about half of the truth.
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        if (t->src[s] && P.params.count(t->src[s])) {
            P.kind[t] = 'a';
            return "adapter";
        }
    }

    if (strncmp(sn, "apply.", 6) == 0) {
        // adapter vs frozen projection: does the node touch a trainable factor?
        bool adapter = false;
        for (int s = 0; s < GGML_MAX_SRC && !adapter; s++) {
            const ggml_tensor * u = t->src[s];
            if (!u) {
                continue;
            }
            if (P.params.count(u)) {
                adapter = true;
            }
            std::unordered_map<const ggml_tensor *, char>::iterator kt = P.kind.find(u);
            if (kt != P.kind.end() && kt->second == 'a') {
                adapter = true;
            }
        }
        P.kind[t] = adapter ? 'a' : 'b';
        return adapter ? "adapter" : "base-proj";
    }

    // The second half of the packed gradient: ggml emits scale(packed, 0) then
    // acc(that, dO). The ACC's srcs are both backward nodes, so nothing tags it —
    // it inherits from the SCALE, and the chain is two nodes long by construction.
    if (site < 0) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) {
                continue;
            }
            std::unordered_map<const ggml_tensor *, std::string>::iterator ci = P.carry.find(t->src[s]);
            if (ci != P.carry.end()) {
                P.carry[t] = ci->second;
                return ci->second;
            }
        }
    }

    // Rule 3: geometry, only for what rules 1-2 left over.
    if (dnp_looks_like_attn(P, t)) {
        *used_fallback = true;
        return std::string("attn.") + (dnp_is_cross_shape(P, t) ? "cross" : "self.unk") + (is_fwd ? ".fwd" : ".bwd");
    }
    return "other";
}

static inline void dnp_add(std::map<std::string, DnpAgg> & m, const std::string & k, long long us) {
    DnpAgg & e = m[k];
    e.us += us;
    e.n++;
}

static bool dnp_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    DitNodeProfile * P = (DitNodeProfile *) ud;
    if (ask) {
        P->t0 = ggml_time_us();
        return true;  // yes: call back once it has actually been computed
    }
    const long long dt     = ggml_time_us() - P->t0;
    const bool      is_fwd = P->fwd.count(t) != 0;

    int                                                site = -1;
    std::unordered_map<const ggml_tensor *, int>::iterator it = P->tag.find(t);
    if (it != P->tag.end()) {
        site = it->second;
    } else {
        for (int s = 0; s < GGML_MAX_SRC && site < 0; s++) {
            if (!t->src[s]) {
                continue;
            }
            std::unordered_map<const ggml_tensor *, int>::iterator si = P->tag.find(t->src[s]);
            if (si != P->tag.end()) {
                site = si->second;
            }
        }
    }

    bool              fb  = false;
    const std::string cls = dnp_class_of(*P, t, site, is_fwd, &fb);
    const char *      opn = ggml_op_name(t->op);

    char key[192];
    snprintf(key, sizeof(key), "%-14s [%lld,%lld,%lld,%lld]", opn, (long long) t->ne[0], (long long) t->ne[1],
             (long long) t->ne[2], (long long) t->ne[3]);

    dnp_add(P->by_class, cls, dt);
    dnp_add(P->by_site, site >= 0 ? P->siteName(site) : "-", dt);
    dnp_add(P->by_op, opn, dt);
    dnp_add(P->by_key, std::string(key), dt);
    dnp_add(P->by_class_op, cls + " | " + opn, dt);
    P->total_us += dt;
    P->n_nodes++;
    if (is_fwd) {
        P->n_fwd++;
    } else {
        P->n_bwd++;
    }
    if (fb) {
        P->n_fallback++;
        P->fallback_us += dt;
    }
    return true;
}

// ─── lifecycle ──────────────────────────────────────────────────────────────

// Reads the env once. `at_step` is the micro-step index (the same counter
// --profile-ops uses), so the graph is warm and the allocator settled.
static inline void dit_node_prof_init(DitNodeProfile * P, const char * out_dir, const char * attn_mode) {
    const char * e = getenv("DIT_PROFILE_NODES");
    if (!e || !*e || !strcmp(e, "0")) {
        return;
    }
    P->enabled   = true;
    const int n  = atoi(e);
    P->at_step   = (n > 1) ? n : 15;
    // One serialised micro-step is a noisy sample — a re-run of the same arm
    // moved a category by 19 %. Ten consecutive steps cost ~3 s and settle it.
    const char * ns = getenv("DIT_PROFILE_NODES_STEPS");
    P->n_steps      = (ns && atoi(ns) > 0) ? atoi(ns) : 10;
    P->attn_mode    = attn_mode ? attn_mode : "exact";
    char path[1024];
    snprintf(path, sizeof(path), "%s/dit_node_profile_%s.json", out_dir && *out_dir ? out_dir : ".",
             P->attn_mode.c_str());
    P->out_path = path;
}

static inline void dit_node_prof_mark_forward(DitNodeProfile * P, ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; i++) {
        P->fwd.insert(ggml_graph_node(gf, i));
    }
}

static void dit_node_prof_print(DitNodeProfile & P) {
    std::vector<std::pair<std::string, DnpAgg>> v(P.by_class.begin(), P.by_class.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string, DnpAgg> & a, const std::pair<std::string, DnpAgg> & b) {
                  return a.second.us > b.second.us;
              });
    const double k = P.done > 0 ? (double) P.done : 1.0;
    fprintf(stderr,
            "[train-dit] node profile (%s), mean of %d serialised micro-steps: %.1f ms of node time over %d nodes "
            "(%d fwd / %d bwd); those steps' whole compute bucket %.1f ms (the rest is the between-node syncs the "
            "callback forces); the un-serialised reference lands in the JSON; shape fallback claimed %d nodes "
            "(%.1f ms)\n",
            P.attn_mode.c_str(), P.done, (double) P.total_us / 1000.0 / k, (int) (P.n_nodes / k),
            (int) (P.n_fwd / k), (int) (P.n_bwd / k), (double) P.step_compute_us / 1000.0 / k,
            (int) (P.n_fallback / k), (double) P.fallback_us / 1000.0 / k);
    for (size_t i = 0; i < v.size(); i++) {
        fprintf(stderr, "[train-dit]   %8.1f ms  %5.1f%%  x%-6d %s\n", (double) v[i].second.us / 1000.0 / k,
                P.total_us > 0 ? 100.0 * (double) v[i].second.us / (double) P.total_us : 0.0,
                (int) (v[i].second.n / k), v[i].first.c_str());
    }
}

static inline void dnp_json_map(FILE * f, const char * name, const std::map<std::string, DnpAgg> & m, bool last) {
    fprintf(f, "  \"%s\": {\n", name);
    size_t i = 0;
    for (std::map<std::string, DnpAgg>::const_iterator it = m.begin(); it != m.end(); ++it, ++i) {
        std::string k = it->first;
        for (size_t j = 0; j < k.size(); j++) {
            if (k[j] == '"' || k[j] == '\\') {
                k[j] = '\'';
            }
        }
        fprintf(f, "    \"%s\": {\"us\": %lld, \"n\": %d}%s\n", k.c_str(), it->second.us, it->second.n,
                (i + 1 == m.size()) ? "" : ",");
    }
    fprintf(f, "  }%s\n", last ? "" : ",");
}

static void dit_node_prof_write(DitNodeProfile & P) {
    if (!P.captured || P.out_path.empty()) {
        return;
    }
    FILE * f = fopen(P.out_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[train-dit] node profile: cannot write %s\n", P.out_path.c_str());
        return;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"hot-step-dit-node-profile-v1\",\n");
    fprintf(f,
            "  \"note\": \"serialised per-node timing via the scheduler eval callback; ggml synchronises between "
            "nodes so absolute times are inflated relative to realComputeMsSameStep - see overheadUsPerNode\",\n");
    fprintf(f, "  \"attnMode\": \"%s\", \"attnPrec\": \"%s\", \"adapter\": \"%s\",\n", P.attn_mode.c_str(),
            P.attn_prec.c_str(), P.adapter.c_str());
    fprintf(f, "  \"geometry\": {\"S\": %d, \"encS\": %d, \"D\": %d, \"Nh\": %d, \"Nkv\": %d, \"B\": %d, "
               "\"layers\": %d, \"crop\": %d},\n",
            P.S, P.enc_S, P.D, P.Nh, P.Nkv, P.B, P.layers, P.crop);
    fprintf(f,
            "  \"profiledSteps\": %d, \"graphNodes\": %d, \"timedNodes\": %d, \"fwdNodes\": %d, \"bwdNodes\": %d, "
            "\"splits\": %d,\n"
            "  \"serialisedUs\": %lld, \"serialisedStepComputeUs\": %lld,\n"
            "  \"realComputeMsPerStep\": %.3f, \"realStepsAveraged\": %lld,\n"
            "  \"shapeFallbackNodes\": %d, \"shapeFallbackUs\": %lld,\n",
            P.done, P.graph_nodes, P.n_nodes, P.n_fwd, P.n_bwd, P.sched_splits, P.total_us, P.step_compute_us,
            P.real_compute_ms, P.real_steps, P.n_fallback, P.fallback_us);
    dnp_json_map(f, "byClass", P.by_class, false);
    dnp_json_map(f, "bySite", P.by_site, false);
    dnp_json_map(f, "byOp", P.by_op, false);
    dnp_json_map(f, "byClassOp", P.by_class_op, false);
    dnp_json_map(f, "byKey", P.by_key, true);
    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "[train-dit] node profile written to %s\n", P.out_path.c_str());
}

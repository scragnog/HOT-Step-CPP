#pragma once
// dit-data.h — `ace-train train-dit` sample cache, crop sampler and the
// rectified-flow objective (plan §3.1).
//
//   DitSample            one preprocessed song, host-resident for the whole run
//   dit_scan_samples()   preprocess_meta.json order -> per-song safetensors
//   dit_load_channel_stats()  channel_stats.json -> inverse-std weights, mean 1
//   dit_sample_crop()    random patch-aligned window (Side-Step parity)
//   dit_sample_t()       logit-normal timestep with an interval-expert window
//   dit_flow_target()    x1 ~ N(0,1), xt = t*x1 + (1-t)*x0, v = x1 - x0   (D15)
//   dit_sa_mask/dit_ca_mask   byte-for-byte the sampler's masks
//   dit_flow_snr_w()     the §3.5 timestep weight
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.1 / §3.5

#include "model-registry.h"  // registry_list_dir
#include "safetensors.h"
#include "train/lm-common.h"  // LmRng, pm_* (via preprocess-io.h), lm_log
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── one preprocessed song ──────────────────────────────────────────────────

struct DitSample {
    std::string        path, id;
    int                T = 0, Oc = 0, Cc = 0;  // frames, 64, 128
    // Preprocess hard-cut this song at --max-duration: the latent's last frame
    // is an artificial cut, not the song's ending. End-anchored crops skip
    // these songs — training "the ending" on a chop teaches an abrupt stop
    // (2026-08-29; a third of the 240s-era corpus carried the cut).
    bool               truncated = false;
    int                enc_S = 0, enc_H = 0;
    int                enc_S_genre = 0;
    std::vector<float> lat, ctxl, enc, enc_mask;    // frame-major
    std::vector<float> enc_genre, enc_mask_genre;   // empty when absent
};

struct DitChannelStats {  // channel_stats.json
    bool               ok = false;
    std::vector<float> weight;  // [64], mean == 1
};

struct DitCrop {
    int start = 0, len = 0;
};

// ─── safetensors helpers ────────────────────────────────────────────────────

static const float * dit_st_f32(const STFile & st, const char * name, const STEntry ** out_e) {
    const STEntry * e = st_find(st, name);
    if (!e || e->dtype != "F32") {
        return nullptr;
    }
    *out_e = e;
    return (const float *) st_data(st, *e);
}

// One cached song. Returns false (with `why`) for anything malformed; the caller
// turns that into a `warn` and skips the song.
static bool dit_load_sample(const std::string & path, DitSample * s, std::string * why) {
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *why = "cannot open";
        return false;
    }
    const STEntry *e_lat = nullptr, *e_ctx = nullptr, *e_enc = nullptr, *e_em = nullptr;
    const float *  lat  = dit_st_f32(st, "target_latents", &e_lat);
    const float *  ctxl = dit_st_f32(st, "context_latents", &e_ctx);
    const float *  enc  = dit_st_f32(st, "encoder_hidden_states", &e_enc);
    const float *  em   = dit_st_f32(st, "encoder_attention_mask", &e_em);
    if (!lat || !ctxl || !enc || e_lat->n_dims != 2 || e_ctx->n_dims != 2 || e_enc->n_dims != 2) {
        *why = "missing or mistyped target_latents/context_latents/encoder_hidden_states";
        st_close(&st);
        return false;
    }
    s->path  = path;
    s->T     = (int) e_lat->shape[0];
    s->Oc    = (int) e_lat->shape[1];
    {
        // Same convention as the LM extractor: duration is llround(T/25) of the
        // STORED latent, so a capped song lands exactly on max_duration.
        std::map<std::string, std::string> md;
        stmd_read(path.c_str(), &md);
        const int dur = atoi(md.count("duration") ? md["duration"].c_str() : "0");
        const int cap = atoi(md.count("max_duration") ? md["max_duration"].c_str() : "0");
        s->truncated  = cap > 0 && dur >= cap;
    }
    s->Cc    = (int) e_ctx->shape[1];
    s->enc_S = (int) e_enc->shape[0];
    s->enc_H = (int) e_enc->shape[1];
    if ((int) e_ctx->shape[0] != s->T || s->T <= 0 || s->Oc <= 0 || s->Cc <= 0 || s->enc_S <= 0 || s->enc_H <= 0) {
        *why = "context/target frame mismatch or empty tensor";
        st_close(&st);
        return false;
    }
    s->lat.assign(lat, lat + (size_t) s->T * (size_t) s->Oc);
    s->ctxl.assign(ctxl, ctxl + (size_t) s->T * (size_t) s->Cc);
    s->enc.assign(enc, enc + (size_t) s->enc_S * (size_t) s->enc_H);
    s->enc_mask.assign((size_t) s->enc_S, 1.0f);
    if (em && e_em->n_dims == 1 && (int) e_em->shape[0] == s->enc_S) {
        s->enc_mask.assign(em, em + s->enc_S);
    }

    // genre-conditioned encoder states (preprocess writes these when the dataset
    // has a genre ratio) — optional, used by --genre-ratio (D14).
    const STEntry *e_g = nullptr, *e_gm = nullptr;
    const float *  eg  = dit_st_f32(st, "encoder_hidden_states_genre", &e_g);
    const float *  egm = dit_st_f32(st, "encoder_attention_mask_genre", &e_gm);
    if (eg && e_g->n_dims == 2 && (int) e_g->shape[1] == s->enc_H) {
        s->enc_S_genre = (int) e_g->shape[0];
        s->enc_genre.assign(eg, eg + (size_t) s->enc_S_genre * (size_t) s->enc_H);
        s->enc_mask_genre.assign((size_t) s->enc_S_genre, 1.0f);
        if (egm && e_gm->n_dims == 1 && (int) e_gm->shape[0] == s->enc_S_genre) {
            s->enc_mask_genre.assign(egm, egm + s->enc_S_genre);
        }
    }
    st_close(&st);
    return true;
}

// `dit_path` recorded by preprocess for this variant (§4.2's base-match rule).
static std::string dit_meta_dit_path(const char * tensors_dir) {
    const std::string p = lm_join(tensors_dir, "preprocess_meta.json");
    FILE *            f = hs_fopen(p, "rb");
    if (!f) {
        return std::string();
    }
    std::string buf;
    char        tmp[8192];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
        buf.append(tmp, n);
    }
    fclose(f);
    yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
    if (!doc) {
        return std::string();
    }
    yyjson_val *      root = yyjson_doc_get_root(doc);
    const std::string out  = (root && yyjson_is_obj(root)) ? pm_js_str(root, "dit_path") : std::string();
    yyjson_doc_free(doc);
    return out;
}

// Scan the variant directory. preprocess_meta.json supplies the order and the
// per-song file names; a variant without it falls back to a directory listing so
// a hand-assembled cache still trains.
static bool dit_scan_samples(const char * tensors_dir, int limit, std::vector<DitSample> * out, std::string * err) {
    out->clear();
    std::vector<std::pair<std::string, std::string>> files;  // (file, id)

    const std::string meta_path = lm_join(tensors_dir, "preprocess_meta.json");
    FILE *            f         = hs_fopen(meta_path, "rb");
    if (f) {
        std::string buf;
        char        tmp[8192];
        size_t      n;
        while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
            buf.append(tmp, n);
        }
        fclose(f);
        yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
        if (doc) {
            yyjson_val * root = yyjson_doc_get_root(doc);
            yyjson_val * arr  = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "samples") : NULL;
            if (arr && yyjson_is_arr(arr)) {
                size_t       idx, mx;
                yyjson_val * it;
                yyjson_arr_foreach(arr, idx, mx, it) {
                    if (!yyjson_is_obj(it)) {
                        continue;
                    }
                    if (!pm_js_bool(it, "ok", false)) {
                        continue;
                    }
                    const std::string file = pm_js_str(it, "file");
                    if (!file.empty()) {
                        files.push_back({ file, pm_js_str(it, "id") });
                    }
                }
            }
            yyjson_doc_free(doc);
        }
    }
    if (files.empty()) {
        std::vector<std::string> names;
        registry_list_dir(tensors_dir, &names);
        std::sort(names.begin(), names.end());
        for (size_t i = 0; i < names.size(); i++) {
            const std::string & nm = names[i];
            if (nm.size() > 12 && nm.compare(nm.size() - 12, 12, ".safetensors") == 0) {
                files.push_back({ nm, nm.substr(0, nm.size() - 12) });
            }
        }
    }
    if (files.empty()) {
        *err = std::string("no cached songs in ") + tensors_dir;
        return false;
    }
    if (limit > 0 && (int) files.size() > limit) {
        files.resize((size_t) limit);
    }

    for (size_t i = 0; i < files.size(); i++) {
        DitSample   s;
        std::string why;
        const std::string path = lm_join(tensors_dir, files[i].first);
        if (!dit_load_sample(path, &s, &why)) {
            lm_log("warn", "SKIP " + files[i].first + ": " + why);
            continue;
        }
        s.id = files[i].second;
        out->push_back(std::move(s));
    }
    if (out->empty()) {
        *err = std::string("every cached song in ") + tensors_dir + " was unusable";
        return false;
    }
    return true;
}

// channel_std[64] -> w = 1/max(std,1e-6), then w /= mean(w) (mean exactly 1).
// Missing/short file => ok=false and channel balancing is disabled.
static bool dit_load_channel_stats(const char * tensors_dir, DitChannelStats * out) {
    out->ok = false;
    out->weight.clear();
    const std::string p = lm_join(tensors_dir, "channel_stats.json");
    FILE *            f = hs_fopen(p, "rb");
    if (!f) {
        return false;
    }
    std::string buf;
    char        tmp[8192];
    size_t      n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) {
        buf.append(tmp, n);
    }
    fclose(f);
    yyjson_doc * doc = yyjson_read(buf.data(), buf.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * arr  = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "channel_std") : NULL;
    if (arr && yyjson_is_arr(arr)) {
        size_t       idx, mx;
        yyjson_val * it;
        yyjson_arr_foreach(arr, idx, mx, it) {
            if (yyjson_is_num(it)) {
                const double sd = yyjson_get_num(it);
                out->weight.push_back(1.0f / std::max(1e-6f, (float) sd));
            }
        }
    }
    yyjson_doc_free(doc);
    if (out->weight.empty()) {
        out->weight.clear();
        return false;
    }
    double mean = 0.0;
    for (size_t i = 0; i < out->weight.size(); i++) {
        mean += (double) out->weight[i];
    }
    mean /= (double) out->weight.size();
    if (!(mean > 0.0)) {
        out->weight.clear();
        return false;
    }
    for (size_t i = 0; i < out->weight.size(); i++) {
        out->weight[i] = (float) ((double) out->weight[i] / mean);
    }
    out->ok = true;
    return true;
}

// ─── crop sampler (D10) ─────────────────────────────────────────────────────

// len = min(crop, T) rounded DOWN to a multiple of `patch`; start uniform over
// the patch-aligned starts in [0, T - len].
//
// DEVIATION (E1, documented in the handoff): §3.1 says "start = uniform(0, T-len)
// rounded down to a multiple of patch". That composition is NOT uniform over the
// aligned starts — the top bucket is short — and would fail T10's own ±3σ
// uniformity assertion. Drawing uniformly over the (T-len)/patch + 1 aligned
// starts is the same support with an exactly uniform distribution.
static DitCrop dit_sample_crop(LmRng * rng, int T, int crop, int patch) {
    DitCrop c;
    if (patch < 1) {
        patch = 1;
    }
    int len = (crop > 0 && crop < T) ? crop : T;
    len -= len % patch;
    if (len < patch) {
        len = std::min(T - (T % patch), patch);
    }
    const int n_starts = (T - len) / patch + 1;
    c.len              = len;
    c.start            = patch * (int) lm_rng_below(rng, (uint64_t) std::max(1, n_starts));
    GGML_ASSERT(c.len > 0 && c.len <= T);
    GGML_ASSERT(c.start >= 0 && c.start + c.len <= T);
    return c;
}

// Structured crop draw (2026-08-29, ported from the MM3 LM crop fix; endpoint
// spike repaired 2026-08-30): a uniform sampler sees the true ending almost
// never, so the arc's endpoints can train on starvation rations. `start_frac`
// of draws land on the song's OPENING REGION and `end_frac` flush against the
// track end (skipped for truncated songs — their "end" is a preprocess chop);
// the rest fall through to the uniform sampler.
//
// WHY THE START SHARE IS A REGION (the 2026-08-30 fix). The first port put the
// whole start share on the single crop start 0. That is fine in the MM3 LM
// trainer it came from, where the crop is ~82% of the song and there are only a
// handful of legal starts — but the DiT crop is a few percent of the track, so
// with patch=2 and a 418-frame crop of a 6625-frame song there are 3104 legal
// starts and frame 0 was drawn 622x more often than any other, so the adapter
// memorised the training set's eleven opening WINDOWS and stamped them on every
// render (measured: the first 55 s of an adapter render carried 0.14x the
// spectral flux of its own body, against 0.89x for the bare DiT).
//
// A CORRECTION TO THE 2026-08-30 THEORY (2026-09-01). That commit blamed
// "absolute RoPE positions" for both spikes and zeroed the end share on that
// basis. The theory is mechanically impossible: RoPE is applied to q/k only
// (cross-attn is unroped, nothing else consumes t_pos), and rotation-composed
// attention scores depend on q-k position DELTAS alone — the graph is exactly
// shift-invariant in the crop offset, so `anchor_song` vs `zero` cannot change
// a single output. What the pinned draws actually did was oversample the same
// few WINDOWS 622x (content overfit); what zeroing the end share actually did
// was leave endings unsupervised — a flush-end draw then only arrives through
// the uniform sampler's single legal start, ~0.03% of draws, ~1.4 times in a
// 500-epoch run. Every adapter of the 2026-08-31 overnight batch trained that
// way still cut off mid-stream with no ending, which is what refuted the
// "base DiT already ends well, nothing to rescue" assumption: a dim-512 LoKr
// across 32 layers freely damages a mapping it never receives gradient on.
// The ending signal the adapter must learn is CONTENT — the ctxl frames of a
// real ending — and that generalises to any render duration precisely because
// position cannot enter.
//
// MM3 splits this share "half at frame 0, half over aligned tiles", but tiles
// land on a handful of exact positions, which at this coverage still leaves a
// ~197x spike. What transfers is the INTENT, not the mechanic: spend the share
// on every crop that TOUCHES the opening — start drawn uniformly over the
// aligned starts in [0, start_window * len). At window=1 that is a 3.0x worst
// spike (down from 622x) while the opening region still sees 18.5% of draws
// against a natural 6.7%, so it is covered without being memorised.
//
// THE END SHARE (restored 2026-09-01) needs a different diversifier, because
// unlike the opening the payload is pinned to one boundary: only a crop whose
// right edge IS the track end contains the decay-to-silence. Spreading over a
// region the way the start share does would hand the flush window ~1/198th of
// the share and starve the one lesson that matters. So the share splits:
//   - flush-jitter (first half): flush against the track end, length jittered
//     over the patch-aligned lengths in [crop/2, crop]. Every draw carries the
//     real ending; the moving left edge varies the window (~100 distinct crops
//     per song at crop=396) so no exact window repeats often enough to memorise.
//   - closing region (second half): full-length crops uniform over the aligned
//     starts in [T - len - win + patch, T - len], the mirror of the opening
//     region — context coverage for the approach to the ending.
// Both fold their variety out of the already-drawn uniform start, keeping the
// two-draw stride.
//
// The share should also track how much of the song a crop covers: a uniform
// draw already gives the opening region crop/T of the mass, which IS its fair
// share, so at small crops the boost must stay small. `dit_crop_endpoint_frac`
// below does that scaling; this function takes the already-scaled fracs.
//
// RNG DISCIPLINE: exactly TWO rng draws per call, on every branch — the resume
// replay and the smoke probe both count on the crop stream advancing by a
// fixed stride per element. The in-window position is therefore folded out of
// the start already drawn by dit_sample_crop rather than costing a third draw.
// `random` mode must keep calling dit_sample_crop directly (ONE draw) so
// pre-2026-08-29 streams replay unchanged.
static DitCrop dit_sample_crop_structured(LmRng * rng, int T, int crop, int patch, float start_frac, float end_frac,
                                          bool truncated, int start_window) {
    // Draw the mode selector and the (possibly discarded) uniform start in a
    // fixed order.
    const double  u  = (double) lm_rng_below(rng, 1u << 24) / (double) (1u << 24);
    const DitCrop un = dit_sample_crop(rng, T, crop, patch);
    if (u < (double) start_frac) {
        DitCrop c;
        c.len = un.len;
        // Opening REGION: uniform over the aligned starts in [0, window*len),
        // i.e. every crop that still contains some of the song's opening. The
        // position is folded out of the already-drawn uniform start, so this
        // branch costs no extra draw.
        const int span = T - c.len;
        int       win  = std::max(1, start_window) * c.len;
        if (win > span + patch) {
            win = span + patch;
        }
        const int slots = std::max(1, win / patch);
        int       st    = patch * ((un.start / patch) % slots);
        if (st > span) {
            st = span - (span % patch);
        }
        c.start = st > 0 ? st : 0;
        return c;
    }
    if (u < (double) start_frac + (double) end_frac && !truncated) {
        DitCrop      c;
        const double mid = (double) start_frac + (double) end_frac * 0.5;
        if (u < mid) {
            // Flush-jitter: the crop's right edge is the track's true end; the
            // length (and so the left edge) is folded out of the uniform start,
            // costing no extra draw. len stays patch-aligned because un.len and
            // patch*k both are.
            int len_min = un.len / 2;
            len_min -= len_min % patch;
            if (len_min < patch) {
                len_min = patch;
            }
            const int len_opts = (un.len - len_min) / patch + 1;
            c.len              = un.len - patch * ((un.start / patch) % std::max(1, len_opts));
        } else {
            // Closing region: full-length crop, start uniform over the aligned
            // starts whose window still touches the closing region — the mirror
            // of the opening branch above.
            c.len          = un.len;
            const int span = T - c.len;
            int       win  = std::max(1, start_window) * c.len;
            if (win > span + patch) {
                win = span + patch;
            }
            const int slots = std::max(1, win / patch);
            int       st    = span - patch * ((un.start / patch) % slots);
            if (st < 0) {
                st = 0;
            }
            c.start = st - (st % patch);
            GGML_ASSERT(c.len > 0 && c.start >= 0 && c.start + c.len <= T);
            return c;
        }
        const int last = T - c.len;
        c.start        = last - (last % patch);
        if (c.start < 0) {
            c.start = 0;
        }
        GGML_ASSERT(c.len > 0 && c.start + c.len <= T);
        return c;
    }
    return un;
}

// Scale an endpoint share to how much of the track a crop actually covers.
//
// The raw 0.2/0.2 defaults came from the MM3 LM trainer, where the crop is most
// of the song. Here a crop is typically a few percent, and a uniform draw
// already gives each endpoint region crop/T of the mass. Boosting past a small
// multiple of that turns "cover the endpoints" into "memorise these eleven
// openings". `k` is that multiple; the result is clamped to the requested frac
// so the knob still means "at most this much".
static float dit_crop_endpoint_frac(float requested, int crop, int T_median, float k) {
    if (requested <= 0.0f || crop <= 0 || T_median <= 0) {
        return 0.0f;
    }
    const float coverage = (float) crop / (float) T_median;
    const float scaled   = k * coverage;
    return scaled < requested ? scaled : requested;
}

// ─── timestep (D12) ─────────────────────────────────────────────────────────

// t = sigmoid(mu + sigma*z), clamped to [1e-5, 1-1e-5]; rejection-resampled
// (max 64 tries, then clamped) into the interval-expert window [t_min, t_max].
static float dit_sample_t(LmRng * rng, float mu, float sigma, float t_min, float t_max) {
    float t = 0.5f;
    for (int i = 0; i < 64; i++) {
        const float z = lm_rng_normal(rng);
        t             = 1.0f / (1.0f + expf(-(mu + sigma * z)));
        if (t < 1e-5f) {
            t = 1e-5f;
        }
        if (t > 1.0f - 1e-5f) {
            t = 1.0f - 1e-5f;
        }
        if (t >= t_min && t <= t_max) {
            return t;
        }
    }
    return std::min(std::max(t, t_min), t_max);
}

// ─── rectified-flow target (D15) ────────────────────────────────────────────
//
// ONE rng stream, drawn in index order. This exact ordering is the verifier's
// E[v^2] = 1.672071 fingerprint (T9) — do not reorder it.
static void dit_flow_target(const float * x0, size_t n, float t, LmRng * rng, std::vector<float> * xt,
                            std::vector<float> * v) {
    xt->resize(n);
    v->resize(n);
    for (size_t i = 0; i < n; i++) {
        const float x1 = lm_rng_normal(rng);
        (*xt)[i]       = t * x1 + (1.0f - t) * x0[i];
        (*v)[i]        = x1 - x0[i];
    }
}

// ─── attention masks (byte-for-byte the sampler's) ──────────────────────────

// Self-attention sliding window, F16, [S(kv), S(q)] for ONE element.
//
// `valid_S` (0 or S = nothing padded) is the element's own valid token count.
// When it is shorter than S, the padded tail is masked OUT AS KEYS for every
// VALID query. That column mask is load-bearing, not cosmetic: padded latent
// frames sit inside the KV window of the valid queries, so without it they feed
// every valid query's attention output and from there the adapter gradients —
// and the design-B4 loss mask cannot undo that, because it only zeroes the
// padded frames' OWN loss contribution, never their influence on the valid ones.
//
// Padded QUERY rows deliberately keep the plain window mask instead of being
// masked out entirely. A row whose every key is -inf makes soft_max produce NaN
// (its max is -inf, so every exp() argument is -inf - -inf), and under a sliding
// window a padded query far past the valid tail would have exactly that. They do
// not need masking to be inert: a padded frame's loss weight is exactly 0, so
// dL/d(its output) is 0, and a zero output gradient contributes exactly zero to
// dQ/dK/dV. With the columns masked they are invisible to the valid rows too, so
// a padded frame is inert in BOTH directions.
static void dit_sa_mask(int S, int win, std::vector<uint16_t> * out, int valid_S = 0) {
    const int vS = (valid_S > 0 && valid_S < S) ? valid_S : S;
    out->assign((size_t) S * (size_t) S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < S; ki++) {
            const int  dist   = (qi > ki) ? (qi - ki) : (ki - qi);
            const bool in_win = (win <= 0) || (S <= win) || (dist <= win);
            const bool pad_kv = (qi < vS) && (ki >= vS);
            (*out)[(size_t) qi * (size_t) S + (size_t) ki] =
                ggml_fp32_to_fp16((in_win && !pad_kv) ? 0.0f : -INFINITY);
        }
    }
}

// Cross-attention encoder-padding mask, F16, [enc_S(kv), S(q)].
static void dit_ca_mask(int enc_S, int S, const std::vector<float> & m, std::vector<uint16_t> * out) {
    out->assign((size_t) enc_S * (size_t) S, 0);
    for (int qi = 0; qi < S; qi++) {
        for (int ki = 0; ki < enc_S; ki++) {
            const bool real = m.empty() || m[(size_t) ki] > 0.5f;
            (*out)[(size_t) qi * (size_t) enc_S + (size_t) ki] = ggml_fp32_to_fp16(real ? 0.0f : -INFINITY);
        }
    }
}

// ─── micro-batch assembly (design B2 / B3 / B4) ─────────────────────────────
//
// A micro-batch is B crops from B DIFFERENT songs (Side-Step DataLoader parity),
// each with its OWN timestep, CFG-dropout state, genre variant and encoder-padding
// mask. Elements are padded to the batch's longest crop; the pad is WRAP-filled
// from the element's own crop (in-distribution content) and carries loss weight
// exactly 0, so it contributes nothing to the loss or the gradients.
//
// RNG ORDER IS LOAD-BEARING. The draws happen in exactly the order the
// pre-batching trainer made them — crop then noise, element by element — so at
// B == 1 the streams are consumed identically and §2.3.1 holds.
//
// SELF-ATTENTION MASK SHAPE. Design B1's [S,S] mask (ne2 == ne3 == 1, so it
// broadcasts over heads AND batch) is kept as the FAST PATH — it is what every
// micro-batch whose elements all crop to the same length uses, which is every
// batch drawn from songs at least `crop` frames long, i.e. the common case.
// The moment ANY element is padded the mask becomes per-element [S,S,1,B]
// (`h->sa_B == B`) with that element's padded tail masked out as KEYS: padded
// frames sit inside the valid queries' KV window, and the loss mask alone does
// NOT stop them reaching the adapter gradients through it (see dit_sa_mask's
// header for the full argument, and self-test rung SB3 for the measurement).
// Wrap-filling the pad is still done, but it is now belt-and-braces rather than
// the mechanism.

struct DitBatchElem {
    const DitSample * s         = nullptr;
    float             t         = 0.5f;
    float             w         = 1.0f;  // (w_i / wbar); always 1 for --loss-weighting none
    bool              cfg_drop  = false;
    bool              use_genre = false;
    // filled in by dit_batch_assemble
    int crop_start = 0;
    int len        = 0;  // valid latent frames for this element
};

struct DitBatchHost {
    std::vector<float>    input, vtgt, enc, lw, lwu;
    std::vector<int32_t>  pos;
    // `sa` is the sliding-window mask (layer_type 0). `sa_pad` is the SAME
    // per-element pad-column mask with NO window, for the full-attention layers
    // (layer_type 1), which take no mask at all when nothing is padded. Empty
    // unless something is padded — which can only happen at B > 1, since a
    // 1-element batch is padded to its own length.
    std::vector<uint16_t> sa, sa_pad, ca;
    int                   len    = 0;  // padded frames per element (the batch max)
    int                   S      = 0;
    int                   B      = 0;
    // Elements carried by `sa`: 1 = the [S,S] broadcast mask (nothing padded),
    // B = a per-element [S,S,1,B] mask. The graph input view shape follows this.
    int                   sa_B   = 1;
    float                 gscale = 1.0f;  // the graph's ggml_scale factor
};

struct DitBatchCfg {
    int  in_ch          = 192;
    int  out_ch         = 64;
    int  enc_H          = 2048;
    int  enc_S          = 0;
    int  patch          = 2;
    int  sliding_window = 0;
    int  crop           = 0;
    bool weighted       = true;  // --loss-weighting flow_snr
    const std::vector<float> * null_cond = nullptr;
    // Crop regime (2026-08-29). Defaults are the FIXED behaviour; `zero` /
    // `random` reproduce the legacy run byte-for-byte (see the sampler note).
    bool  anchor_song = true;   // RoPE positions carry the crop's true offset
    bool  structured  = true;   // start/end-weighted crop draws
    // Effective shares — already scaled by dit_crop_endpoint_frac(). The
    // trainer scales the user's request by crop coverage before filling these.
    float start_frac  = 0.2f;
    float end_frac    = 0.0f;   // 0 by default: the base DiT already ends well
    // Opening window the start share spreads over, in crop lengths. 1 = every
    // crop that still touches the song's opening; larger dilutes the boost.
    int   start_window = 1;
};

static void dit_batch_assemble(const DitBatchCfg & cfg, std::vector<DitBatchElem> & els, LmRng * rng_crop,
                               LmRng * rng_noise, DitBatchHost * h) {
    const int B  = (int) els.size();
    const int Oc = cfg.out_ch;

    // 1) crops, in element order.
    int padded = 0;
    for (int b = 0; b < B; b++) {
        const DitCrop cr = cfg.structured
            ? dit_sample_crop_structured(rng_crop, els[(size_t) b].s->T, cfg.crop, cfg.patch, cfg.start_frac,
                                         cfg.end_frac, els[(size_t) b].s->truncated, cfg.start_window)
            : dit_sample_crop(rng_crop, els[(size_t) b].s->T, cfg.crop, cfg.patch);
        els[(size_t) b].crop_start = cr.start;
        els[(size_t) b].len        = cr.len;
        padded                     = std::max(padded, cr.len);
    }
    h->B   = B;
    h->len = padded;
    h->S   = padded / cfg.patch;

    h->input.assign((size_t) cfg.in_ch * (size_t) padded * (size_t) B, 0.0f);
    h->vtgt.assign((size_t) Oc * (size_t) padded * (size_t) B, 0.0f);
    h->enc.assign((size_t) cfg.enc_H * (size_t) cfg.enc_S * (size_t) B, 0.0f);
    h->lw.assign((size_t) padded * (size_t) B, 0.0f);
    h->lwu.assign((size_t) padded * (size_t) B, 0.0f);
    h->pos.assign((size_t) h->S * (size_t) B, 0);
    h->ca.assign((size_t) cfg.enc_S * (size_t) h->S * (size_t) B, 0);

    // 2) per element: rectified-flow target over the VALID frames only (so the
    //    noise stream draws exactly what the pre-batching trainer drew), then the
    //    wrap-fill, the encoder slice and that element's cross-attention mask.
    std::vector<float>    xt, v;
    std::vector<uint16_t> ca_one;
    for (int b = 0; b < B; b++) {
        DitBatchElem &    e = els[(size_t) b];
        const DitSample & s = *e.s;
        dit_flow_target(s.lat.data() + (size_t) e.crop_start * (size_t) s.Oc, (size_t) e.len * (size_t) s.Oc, e.t,
                        rng_noise, &xt, &v);
        for (int f = 0; f < padded; f++) {
            const int sf  = f % e.len;  // wrap-fill past the valid frames
            float *   dst = &h->input[((size_t) b * (size_t) padded + (size_t) f) * (size_t) cfg.in_ch];
            memcpy(dst, &s.ctxl[(size_t) (e.crop_start + sf) * (size_t) s.Cc], (size_t) s.Cc * sizeof(float));
            memcpy(dst + s.Cc, &xt[(size_t) sf * (size_t) s.Oc], (size_t) s.Oc * sizeof(float));
            memcpy(&h->vtgt[((size_t) b * (size_t) padded + (size_t) f) * (size_t) Oc],
                   &v[(size_t) sf * (size_t) s.Oc], (size_t) Oc * sizeof(float));
        }

        // encoder conditioning: genre variant | null (CFG dropout) | caption. The
        // padding MASK stays the song's own even when the states are replaced,
        // which is what the pre-batching trainer did.
        const std::vector<float> * src_enc  = &s.enc;
        const std::vector<float> * src_mask = &s.enc_mask;
        if (e.use_genre && !s.enc_genre.empty()) {
            src_enc  = &s.enc_genre;
            src_mask = &s.enc_mask_genre;
        }
        float * edst = &h->enc[(size_t) b * (size_t) cfg.enc_H * (size_t) cfg.enc_S];
        if (e.cfg_drop && cfg.null_cond && (int) cfg.null_cond->size() == cfg.enc_H) {
            for (int i = 0; i < cfg.enc_S; i++) {
                memcpy(edst + (size_t) i * (size_t) cfg.enc_H, cfg.null_cond->data(),
                       (size_t) cfg.enc_H * sizeof(float));
            }
        } else {
            memcpy(edst, src_enc->data(),
                   std::min(src_enc->size(), (size_t) cfg.enc_H * (size_t) cfg.enc_S) * sizeof(float));
        }
        dit_ca_mask(cfg.enc_S, h->S, *src_mask, &ca_one);
        memcpy(&h->ca[(size_t) b * (size_t) cfg.enc_S * (size_t) h->S], ca_one.data(),
               ca_one.size() * sizeof(uint16_t));

        // Song-anchored RoPE (the MM3 crop-anchor fix, ported 2026-08-29): a
        // crop from 60 s in is PRESENTED at 60 s, not passed off as the song's
        // opening. crop_start is patch-aligned, so the division is exact. The
        // wrap-filled pad frames continue past the valid span; their loss
        // weight is zero, same as before. `zero` reproduces the legacy lie.
        const int pos0 = cfg.anchor_song ? e.crop_start / cfg.patch : 0;
        for (int i = 0; i < h->S; i++) {
            h->pos[(size_t) b * (size_t) h->S + (size_t) i] = pos0 + i;
        }
    }
    // Self-attention mask: the shared [S,S] broadcast when every element fills
    // the padded length, per-element [S,S,1,B] the moment one does not.
    bool any_pad = false;
    for (int b = 0; b < B; b++) {
        any_pad = any_pad || (els[(size_t) b].len < padded);
    }
    h->sa_B = any_pad ? B : 1;
    h->sa_pad.clear();
    if (!any_pad) {
        dit_sa_mask(h->S, cfg.sliding_window, &h->sa);
    } else {
        // TWO masks, because dit_train_layer gives layer_type 0 (sliding window)
        // and layer_type 1 (full attention) different masks — and full attention
        // gets NO mask at all in the unpadded case. A pad that is masked only in
        // the windowed layers still leaks through every full-attention layer, so
        // those need a window-free copy carrying the same padded KV columns.
        const size_t per = (size_t) h->S * (size_t) h->S;
        h->sa.assign(per * (size_t) B, 0);
        h->sa_pad.assign(per * (size_t) B, 0);
        std::vector<uint16_t> sa_one;
        for (int b = 0; b < B; b++) {
            const int valid_S = els[(size_t) b].len / cfg.patch;
            dit_sa_mask(h->S, cfg.sliding_window, &sa_one, valid_S);
            memcpy(&h->sa[(size_t) b * per], sa_one.data(), sa_one.size() * sizeof(uint16_t));
            dit_sa_mask(h->S, 0 /* no window */, &sa_one, valid_S);
            memcpy(&h->sa_pad[(size_t) b * per], sa_one.data(), sa_one.size() * sizeof(uint16_t));
        }
    }

    // 3) design-B4 loss weights.
    if (B == 1) {
        // THE BYTE-IDENTITY ANCHOR (§2.3.1). One element cannot be padded, so the
        // mask is all ones and x*1.0f is exact; the scale is the pre-batching
        // 1/(Oc*len) and the flow_snr weight stays in the scalar t_lossgrad.
        for (int f = 0; f < padded; f++) {
            h->lw[(size_t) f]  = (f < els[0].len) ? 1.0f : 0.0f;
            h->lwu[(size_t) f] = h->lw[(size_t) f];
        }
        h->gscale = 1.0f / (float) ((int64_t) Oc * (int64_t) els[0].len);
    } else if (cfg.weighted) {
        // flow_snr: mean of per-sample means, each scaled by w_i/wbar
        // (Side-Step lora_module.py:578-593).
        for (int b = 0; b < B; b++) {
            const double denom = (double) Oc * (double) els[(size_t) b].len * (double) B;
            const float  wl    = (float) ((double) els[(size_t) b].w / denom);
            const float  ul    = (float) (1.0 / denom);
            for (int f = 0; f < padded; f++) {
                const bool valid = f < els[(size_t) b].len;
                h->lw[(size_t) b * (size_t) padded + (size_t) f]  = valid ? wl : 0.0f;
                h->lwu[(size_t) b * (size_t) padded + (size_t) f] = valid ? ul : 0.0f;
            }
        }
        h->gscale = 1.0f;
    } else {
        // "none": masked GLOBAL element mean over the whole micro-batch.
        double total = 0.0;
        for (int b = 0; b < B; b++) {
            total += (double) Oc * (double) els[(size_t) b].len;
        }
        const float g = (float) (1.0 / std::max(1.0, total));
        for (int b = 0; b < B; b++) {
            for (int f = 0; f < padded; f++) {
                const bool valid = f < els[(size_t) b].len;
                h->lw[(size_t) b * (size_t) padded + (size_t) f]  = valid ? g : 0.0f;
                h->lwu[(size_t) b * (size_t) padded + (size_t) f] = valid ? g : 0.0f;
            }
        }
        h->gscale = 1.0f;
    }
}

// ─── flow_snr timestep weight (§3.5) ────────────────────────────────────────
//
// w(t) = ((1-t)^t_bias) / (t * (1-t)), clamped to snr_gamma.
// Side-Step clamps t to [1e-4, 1-1e-4] before the division (lora_module.py:579).
static float dit_flow_snr_w(float t, float t_bias, float snr_gamma) {
    float tf = t;
    if (tf < 1e-4f) {
        tf = 1e-4f;
    }
    if (tf > 1.0f - 1e-4f) {
        tf = 1.0f - 1e-4f;
    }
    float w = powf(1.0f - tf, t_bias) / (tf * (1.0f - tf));
    if (w > snr_gamma) {
        w = snr_gamma;
    }
    return w;
}

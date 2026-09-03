#pragma once
// artist-token-runtime.h — load a trained artist token (textual inversion) and
// apply it at inference.
//
// The trainer writes artist_token.safetensors + artist_token.json
// (train/artist-token-io.h). This is the reader, and the ONLY consumer that
// matters for an ear test: nothing else in the engine looks at those files.
//
// HOW IT FINDS ITS POSITIONS. The prompt builder splices k copies of a
// placeholder id into the caption span (prompt.h lm_append_user_span), so the
// vectors' positions are wherever that run of ids landed. Rather than thread an
// offset through lm_forward and every caller, the applier SCANS the batch for k
// consecutive placeholder ids. That makes the uncond/CFG prompt — which is
// built without the splice and therefore contains no such run — automatically
// unaffected, which is what we want: the token conditions the positive branch
// only.
//
// The scan can in principle false-positive on a caption that genuinely repeats
// the seed word k times in a row. With k=8 and a seed like "band" that is not a
// real risk, but it IS the reason to prefer an unusual seed word.
//
// LOADED FROM AN ENV VAR, deliberately. HOTSTEP_ARTIST_TOKEN=<dir> is enough to
// A/B two ace-server instances against each other without touching the server,
// the request schema or the UI — none of which should grow a knob for a feature
// that has never been heard. Wire it properly once it earns its place.

#include "safetensors.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct ArtistTokenRT {
    bool                  loaded      = false;
    bool                  tried       = false;
    int                   k           = 0;
    int                   placeholder = -1;
    int                   hidden      = 0;
    std::string           name, site;
    std::vector<float>    vec;  // [k][hidden] row-major

    // Backend residency, allocated on first use against the model's backend.
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

// ONE instance across every translation unit. `static` at namespace scope in a
// header gives each TU its OWN copy, which is not a style nit here: the LM
// loader would fill its copy and print "[ArtistToken] loaded", while
// pipeline-lm.cpp's prompt builder and the forward hook read a different, empty
// one — producing a confident log line and exactly zero effect. `inline` gives
// the vague linkage that collapses them into one object.
inline ArtistTokenRT & art_rt() {
    static ArtistTokenRT s;
    return s;
}

// Minimal scalar reads out of artist_token.json. Not a JSON parser: the file is
// written by artist_token_write() and has a known, flat shape.
inline bool art_json_int(const std::string & j, const char * key, int * out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t            p   = j.find(pat);
    if (p == std::string::npos) {
        return false;
    }
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) {
        return false;
    }
    *out = atoi(j.c_str() + p + 1);
    return true;
}

inline bool art_json_str(const std::string & j, const char * key, std::string * out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t            p   = j.find(pat);
    if (p == std::string::npos) {
        return false;
    }
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) {
        return false;
    }
    p = j.find('"', p);
    if (p == std::string::npos) {
        return false;
    }
    const size_t e = j.find('"', p + 1);
    if (e == std::string::npos) {
        return false;
    }
    *out = j.substr(p + 1, e - p - 1);
    return true;
}

// `want_site` is checked, not assumed: an as15_dit file is 2048-wide and an
// as15_lm file is the LM's hidden size, and applying one where the other
// belongs would silently add a meaningless vector rather than fail.
inline bool artist_token_load(const std::string & dir, const char * want_site, int want_hidden, std::string * err) {
    const std::string jp = dir + "/artist_token.json";
    const std::string sp = dir + "/artist_token.safetensors";

    FILE * f = fopen(jp.c_str(), "rb");
    if (!f) {
        *err = "cannot open " + jp;
        return false;
    }
    std::string js;
    char        rb[4096];
    size_t      n;
    while ((n = fread(rb, 1, sizeof(rb), f)) > 0) {
        js.append(rb, n);
    }
    fclose(f);

    ArtistTokenRT a;
    art_json_str(js, "name", &a.name);
    art_json_str(js, "site", &a.site);
    art_json_int(js, "k", &a.k);
    art_json_int(js, "placeholder", &a.placeholder);
    art_json_int(js, "hidden", &a.hidden);

    if (a.site != want_site) {
        *err = "artist token site \"" + a.site + "\" is not \"" + want_site + "\"";
        return false;
    }
    if (a.hidden != want_hidden) {
        char b[160];
        snprintf(b, sizeof(b), "artist token hidden %d != model hidden %d", a.hidden, want_hidden);
        *err = b;
        return false;
    }
    if (a.k < 1 || a.placeholder < 0) {
        *err = "artist token json is missing k or placeholder";
        return false;
    }

    STFile st;
    if (!st_open(&st, sp.c_str())) {
        *err = "cannot open " + sp;
        return false;
    }
    const STEntry * e = st_find(st, "artist_token.vec");
    if (!e || e->n_dims != 2 || e->shape[0] != a.k || e->shape[1] != a.hidden || e->dtype != "F32") {
        st_close(&st);
        *err = "artist_token.vec missing or not F32 [k, hidden]";
        return false;
    }
    a.vec.assign((size_t) a.k * (size_t) a.hidden, 0.0f);
    memcpy(a.vec.data(), st_data(st, *e), a.vec.size() * sizeof(float));
    st_close(&st);

    a.loaded = true;
    a.tried  = true;  // the assignment below overwrites the flag load_env just set
    art_rt() = a;
    fprintf(stderr, "[ArtistToken] loaded \"%s\" (%s) k=%d placeholder=%d hidden=%d from %s\n", a.name.c_str(),
            a.site.c_str(), a.k, a.placeholder, a.hidden, dir.c_str());
    return true;
}

// Called once per model load. Silent no-op when the env var is unset, which is
// every existing run.
inline void artist_token_load_env(const char * want_site, int want_hidden) {
    if (art_rt().tried) {
        return;
    }
    art_rt().tried = true;
    const char * dir     = getenv("HOTSTEP_ARTIST_TOKEN");
    if (!dir || !*dir) {
        return;
    }
    std::string err;
    if (!artist_token_load(dir, want_site, want_hidden, &err)) {
        fprintf(stderr, "[ArtistToken] NOT LOADED: %s\n", err.c_str());
    }
}

// Index of the first run of k consecutive placeholder ids, or -1.
inline int artist_token_find(const int * ids, int n) {
    const ArtistTokenRT & a = art_rt();
    if (!a.loaded || a.k <= 0 || n < a.k) {
        return -1;
    }
    for (int i = 0; i + a.k <= n; i++) {
        int j = 0;
        while (j < a.k && ids[i + j] == a.placeholder) {
            j++;
        }
        if (j == a.k) {
            // Once per process. "Loaded" and "applied" are different claims,
            // and the gap between them is exactly where the first attempt at
            // this failed silently — the file loaded into one translation
            // unit's copy of the state while the forward read another's.
            static bool said = false;
            if (!said) {
                said = true;
                fprintf(stderr, "[ArtistToken] applied at prompt offset %d (k=%d)\n", i, a.k);
            }
            return i;
        }
    }
    return -1;
}

// Materialise the [hidden, k] delta on `backend`, once.
inline ggml_tensor * artist_token_tensor(ggml_backend_t backend) {
    ArtistTokenRT & a = art_rt();
    if (!a.loaded) {
        return nullptr;
    }
    if (a.t) {
        return a.t;
    }
    ggml_init_params p = { 2 * ggml_tensor_overhead(), nullptr, true };
    a.ctx              = ggml_init(p);
    if (!a.ctx) {
        return nullptr;
    }
    a.t = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, a.hidden, a.k);
    ggml_set_name(a.t, "artist_token");
    a.buf = ggml_backend_alloc_ctx_tensors(a.ctx, backend);
    if (!a.buf) {
        ggml_free(a.ctx);
        a.ctx = nullptr;
        a.t   = nullptr;
        return nullptr;
    }
    // vec is [k][hidden] row-major, which is exactly [hidden, k] in ggml.
    ggml_backend_tensor_set(a.t, a.vec.data(), 0, a.vec.size() * sizeof(float));
    return a.t;
}

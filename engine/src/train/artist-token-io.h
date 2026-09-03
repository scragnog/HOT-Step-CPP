#pragma once
// artist-token-io.h — the on-disk format for a learned artist token
// (textual inversion), shared by every trainer that can produce one.
//
// ONE writer, deliberately. The LM trainer and the DiT trainer learn vectors in
// different spaces, and the format's whole job is to make loading the wrong one
// impossible — a guarantee that evaporates the moment two copies of this code
// drift apart. Same reason prompt.h grew lm_append_user_span() rather than
// letting the training and sampling prompts be built by two similar functions.
//
//   artist_token.safetensors   artist_token.vec  [k, H]  torch row-major
//   artist_token.json          name, k, hidden, site, base model, placeholder
//
// `site` is load-bearing, not bookkeeping:
//
//   as15_lm    [H=lm hidden] a DELTA on `placeholder`'s embedding, spliced into
//              the caption's token span. Meaningless against a different
//              placeholder id or a different vocabulary.
//   as15_dit   [H=enc_H] extra rows of the DiT's encoder sequence, read through
//              the frozen cond_emb. No placeholder (the field is -1).
//
// The two are the same shape only by coincidence of dimension. A loader must
// refuse a site it did not ask for rather than apply the vectors anyway.

#include "train/lm-common.h"      // lm_join, lm_json_escape
#include "train/preprocess-io.h"  // pm_mkdir_p, pm_write_atomic
#include "train/st-write.h"
#include "version.h"

#include <string>
#include <vector>

struct ArtistTokenMeta {
    std::string name;
    std::string site;         // "as15_lm" | "as15_dit"
    std::string base_model;
    int         k           = 0;
    int         placeholder = -1;  // as15_lm only; -1 when the site has none
};

static bool artist_token_write(ggml_tensor * t, const ArtistTokenMeta & m, const std::string & out_dir,
                               std::string * err) {
    if (!t) {
        *err = "no artist-token tensor to export";
        return false;
    }
    if (!pm_mkdir_p(out_dir)) {
        *err = "cannot create " + out_dir;
        return false;
    }

    std::vector<float> buf((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));

    STWTensor st;
    st.name  = "artist_token.vec";
    st.shape = { t->ne[1], t->ne[0] };  // torch [k, H]
    st.data  = buf.data();

    std::vector<STWTensor>                           tensors{ st };
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "producer", std::string("ace-train ") + ACE_VERSION });
    md.push_back({ "hot_step_artist_token", "v1" });

    const std::string sf = lm_join(out_dir, "artist_token.safetensors");
    if (!st_write_file(sf.c_str(), tensors, md, STW_F32)) {
        *err = "cannot write " + sf;
        return false;
    }

    std::string j;
    char        b[128];
    j += "{\n";
    j += "  \"name\": \"" + lm_json_escape(m.name) + "\",\n";
    snprintf(b, sizeof(b), "  \"k\": %d,\n", m.k);
    j += b;
    snprintf(b, sizeof(b), "  \"placeholder\": %d,\n", m.placeholder);
    j += b;
    snprintf(b, sizeof(b), "  \"hidden\": %d,\n", (int) t->ne[0]);
    j += b;
    j += "  \"site\": \"" + m.site + "\",\n";
    j += "  \"base_model\": \"" + lm_json_escape(m.base_model) + "\"\n";
    j += "}\n";
    if (!pm_write_atomic(lm_join(out_dir, "artist_token.json"), j)) {
        *err = "cannot write artist_token.json in " + out_dir;
        return false;
    }
    return true;
}

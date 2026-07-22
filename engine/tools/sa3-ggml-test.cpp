// sa3-ggml-test.cpp: parity test CLI for the StableStep GGML SA3 modules.
//
// Runs GGML ports of the SA3 conditioning modules against golden vectors
// dumped from the validated ONNX graphs (see tools/onnx-export/), and reports
// cosine similarity + max abs diff per component.
//
// Usage:
//   sa3-ggml-test --models <dir-with-sa3-*.gguf> --goldens <dir-with-manifest.json>
//                 [--component text_enc|seconds|all]
//
// Components:
//   text_enc: sa3-text-enc-BF16.gguf, T5Gemma encoder + learned padding
//             substitution. Inputs input_ids [1,S] i64 + attention_mask [1,S]
//             u8, expected embeddings [1,S,768] f32.
//   seconds:  sa3-dit-BF16.gguf (embedder tensors only), NumberConditioner
//             expo Fourier embedder. Input [1] f32, expected [1,768] f32.
//
// Exit code 0 only if every run component passes cosine > 0.999.

#include "sa3-t5gemma-enc.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double PASS_COSINE = 0.999;

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[Test] cannot open %s\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) sz);
    size_t rd = fread(out.data(), 1, (size_t) sz, f);
    fclose(f);
    if (rd != (size_t) sz) {
        fprintf(stderr, "[Test] short read on %s\n", path.c_str());
        return false;
    }
    return true;
}

struct Metrics {
    double cosine;
    double max_abs_diff;
};

static Metrics compare(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0, mad = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
        double d = fabs((double) a[i] - b[i]);
        if (d > mad) {
            mad = d;
        }
    }
    Metrics m;
    m.cosine       = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    m.max_abs_diff = mad;
    return m;
}

// Manifest helpers: get golden["<component>"]["inputs"/"outputs"]["<name>"]["file"]
static std::string manifest_file(yyjson_val * root, const char * comp, const char * io, const char * name) {
    yyjson_val * c = yyjson_obj_get(root, comp);
    yyjson_val * g = c ? yyjson_obj_get(c, io) : NULL;
    yyjson_val * t = g ? yyjson_obj_get(g, name) : NULL;
    yyjson_val * f = t ? yyjson_obj_get(t, "file") : NULL;
    return (f && yyjson_is_str(f)) ? yyjson_get_str(f) : "";
}

static int64_t manifest_shape_prod(yyjson_val * root, const char * comp, const char * io, const char * name) {
    yyjson_val * c = yyjson_obj_get(root, comp);
    yyjson_val * g = c ? yyjson_obj_get(c, io) : NULL;
    yyjson_val * t = g ? yyjson_obj_get(g, name) : NULL;
    yyjson_val * s = t ? yyjson_obj_get(t, "shape") : NULL;
    if (!s || !yyjson_is_arr(s)) {
        return 0;
    }
    int64_t      prod = 1;
    size_t       idx, max;
    yyjson_val * d;
    yyjson_arr_foreach(s, idx, max, d) {
        prod *= yyjson_get_int(d);
    }
    return prod;
}

static bool run_text_enc(const std::string & models, const std::string & goldens, yyjson_val * root, Metrics * out) {
    std::string ids_f  = manifest_file(root, "text_enc", "inputs", "input_ids");
    std::string mask_f = manifest_file(root, "text_enc", "inputs", "attention_mask");
    std::string exp_f  = manifest_file(root, "text_enc", "outputs", "embeddings");
    if (ids_f.empty() || mask_f.empty() || exp_f.empty()) {
        fprintf(stderr, "[Test] text_enc: manifest missing entries\n");
        return false;
    }
    int64_t S       = manifest_shape_prod(root, "text_enc", "inputs", "input_ids");
    int64_t n_out   = manifest_shape_prod(root, "text_enc", "outputs", "embeddings");
    if (S <= 0 || n_out <= 0 || n_out % S != 0) {
        fprintf(stderr, "[Test] text_enc: bad shapes in manifest\n");
        return false;
    }
    int64_t H = n_out / S;

    std::vector<uint8_t> ids_raw, mask_raw, exp_raw;
    if (!read_file(goldens + "/" + ids_f, ids_raw) || !read_file(goldens + "/" + mask_f, mask_raw) ||
        !read_file(goldens + "/" + exp_f, exp_raw)) {
        return false;
    }
    if (ids_raw.size() != (size_t) S * 8 || mask_raw.size() != (size_t) S || exp_raw.size() != (size_t) n_out * 4) {
        fprintf(stderr, "[Test] text_enc: golden file sizes do not match manifest shapes\n");
        return false;
    }

    std::vector<int32_t> ids((size_t) S);
    const int64_t *      ids64 = (const int64_t *) ids_raw.data();
    for (int64_t i = 0; i < S; i++) {
        ids[(size_t) i] = (int32_t) ids64[i];
    }

    SA3T5GemmaEnc enc = {};
    if (!sa3_t5gemma_load(&enc, (models + "/sa3-text-enc-BF16.gguf").c_str())) {
        return false;
    }
    if ((int64_t) enc.cfg.hidden_size != H) {
        fprintf(stderr, "[Test] text_enc: model H=%d but golden H=%lld\n", enc.cfg.hidden_size, (long long) H);
        sa3_t5gemma_free(&enc);
        return false;
    }
    const char * env_layers = getenv("SA3_T5G_LAYERS");
    if (env_layers) {
        enc.debug_n_layers = atoi(env_layers);
        fprintf(stderr, "[Test] text_enc: DEBUG truncated to %d layers\n", enc.debug_n_layers);
    }

    std::vector<float> got((size_t) n_out);
    sa3_t5gemma_forward(&enc, ids.data(), mask_raw.data(), (int) S, got.data());
    sa3_t5gemma_free(&enc);

    *out = compare(got.data(), (const float *) exp_raw.data(), (size_t) n_out);
    return true;
}

static bool run_seconds(const std::string & models, const std::string & goldens, yyjson_val * root, Metrics * out) {
    std::string in_f  = manifest_file(root, "seconds", "inputs", "seconds");
    std::string exp_f = manifest_file(root, "seconds", "outputs", "embed");
    if (in_f.empty() || exp_f.empty()) {
        fprintf(stderr, "[Test] seconds: manifest missing entries\n");
        return false;
    }
    std::vector<uint8_t> in_raw, exp_raw;
    if (!read_file(goldens + "/" + in_f, in_raw) || !read_file(goldens + "/" + exp_f, exp_raw)) {
        return false;
    }
    if (in_raw.size() != 4) {
        fprintf(stderr, "[Test] seconds: bad input size\n");
        return false;
    }
    float   seconds = *(const float *) in_raw.data();
    int64_t n_out   = manifest_shape_prod(root, "seconds", "outputs", "embed");
    if (n_out <= 0 || exp_raw.size() != (size_t) n_out * 4) {
        fprintf(stderr, "[Test] seconds: golden size mismatch\n");
        return false;
    }

    SA3SecondsEmbedder emb;
    if (!sa3_seconds_embedder_load(&emb, (models + "/sa3-dit-BF16.gguf").c_str())) {
        return false;
    }
    if ((int64_t) emb.out_dim != n_out) {
        fprintf(stderr, "[Test] seconds: model out=%d but golden %lld\n", emb.out_dim, (long long) n_out);
        return false;
    }
    std::vector<float> got((size_t) n_out);
    sa3_seconds_embed(emb, seconds, got.data());

    *out = compare(got.data(), (const float *) exp_raw.data(), (size_t) n_out);
    return true;
}

int main(int argc, char ** argv) {
    std::string models, goldens, component = "all";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models = argv[++i];
        } else if (!strcmp(argv[i], "--goldens") && i + 1 < argc) {
            goldens = argv[++i];
        } else if (!strcmp(argv[i], "--component") && i + 1 < argc) {
            component = argv[++i];
        } else {
            fprintf(stderr,
                    "Usage: sa3-ggml-test --models <dir> --goldens <dir> [--component text_enc|seconds|all]\n");
            return 2;
        }
    }
    if (models.empty() || goldens.empty()) {
        fprintf(stderr, "Usage: sa3-ggml-test --models <dir> --goldens <dir> [--component text_enc|seconds|all]\n");
        return 2;
    }

    std::vector<uint8_t> manifest_raw;
    if (!read_file(goldens + "/manifest.json", manifest_raw)) {
        return 2;
    }
    yyjson_doc * doc = yyjson_read((const char *) manifest_raw.data(), manifest_raw.size(), 0);
    if (!doc) {
        fprintf(stderr, "[Test] cannot parse manifest.json\n");
        return 2;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);

    bool all_pass = true;
    bool any_run  = false;

    if (component == "all" || component == "text_enc") {
        Metrics m;
        any_run = true;
        if (run_text_enc(models, goldens, root, &m)) {
            bool pass = m.cosine > PASS_COSINE;
            printf("text_enc: cosine=%.6f max_abs_diff=%.6f  %s\n", m.cosine, m.max_abs_diff,
                   pass ? "PASS" : "FAIL");
            all_pass = all_pass && pass;
        } else {
            printf("text_enc: ERROR\n");
            all_pass = false;
        }
    }
    if (component == "all" || component == "seconds") {
        Metrics m;
        any_run = true;
        if (run_seconds(models, goldens, root, &m)) {
            bool pass = m.cosine > PASS_COSINE;
            printf("seconds:  cosine=%.6f max_abs_diff=%.6f  %s\n", m.cosine, m.max_abs_diff,
                   pass ? "PASS" : "FAIL");
            all_pass = all_pass && pass;
        } else {
            printf("seconds:  ERROR\n");
            all_pass = false;
        }
    }
    yyjson_doc_free(doc);

    if (!any_run) {
        fprintf(stderr, "[Test] unknown component '%s'\n", component.c_str());
        return 2;
    }
    return all_pass ? 0 : 1;
}

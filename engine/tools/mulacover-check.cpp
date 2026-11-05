// mulacover-check: Stage 3a receipts for the MuLaCover port.
//
//   mulacover-check --models DIR --qwen QWEN --captures DIR
//
// 1. Qwen style vector (qwen3-enc.h qwen3_style_vector) against every capture case
//    under --captures (<case>/input.json + <case>/observed/*.npy): token ids and mask
//    exact, the pooled last row within the BF16 rule (|d| <= 0.0625 + 0.02 |ref|),
//    the normalised vector within 1e-3 absolute. QWEN is a .gguf or a safetensors dir.
// 2. The converted pair in DIR (heartmula/ or the root, as heartmula_discover scans):
//    both files compatible in the catalogue, then heartmula_load_lm loads the LM and
//    its cover modules.
// 3. heartmula_cover_load refuses a missing, misshaped and wrongly typed tensor with
//    a message naming it (three tiny GGUFs written to DIR and deleted).
// Exit 0 when everything passes.
#include "heartmula/pipeline.h"
#include "qwen3-enc.h"

#include <fstream>
#include <sstream>

// ── .npy fixtures (as heartmula-probe.cpp) ──────────────────────────────────

struct Npy {
    std::vector<int64_t> shape;
    std::vector<double>  v;
};

static Npy read_npy(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic + 1, 5) != "NUMPY") throw std::runtime_error(path + ": not .npy");
    uint32_t hlen = 0;
    if (magic[6] == 1) { uint16_t h; f.read((char *) &h, 2); hlen = h; } else { f.read((char *) &hlen, 4); }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    const auto field = [&](const char * key) {
        const size_t at = header.find(key);
        if (at == std::string::npos) throw std::runtime_error(path + ": header lacks " + key);
        return at + strlen(key);
    };
    const size_t      d     = header.find('\'', field("'descr':")) + 1;
    const std::string descr = header.substr(d, header.find('\'', d) - d);
    const size_t      fo    = field("'fortran_order':");
    if (header.find("True", fo) < header.find(',', fo)) throw std::runtime_error(path + ": Fortran order");
    Npy          out;
    const size_t sp = header.find('(', field("'shape':")) + 1, se = header.find(')', sp);
    int64_t      n  = 1;
    for (std::string dims = header.substr(sp, se - sp); !dims.empty();) {
        const size_t      c   = dims.find(',');
        const std::string tok = dims.substr(0, c);
        if (tok.find_first_not_of(' ') != std::string::npos) { out.shape.push_back(std::stoll(tok)); n *= out.shape.back(); }
        dims = c == std::string::npos ? "" : dims.substr(c + 1);
    }
    out.v.resize((size_t) n);
    const auto read_all = [&](auto tag) {
        std::vector<decltype(tag)> raw((size_t) n);
        f.read((char *) raw.data(), (std::streamsize) (raw.size() * sizeof(tag)));
        for (int64_t i = 0; i < n; i++) out.v[(size_t) i] = (double) raw[(size_t) i];
    };
    if (descr == "<f4") read_all(float{});
    else if (descr == "<i8") read_all(int64_t{});
    else if (descr == "|b1") read_all(uint8_t{});
    else throw std::runtime_error(path + ": unsupported dtype " + descr);
    if (!f) throw std::runtime_error(path + ": truncated");
    return out;
}

static int g_fail = 0;

static void verdict(bool ok, const std::string & what) {
    fprintf(stderr, "%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    g_fail += !ok;
}

// ── 1. Qwen style vector ────────────────────────────────────────────────────

static std::string read_tags(const std::filesystem::path & input) {
    std::ifstream     f(input, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    yyjson_doc *      doc  = yyjson_read(text.data(), text.size(), 0);
    yyjson_val *      tags = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "tags") : nullptr;
    std::string       out  = yyjson_is_str(tags) ? std::string(yyjson_get_str(tags), yyjson_get_len(tags)) : "";
    yyjson_doc_free(doc);
    return out;
}

static void check_qwen(const std::string & qwen_path, const HeartMuLaTokenizer & hm_tok, const std::string & captures) {
    namespace fs = std::filesystem;
    Qwen3GGML    qm  = {};
    BPETokenizer bpe = {};
    const bool   dir = fs::is_directory(qwen_path);
    const bool   tok = dir ? load_bpe_from_files(&bpe, (qwen_path + "/vocab.json").c_str(),
                                                 (qwen_path + "/merges.txt").c_str())
                           : load_bpe_from_gguf(&bpe, qwen_path.c_str());
    if (!tok || !qwen3_load_text_encoder(&qm, qwen_path.c_str())) {
        verdict(false, "qwen: cannot load " + qwen_path);
        return;
    }
    const int H = qm.cfg.hidden_size;
    std::vector<fs::path> cases;
    for (const auto & e : fs::directory_iterator(captures)) {
        if (fs::exists(e.path() / "observed" / "qwen.input_ids.npy")) cases.push_back(e.path());
    }
    std::sort(cases.begin(), cases.end());
    verdict(cases.size() >= 2, "qwen: " + std::to_string(cases.size()) + " capture cases under " + captures);
    for (const auto & c : cases) {
        const fs::path    o    = c / "observed";
        const std::string name = "qwen[" + fs::path(qwen_path).filename().string() + "] " + c.filename().string();
        // Upstream pipeline: tags.strip().lower(), then _encode_style.
        std::string tags = read_tags(c / "input.json");
        tags.erase(0, tags.find_first_not_of(" \t\r\n\v\f"));
        tags.erase(tags.find_last_not_of(" \t\r\n\v\f") + 1);
        tags = heartmula_tokenizer_detail::lowercase(hm_tok, tags);

        std::vector<float>     vec((size_t) H);
        const std::vector<int> ids = qwen3_style_vector(&qm, &bpe, tags, vec.data());
        const Npy ref_ids = read_npy((o / "qwen.input_ids.npy").string());
        const Npy ref_mask = read_npy((o / "qwen.attention_mask.npy").string());
        bool ids_ok = ref_ids.v.size() == ids.size() && ref_mask.v.size() == ids.size();
        for (size_t i = 0; ids_ok && i < ids.size(); i++) ids_ok = ref_ids.v[i] == ids[i] && ref_mask.v[i] == 1.0;
        verdict(ids_ok, name + " input_ids + attention_mask exact (" + std::to_string(ids.size()) + " tokens)");

        // The pooled row before the norm, from the same forward.
        const int          S = (int) ids.size();
        std::vector<float> hidden((size_t) H * S);
        qwen3_forward(&qm, ids.data(), S, hidden.data());
        const Npy ref_row = read_npy((o / "qwen.last-row.npy").string());
        double    max_abs = 0, max_rel = 0;
        bool      row_ok  = ref_row.v.size() == (size_t) H;
        for (int i = 0; row_ok && i < H; i++) {
            const double r = ref_row.v[(size_t) i], d = std::fabs(hidden[(size_t) H * (S - 1) + i] - r);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(std::fabs(r), 1e-30));
            row_ok  = row_ok && d <= 0.0625 + 0.02 * std::fabs(r);
        }
        char buf[160];
        snprintf(buf, sizeof(buf), " last-row BF16 rule: max abs %.5f, max rel %.5f", max_abs, max_rel);
        verdict(row_ok, name + buf);

        const Npy ref_vec = read_npy((o / "input.qwen_embedding.npy").string());
        bool      vec_ok  = !ref_vec.v.empty() && ref_vec.v.size() % (size_t) H == 0;
        double    vec_abs = 0;
        for (size_t i = 0; vec_ok && i < ref_vec.v.size(); i++) {
            vec_abs = std::max(vec_abs, std::fabs(vec[i % (size_t) H] - ref_vec.v[i]));
        }
        vec_ok = vec_ok && vec_abs < 1e-3;
        snprintf(buf, sizeof(buf), " qwen_embedding (%zu rows) max abs %.2e < 1e-3", ref_vec.v.size() / (size_t) H,
                 vec_abs);
        verdict(vec_ok, name + buf);
    }
    std::vector<float> zero((size_t) H, 1.0f);
    verdict(qwen3_style_vector(&qm, &bpe, "", zero.data()).empty() &&
                std::all_of(zero.begin(), zero.end(), [](float x) { return x == 0.0f; }),
            "qwen: empty tags give a zero vector");
    qwen3_free(&qm);
}

// ── 2 and 3. Catalogue, load, refusals ──────────────────────────────────────

// A tensor-only GGUF carrying the loaded global's identity, so heartmula_cover_load
// gets past the header and fingerprint checks to the tensor check under test.
static std::string write_probe(const std::string & dir, const char * tag, const HeartMuLaModel & m, const char * name,
                               ggml_type type, int64_t ne0, int64_t ne1) {
    const std::string path = dir + "/mulacover-check-" + tag + ".gguf";
    ggml_init_params  ip   = { 1 << 20, nullptr, false };
    ggml_context *    ctx  = ggml_init(ip);
    ggml_tensor *     t    = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, name);
    memset(t->data, 0, ggml_nbytes(t));
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "heartmula-global");
    gguf_set_val_u32(g, "heartmula.format_version", HEARTMULA_FORMAT_VERSION);
    gguf_set_val_str(g, "heartmula.base_fingerprint", m.base_fingerprint.c_str());
    gguf_set_val_str(g, "heartmula.variant", m.variant.c_str());
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static void check_pack(const std::string & models) {
    const HeartMuLaCatalogue cat = heartmula_discover(models);
    HeartMuLaResolved        p;
    for (const auto & f : cat.files) {
        if (heartmula_pipeline_detail::lower(f.variant) != "mulacover") continue;
        verdict(f.compatible, "catalogue " + f.filename + " (" + k_heartmula_roles[f.role] + ", " +
                                  std::to_string(f.gpu_bytes >> 20) + " MiB)" + (f.compatible ? "" : ": " + f.error));
        if (f.role == HM_GLOBAL || f.role == HM_LOCAL) p.file[f.role] = f;
    }
    if (p.file[HM_GLOBAL].path.empty() || p.file[HM_LOCAL].path.empty()) {
        verdict(false, "catalogue: no MuLaCover global + local under " + models);
        return;
    }
    HeartMuLaRuntime rt;
    std::string      err;
    const auto       t0 = std::chrono::steady_clock::now();
    const bool       ok = heartmula_load_lm(&rt, p, &err);
    const double     ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    char             buf[200];
    snprintf(buf, sizeof(buf), "heartmula_load_lm: variant %s, cover modules %s (%zu layers), %.0f MiB resident, %.0f ms",
             rt.lm.variant.c_str(), rt.cover.loaded ? "loaded" : "absent", rt.cover.layers.size(),
             rt.lm_bytes / 1048576.0, ms);
    verdict(ok && rt.cover.loaded && rt.cover.layers.size() == HEARTMULA_COVER_LAYERS, ok ? buf : "heartmula_load_lm: " + err);
    if (!ok) return;

    const std::string dir  = std::filesystem::path(p.file[HM_GLOBAL].path).parent_path().string();
    const char *      want = "peft_layers.20.adaptor.pianoroll_proj.weight";
    struct Case {
        const char * tag, * name, * expect;
        ggml_type    type;
        int64_t      ne0;
    };
    for (const Case & c : { Case{ "missing", "unrelated.weight", "is missing", GGML_TYPE_F32, 256 },
                            Case{ "shape", want, "has shape [255, 24], expected [256, 24]", GGML_TYPE_F32, 255 },
                            Case{ "type", want, "has unsupported type i32", GGML_TYPE_I32, 256 } }) {
        const std::string path = write_probe(dir, c.tag, rt.lm, c.name, c.type, c.ne0, 24);
        HeartMuLaCover    cover;
        std::string       e;
        const bool        refused = !heartmula_cover_load(&cover, rt.lm, path.c_str(), &e);
        const bool named = e.find(std::string("tensor '") + want + "'") != std::string::npos &&
                           e.find(c.expect) != std::string::npos;
        verdict(refused && named, std::string("refuses ") + c.tag + ": " +
                                      (named ? e.substr(e.find("tensor '")) : e));
        std::filesystem::remove(path);
    }
    heartmula_free_all(&rt);
}

int main(int argc, char ** argv) {
    std::string models, qwen, captures;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--models") models = argv[i + 1];
        else if (k == "--qwen") qwen = argv[i + 1];
        else if (k == "--captures") captures = argv[i + 1];
    }
    if (models.empty() || qwen.empty() || captures.empty()) {
        fprintf(stderr, "usage: mulacover-check --models DIR --qwen QWEN(.gguf|dir) --captures DIR\n");
        return 2;
    }
    try {
        HeartMuLaTokenizer hm_tok;
        std::string        err;
        const std::string  tok_path = models + "/heartmula/heartmula-tokenizer.gguf";
        if (!heartmula_tokenizer_load(&hm_tok, tok_path.c_str(), &err)) {
            verdict(false, "tokenizer: " + err);
        } else {
            check_qwen(qwen, hm_tok, captures);
        }
        check_pack(models);
    } catch (const std::exception & e) {
        verdict(false, std::string("exception: ") + e.what());
    }
    fprintf(stderr, "MULACOVER CHECK: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

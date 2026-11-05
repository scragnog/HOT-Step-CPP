// mulacover-check: Stage 3a and 3b receipts for the MuLaCover port.
//
//   mulacover-check --models DIR --qwen QWEN --captures DIR [--only qwen,pack,parity] [--fp32 ROOT] [--table OUT.md]
//
// 1. Qwen style vector (qwen3-enc.h qwen3_style_vector) against every capture case
//    under --captures (<case>/input.json + <case>/observed/*.npy): token ids and mask
//    exact, the pooled last row within the BF16 rule (|d| <= 0.0625 + 0.02 |ref|),
//    the normalised vector within 1e-3 absolute; empty tags give zeros and the
//    8192-token limit holds at its boundary. QWEN is a .gguf or a safetensors dir.
// 2. The converted pair in DIR (heartmula/ or the root, as heartmula_discover scans):
//    both files compatible in the catalogue, then heartmula_load_lm loads the LM and
//    its cover modules.
// 3. heartmula_cover_load refuses a missing, misshaped and wrongly typed tensor with
//    a message naming it (three tiny GGUFs written to DIR and deleted).
// 4. Stage 3b parity (section 4 below): prompt, rolls, CFG rows, context, GACA at
//    layers 20 and 27, prefill and one cached decode step, both local decoders, against
//    every key of every case's observed/tensors.json; --table writes the full table.
// Exit 0 when everything passes.
#include "heartmula/mulacover.h"
#include "heartmula/pipeline.h"
#include "qwen3-enc.h"

#include <array>
#include <fstream>
#include <map>
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
    else if (descr == "<i4") read_all(int32_t{});
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

        std::vector<float> vec((size_t) H);
        std::vector<int>   ids;
        std::string        err;
        if (!qwen3_style_vector(&qm, &bpe, tags, vec.data(), &ids, &err) || ids.empty()) {
            verdict(false, name + ": " + err);
            continue;
        }
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
    std::vector<float> v((size_t) H, 1.0f);
    std::vector<int>   ids;
    std::string        err;
    verdict(qwen3_style_vector(&qm, &bpe, "", v.data(), &ids, &err) && ids.empty() &&
                std::all_of(v.begin(), v.end(), [](float x) { return x == 0.0f; }),
            "qwen: empty tags give a zero vector");

    // Upstream's limit: "a" + n x " a" is n + 1 word tokens, plus the trailing EOS.
    for (const int n : { QWEN3_STYLE_MAX_TOKENS, QWEN3_STYLE_MAX_TOKENS + 1 }) {
        std::string tags = "a";
        for (int i = 0; i < n - 2; i++) tags += " a";
        std::fill(v.begin(), v.end(), 1.0f);
        err.clear();
        const bool ran  = qwen3_style_vector(&qm, &bpe, tags, v.data(), &ids, &err);
        double     norm = 0;
        for (float x : v) norm += (double) x * x;
        const bool want = n <= QWEN3_STYLE_MAX_TOKENS;
        verdict(bpe_encode(&bpe, tags, true).size() == (size_t) n && ran == want &&
                    (want ? ids.size() == (size_t) n && std::fabs(norm - 1.0) < 1e-5
                          : ids.empty() && norm == 0.0 && !err.empty()),
                "qwen: " + std::to_string(n) + " tokens " + (ran ? "run, unit norm" : "refused: " + err));
    }
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

// The MuLaCover global + local in the catalogue, or false.
static bool find_pack(const std::string & models, HeartMuLaResolved * p, bool report) {
    const HeartMuLaCatalogue cat = heartmula_discover(models);
    for (const auto & f : cat.files) {
        if (heartmula_pipeline_detail::lower(f.variant) != "mulacover") continue;
        if (report) {
            verdict(f.compatible, "catalogue " + f.filename + " (" + k_heartmula_roles[f.role] + ", " +
                                      std::to_string(f.gpu_bytes >> 20) + " MiB)" + (f.compatible ? "" : ": " + f.error));
        }
        if (f.role == HM_GLOBAL || f.role == HM_LOCAL) p->file[f.role] = f;
    }
    if (p->file[HM_GLOBAL].path.empty() || p->file[HM_LOCAL].path.empty()) {
        verdict(false, "catalogue: no MuLaCover global + local under " + models);
        return false;
    }
    return true;
}

static void check_pack(const std::string & models) {
    HeartMuLaResolved p;
    if (!find_pack(models, &p, true)) return;
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

// ── 4. Stage 3b: conditioning, GACA and prefill/cache parity ────────────────
//
// Per case: the prompt and the rolls are rebuilt from input.json (pipeline.py
// _build_prompt, symbolic.py to_tensors, plus the 4 padding rows capture.py appends),
// the golden input.qwen_embedding is injected as the style vector (Lead's ruling on
// HOTSTEP-980: the 2c Qwen ran in BF16), and the golden codes are fed back (teacher
// forcing) through the prefill, frame 0's local decoder, one cached global step and
// frame 1's local decoder. Every key of observed/tensors.json is then compared; a key
// the engine does not produce fails. Rules: discrete and roll values exact; torch
// bfloat16 tensors |d| <= 0.0625 + 0.02 |ref|; torch float32 tensors |d| < 1e-3;
// sampled codes cannot reproduce torch's Philox draws, so a code passes when the fed
// golden code equals the reference and lies inside the engine's own top-k (it could
// have been drawn); the Qwen last row and vector are informational here (3a's checks).
//
// The 2c run used BF16 autocast, and torch itself in FP32 misses those float rules by
// the margins the engine does (Lead's tolerance amendment on HOTSTEP-980). Each case's
// <case>/fp32/observed/ holds the same run in torch FP32 (fp32_ref.py beside capture.py;
// --fp32 ROOT reads ROOT/<case>/fp32 instead), and a case without it fails. Every float
// key also gets heartmula-probe's drift rule: engine vs torch FP32
// within 2x torch's own FP32-vs-BF16 drift, in relative L2 and in max abs (floor 1e-3).
// Logits keys also take that precedent's distribution checks: an argmax that differs
// from torch FP32 must be a near-tie (FP32 top-2 margin within 2x drift max abs), and
// per case the engine's cross-entropy gap to FP32 at the fed codes must not exceed
// torch BF16's (sum, and max within 2x), nor its argmax flips torch BF16's.
// For float keys the drift and distribution checks are the verdict and need a valid
// FP32 tensor (missing or misshaped fails); the declared rule is reported, never gated.
// A manifest that cannot be read, is empty, or lacks a key the run produces fails.

using Vals = std::map<std::string, std::vector<double>>;

struct CaseRun {
    Vals                        v;
    std::map<std::string, int> rank;  // codes: rank of the fed code in the engine's guided logits
    int                         topk = MULACOVER_DEFAULT_TOPK;
};

static yyjson_doc * read_json(const std::filesystem::path & p) {
    std::ifstream     f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    return yyjson_read(text.data(), text.size(), 0);
}

static std::string json_str(yyjson_val * root, const char * key) {
    yyjson_val * v = yyjson_obj_get(root, key);
    return yyjson_is_str(v) ? std::string(yyjson_get_str(v), yyjson_get_len(v)) : "";
}

static std::vector<std::array<int64_t, 3>> json_triplets(yyjson_val * root, const char * key) {
    std::vector<std::array<int64_t, 3>> out;
    size_t                              i, n;
    yyjson_val *                        row;
    yyjson_arr_foreach(yyjson_obj_get(root, key), i, n, row) {
        out.push_back({ yyjson_get_sint(yyjson_arr_get(row, 0)), yyjson_get_sint(yyjson_arr_get(row, 1)),
                        yyjson_get_sint(yyjson_arr_get(row, 2)) });
    }
    return out;
}

// symbolic.py _validate_triplets + to_tensors, then capture.py's 4 padding rows.
static HeartMuLaCoverInput build_rolls(yyjson_val * root) {
    auto mel = json_triplets(root, "melody"), cho = json_triplets(root, "chords"), dru = json_triplets(root, "drums");
    std::vector<std::array<int64_t, 3>> chords;
    for (auto c : cho) {
        if (c[2] > 0) chords.push_back({ c[0], c[1] % 12, c[2] });
    }
    for (auto * v : { &mel, &dru }) {
        for (auto & n : *v) n[2] = std::max<int64_t>(n[2], 1);
    }
    int64_t len = 0;
    for (const auto * v : { &mel, &dru, &chords }) {
        for (const auto & n : *v) len = std::max(len, n[0] + n[2]);
    }
    const int64_t       S = len + 4;
    HeartMuLaCoverInput in;
    in.steps = S;
    in.melody.assign((size_t) S * 256, 0.0f);
    in.drums.assign((size_t) S * 128, 0.0f);
    in.chords.assign((size_t) S * 12, 0.0f);
    in.pad.assign((size_t) S, 0);
    for (int64_t t = len; t < S; t++) in.pad[(size_t) t] = 1;
    for (const auto & n : mel) {
        in.melody[(size_t) (n[0] * 256 + n[1] * 2)] = 1.0f;
        for (int64_t t = n[0] + 1; t < n[0] + n[2]; t++) in.melody[(size_t) (t * 256 + n[1] * 2 + 1)] = 1.0f;
    }
    for (const auto & n : dru) {
        for (int64_t t = n[0]; t < n[0] + n[2]; t++) in.drums[(size_t) (t * 128 + n[1])] = 1.0f;
    }
    for (const auto & n : chords) {
        for (int64_t t = n[0]; t < n[0] + n[2]; t++) in.chords[(size_t) (t * 12 + n[1])] = 1.0f;
    }
    return in;
}

template <typename T> static std::vector<double> rep(const std::vector<T> & x, int B) {
    std::vector<double> out;
    for (int b = 0; b < B; b++) out.insert(out.end(), x.begin(), x.end());
    return out;
}

template <typename T> static std::vector<double> dbl(const std::vector<T> & x) { return { x.begin(), x.end() }; }

static int g_cpu_nodes = 0;

static void placement(const char * name, const HeartMuLaSlot & s, const BackendPair & bp) {
    int       cpu = 0;
    const int n   = ggml_graph_n_nodes(s.graph);
    for (int i = 0; i < n; i++) {
        cpu += bp.cpu_backend != bp.backend &&
               ggml_backend_sched_get_tensor_backend(s.sched, ggml_graph_node(s.graph, i)) == bp.cpu_backend;
    }
    fprintf(stderr, "placement %s: %d nodes, %d splits, %d on CPU (%s)\n", name, n,
            ggml_backend_sched_get_n_splits(s.sched), cpu, ggml_backend_name(bp.backend));
    g_cpu_nodes += cpu;
}

// One teacher-forced run of a case. `style` is the injected style vector.
static bool run_case(HeartMuLaRuntime & rt, const HeartMuLaTokenizer & tok, const std::filesystem::path & dir,
                     const std::vector<float> & style, bool report_placement, CaseRun * out, std::string * err) {
    yyjson_doc * doc = read_json(dir / "input.json");
    if (!doc) {
        *err = "cannot read input.json";
        return false;
    }
    yyjson_val *          root   = yyjson_doc_get_root(doc);
    const float           cfg    = (float) yyjson_get_num(yyjson_obj_get(root, "cfg"));
    const MuLaCoverPrompt prompt = mulacover_prompt(tok, json_str(root, "tags"), json_str(root, "lyrics"));
    HeartMuLaCoverInput   in     = build_rolls(root);
    in.style                     = style;
    yyjson_doc_free(doc);

    const int     B = cfg != 1.0f ? 2 : 1, V = HEARTMULA_AUDIO_VOCAB, K = HEARTMULA_CODEBOOKS;
    const int64_t T = (int64_t) prompt.ids.size(), S = in.steps, H = rt.lm.global.cfg.n_embd;
    Vals &        v = out->v;

    // Prompt (pipeline.py preprocess), as the engine feeds heartmula_lm_step.
    std::vector<HeartMuLaPosition> pp((size_t) T);
    std::vector<int64_t>           pos((size_t) T);
    std::vector<int32_t>           tokens, tmask;
    for (int64_t i = 0; i < T; i++) {
        pp[(size_t) i].id[HEARTMULA_COLUMNS - 1]   = prompt.ids[(size_t) i];
        pp[(size_t) i].mask[HEARTMULA_COLUMNS - 1] = true;
        pos[(size_t) i]                            = i;
        for (int c = 0; c < HEARTMULA_COLUMNS; c++) {
            tokens.push_back(pp[(size_t) i].id[c]);
            tmask.push_back(pp[(size_t) i].mask[c]);
        }
    }
    v["input.tokens"]       = rep(tokens, B);
    v["input.tokens_mask"]  = rep(tmask, B);
    v["input.input_pos"]    = rep(pos, B);
    v["input.qwen_indices"] = std::vector<double>((size_t) B, (double) prompt.qwen_idx);
    v["input.muq_indices"]  = std::vector<double>((size_t) B, (double) prompt.muq_idx);
    v["input.qwen_embedding"] = rep(style, B);
    v["input.muq_embedding"]  = std::vector<double>((size_t) B * 512, 0.0);  // muq_linear(0): heartmula_lm_step's MuQ gate
    v["input.pianoroll"]      = rep(in.melody, B);
    v["input.drum_pianoroll"] = rep(in.drums, B);
    v["input.chord"]          = rep(in.chords, B);
    v["input.context_mask"]   = rep(in.pad, B);
    std::vector<int64_t> kpos((size_t) S);
    for (int64_t j = 0; j < S; j++) kpos[(size_t) j] = j;
    for (const char * s : { "s0", "s1" }) {
        const std::string p = s;
        v[p + ".cfg.melody"] = dbl(heartmula_cover_rows(in.melody, B));
        v[p + ".cfg.drums"]  = dbl(heartmula_cover_rows(in.drums, B));
        v[p + ".cfg.chords"] = dbl(heartmula_cover_rows(in.chords, B));
        v[p + ".cfg.mask"]   = rep(in.pad, B);
        for (const char * l : { "20", "27" }) {
            v[p + ".layer" + l + ".mask"]          = rep(in.pad, B);
            v[p + ".layer" + l + ".key-positions"] = dbl(kpos);
        }
    }

    const Npy frames = read_npy((dir / "observed" / "frames.npy").string());  // [2][B][8]
    if (frames.v.size() != (size_t) (2 * B * K)) {
        *err = "frames.npy shape";
        return false;
    }
    HeartMuLaLmGraph    g;
    HeartMuLaLocalGraph lg;
    auto                done = [&](bool ok) {
        heartmula_local_free(&lg);
        heartmula_lm_free(&g);
        return ok;
    };
    if (!heartmula_lm_reserve(rt.lm, &g, B, T + 2, err) || !heartmula_local_prepare(rt.lm, &lg, B, err)) return done(false);
    g.taps = true;
    std::vector<std::vector<float>> contexts;
    if (!heartmula_cover_bind(rt.lm, rt.cover, &g, in, prompt.qwen_idx, &contexts, err)) return done(false);
    for (const char * s : { "s0", "s1" }) {
        v[std::string(s) + ".layer20.context"] = dbl(contexts[0]);
        v[std::string(s) + ".layer27.context"] = dbl(contexts[HEARTMULA_COVER_LAYERS - 1]);
    }

    std::vector<float> hidden((size_t) (B * H)), c0((size_t) (B * V)), tap;
    std::vector<int32_t> fed;
    for (int step = 0; step < 2; step++) {
        const std::string p = "s" + std::to_string(step);
        int64_t           at = T;
        HeartMuLaPosition frame;
        bool              ok;
        if (step == 0) {
            ok = heartmula_lm_step(rt.lm, &g, pp.data(), pos.data(), T, prompt.muq_idx, hidden.data(), c0.data(), err);
            v[p + ".layer20.query-positions"] = rep(pos, B);
            v[p + ".layer27.query-positions"] = rep(pos, B);
        } else {
            for (int k = 0; k < K; k++) {
                frame.id[k]   = (int32_t) frames.v[(size_t) k];
                frame.mask[k] = true;
            }
            ok = heartmula_lm_step(rt.lm, &g, &frame, &at, 1, -1, hidden.data(), c0.data(), err);
            v[p + ".layer20.query-positions"] = std::vector<double>((size_t) B, (double) at);
            v[p + ".layer27.query-positions"] = std::vector<double>((size_t) B, (double) at);
        }
        if (!ok) return done(false);
        if (report_placement) placement(step == 0 ? "prefill" : "decode", step == 0 ? g.prefill : g.decode, rt.lm.bp);
        for (const char * t : { "assembled", "pre20", "post20", "pre27", "post27" }) {
            if (!heartmula_lm_tap(g, step == 0, t, &tap)) {
                *err = std::string("no tap ") + t;
                return done(false);
            }
            const std::string n = t;
            const std::string key = n == "assembled" ? p + ".assembled-hidden"
                                                     : p + ".layer" + n.substr(n.size() - 2) + (n[1] == 'r' ? ".pre-gaca" : ".post-gaca");
            v[key] = dbl(tap);
            if (step == 0 && n == "assembled") {
                // qwen.projected (both rows project the same vector): the conditioned row's style slot.
                std::vector<float> row(tap.begin() + prompt.qwen_idx * H, tap.begin() + (prompt.qwen_idx + 1) * H);
                v["qwen.projected"] = rep(row, B);
            }
        }
        const HeartMuLaChooser teacher = [&](int k, const float * logits, int rows) {
            const std::string  q = p + ".codebook" + std::to_string(k);
            const int32_t      code = (int32_t) frames.v[(size_t) (step * B * K + k)];
            std::vector<float> guided(logits, logits + V);
            if (rows == 2) heartmula_cfg_blend(logits, logits + V, V, cfg, guided.data());
            v[q + ".logits"]        = dbl(std::vector<float>(logits, logits + (size_t) rows * V));
            v[q + ".guided-logits"] = dbl(guided);
            v[q + ".codes"]         = std::vector<double>((size_t) rows, (double) code);
            out->rank[q + ".codes"] = (int) std::count_if(guided.begin(), guided.end(), [&](float x) { return x > guided[(size_t) code]; });
            fed.push_back(code);
            return code;
        };
        int32_t codes[HEARTMULA_CODEBOOKS];
        if (!heartmula_local_frame(rt.lm, &lg, hidden.data(), c0.data(), teacher, codes, nullptr, err)) return done(false);
    }
    std::vector<double> fr;
    for (int s = 0; s < 2; s++) {
        for (int b = 0; b < B; b++) fr.insert(fr.end(), fed.begin() + s * K, fed.begin() + (s + 1) * K);
    }
    v["frames"] = fr;
    return done(true);
}

struct Err {
    double max_abs = 0, max_rel = 0, rel_l2 = 0;
};

static Err err_of(const std::vector<double> & got, const std::vector<double> & ref) {
    Err    e;
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double d = std::fabs(got[i] - ref[i]), a = std::fabs(ref[i]);
        e.max_abs      = std::max(e.max_abs, d);
        if (a > 0) e.max_rel = std::max(e.max_rel, d / a);
        num += d * d;
        den += a * a;
    }
    e.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    return e;
}

struct Row {
    std::string key, rule;
    size_t      n = 0;
    Err         golden, fp32, drift;  // engine vs golden, engine vs torch FP32, torch FP32 vs golden
    bool        has_fp32 = false, declared = false, drift_ok = false, ok = false;
    std::string note;
};

static std::string rule_for(const std::string & k, const std::string & dtype) {
    if (k == "qwen.last-row" || k == "input.qwen_embedding") return "info";
    if (k.find(".codes") != std::string::npos || k == "frames") return "teacher";
    if (dtype.find("int") != std::string::npos || dtype.find("bool") != std::string::npos) return "exact";
    if (k.rfind("input.", 0) == 0 || k.find(".cfg.") != std::string::npos) return "exact";
    return dtype == "torch.bfloat16" ? "bf16" : "fp32";
}

static double cross_entropy(const double * x, int64_t n, int64_t target) {
    const double top = *std::max_element(x, x + n);
    double       sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::exp(x[i] - top);
    return std::log(sum) + top - x[target];
}

// heartmula-probe's distribution checks, accumulated over a case's logits keys.
struct Dist {
    int    flips_ours = 0, flips_torch = 0, rows = 0;
    double ce_ours = 0, ce_torch = 0, ce_ours_max = 0, ce_torch_max = 0;
};

// Per row: engine and torch BF16 argmax against torch FP32's, the near-tie test and the
// cross-entropy gaps at the fed code. Returns whether every engine flip is a near-tie.
static bool logits_dist(const std::vector<double> & got, const std::vector<double> & bf16, const std::vector<double> & f32,
                        int64_t target, double drift_max, Dist * d, std::string * note) {
    const int64_t V    = HEARTMULA_AUDIO_VOCAB;
    bool          ok   = true;
    double        gap  = 0, tgap = 0;
    int           flip = 0;
    for (size_t r = 0; r * V < got.size(); r++) {
        const double * w = f32.data() + r * V;
        const int64_t  a = std::max_element(w, w + V) - w;
        double         m2 = -INFINITY;
        for (int64_t i = 0; i < V; i++) {
            if (i != a) m2 = std::max(m2, w[i]);
        }
        const bool fo = std::max_element(got.data() + r * V, got.data() + (r + 1) * V) - (got.data() + r * V) != a;
        const bool ft = std::max_element(bf16.data() + r * V, bf16.data() + (r + 1) * V) - (bf16.data() + r * V) != a;
        ok            = ok && (!fo || w[a] - m2 <= 2.0 * drift_max);
        flip += fo;
        d->flips_ours += fo;
        d->flips_torch += ft;
        d->rows++;
        const double ce = cross_entropy(w, V, target);
        gap             = std::max(gap, std::fabs(cross_entropy(got.data() + r * V, V, target) - ce));
        tgap            = std::max(tgap, std::fabs(cross_entropy(bf16.data() + r * V, V, target) - ce));
    }
    d->ce_ours += gap;
    d->ce_torch += tgap;
    d->ce_ours_max  = std::max(d->ce_ours_max, gap);
    d->ce_torch_max = std::max(d->ce_torch_max, tgap);
    char buf[120];
    snprintf(buf, sizeof(buf), "dCE %.4f (torch BF16 %.4f)%s", gap, tgap, flip ? (ok ? ", argmax near-tie" : ", ARGMAX FLIP") : "");
    *note = buf;
    return ok;
}

#define MULACOVER_CAPTURE_KEYS 98  // keys per case in the 2c captures

static Row manifest_fail(const std::string & key, const std::string & why) {
    Row r;
    r.key  = key;
    r.rule = "manifest";
    r.note = why;
    return r;
}

// Every key in observed/tensors.json against the run. Returns the rows. `fp32` is the
// case's torch FP32 folder, or empty; `dist` collects the logits distribution checks.
static std::vector<Row> compare_case(const std::filesystem::path & dir, const CaseRun & run, const Vals & extra,
                                     const std::filesystem::path & fp32 = {}, Dist * dist = nullptr) {
    std::vector<Row> rows;
    Dist             local;
    if (!dist) dist = &local;
    yyjson_doc * doc  = read_json(dir / "observed" / "tensors.json");
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!yyjson_is_obj(root) || yyjson_obj_size(root) < MULACOVER_CAPTURE_KEYS) {
        rows.push_back(manifest_fail("tensors.json", !root ? "manifest unreadable"
                                                           : !yyjson_is_obj(root) ? "manifest is not an object"
                                                           : "manifest has " + std::to_string(yyjson_obj_size(root)) +
                                                                 " keys; the 2c captures have " +
                                                                 std::to_string(MULACOVER_CAPTURE_KEYS)));
        yyjson_doc_free(doc);
        return rows;
    }
    // Coverage both ways: a key the engine produces but the manifest lacks fails too.
    for (const Vals * vals : { &run.v, &extra }) {
        for (const auto & kv : *vals) {
            if (!yyjson_obj_get(root, kv.first.c_str())) rows.push_back(manifest_fail(kv.first, "absent from the manifest"));
        }
    }
    yyjson_obj_iter it;
    yyjson_obj_iter_init(root, &it);
    for (yyjson_val * key; (key = yyjson_obj_iter_next(&it));) {
        Row r;
        r.key  = yyjson_get_str(key);
        r.rule = rule_for(r.key, json_str(yyjson_obj_iter_get_val(key), "dtype"));
        const Npy                   ref = read_npy((dir / "observed" / (r.key + ".npy")).string());
        const std::vector<double> * hit = nullptr;
        if (run.v.count(r.key)) hit = &run.v.at(r.key);
        else if (extra.count(r.key)) hit = &extra.at(r.key);
        if (!hit) {
            r.note = "missing from the engine run";
            rows.push_back(r);
            continue;
        }
        const std::vector<double> & got = *hit;
        if (got.size() != ref.v.size()) {
            r.note = "size " + std::to_string(got.size()) + " vs " + std::to_string(ref.v.size());
            rows.push_back(r);
            continue;
        }
        r.n       = got.size();
        r.golden  = err_of(got, ref.v);
        bool rule = true;
        for (size_t i = 0; i < got.size(); i++) {
            const double d = std::fabs(got[i] - ref.v[i]), a = std::fabs(ref.v[i]);
            if (r.rule == "exact" || r.rule == "teacher") rule = rule && d == 0;
            else if (r.rule == "bf16") rule = rule && d <= 0.0625 + 0.02 * a;
            else if (r.rule == "fp32") rule = rule && d < 1e-3;
        }
        r.declared = rule;
        const std::filesystem::path fp = fp32 / "observed" / (r.key + ".npy");
        if (r.rule == "bf16" || r.rule == "fp32") {
            rule = false;  // Lead's amendment: the FP32 drift result alone decides
            const Npy f = !fp32.empty() && std::filesystem::exists(fp) ? read_npy(fp.string()) : Npy{};
            if (f.v.size() != got.size()) {
                r.note = f.v.empty() ? "no torch FP32 tensor" : "torch FP32 size " + std::to_string(f.v.size());
            } else {
                r.has_fp32 = true;
                r.fp32     = err_of(got, f.v);
                r.drift    = err_of(f.v, ref.v);
                r.drift_ok = r.fp32.rel_l2 <= 2 * r.drift.rel_l2 && r.fp32.max_abs <= 2 * std::max(r.drift.max_abs, 1e-3);
                const size_t lg = r.key.rfind(".logits") != std::string::npos ? r.key.rfind(".logits")
                                                                              : r.key.rfind(".guided-logits");
                if (lg != std::string::npos) {
                    const Npy code = read_npy((dir / "observed" / (r.key.substr(0, lg) + ".codes.npy")).string());
                    r.drift_ok     = logits_dist(got, ref.v, f.v, (int64_t) code.v.at(0), r.drift.max_abs, dist, &r.note) &&
                                 r.drift_ok;
                }
                rule = r.drift_ok;
            }
        }
        if (r.rule == "teacher" && r.key != "frames") {
            const int rank = run.rank.at(r.key);
            r.note         = "rank " + std::to_string(rank) + " of top-" + std::to_string(run.topk);
            rule           = rule && rank < run.topk;
        }
        r.ok = r.rule == "info" || rule;
        rows.push_back(r);
    }
    yyjson_doc_free(doc);
    return rows;
}

// A 5000-step lead sheet binds and runs a prefill and a decode step; 5001 steps and an
// all-padding sheet are refused with an error. Reports the bind time and cache size.
static void check_cover_limits(HeartMuLaRuntime & rt) {
    for (const int64_t S : { (int64_t) HEARTMULA_COVER_MAX_STEPS, (int64_t) HEARTMULA_COVER_MAX_STEPS + 1, (int64_t) 0 }) {
        const bool          all_pad = S == 0;
        HeartMuLaCoverInput in;
        in.steps = all_pad ? 16 : S;
        in.melody.assign((size_t) in.steps * 256, 0.0f);
        in.drums.assign((size_t) in.steps * 128, 0.0f);
        in.chords.assign((size_t) in.steps * 12, 0.0f);
        in.pad.assign((size_t) in.steps, all_pad ? 1 : 0);
        in.style.assign(HEARTMULA_COVER_STYLE_DIM, 1.0f / 32.0f);
        for (int64_t t = 0; t < in.steps; t++) {
            in.melody[(size_t) (t * 256 + 2 * (60 + t % 12) + (t % 4 != 0))] = 1.0f;
            in.chords[(size_t) (t * 12 + (t / 16) % 12)]                     = 1.0f;
        }
        HeartMuLaLmGraph               g;
        std::vector<HeartMuLaPosition> pp(8);
        std::vector<int64_t>           pos(8);
        for (int i = 0; i < 8; i++) {
            pp[(size_t) i].id[HEARTMULA_COLUMNS - 1]   = i ? 1000 + i : MULACOVER_TAG_START;
            pp[(size_t) i].mask[HEARTMULA_COLUMNS - 1] = true;
            pos[(size_t) i]                            = i;
        }
        std::vector<float> hidden((size_t) 2 * rt.lm.global.cfg.n_embd), c0((size_t) 2 * HEARTMULA_AUDIO_VOCAB);
        std::string        err;
        const auto         t0  = std::chrono::steady_clock::now();
        bool               ok  = heartmula_lm_reserve(rt.lm, &g, 2, 9, &err) &&
                  heartmula_cover_bind(rt.lm, rt.cover, &g, in, 2, nullptr, &err);
        const double       ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double       mib = g.cover_buf ? ggml_backend_buffer_get_size(g.cover_buf) / 1048576.0 : 0.0;
        const int64_t      at  = 8;
        HeartMuLaPosition  frame;
        for (int k = 0; k < HEARTMULA_CODEBOOKS; k++) frame.mask[k] = true;
        ok = ok && heartmula_lm_step(rt.lm, &g, pp.data(), pos.data(), 8, -1, hidden.data(), c0.data(), &err) &&
             heartmula_lm_step(rt.lm, &g, &frame, &at, 1, -1, hidden.data(), c0.data(), &err) &&
             std::all_of(c0.begin(), c0.end(), [](float x) { return std::isfinite(x); });
        heartmula_lm_free(&g);
        char buf[240];
        if (S == HEARTMULA_COVER_MAX_STEPS) {
            snprintf(buf, sizeof(buf), "limits: %lld steps bind (%.0f ms incl. reserve, cache %.0f MiB), prefill + decode finite",
                     (long long) S, ms, mib);
            verdict(ok, ok ? buf : "limits: " + std::to_string((long long) S) + " steps: " + err);
        } else {
            verdict(!ok && !err.empty(), std::string("limits: ") + (all_pad ? "all-padding sheet" : "5001 steps") +
                                             " refused: " + err);
        }
    }
}

static void check_parity(HeartMuLaRuntime & rt, const HeartMuLaTokenizer & tok, const std::string & qwen_path,
                         const std::string & captures, const std::string & fp32_dir, const std::string & table) {
    namespace fs = std::filesystem;
    std::vector<fs::path> cases;
    for (const auto & e : fs::directory_iterator(captures)) {
        if (fs::exists(e.path() / "observed" / "tensors.json")) cases.push_back(e.path());
    }
    std::sort(cases.begin(), cases.end());
    verdict(cases.size() == 5, "parity: " + std::to_string(cases.size()) + " capture cases");

    // 3a's Qwen keys from the engine encoder, for coverage (informational against the BF16 captures).
    Qwen3GGML    qm  = {};
    BPETokenizer bpe = {};
    const bool   qdir = fs::is_directory(qwen_path);
    const bool   qok  = (qdir ? load_bpe_from_files(&bpe, (qwen_path + "/vocab.json").c_str(), (qwen_path + "/merges.txt").c_str())
                              : load_bpe_from_gguf(&bpe, qwen_path.c_str())) &&
                     qwen3_load_text_encoder(&qm, qwen_path.c_str());
    verdict(qok, "parity: Qwen encoder " + qwen_path);
    if (!qok) return;

    std::string md =
        "| case | key | rule | n | vs golden: max abs | rel L2 | declared | vs torch FP32: max abs | rel L2 | "
        "torch FP32 vs golden: max abs | rel L2 | drift rule | verdict | note |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    auto line = [](const std::string & cname, const Row & r) {
        char buf[600];
        if (r.has_fp32) {
            snprintf(buf, sizeof(buf), "| %s | %s | %s | %zu | %.3g | %.3g | %s | %.3g | %.3g | %.3g | %.3g | %s | %s | %s |\n",
                     cname.c_str(), r.key.c_str(), r.rule.c_str(), r.n, r.golden.max_abs, r.golden.rel_l2,
                     r.declared ? "pass" : "fail", r.fp32.max_abs, r.fp32.rel_l2, r.drift.max_abs, r.drift.rel_l2,
                     r.drift_ok ? "pass" : "fail", r.ok ? "PASS" : "FAIL", r.note.c_str());
        } else {
            snprintf(buf, sizeof(buf), "| %s | %s | %s | %zu | %.3g | %.3g | %s | | | | | | %s | %s |\n", cname.c_str(),
                     r.key.c_str(), r.rule.c_str(), r.n, r.golden.max_abs, r.golden.rel_l2, r.declared ? "pass" : "fail",
                     r.ok ? "PASS" : "FAIL", r.note.c_str());
        }
        return std::string(buf);
    };
    int         compared = 0;
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const fs::path &  c    = cases[ci];
        const std::string name = c.filename().string();
        const Npy         emb  = read_npy((c / "observed" / "input.qwen_embedding.npy").string());
        const std::vector<float> golden(emb.v.begin(), emb.v.begin() + HEARTMULA_COVER_STYLE_DIM);

        // Engine Qwen: ids, the raw last row and the vector, from the prompt's tags.
        yyjson_doc *          doc    = read_json(c / "input.json");
        const MuLaCoverPrompt prompt = mulacover_prompt(tok, json_str(yyjson_doc_get_root(doc), "tags"), "");
        yyjson_doc_free(doc);
        std::vector<float> vec(HEARTMULA_COVER_STYLE_DIM);
        std::vector<int>   ids;
        std::string        err;
        Vals               extra;
        if (qwen3_style_vector(&qm, &bpe, prompt.tags, vec.data(), &ids, &err)) {
            std::vector<float> hid((size_t) HEARTMULA_COVER_STYLE_DIM * ids.size());
            qwen3_forward(&qm, ids.data(), (int) ids.size(), hid.data());
            extra["qwen.input_ids"]      = dbl(ids);
            extra["qwen.attention_mask"] = std::vector<double>(ids.size(), 1.0);
            extra["qwen.last-row"]       = dbl(std::vector<float>(hid.end() - HEARTMULA_COVER_STYLE_DIM, hid.end()));
        }

        CaseRun    run;
        const bool ok = run_case(rt, tok, c, golden, ci == 0, &run, &err);
        verdict(ok, "parity " + name + ": teacher-forced run" + (ok ? "" : ": " + err));
        if (!ok) continue;
        int        fails = 0, n = 0, declared_fails = 0, drift_rows = 0;
        const auto fp    = (fp32_dir.empty() ? c : fs::path(fp32_dir) / name) / "fp32";
        if (!fs::exists(fp / "observed" / "tensors.json")) {
            verdict(false, "parity " + name + ": no torch FP32 reference at " + fp.string() + " (fp32_ref.py)");
            continue;
        }
        Dist       dist;
        for (const Row & r : compare_case(c, run, extra, fp, &dist)) {
            n++;
            fails += !r.ok;
            declared_fails += !r.declared && r.rule != "info";
            drift_rows += r.has_fp32;
            md += line(name, r);
            if (!r.ok) {
                fprintf(stderr, "  FAIL %s %s (%s): vs golden %.4g / rel L2 %.3g; vs FP32 %.4g / %.3g; drift %.4g / %.3g %s\n",
                        name.c_str(), r.key.c_str(), r.rule.c_str(), r.golden.max_abs, r.golden.rel_l2, r.fp32.max_abs,
                        r.fp32.rel_l2, r.drift.max_abs, r.drift.rel_l2, r.note.c_str());
            }
        }
        compared += n;
        if (dist.rows) {
            const bool dok = dist.flips_ours <= dist.flips_torch && dist.ce_ours <= dist.ce_torch &&
                            dist.ce_ours_max <= 2.0 * dist.ce_torch_max;
            char buf[300];
            snprintf(buf, sizeof(buf),
                     "parity %s: logits vs torch FP32 over %d rows: argmax flips engine %d, torch BF16 %d; dCE sum %.4f "
                     "vs %.4f, max %.4f vs %.4f",
                     name.c_str(), dist.rows, dist.flips_ours, dist.flips_torch, dist.ce_ours, dist.ce_torch,
                     dist.ce_ours_max, dist.ce_torch_max);
            verdict(dok, buf);
            md += std::string("| ") + name + " | logits distribution | drift | " + std::to_string(dist.rows) +
                  " rows | | | | | | | | " + (dok ? "pass" : "fail") + " | " + (dok ? "PASS" : "FAIL") + " | " +
                  (buf + name.size() + 9) + " |\n";
        } else {
            verdict(false, "parity " + name + ": no logits checked against torch FP32");
        }
        verdict(fails == 0, "parity " + name + ": " + std::to_string(n) + " keys compared, " + std::to_string(fails) +
                                " failed (" + std::to_string(declared_fails) + " miss the declared rule, " +
                                std::to_string(drift_rows) + " checked against torch FP32)");

        if (name == "synthetic-drums-cfg1.5") {
            // Informational (Lead's amendment): the same run with the engine's own Qwen vector.
            CaseRun er;
            if (run_case(rt, tok, c, vec, false, &er, &err)) {
                for (Row r : compare_case(c, er, extra, fp)) {
                    if (r.rule == "bf16" || r.rule == "fp32" || r.key == "input.qwen_embedding") {
                        r.rule = "info";
                        md += line(name + " (engine Qwen)", r);
                    }
                }
            }
        }
    }
    verdict(g_cpu_nodes == 0, "parity: prefill and decode graphs have " + std::to_string(g_cpu_nodes) + " CPU nodes");
    check_cover_limits(rt);
    fprintf(stderr, "parity: %d keys compared across %zu cases\n", compared, cases.size());
    qwen3_free(&qm);
    if (!table.empty()) {
        std::ofstream(table, std::ios::binary) << md;
        fprintf(stderr, "parity table: %s\n", table.c_str());
    }
}

int main(int argc, char ** argv) {
    std::string models, qwen, captures, only = "qwen,pack,parity", table, fp32;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--models") models = argv[i + 1];
        else if (k == "--qwen") qwen = argv[i + 1];
        else if (k == "--captures") captures = argv[i + 1];
        else if (k == "--only") only = argv[i + 1];
        else if (k == "--table") table = argv[i + 1];
        else if (k == "--fp32") fp32 = argv[i + 1];
    }
    const auto run = [&](const char * part) { return only.find(part) != std::string::npos; };
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
        } else if (run("qwen")) {
            check_qwen(qwen, hm_tok, captures);
        }
        if (run("pack")) check_pack(models);
        HeartMuLaResolved p;
        if (run("parity") && hm_tok.loaded && find_pack(models, &p, false)) {
            HeartMuLaRuntime rt;
            if (heartmula_load_lm(&rt, p, &err) && rt.cover.loaded) {
                check_parity(rt, hm_tok, qwen, captures, fp32, table);
            } else {
                verdict(false, "parity: heartmula_load_lm: " + err);
            }
            heartmula_free_all(&rt);
        }
    } catch (const std::exception & e) {
        verdict(false, std::string("exception: ") + e.what());
    }
    fprintf(stderr, "MULACOVER CHECK: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

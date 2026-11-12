#pragma once
// heartmula/server.h: the /heartmula/* routes (engine/docs/HEARTMULA_CONTRACT.md).
//
// HOT-Step file. Included mid-file in hot-step-server.cpp, after yue2/yue2-server.h,
// because it needs the job system (Job, job_create, job_set_phase, work_push,
// on_worker), g_store and g_keep_loaded, and reclaims idle MM3 and YuE2 weights
// through their own unload handlers. The call site is
// heartmula_register_routes(svr, models_dir), next to yue2_register_routes.
//
// Threads. Generation, warm and unload run on the shared GPU worker, so HeartMuLa
// serialises with every other family; g_hm_rt is touched only there. The HTTP
// threads read the catalogue, the pick and a residency snapshot under g_hm.mtx;
// the worker publishes that snapshot at every stage change.
//
// Covers. POST /heartmula/cover is /heartmula/synth plus `rolls` (pipeline.h note F),
// on the MuLaCover variant, which plain synth and select-model refuse. Before the LM
// phase the worker runs the tags through the BF16 Qwen3 embedding (the ACE text
// encoder file; Q8_0 misses the FP32 reference by 6e-3) and frees it again.

#include "heartmula/pipeline.h"
#include "heartmula/codec-plugins.h"
#include "bpe.h"
#include "qwen3-enc.h"

#include "audio-io.h"
#include "httplib.h"
#include "yyjson.h"

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>

// Detail for GET /heartmula/job. Guarded by its own mutex.
struct HeartMuLaJobInfo {
    std::mutex  mtx;
    std::string stage = "queued", end_reason, error_code, error;
    int         prompt_tokens = 0, prefill_done = 0, frames_done = 0, max_frames = 0;
    int         codec_chunk = 0, codec_chunks = 0, codec_step = 0, codec_steps = 0;
    uint64_t    seed = 0, noise_seed = 0;
    HeartMuLaPick pick;
    std::string pick_types[4];  // resolved global, local, flow, scalar decoder
    std::string variant;
    std::optional<HeartMuLaAdmission> admission;
    double      t_admission = 0, t_load = 0, t_prefill = 0, t_frames = 0, t_codec = 0, t_encode = 0;
};

struct HeartMuLaServer {
    std::mutex         mtx;
    std::string        models_dir;
    HeartMuLaCatalogue cat;
    HeartMuLaPick      requested;
    std::shared_ptr<const HeartMuLaTokenizer> tok;
    std::string        tok_path;
    bool               resident[HM_TOKENIZER + 1] = {};
    std::string        adapter_loaded;
    ggml_backend_dev_t device = nullptr;  // published by the worker; registry-owned
    uint64_t           resident_bytes = 0;
    std::map<std::string, std::shared_ptr<HeartMuLaJobInfo>> jobs;
    std::deque<std::string>                                  job_order;
};

static HeartMuLaServer  g_hm;
static HeartMuLaRuntime g_hm_rt;            // GPU worker thread only
static std::atomic<int> g_hm_active{ 0 };   // jobs loading or running HeartMuLa weights

// ── JSON helpers ────────────────────────────────────────────────────────────

static std::string hm_write(yyjson_mut_doc * doc) {
    char *      j = yyjson_mut_write(doc, 0, NULL);
    std::string s = j ? j : "{}";
    free(j);
    yyjson_mut_doc_free(doc);
    return s;
}

static void hm_add_admission(yyjson_mut_doc * doc, yyjson_mut_val * obj, const HeartMuLaAdmission & a) {
    yyjson_mut_val * o = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, obj, "admission", o);
    yyjson_mut_obj_add_strcpy(doc, o, "phase", a.phase.c_str());
    yyjson_mut_obj_add_uint(doc, o, "need_bytes", a.need);
    yyjson_mut_obj_add_uint(doc, o, "free_bytes", a.free);
    yyjson_mut_obj_add_uint(doc, o, "reclaimable_bytes", a.reclaimable);
    yyjson_mut_obj_add_uint(doc, o, "reserve_bytes", a.reserve);
    yyjson_mut_obj_add_uint(doc, o, "weights_bytes", a.weights);
    yyjson_mut_obj_add_uint(doc, o, "kv_bytes", a.kv);
    yyjson_mut_obj_add_int(doc, o, "positions", a.positions);
    yyjson_mut_obj_add_bool(doc, o, "cfg", a.cfg);
}

static void hm_error(httplib::Response & res, int status, const char * code, const std::string & msg,
                     const HeartMuLaAdmission * a = nullptr) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "error", msg.c_str());
    yyjson_mut_obj_add_str(doc, root, "code", code);
    if (a) hm_add_admission(doc, root, *a);
    res.status = status;
    res.set_content(hm_write(doc), "application/json");
}

static const char * hm_status_code(int status) {
    return status == 404 ? "not_found" : status == 409 ? "incompatible" : status == 501 ? "not_implemented" : "bad_request";
}

// ── State helpers ───────────────────────────────────────────────────────────

// Caller holds g_hm.mtx. Loads (or swaps) the host-side tokenizer for `path`.
static std::shared_ptr<const HeartMuLaTokenizer> hm_tokenizer_locked(const std::string & path, std::string * err) {
    if (g_hm.tok && g_hm.tok_path == path) return g_hm.tok;
    auto t = std::make_shared<HeartMuLaTokenizer>();
    if (!heartmula_tokenizer_load(t.get(), path.c_str(), err)) return nullptr;
    g_hm.tok      = t;
    g_hm.tok_path = path;
    return t;
}

// Worker thread: copy the runtime's residency into the snapshot props reads.
static void hm_publish() {
    std::lock_guard<std::mutex> lock(g_hm.mtx);
    const bool lm = g_hm_rt.lm.loaded, codec = !g_hm_rt.codec_key.empty();
    g_hm.resident[HM_GLOBAL] = g_hm.resident[HM_LOCAL] = lm;
    g_hm.resident[HM_FLOW] = g_hm.resident[HM_SCALAR] = g_hm.resident[HM_RVQ] = codec;
    g_hm.resident[HM_TOKENIZER] = g_hm.tok != nullptr;
    g_hm.resident_bytes         = heartmula_resident_bytes(g_hm_rt);
    g_hm.adapter_loaded         = g_hm_rt.adapter.loaded() ? g_hm_rt.adapter.path : "";
    if (g_hm_rt.bp_ref) g_hm.device = ggml_backend_get_device(g_hm_rt.bp.backend);
}

// Idle VRAM other backends hold, and its eviction (contract "Residency and VRAM
// admission"). Runs on the worker, so no ACE job is in flight; MM3 and YuE2 go
// through their own unload handlers, which drain YuE2's render lane first.
static uint64_t hm_reclaimable() {
    uint64_t b = g_store ? store_vram_bytes(g_store) : 0;
    {
        std::lock_guard<std::mutex> lock(g_mm3_mutex);
        b += mm3_vram_bytes(g_mm3) + g_mm3_lm.kv_bytes;
    }
    {
        std::lock_guard<std::mutex> lock(g_yue2_mutex);
        b += yue2_vram_bytes(g_yue2);
    }
    return b;
}

static void hm_reclaim() {
    int still = 0;
    if (g_store) store_evict_all(g_store, &still);
    httplib::Request  mreq, yreq;
    httplib::Response mres, yres;
    mreq.body = "{\"keep_ar_cache\":true}";  // VRAM only; the host-side AR slot stays
    mm3_handle_unload(mreq, mres);
    yue2_handle_unload(yreq, yres);
    fprintf(stderr, "[HeartMuLa] reclaimed idle VRAM from other backends (%d store module(s) still referenced)\n", still);
}

static HeartMuLaHooks hm_reclaim_hooks() {
    HeartMuLaHooks h;
    h.reclaimable = hm_reclaimable;
    h.reclaim     = hm_reclaim;
    return h;
}

// Caller holds g_hm.mtx.
static bool hm_resolve_locked(const HeartMuLaPick & pick, HeartMuLaResolved * out, int * status, std::string * err) {
    return heartmula_resolve(g_hm.cat, pick, out, status, err);
}

static bool hm_synth_ready_locked(HeartMuLaResolved * out, std::string * err) {
    HeartMuLaPick p = g_hm.requested;
    p.adapter.clear();  // an adapter problem is a pick error, not missing weights
    int status = 0;
    return hm_resolve_locked(p, out, &status, err);
}

#define HM_COVER_VARIANT "MuLaCover"
#define HM_COVER_QWEN    "Qwen3-Embedding-0.6B-BF16.gguf"  // registry entry embedding-bf16

static bool hm_is_cover(const std::string & variant) {
    return heartmula_pipeline_detail::lower(variant) == "mulacover";
}

// A cover's pick: MuLaCover's own global/local (the current pick's types belong to
// another variant), the current codec choice, no adapter; then the request's fields.
static HeartMuLaPick hm_cover_pick(const HeartMuLaRequest & r) {
    HeartMuLaPick base = g_hm.requested;
    base.global.clear();
    base.local.clear();
    base.adapter.clear();
    HeartMuLaPick p = heartmula_request_pick(base, r);
    p.variant       = HM_COVER_VARIANT;
    return p;
}

// Caller holds g_hm.mtx. Missing-model contract: false with *status 503 and code
// model_missing when the MuLaCover pair or the BF16 Qwen file is not installed, or the
// resolver's 404/409 for a bad pick.
static bool hm_cover_ready_locked(const HeartMuLaRequest & r, HeartMuLaResolved * out, int * status, const char ** code,
                                  std::string * err) {
    const bool installed = std::any_of(g_hm.cat.files.begin(), g_hm.cat.files.end(),
                                       [](const HeartMuLaFile & f) { return f.role == HM_GLOBAL && hm_is_cover(f.variant); });
    std::error_code ec;
    const bool      qwen = std::filesystem::exists(std::filesystem::path(g_hm.models_dir) / HM_COVER_QWEN, ec);
    if (!installed || !qwen) {
        *status = 503;
        *code   = "model_missing";
        *err    = std::string("HeartMuLa covers need ") +
               (!installed ? "the MuLaCover pack (heartmula-mulacover-global and -local GGUFs in the heartmula models folder)" : "") +
               (!installed && !qwen ? " and " : "") +
               (!qwen ? "the Qwen3 Embedding BF16 text encoder (" HM_COVER_QWEN "); the Q8_0 one is not precise enough" : "");
        return false;
    }
    if (!hm_resolve_locked(hm_cover_pick(r), out, status, err)) {
        *code = hm_status_code(*status);
        return false;
    }
    return true;
}

static void hm_add_pick(yyjson_mut_doc * doc, yyjson_mut_val * root, const char * key, const std::string & variant,
                        const std::string * types) {
    yyjson_mut_val * o = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, key, o);
    yyjson_mut_obj_add_strcpy(doc, o, "variant", variant.c_str());
    const char * keys[4] = { "global", "local", "flow", "scalar_decoder" };
    for (int i = 0; i < 4; i++) yyjson_mut_obj_add_strcpy(doc, o, keys[i], types[i].c_str());
}

// Caller holds g_hm.mtx. The selected, requested and adapter blocks.
static void hm_add_pick_blocks(yyjson_mut_doc * doc, yyjson_mut_val * root) {
    HeartMuLaResolved r;
    std::string       err;
    const bool        ok = hm_synth_ready_locked(&r, &err);
    const std::string sel[4] = { ok ? r.file[HM_GLOBAL].type : "", ok ? r.file[HM_LOCAL].type : "",
                                 ok ? r.file[HM_FLOW].type : "", ok ? r.file[HM_SCALAR].type : "" };
    hm_add_pick(doc, root, "selected", ok ? r.variant : "", sel);
    const HeartMuLaPick & q = g_hm.requested;
    const std::string req[4] = { q.global, q.local, q.flow, q.scalar };
    hm_add_pick(doc, root, "requested", q.variant, req);
    yyjson_mut_val * a = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "adapter", a);
    yyjson_mut_obj_add_strcpy(doc, a, "requested", q.adapter.c_str());
    yyjson_mut_obj_add_strcpy(doc, a, "loaded", g_hm.adapter_loaded.c_str());
    yyjson_mut_obj_add_real(doc, a, "scale", q.adapter_scale);
    yyjson_mut_obj_add_bool(doc, a, "in_force", !g_hm.adapter_loaded.empty() && g_hm.adapter_loaded == q.adapter &&
                                                    q.adapter_scale != 0.0f);
}

// ── Routes ──────────────────────────────────────────────────────────────────

static void heartmula_handle_props(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_hm.mtx);
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    HeartMuLaResolved r;
    std::string       err;
    const bool        ready = hm_synth_ready_locked(&r, &err);
    yyjson_mut_obj_add_str(doc, root, "backend", "heartmula");
    yyjson_mut_obj_add_bool(doc, root, "available", !g_hm.cat.files.empty());
    yyjson_mut_obj_add_bool(doc, root, "synth_ready", ready);
    yyjson_mut_obj_add_int(doc, root, "max_duration_s", HEARTMULA_MAX_DURATION_S);
    yyjson_mut_obj_add_int(doc, root, "context_length", heartmula_global_config().n_ctx);
    {
        HeartMuLaResolved cr;
        int               cs = 0;
        const char *      cc = "";
        std::string       ce;
        const bool        cover_ready = hm_cover_ready_locked(HeartMuLaRequest{}, &cr, &cs, &cc, &ce);
        yyjson_mut_obj_add_bool(doc, root, "cover_ready", cover_ready);
        yyjson_mut_obj_add_strcpy(doc, root, "cover_error", cover_ready ? "" : ce.c_str());
        yyjson_mut_obj_add_int(doc, root, "cover_max_duration_s", HEARTMULA_COVER_MAX_DURATION_S);
        yyjson_mut_obj_add_int(doc, root, "cover_max_steps", HEARTMULA_COVER_MAX_STEPS);
    }
    yyjson_mut_val * d = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "defaults", d);
    yyjson_mut_obj_add_real(doc, d, "temperature", 1.0);
    yyjson_mut_obj_add_int(doc, d, "topk", 50);
    yyjson_mut_obj_add_real(doc, d, "cfg_scale", 1.5);
    yyjson_mut_obj_add_int(doc, d, "codec_steps", 10);
    yyjson_mut_obj_add_real(doc, d, "codec_guidance", 1.25);
    yyjson_mut_obj_add_int(doc, d, "duration_s", 120);
    hm_add_pick_blocks(doc, root);
    yyjson_mut_val * rs = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "resident", rs);
    const char * keys[HM_TOKENIZER + 1] = { "global", "local", "flow", "scalar_decoder", "rvq", "tokenizer" };
    for (int i = 0; i <= HM_TOKENIZER; i++) yyjson_mut_obj_add_bool(doc, rs, keys[i], g_hm.resident[i]);
    yyjson_mut_obj_add_bool(doc, root, "encode_ready", false);
    yyjson_mut_val * v = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "vram", v);
    yyjson_mut_obj_add_uint(doc, v, "resident_bytes", g_hm.resident_bytes);
    yyjson_mut_obj_add_uint(doc, v, "free_bytes", heartmula_device_free(g_hm.device));
    yyjson_mut_obj_add_uint(doc, v, "reserve_bytes", heartmula_reserve_bytes());
    yyjson_mut_val * errs = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "errors", errs);
    for (const auto & e : g_hm.cat.errors) yyjson_mut_arr_add_strcpy(doc, errs, e.c_str());
    if (!ready && !g_hm.cat.files.empty()) yyjson_mut_arr_add_strcpy(doc, errs, err.c_str());
    res.set_content(hm_write(doc), "application/json");
}

static void heartmula_handle_models(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_hm.mtx);
    if (req.has_param("rescan") && req.get_param_value("rescan") == "1") g_hm.cat = heartmula_discover(g_hm.models_dir);
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val *         variants = yyjson_mut_arr(doc);
    std::vector<std::string> seen;
    for (const auto & f : g_hm.cat.files) {
        // MuLaCover is not a plain-synth variant (it renders covers only); its files stay under components.
        if ((f.role == HM_GLOBAL || f.role == HM_LOCAL) && !f.variant.empty() && !hm_is_cover(f.variant) &&
            std::find(seen.begin(), seen.end(), f.variant) == seen.end()) {
            seen.push_back(f.variant);
            yyjson_mut_arr_add_strcpy(doc, variants, f.variant.c_str());
        }
    }
    yyjson_mut_obj_add_val(doc, root, "variants", variants);
    yyjson_mut_val * comps = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "components", comps);
    for (int r = 0; r < HM_ROLE_COUNT; r++) {
        std::string key = k_heartmula_roles[r];
        std::replace(key.begin(), key.end(), '-', '_');
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        yyjson_mut_obj_add(comps, yyjson_mut_strcpy(doc, key.c_str()), arr);
        for (const auto & f : g_hm.cat.files) {
            if (f.role != r) continue;
            yyjson_mut_val * o = yyjson_mut_arr_add_obj(doc, arr);
            yyjson_mut_obj_add_strcpy(doc, o, "type", f.type.c_str());
            yyjson_mut_obj_add_strcpy(doc, o, "filename", f.filename.c_str());
            yyjson_mut_obj_add_uint(doc, o, "bytes", f.bytes);
            yyjson_mut_obj_add_strcpy(doc, o, "variant", f.variant.c_str());
            yyjson_mut_obj_add_uint(doc, o, "file_type", f.file_type);
            yyjson_mut_obj_add_strcpy(doc, o, "base_fingerprint", f.fingerprint.c_str());
            yyjson_mut_obj_add_bool(doc, o, "compatible", f.compatible);
            yyjson_mut_obj_add_strcpy(doc, o, "error", f.error.c_str());
        }
    }
    res.set_content(hm_write(doc), "application/json");
}

// Worker: free resident components the pick no longer names (select-model's unload).
static void hm_drop_stale() {
    HeartMuLaResolved r;
    std::string       err;
    bool              ok;
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        ok = hm_synth_ready_locked(&r, &err);
    }
    if (ok) {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        r.adapter = g_hm.requested.adapter;  // synth_ready resolves without it
    }
    heartmula_drop_stale(&g_hm_rt, ok ? &r : nullptr);
    hm_publish();
}

static void heartmula_handle_select_model(const httplib::Request & req, httplib::Response & res) {
    if (g_hm_active.load() > 0) {
        hm_error(res, 503, "busy", "a HeartMuLa job is loading or running; retry, or put the pick on the synth request");
        return;
    }
    // The pick fields of a synth request, with the same rules.
    HeartMuLaRequest r;
    std::string      err;
    yyjson_doc *     d    = yyjson_read(req.body.data(), req.body.size(), 0);
    yyjson_val *     root = d ? yyjson_doc_get_root(d) : nullptr;
    if (!yyjson_is_obj(root)) {
        err = "body must be a JSON object";
    } else {
        size_t       idx, max;
        yyjson_val * k, * v;
        yyjson_obj_foreach(root, idx, max, k, v) {
            const std::string key = yyjson_get_str(k);
            std::optional<std::string> * field = key == "variant" ? &r.variant : key == "global_type" ? &r.global
                                               : key == "local_type" ? &r.local : key == "flow_type" ? &r.flow
                                               : key == "scalar_decoder_type" ? &r.scalar : key == "adapter" ? &r.adapter
                                               : nullptr;
            if (field && (yyjson_is_str(v) || yyjson_is_null(v))) {
                *field = yyjson_is_str(v) ? yyjson_get_str(v) : "";
            } else if (key == "adapter_scale" && yyjson_is_num(v) && yyjson_get_num(v) >= 0 && yyjson_get_num(v) <= 2) {
                r.adapter_scale = (float) yyjson_get_num(v);
            } else if (err.empty()) {
                err = field || key == "adapter_scale" ? key + " has the wrong type or range" : "unknown field '" + key + "'";
            }
        }
    }
    yyjson_doc_free(d);
    if (!err.empty()) {
        hm_error(res, 400, "bad_request", err);
        return;
    }
    std::lock_guard<std::mutex> lock(g_hm.mtx);
    const HeartMuLaPick next = heartmula_request_pick(g_hm.requested, r);
    if (hm_is_cover(next.variant)) {
        hm_error(res, 409, "incompatible", "MuLaCover renders covers only (POST /heartmula/cover); pick another variant");
        return;
    }
    HeartMuLaResolved   resolved;
    int                 status = 0;
    if (!hm_resolve_locked(next, &resolved, &status, &err)) {
        hm_error(res, status, hm_status_code(status), err);
        return;
    }
    g_hm.requested = next;
    work_push(hm_drop_stale);
    yyjson_mut_doc * doc   = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * rootw = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, rootw);
    hm_add_pick_blocks(doc, rootw);
    res.set_content(hm_write(doc), "application/json");
}

// Worker (on_worker): admission, then the LM phase's weights for the current pick.
static void heartmula_handle_warm(const httplib::Request &, httplib::Response & res) {
    HeartMuLaResolved r;
    std::string       err;
    int               status = 0;
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        if (!hm_resolve_locked(g_hm.requested, &r, &status, &err)) {
            hm_error(res, status, hm_status_code(status), err);
            return;
        }
        if (!hm_tokenizer_locked(r.file[HM_TOKENIZER].path, &err)) {
            hm_error(res, 500, "load_failed", err);
            return;
        }
    }
    heartmula_ensure_backend(&g_hm_rt);
    heartmula_drop_stale(&g_hm_rt, &r);
    const uint64_t need = heartmula_lm_need(g_hm_rt, r);
    if (!g_keep_loaded && need) heartmula_free_codec(&g_hm_rt);
    const HeartMuLaAdmission a = heartmula_admit(g_hm_rt, "lm", need, 0, 0, true, hm_reclaim_hooks());
    if (!a.ok) {
        hm_error(res, 507, "vram_insufficient", a.message(), &a);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = heartmula_load_lm(&g_hm_rt, r, &err) && heartmula_load_adapter(&g_hm_rt, r, &err);
    hm_publish();
    if (!ok) {
        hm_error(res, 500, err.find("alloc") != std::string::npos ? "vram_alloc_failed" : "load_failed", err);
        return;
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "{\"loaded\":true,\"already_loaded\":%s,\"resident_bytes\":%llu,\"load_ms\":%.0f}",
             need ? "false" : "true", (unsigned long long) heartmula_resident_bytes(g_hm_rt),
             heartmula_pipeline_detail::ms_since(t0));
    res.set_content(buf, "application/json");
}

// Worker (on_worker): free every HeartMuLa buffer. The tokenizer is host memory and stays.
static void heartmula_handle_unload(const httplib::Request &, httplib::Response & res) {
    heartmula_free_all(&g_hm_rt);
    hm_publish();
    res.set_content("{\"unloaded\":true}", "application/json");
}

// Stage -> shared phase (contract "Progress").
static JobPhase hm_phase(const std::string & s) {
    if (s == "style") return JobPhase::LOADING_TEXT_ENC;
    if (s == "admission" || s == "loading_lm") return JobPhase::LOADING_DIT;
    if (s == "loading_adapter") return JobPhase::LOADING_ADAPTER;
    if (s == "prefill") return JobPhase::ENCODING_COND;
    if (s == "frames") return JobPhase::DIT_INFERENCE;
    if (s == "loading_codec") return JobPhase::LOADING_VAE;
    if (s == "codec_decode") return JobPhase::VAE_DECODE;
    if (s == "encoding_output") return JobPhase::ENCODING_OUTPUT;
    return JobPhase::QUEUED;
}

// Worker: the cover's style vector (upstream _encode_style) from the tags as the prompt
// reads them. The BF16 encoder is loaded for this alone, after admission, and freed.
static bool hm_cover_style(HeartMuLaRequest * req, const HeartMuLaTokenizer & tok, const HeartMuLaHooks & hooks,
                           const char ** code, std::string * err) {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        path = (std::filesystem::path(g_hm.models_dir) / HM_COVER_QWEN).string();
    }
    hooks.progress("style", 0, 1);
    std::error_code ec;
    const uint64_t  bytes = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        *code = "model_missing";
        *err  = "HeartMuLa covers need the Qwen3 Embedding BF16 text encoder (" HM_COVER_QWEN ")";
        return false;
    }
    heartmula_ensure_backend(&g_hm_rt);
    const HeartMuLaAdmission a = heartmula_admit(g_hm_rt, "style", bytes, 0, 0, false, hooks);
    if (!a.ok) {
        *code = "vram_insufficient";
        *err  = a.message();
        return false;
    }
    const auto       t0  = std::chrono::steady_clock::now();
    Qwen3GGML        qm  = {};
    BPETokenizer     bpe = {};
    std::vector<int> ids;
    req->cover->style.assign(HEARTMULA_COVER_STYLE_DIM, 0.0f);
    *code   = "load_failed";
    bool ok = load_bpe_from_gguf(&bpe, path.c_str());
    if (ok) {
        ok = qwen3_load_text_encoder(&qm, path.c_str());
        if (ok) {
            ok = qwen3_style_vector(&qm, &bpe, mulacover_prompt(tok, req->tags, "").tags, req->cover->style.data(), &ids, err);
            *code = "bad_request";  // the only refusal left is the 8192-token limit
        } else {
            *err = "cannot load " + path;
        }
        qwen3_free(&qm);
    } else {
        *err = "cannot read the tokenizer in " + path;
    }
    fprintf(stderr, "[HeartMuLa] style vector: %zu Qwen tokens, %.0f ms%s\n", ids.size(),
            heartmula_pipeline_detail::ms_since(t0), ok ? "" : " (failed)");
    hooks.progress("style", 1, 1);
    return ok;
}

static void hm_synth_worker(std::shared_ptr<Job> job, HeartMuLaRequest req, std::shared_ptr<HeartMuLaJobInfo> info) {
    auto end = [&](int status, const char * code, const std::string & msg) {
        {
            std::lock_guard<std::mutex> lock(info->mtx);
            if (status == 3) {
                info->stage      = "cancelled";
                info->end_reason = "cancelled";
            } else if (status == 2) {
                info->stage      = "failed";
                info->error_code = code;
                info->error      = msg;
            }
        }
        if (status == 3) job->result_end_reason = "cancelled";
        job_set_phase(*job, status == 1 ? JobPhase::DONE : status == 2 ? JobPhase::FAILED : JobPhase::CANCELLED);
        job->status.store(status);
    };
    if (job->cancel.load()) {
        end(3, "cancelled", "");
        return;
    }
    g_hm_active++;
    struct Done { ~Done() { hm_publish(); g_hm_active--; } } done;
    HeartMuLaResolved r;
    std::shared_ptr<const HeartMuLaTokenizer> tok;
    std::string err;
    int         status = 0;
    {
        // Files can change between the request and the job (load-time checks run again).
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        g_hm.cat = heartmula_discover(g_hm.models_dir);
        const char * code = "";
        const bool   ok   = req.cover ? hm_cover_ready_locked(req, &r, &status, &code, &err)
                                      : hm_resolve_locked(heartmula_request_pick(g_hm.requested, req), &r, &status, &err);
        if (!ok) {
            end(2, req.cover ? code : hm_status_code(status), err);
            return;
        }
        tok = hm_tokenizer_locked(r.file[HM_TOKENIZER].path, &err);
        if (!tok) {
            end(2, "load_failed", err);
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(info->mtx);
        info->variant = r.variant;
        for (int i = 0; i < 4; i++) info->pick_types[i] = r.file[i].type;
        info->codec_steps = req.codec_steps;
    }
    HeartMuLaHooks hooks = hm_reclaim_hooks();
    {
        MM3PluginSel sel;
        sel.solver             = req.infer_method;
        sel.scheduler          = req.scheduler;
        sel.guidance           = req.guidance_mode;
        sel.shift              = req.flow_shift;
        sel.apg_norm_threshold = req.apg_norm_threshold;
        sel.params.insert(req.plugin_params.begin(), req.plugin_params.end());
        hooks.solve = heartcodec::plugin_solver(sel);
    }
    std::string    last;
    hooks.cancelled = [&]() { return job->cancel.load(); };
    hooks.progress  = [&](const char * stage, int step, int total) {
        {
            std::lock_guard<std::mutex> lock(info->mtx);
            info->stage = stage;
            if (!strcmp(stage, "prefill")) {
                info->prefill_done  = step;
                info->prompt_tokens = total;
            } else if (!strcmp(stage, "frames")) {
                info->frames_done = step;
                info->max_frames  = total;
            } else if (!strcmp(stage, "codec_decode")) {
                info->codec_step   = step;
                info->codec_chunks = req.codec_steps ? total / req.codec_steps : 0;
                info->codec_chunk  = req.codec_steps ? step / req.codec_steps : 0;
            }
        }
        job_set_phase(*job, hm_phase(stage), step, total);
        if (last != stage) hm_publish();
        last = stage;
    };
    if (req.cover) {
        const char * code = "";
        if (!hm_cover_style(&req, *tok, hooks, &code, &err)) {
            end(2, code, err);
            return;
        }
        if (job->cancel.load()) {
            end(3, "cancelled", "");
            return;
        }
    }
    HeartMuLaResult res;
    const bool      ok = heartmula_generate(&g_hm_rt, r, *tok, req, g_keep_loaded, hooks, &res);
    {
        std::lock_guard<std::mutex> lock(info->mtx);
        info->admission = res.admission;
        info->prompt_tokens = (int) res.prompt_tokens;
        info->max_frames    = (int) res.max_frames;
        info->frames_done   = (int) res.frames;
        info->t_admission   = res.t_admission;
        info->t_load        = res.t_load;
        info->t_prefill     = res.t_prefill;
        info->t_frames      = res.t_frames;
        info->t_codec       = res.t_codec;
        info->end_reason    = res.end_reason;
    }
    if (job->cancel.load() || res.error_code == "cancelled") {
        end(3, "cancelled", "");
        return;
    }
    if (!ok) {
        end(2, res.error_code.c_str(), res.error);
        return;
    }
    hooks.progress("encoding_output", 0, 0);
    const auto t0 = std::chrono::steady_clock::now();
    std::string body, mime;
    if (req.codes_only) {
        body = "{\"frames\":" + std::to_string(res.frames) + ",\"codebooks\":8,\"frame_rate\":12.5,\"codes\":[";
        for (int64_t t = 0; t < res.frames; t++) {
            body += t ? ",[" : "[";
            for (int q = 0; q < HEARTMULA_CODEBOOKS; q++) {
                body += (q ? "," : "") + std::to_string(res.codes[(size_t) (t * HEARTMULA_CODEBOOKS + q)]);
            }
            body += "]";
        }
        body += "]}";
        mime = "application/json";
    } else {
        body = audio_encode_wav_s16(res.audio.data(), (int) res.samples, 48000);
        mime = "audio/wav";
    }
    {
        std::lock_guard<std::mutex> lock(info->mtx);
        info->t_encode = heartmula_pipeline_detail::ms_since(t0);
        info->stage    = "done";
    }
    if (job->cancel.load()) {  // a cancel before the job is marked done always wins
        end(3, "cancelled", "");
        return;
    }
    job->result_body       = std::move(body);
    job->result_mime       = mime;
    job->result_end_reason = res.end_reason;
    end(1, "", "");
}

// POST /heartmula/synth, or with `cover` POST /heartmula/cover (same job path).
static void hm_submit(const httplib::Request & req, httplib::Response & res, bool cover) {
    HeartMuLaRequest r;
    std::string      err;
    if (!heartmula_parse_request(req.body, &r, &err, cover)) {
        hm_error(res, 400, "bad_request", err);
        return;
    }
    if (cover && r.variant && !r.variant->empty() && !hm_is_cover(*r.variant)) {
        hm_error(res, 400, "bad_request", "variant must be " HM_COVER_VARIANT " or omitted");
        return;
    }
    if (cover && r.adapter && !r.adapter->empty()) {
        hm_error(res, 400, "bad_request", "HeartMuLa adapters do not apply to MuLaCover covers");
        return;
    }
    auto info = std::make_shared<HeartMuLaJobInfo>();
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        HeartMuLaResolved resolved;
        int               status = 0;
        const char *      code   = "";
        if (cover) {
            if (!hm_cover_ready_locked(r, &resolved, &status, &code, &err)) {
                hm_error(res, status, code, err);
                return;
            }
        } else if (!hm_synth_ready_locked(&resolved, &err)) {
            hm_error(res, 503, "unavailable", "HeartMuLa is not ready: " + err);
            return;
        } else if (hm_is_cover(heartmula_request_pick(g_hm.requested, r).variant)) {
            hm_error(res, 409, "incompatible", "MuLaCover renders covers only (POST /heartmula/cover)");
            return;
        } else if (!hm_resolve_locked(heartmula_request_pick(g_hm.requested, r), &resolved, &status, &err)) {
            hm_error(res, status, hm_status_code(status), err);
            return;
        }
        if (r.codes.empty()) {
            auto tok = hm_tokenizer_locked(resolved.file[HM_TOKENIZER].path, &err);
            if (!tok) {
                hm_error(res, 500, "load_failed", err);
                return;
            }
            int64_t       muq = 0;
            const int64_t T   = (int64_t) (cover ? mulacover_prompt(*tok, r.tags, r.lyrics).ids
                                                 : heartmula_prompt_ids(*tok, r.tags, r.lyrics, &muq)).size();
            const int64_t F   = heartmula_max_frames(r.duration_s);
            const int64_t lim = heartmula_global_config().n_ctx;
            if (T + F + 1 > lim) {
                yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
                yyjson_mut_val * root = yyjson_mut_obj(doc);
                yyjson_mut_doc_set_root(doc, root);
                yyjson_mut_obj_add_str(doc, root, "error", "the prompt and the requested duration exceed the context length");
                yyjson_mut_obj_add_str(doc, root, "code", "context_overflow");
                yyjson_mut_obj_add_int(doc, root, "prompt_tokens", T);
                yyjson_mut_obj_add_int(doc, root, "frames", F);
                yyjson_mut_obj_add_int(doc, root, "limit", lim);
                res.status = 400;
                res.set_content(hm_write(doc), "application/json");
                return;
            }
            info->prompt_tokens = (int) T;
            info->max_frames    = (int) F;
        }
        info->pick = cover ? hm_cover_pick(r) : heartmula_request_pick(g_hm.requested, r);
    }
    info->seed       = r.seed;
    info->noise_seed = r.noise_seed;
    auto job         = job_create();
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        g_hm.jobs[job->id] = info;
        g_hm.job_order.push_back(job->id);
        while (g_hm.job_order.size() > 64) {
            g_hm.jobs.erase(g_hm.job_order.front());
            g_hm.job_order.pop_front();
        }
    }
    fprintf(stderr, "[HeartMuLa] job %s queued: %s, %.0f s, seed %llu\n", job->id.c_str(), r.id.c_str(), r.duration_s,
            (unsigned long long) r.seed);
    if (r.cover) {
        fprintf(stderr, "[HeartMuLa] cover: %lld lead-sheet steps, cfg %.2f, top-k %d, temperature %.2f\n",
                (long long) r.cover->steps, r.cfg_scale, r.topk, r.temperature);
    }
    // What the model is conditioned on, before the pipeline lower-cases and wraps it.
    fprintf(stderr, "[HeartMuLa] tags: %s\n", r.tags.c_str());
    if (r.lyrics.empty()) {
        fprintf(stderr, "[HeartMuLa] lyrics: (none)\n");
    } else {
        size_t from = 0;
        fprintf(stderr, "[HeartMuLa] lyrics:\n");
        while (from <= r.lyrics.size()) {
            const size_t nl = r.lyrics.find('\n', from);
            fprintf(stderr, "[HeartMuLa]   %s\n", r.lyrics.substr(from, nl == std::string::npos ? std::string::npos : nl - from).c_str());
            if (nl == std::string::npos) break;
            from = nl + 1;
        }
    }
    work_push([job, r, info]() mutable { hm_synth_worker(job, std::move(r), info); });
    res.set_content("{\"id\":\"" + job->id + "\"}", "application/json");
}

static void heartmula_handle_synth(const httplib::Request & req, httplib::Response & res) {
    hm_submit(req, res, false);
}

static void heartmula_handle_cover(const httplib::Request & req, httplib::Response & res) {
    hm_submit(req, res, true);
}

static void heartmula_handle_job(const httplib::Request & req, httplib::Response & res) {
    const std::string id  = req.has_param("id") ? req.get_param_value("id") : "";
    auto              job = job_find(id);
    std::shared_ptr<HeartMuLaJobInfo> info;
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        auto it = g_hm.jobs.find(id);
        if (it != g_hm.jobs.end()) info = it->second;
    }
    if (!job || !info) {
        hm_error(res, 404, "not_found", "no HeartMuLa job " + id);
        return;
    }
    std::lock_guard<std::mutex> lock(info->mtx);
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "id", id.c_str());
    yyjson_mut_obj_add_str(doc, root, "status", job_status_str(job->status.load()));
    yyjson_mut_obj_add_str(doc, root, "phase", job_phase_str(job->phase.load()));
    yyjson_mut_obj_add_strcpy(doc, root, "stage", info->stage.c_str());
    yyjson_mut_obj_add_int(doc, root, "prompt_tokens", info->prompt_tokens);
    yyjson_mut_obj_add_int(doc, root, "prefill_done", info->prefill_done);
    yyjson_mut_obj_add_int(doc, root, "frames_done", info->frames_done);
    yyjson_mut_obj_add_int(doc, root, "max_frames", info->max_frames);
    yyjson_mut_obj_add_int(doc, root, "codec_chunk", info->codec_chunk);
    yyjson_mut_obj_add_int(doc, root, "codec_chunks", info->codec_chunks);
    yyjson_mut_obj_add_int(doc, root, "codec_step", info->codec_step);
    yyjson_mut_obj_add_int(doc, root, "codec_steps", info->codec_steps);
    yyjson_mut_obj_add_strcpy(doc, root, "seed", std::to_string(info->seed).c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "noise_seed", std::to_string(info->noise_seed).c_str());
    yyjson_mut_val * p = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "pick", p);
    yyjson_mut_obj_add_strcpy(doc, p, "variant", info->variant.c_str());
    const char * keys[4] = { "global", "local", "flow", "scalar_decoder" };
    for (int i = 0; i < 4; i++) yyjson_mut_obj_add_strcpy(doc, p, keys[i], info->pick_types[i].c_str());
    yyjson_mut_obj_add_strcpy(doc, p, "adapter", info->pick.adapter.c_str());
    yyjson_mut_obj_add_real(doc, p, "adapter_scale", info->pick.adapter_scale);
    if (info->admission) hm_add_admission(doc, root, *info->admission);
    else yyjson_mut_obj_add_null(doc, root, "admission");
    yyjson_mut_val * t = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "timings_ms", t);
    yyjson_mut_obj_add_int(doc, t, "admission", (int64_t) info->t_admission);
    yyjson_mut_obj_add_int(doc, t, "load", (int64_t) info->t_load);
    yyjson_mut_obj_add_int(doc, t, "prefill", (int64_t) info->t_prefill);
    yyjson_mut_obj_add_int(doc, t, "frames", (int64_t) info->t_frames);
    yyjson_mut_obj_add_int(doc, t, "codec", (int64_t) info->t_codec);
    yyjson_mut_obj_add_int(doc, t, "encode", (int64_t) info->t_encode);
    yyjson_mut_obj_add_strcpy(doc, root, "end_reason", info->end_reason.c_str());
    if (info->error_code.empty()) {
        yyjson_mut_obj_add_null(doc, root, "error");
    } else {
        yyjson_mut_val * e = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, root, "error", e);
        yyjson_mut_obj_add_strcpy(doc, e, "code", info->error_code.c_str());
        yyjson_mut_obj_add_strcpy(doc, e, "message", info->error.c_str());
    }
    res.set_content(hm_write(doc), "application/json");
}

static void heartmula_handle_not_implemented(const httplib::Request &, httplib::Response & res) {
    hm_error(res, 501, "not_implemented", "not available until its HeartMuLa slice lands");
}

static void heartmula_register_routes(httplib::Server & svr, const char * models_dir) {
    {
        std::lock_guard<std::mutex> lock(g_hm.mtx);
        g_hm.models_dir = models_dir;
        g_hm.cat        = heartmula_discover(models_dir);
    }
    svr.Get("/heartmula/props", heartmula_handle_props);
    svr.Get("/heartmula/models", heartmula_handle_models);
    svr.Post("/heartmula/select-model", heartmula_handle_select_model);
    svr.Post("/heartmula/warm", on_worker(heartmula_handle_warm));
    svr.Post("/heartmula/unload", on_worker(heartmula_handle_unload));
    svr.Post("/heartmula/synth", heartmula_handle_synth);
    svr.Post("/heartmula/cover", heartmula_handle_cover);
    svr.Get("/heartmula/job", heartmula_handle_job);
    svr.Post("/heartmula/encode", heartmula_handle_not_implemented);
    svr.Post("/heartmula/imatrix", heartmula_handle_not_implemented);
    fprintf(stderr, "[Server] HeartMuLa routes registered (models_dir=%s, %zu file(s))\n", models_dir, g_hm.cat.files.size());
}

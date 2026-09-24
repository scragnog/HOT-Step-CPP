// lm-common.h (pulled in through yue2-aitk-joint-step.h -> lm-optim.h)
// forward-declares jl(); the ace-train CLI defines it against its own --jsonl
// stream. This standalone runtime streams its own event() lines and the lm_*
// log helpers are never called on this path, so a stub keeps the include
// chain self-contained for the kernels translation unit.
static void jl(const char * fmt, ...) { (void) fmt; }

#include "yue2-aitk-runtime.h"
#include "yue2-aitk-dataset.h"
#include "yue2-aitk-sha256.h"
#include "yue2-aitk-sampler.h"
#include "yue2-aitk-joint-step.h"
#include "yue2-aitk-resume.h"
#include "yue2-aitk-embedding-dequant.h"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <chrono>
#include "yyjson.h"

namespace {
volatile std::sig_atomic_t g_cancel = 0;
void on_sigint(int) { g_cancel = 1; }

bool fail(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}

bool fresh_output(const std::filesystem::path & path, std::string * error) {
    std::error_code ec;
    if (path.empty() || std::filesystem::exists(path, ec) || ec)
        return fail(error, "joint-train output must be a new directory");
    return true;
}
void event(const char * stage, int step = -1) { std::cout << "{\"stage\":\"" << stage << "\"" << (step >= 0 ? ",\"step\":" + std::to_string(step) : "") << "}\n" << std::flush; }
std::string lower_hash(std::string value) {
    for (char & c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

struct ResumePlan { yue2_aitk::ResumeRecord record; std::vector<size_t> order; size_t cursor = 0; int completed = 0; std::string sampler; };
bool str_field(yyjson_val * v, std::string * out) { if (!yyjson_is_str(v) || yyjson_get_len(v) != std::strlen(yyjson_get_str(v))) return false; out->assign(yyjson_get_str(v), yyjson_get_len(v)); return true; }
// LmOptim scalars that travel in the metadata JSON rather than the binary
// record: the bias-correction counter and Prodigy's adaptive d / r.
struct ResumeBinding {
    int opt_iter = -1;
    double prodigy_d = -1.0;
    double prodigy_r = -1.0;
    // The stop windows' recent values. Without them every preview resume
    // restarted the windows empty, so no target stop could fire for the first
    // window's worth of steps of each segment: up to 20 steps of overshoot.
    std::vector<double> kl_history, loss_history;
    // Step at which the planner was frozen (--nar-extra-steps); -1 = live.
    int planner_frozen_at = -1;
    // Spike guard state: recent applied gradient norms and recent skip steps.
    std::vector<double> gnorm_history, spike_steps;
    // Reconstruction meter at each checkpoint since the planner froze.
    std::vector<double> recon_history;
    // The last KL rung a checkpoint was written for (--kl-checkpoint-every);
    // the next segment continues from the rung after it.
    double kl_mark_last = 0.0;
    // Planner refinement: the step the planner was unfrozen at (warmup counts
    // from it) and the adaptive rate multiplier.
    int unfrozen_at = -1;
    double rung_lr_mult = 1.0;
};
// Steps in the KL trend fit. Kept in step with Yue2AitkTrainCard's chart.
constexpr int kKlTrendWindow = 30;
// A least-squares line through `v`, read at its last point.
double trend_at_end(const std::vector<double> & v) {
    const double n = (double) v.size(), xm = (n - 1.0) / 2.0;
    double ym = 0.0; for (double y : v) ym += y; ym /= n;
    double sxy = 0.0, sxx = 0.0;
    for (size_t i = 0; i < v.size(); ++i) { const double dx = (double) i - xm; sxy += dx * (v[i] - ym); sxx += dx * dx; }
    return ym + (sxx > 0.0 ? sxy / sxx : 0.0) * (n - 1.0 - xm);
}
bool parse_resume_meta(const std::string & text, const std::string & checkpoint, const std::string & dataset, const std::string & source,
                       uint64_t seed, int cuda_index, size_t item_count, ResumePlan * plan, std::string * error,
                       float * cursor_weight, bool cursor_explicit,
                       const yue2_aitk_runtime::Config & config, ResumeBinding * binding) {
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0); if (!doc) return fail(error, "resume metadata is invalid JSON");
    struct Guard { yyjson_doc * d; ~Guard() { yyjson_doc_free(d); } } guard{doc}; yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root) || !yue2_aitk::dataset_detail::unique_keys(root)) return fail(error, "resume metadata object is malformed");
    auto * cursor_value=yyjson_obj_get(root,"cursor_weight");
    const float saved_cursor=cursor_value?float(yyjson_get_num(cursor_value)):0.0f;
    if((cursor_value&&!yyjson_is_num(cursor_value)) || !std::isfinite(saved_cursor) || saved_cursor<0 || saved_cursor>10 ||
       (cursor_explicit&&*cursor_weight!=saved_cursor)) return fail(error,"resume lyric timing configuration mismatch");
    *cursor_weight=saved_cursor;
    std::string recipe, cp, ds, sm, sampler;
    // Records written before the optimizer fields existed default to the
    // legacy constants (AdamW8bit, rank 32, alpha 32), so an old record still
    // resumes a default-shaped run.
    std::string rec_optimizer="adamw"; int rec_rank=32; float rec_alpha=32.0f;
    if(yyjson_val * v=yyjson_obj_get(root,"optimizer")) { if(!str_field(v,&rec_optimizer)) return fail(error,"resume optimizer field is malformed"); }
    if(yyjson_val * v=yyjson_obj_get(root,"rank")) { if(!yyjson_is_int(v)) return fail(error,"resume rank field is malformed"); rec_rank=int(yyjson_get_sint(v)); }
    if(yyjson_val * v=yyjson_obj_get(root,"alpha")) { if(!yyjson_is_num(v) || !std::isfinite(float(yyjson_get_num(v)))) return fail(error,"resume alpha field is malformed"); rec_alpha=float(yyjson_get_num(v)); }
    if(rec_optimizer!=config.optimizer || rec_rank!=config.rank || rec_alpha!=config.alpha) {
        return fail(error,"resume optimizer, rank or alpha mismatch; start a new run");
    }
    // Records written before LoKr existed carry no adapter_type: they are LoRA.
    std::string rec_adapter="lora"; int rec_lokr_dim=0, rec_lokr_factor=0;
    if(yyjson_val * v=yyjson_obj_get(root,"adapter_type")) { if(!str_field(v,&rec_adapter)) return fail(error,"resume adapter_type field is malformed"); }
    if(yyjson_val * v=yyjson_obj_get(root,"lokr_dim")) { if(!yyjson_is_int(v)) return fail(error,"resume lokr_dim field is malformed"); rec_lokr_dim=int(yyjson_get_sint(v)); }
    if(yyjson_val * v=yyjson_obj_get(root,"lokr_factor")) { if(!yyjson_is_int(v)) return fail(error,"resume lokr_factor field is malformed"); rec_lokr_factor=int(yyjson_get_sint(v)); }
    if(rec_adapter!=config.adapter_type || (config.adapter_type=="lokr" && (rec_lokr_dim!=config.lokr_dim || rec_lokr_factor!=config.lokr_factor))) {
        return fail(error,"resume adapter type or LoKr shape mismatch; start a new run");
    }
    // Same rule as the optimizer name: a modifier changes the update rule, so
    // a run resumed under a different one is a different experiment.
    bool rec_cautious=false;
    if(yyjson_val * v=yyjson_obj_get(root,"cautious")) { if(!yyjson_is_bool(v)) return fail(error,"resume cautious field is malformed"); rec_cautious=yyjson_get_bool(v); }
    if(rec_cautious!=config.cautious) return fail(error,"resume optimizer modifier (cautious) mismatch; start a new run");
    // Schedule knobs that only reshape the run from here on: note, don't refuse.
    { yyjson_val * v=yyjson_obj_get(root,"lr"); if(v&&yyjson_is_num(v)&&double(yyjson_get_num(v))!=double(config.lr)) std::fprintf(stderr,"[yue2-aitk] resume note: lr changed from %.9g to %.9g; the run continues with the new value\n", double(yyjson_get_num(v)), (double)config.lr); }
    { yyjson_val * v=yyjson_obj_get(root,"warmup"); if(v&&yyjson_is_int(v)&&int(yyjson_get_sint(v))!=config.warmup) std::fprintf(stderr,"[yue2-aitk] resume note: warmup changed from %d to %d; the run continues with the new value\n", int(yyjson_get_sint(v)), config.warmup); }
    { yyjson_val * v=yyjson_obj_get(root,"weight_decay"); if(v&&yyjson_is_num(v)&&double(yyjson_get_num(v))!=double(config.weight_decay)) std::fprintf(stderr,"[yue2-aitk] resume note: weight-decay changed from %.9g to %.9g; the run continues with the new value\n", double(yyjson_get_num(v)), (double)config.weight_decay); }
    yyjson_val * vrecipe=yyjson_obj_get(root,"recipe"), *vcp=yyjson_obj_get(root,"checkpoint_sha256"), *vds=yyjson_obj_get(root,"dataset_sha256"), *vsm=yyjson_obj_get(root,"source_manifest_sha256"), *vseed=yyjson_obj_get(root,"seed"), *vdev=yyjson_obj_get(root,"cuda_index"), *vstep=yyjson_obj_get(root,"completed_step"), *vcursor=yyjson_obj_get(root,"order_cursor"), *vorder=yyjson_obj_get(root,"order"), *vsampler=yyjson_obj_get(root,"sampler_state");
    if (!str_field(vrecipe,&recipe) || recipe!="yue2-aitk-runtime-v1" || !str_field(vcp,&cp) || cp!=checkpoint || !str_field(vds,&ds) || ds!=dataset || !str_field(vsm,&sm) || sm!=source || !yyjson_is_uint(vseed) || yyjson_get_uint(vseed)!=seed || !yyjson_is_int(vdev) || yyjson_get_sint(vdev)!=cuda_index || !yyjson_is_int(vstep) || yyjson_get_sint(vstep)<0 || yyjson_get_sint(vstep)>INT_MAX || !yyjson_is_uint(vcursor) || !yyjson_is_arr(vorder) || yyjson_arr_size(vorder)!=item_count || !str_field(vsampler,&sampler)) return fail(error,"resume metadata binding mismatch");
    plan->completed=static_cast<int>(yyjson_get_sint(vstep)); plan->cursor=static_cast<size_t>(yyjson_get_uint(vcursor)); plan->sampler=std::move(sampler); plan->order.clear(); std::unordered_set<size_t> seen; size_t i=0,max=0; yyjson_val * x=nullptr; yyjson_arr_foreach(vorder,i,max,x) { if(!yyjson_is_uint(x) || yyjson_get_uint(x)>=item_count || !seen.insert(static_cast<size_t>(yyjson_get_uint(x))).second) return fail(error,"resume order is not a permutation"); plan->order.push_back(static_cast<size_t>(yyjson_get_uint(x))); }
    if (plan->cursor>item_count || plan->completed<0) return fail(error,"resume cursor is out of range");
    if (binding) {
        auto history=[&](const char * key, std::vector<double> * out) {
            yyjson_val * a=yyjson_obj_get(root,key); if(!yyjson_is_arr(a)) return;
            size_t hi=0, hmax=0; yyjson_val * hv=nullptr;
            yyjson_arr_foreach(a,hi,hmax,hv) if(yyjson_is_num(hv) && std::isfinite(yyjson_get_num(hv))) out->push_back(yyjson_get_num(hv));
        };
        history("kl_history",&binding->kl_history); history("loss_history",&binding->loss_history);
        history("gnorm_history",&binding->gnorm_history); history("spike_steps",&binding->spike_steps);
        history("recon_history",&binding->recon_history);
        if(yyjson_val * v=yyjson_obj_get(root,"kl_mark_last")) { if(yyjson_is_num(v) && std::isfinite(yyjson_get_num(v))) binding->kl_mark_last=yyjson_get_num(v); }
        if(yyjson_val * v=yyjson_obj_get(root,"unfrozen_at")) { if(yyjson_is_int(v) && yyjson_get_sint(v)>=0) binding->unfrozen_at=int(yyjson_get_sint(v)); }
        if(yyjson_val * v=yyjson_obj_get(root,"rung_lr_mult")) { if(yyjson_is_num(v) && std::isfinite(yyjson_get_num(v)) && yyjson_get_num(v)>0) binding->rung_lr_mult=yyjson_get_num(v); }
        if(yyjson_val * v=yyjson_obj_get(root,"planner_frozen_at")) {
            if(!yyjson_is_int(v) || yyjson_get_sint(v)<0 || yyjson_get_sint(v)>plan->completed) return fail(error,"resume planner_frozen_at field is malformed");
            binding->planner_frozen_at=int(yyjson_get_sint(v));
        }
    }
    if (config.optimizer!="adamw" && binding) {
        yyjson_val * vopt=yyjson_obj_get(root,"opt_iter");
        if(!vopt || !yyjson_is_int(vopt) || yyjson_get_sint(vopt)<0) return fail(error,"resume record is missing the optimizer iteration counter");
        binding->opt_iter=int(yyjson_get_sint(vopt));
        if (config.optimizer=="prodigy") {
            yyjson_val * vd=yyjson_obj_get(root,"prodigy_d"), *vr=yyjson_obj_get(root,"prodigy_r");
            if(!vd || !yyjson_is_num(vd) || !std::isfinite(double(yyjson_get_num(vd))) || double(yyjson_get_num(vd))<=0.0 ||
               !vr || !yyjson_is_num(vr) || !std::isfinite(double(yyjson_get_num(vr)))) return fail(error,"resume record is missing Prodigy state");
            binding->prodigy_d=double(yyjson_get_num(vd)); binding->prodigy_r=double(yyjson_get_num(vr));
        }
    }
    return true;
}
std::string make_resume_meta(const std::string & cp, const std::string & ds, const std::string & sm, uint64_t seed, int device, int completed, size_t cursor, const std::vector<size_t> & order, const std::string & sampler, float cursor_weight,
                             const yue2_aitk_runtime::Config & config, int opt_iter, double prodigy_d, double prodigy_r,
                             const std::vector<double> & kl_history, const std::vector<double> & loss_history, int planner_frozen_at,
                             const std::vector<double> & gnorm_history, const std::vector<double> & spike_steps,
                             const std::vector<double> & recon_history = {}, double kl_mark_last = 0.0, int unfrozen_at = -1, double rung_lr_mult = 1.0) {
    yyjson_mut_doc * doc=yyjson_mut_doc_new(nullptr);
    if (!doc) return {};
    yyjson_mut_val * root=yyjson_mut_obj(doc), * arr=yyjson_mut_arr(doc);
    if (!root || !arr) { yyjson_mut_doc_free(doc); return {}; }
    yyjson_mut_doc_set_root(doc,root);
    if(cursor_weight>0) yyjson_mut_obj_add_real(doc,root,"cursor_weight",cursor_weight);
    yyjson_mut_obj_add_strcpy(doc,root,"recipe","yue2-aitk-runtime-v1"); yyjson_mut_obj_add_strcpy(doc,root,"checkpoint_sha256",cp.c_str()); yyjson_mut_obj_add_strcpy(doc,root,"dataset_sha256",ds.c_str()); yyjson_mut_obj_add_strcpy(doc,root,"source_manifest_sha256",sm.c_str()); yyjson_mut_obj_add_uint(doc,root,"seed",seed); yyjson_mut_obj_add_int(doc,root,"cuda_index",device); yyjson_mut_obj_add_int(doc,root,"completed_step",completed); yyjson_mut_obj_add_uint(doc,root,"order_cursor",cursor); for(size_t x:order) yyjson_mut_arr_add_uint(doc,arr,x); yyjson_mut_obj_add_val(doc,root,"order",arr); yyjson_mut_obj_add_strcpy(doc,root,"sampler_state",sampler.c_str());
    yyjson_mut_obj_add_strcpy(doc,root,"optimizer",config.optimizer.c_str()); yyjson_mut_obj_add_int(doc,root,"rank",config.rank); yyjson_mut_obj_add_real(doc,root,"alpha",config.alpha); yyjson_mut_obj_add_real(doc,root,"lr",config.lr); yyjson_mut_obj_add_int(doc,root,"warmup",config.warmup); yyjson_mut_obj_add_real(doc,root,"weight_decay",config.weight_decay); yyjson_mut_obj_add_real(doc,root,"prodigy_d0",config.prodigy_d0); yyjson_mut_obj_add_real(doc,root,"muon_lr_scale",config.muon_lr_scale); yyjson_mut_obj_add_int(doc,root,"muon_ns_steps",config.muon_ns_steps);
    // Only LoKr records carry these keys, so a LoRA record is byte-identical
    // to one written before LoKr existed (the reader defaults to lora).
    if (config.adapter_type=="lokr") { yyjson_mut_obj_add_strcpy(doc,root,"adapter_type","lokr"); yyjson_mut_obj_add_int(doc,root,"lokr_dim",config.lokr_dim); yyjson_mut_obj_add_int(doc,root,"lokr_factor",config.lokr_factor); }
    if (config.cautious) yyjson_mut_obj_add_bool(doc,root,"cautious",true);
    yyjson_mut_obj_add_real(doc,root,"kl_weight",config.kl_weight); yyjson_mut_obj_add_real(doc,root,"abc_dropout",config.abc_dropout); yyjson_mut_obj_add_real(doc,root,"caption_dropout",config.caption_dropout); yyjson_mut_obj_add_real(doc,root,"planner_lr_scale",config.planner_lr_scale); yyjson_mut_obj_add_real(doc,root,"target_kl",config.target_kl);
    // Only runs with a target stop carry these, so other records are unchanged.
    for (auto [key, hist] : { std::pair<const char *, const std::vector<double> *>{"kl_history",&kl_history}, {"loss_history",&loss_history} }) {
        if (hist->empty()) continue;
        yyjson_mut_val * a=yyjson_mut_arr(doc); for(double v:*hist) yyjson_mut_arr_add_real(doc,a,v); yyjson_mut_obj_add_val(doc,root,key,a);
    }
    if (config.target_kl_mode!="mean") yyjson_mut_obj_add_strcpy(doc,root,"target_kl_mode",config.target_kl_mode.c_str());
    if (config.nar_lr_scale!=1.0f) yyjson_mut_obj_add_real(doc,root,"nar_lr_scale",config.nar_lr_scale);  // absent = 1.0, keeps old records byte-identical
    if (planner_frozen_at>=0) { yyjson_mut_obj_add_int(doc,root,"planner_frozen_at",planner_frozen_at); yyjson_mut_obj_add_int(doc,root,"nar_extra_steps",config.nar_extra_steps); }
    // Only guarded runs carry these, so other records are unchanged.
    if (config.spike_factor > 0.0f) for (auto [key, hist] : { std::pair<const char *, const std::vector<double> *>{"gnorm_history",&gnorm_history}, {"spike_steps",&spike_steps} }) {
        yyjson_mut_val * a=yyjson_mut_arr(doc); for(double v:*hist) yyjson_mut_arr_add_real(doc,a,v); yyjson_mut_obj_add_val(doc,root,key,a);
    }
    if (!recon_history.empty()) { yyjson_mut_val * a=yyjson_mut_arr(doc); for(double v:recon_history) yyjson_mut_arr_add_real(doc,a,v); yyjson_mut_obj_add_val(doc,root,"recon_history",a); }
    if (kl_mark_last > 0.0) yyjson_mut_obj_add_real(doc,root,"kl_mark_last",kl_mark_last);
    if (unfrozen_at >= 0) { yyjson_mut_obj_add_int(doc,root,"unfrozen_at",unfrozen_at); yyjson_mut_obj_add_real(doc,root,"rung_lr_mult",rung_lr_mult); }
    if (config.optimizer!="adamw") {
        yyjson_mut_obj_add_int(doc,root,"opt_iter",opt_iter);
        if (config.optimizer=="prodigy") {
            yyjson_mut_obj_add_real(doc,root,"prodigy_d",float(prodigy_d));
            yyjson_mut_obj_add_real(doc,root,"prodigy_r",float(prodigy_r));
        }
    }
    size_t n=0; char * raw=yyjson_mut_write(doc,0,&n); std::string out=raw?std::string(raw,n):std::string(); std::free(raw); yyjson_mut_doc_free(doc); return out;
}
}

namespace yue2_aitk_runtime {

void yue2_aitk_install_sigint_handler() { std::signal(SIGINT, on_sigint); }
bool yue2_aitk_cancel_requested() { return g_cancel != 0; }
void yue2_aitk_clear_cancel() { g_cancel = 0; }

static int run_impl(Config config, std::string * error) {
    if (!fresh_output(std::filesystem::u8path(config.output), error)) return 1;
    if (config.seed > UINT32_MAX) { fail(error, "native-v1 runtime seed must fit uint32_t"); return 1; }
    if (config.steps <= 0 || config.save_every <= 0 || config.cuda_index < 0 || config.cuda_index > 127) { fail(error, "invalid runtime configuration"); return 1; }
    if (config.rank < 1 || config.rank > 65536 || !std::isfinite(config.alpha) || config.alpha <= 0.0f || config.alpha > 1e6f ||
        (config.optimizer != "adamw" && config.optimizer != "adamw-lm" && config.optimizer != "prodigy" && config.optimizer != "muon") ||
        !std::isfinite(config.lr) || config.lr <= 0.0f ||
        config.warmup < 0 || config.warmup > config.steps ||
        !std::isfinite(config.weight_decay) || config.weight_decay < 0.0f ||
        !std::isfinite(config.prodigy_d0) || config.prodigy_d0 <= 0.0f ||
        !std::isfinite(config.muon_lr_scale) || config.muon_lr_scale <= 0.0f ||
        config.muon_ns_steps < 1 || config.muon_ns_steps > 20 ||
        !std::isfinite(config.target_loss) || config.target_loss < 0.0f ||
        !std::isfinite(config.target_kl) || config.target_kl < 0.0f ||
        config.target_loss_window < 1 ||
        !std::isfinite(config.kl_weight) || config.kl_weight < 0.0f ||
        !std::isfinite(config.abc_dropout) || config.abc_dropout < 0.0f || config.abc_dropout > 1.0f ||
        !std::isfinite(config.caption_dropout) || config.caption_dropout < 0.0f || config.caption_dropout > 1.0f ||
        !std::isfinite(config.planner_lr_scale) || config.planner_lr_scale <= 0.0f ||
        !std::isfinite(config.nar_lr_scale) || config.nar_lr_scale <= 0.0f || config.nar_extra_steps < 0) {
        fail(error, "invalid runtime configuration"); return 1;
    }
    if (config.nar_extra_steps > 0 && !(config.target_kl > 0.0f)) {
        fail(error, "--nar-extra-steps needs --target-kl: the planner freezes when it reaches that KL"); return 1;
    }
    if (config.optimizer == "muon" && (config.planner_lr_scale != 1.0f || config.nar_lr_scale != 1.0f)) {
        // Muon's update is bucketed by shape and scaled once per bucket
        // (lm-optim.h), so lr_mul would apply to the AdamW-ruled parameters
        // only — a HALF-honoured split is worse than a refused one.
        fail(error, "--planner-lr-scale / --nar-lr-scale are not supported with --optimizer muon"); return 1;
    }
    if (config.cautious && config.optimizer == "adamw") {
        fail(error, "--cautious needs an LmOptim optimizer: --optimizer adamw-lm, prodigy or muon (the native AdamW8bit kernel has no update tensor to mask)"); return 1;
    }
    const bool lokr = config.adapter_type == "lokr";
    if (!lokr && config.adapter_type != "lora") { fail(error, "invalid runtime configuration"); return 1; }
    if (lokr) {
        if (config.lokr_dim < 1 || config.lokr_dim > 65536 || config.lokr_factor < 1 || config.lokr_factor > 65536) {
            fail(error, "--lokr-dim and --lokr-factor must be within [1, 65536]"); return 1;
        }
        // LyCORIS convention: alpha == dim is scale 1. Resolved here so the
        // resume record and the export carry the number actually trained with.
        if (!config.alpha_explicit) config.alpha = (float) config.lokr_dim;
    }
    event("preflight");
    yue2_aitk::Dataset dataset;
    if (!yue2_aitk::read_dataset(config.dataset, &dataset, error)) return 1;
    if (config.caption_dropout > 0.0f) {
        // Refuse up front rather than at the first draw, a model load later:
        // a dataset prepared before the trigger-only prefixes existed cannot
        // train with caption dropout and must be re-prepared.
        for (const auto & item : dataset.items) {
            if (!item.prompt.has_nocap()) { fail(error, "--caption-dropout needs trigger-only prefixes for every item; this dataset was prepared before they existed — re-run preparation"); return 1; }
            if (config.cursor_weight > 0.0f && item.prompt.cursor.present && !item.prompt.cursor.instrumental && item.prompt.cursor.enabled && !item.prompt.cursor.has_nocap()) {
                fail(error, "--caption-dropout with lyric timing needs trigger-only cursor bindings; re-run preparation"); return 1;
            }
        }
        std::fprintf(stderr, "[yue2-aitk] caption dropout %.2f: trigger-only style on that share of steps\n", (double) config.caption_dropout);
    }
    yue2_aitk::sha256::digest checkpoint_hash, dataset_hash;
    if (!yue2_aitk::sha256::file(std::filesystem::u8path(config.checkpoint), checkpoint_hash, error)) return 1;
    if (checkpoint_hash.hex() != lower_hash(dataset.base_sha256)) { fail(error, "checkpoint SHA-256 does not match dataset base_sha256"); return 1; }
    if (!yue2_aitk::sha256::file(std::filesystem::u8path(config.dataset), dataset_hash, error)) return 1;
    const auto source_copy = std::filesystem::u8path(config.dataset).parent_path() / "source-manifest.json";
    yue2_aitk::sha256::digest source_hash;
    if (!yue2_aitk::sha256::file(source_copy, source_hash, error)) return 1;
    if (source_hash.hex() != lower_hash(dataset.source_manifest_sha256)) { fail(error, "dataset source-manifest.json SHA-256 does not match source_manifest_sha256"); return 1; }
    yue2_aitk::Yue2NativeSampler sampler(config.seed);
    std::vector<size_t> order(dataset.items.size()); std::iota(order.begin(), order.end(), 0);
    size_t cursor = 0; int completed = 0;
    float cursor_weight=config.cursor_weight;
    if(!std::isfinite(cursor_weight)||cursor_weight<0||cursor_weight>10) { fail(error,"invalid lyric timing weight"); return 1; }
    ResumePlan resume_plan;
    ResumeBinding resume_binding;
    if (!config.resume.empty()) {
        if (!yue2_aitk::yue2_aitk_read_resume(config.resume.c_str(), &resume_plan.record)) { fail(error, "cannot read resume record"); return 1; }
        if (!parse_resume_meta(resume_plan.record.runner_metadata, checkpoint_hash.hex(), dataset_hash.hex(), source_hash.hex(), config.seed, config.cuda_index, dataset.items.size(), &resume_plan, error,&cursor_weight,config.cursor_weight_explicit,config,&resume_binding)) return 1;
        if (resume_plan.completed > config.steps || resume_plan.completed > INT_MAX || resume_plan.record.state.step != resume_plan.completed) { fail(error, "resume completed step is invalid"); return 1; }
        if (resume_plan.cursor >= dataset.items.size()) { fail(error, "resume order cursor is out of range"); return 1; }
        if (!sampler.import_rng_state(resume_plan.sampler)) { fail(error, "resume sampler state is invalid"); return 1; }
        if (resume_binding.planner_frozen_at >= 0 && config.nar_extra_steps <= 0 && !config.unfreeze_planner) { fail(error, "this run's planner is frozen; resume it with --nar-extra-steps (or --unfreeze-planner)"); return 1; }
        if (config.freeze_planner_now && config.nar_extra_steps <= 0) { fail(error, "--freeze-planner-now needs --nar-extra-steps (the decoder's budget after the freeze)"); return 1; }
        order = resume_plan.order; cursor = resume_plan.cursor; completed = resume_plan.completed;
    } else if (config.freeze_planner_now) {
        fail(error, "--freeze-planner-now needs --resume: it freezes the checkpoint's planner"); return 1;
    } else { sampler.rng().shuffle(order); }
    if(cursor_weight>0) for(const auto & item:dataset.items) {
        const auto & binding=item.prompt.cursor;
        // present && !enabled && !instrumental = lyrical but untimed: allowed,
        // it simply contributes no timing loss.
        if(!binding.present || (!binding.instrumental && binding.enabled &&
           (binding.L<=0 || binding.full_frame_ranges.size()!=item.song.semantic_tokens.size() ||
            binding.off_frame_ranges.size()!=item.song.semantic_tokens.size()))) {
            if(error)*error="lyric timing requires valid alignment for track "+item.id+"; prepare with alignment or use --cursor-weight 0";
            return 1;
        }
    }
    if (config.pause_at < 0 || config.pause_at > config.steps ||
        (config.pause_at > 0 && config.pause_at <= completed)) {
        fail(error, "pause boundary must be after the resumed step and within total steps"); return 1;
    }
#ifdef _WIN32
    if (_putenv_s("NVIDIA_TF32_OVERRIDE", "0") != 0) { fail(error, "cannot set NVIDIA_TF32_OVERRIDE=0"); return 1; }
#else
    if (setenv("NVIDIA_TF32_OVERRIDE", "0", 1) != 0) { fail(error, "cannot set NVIDIA_TF32_OVERRIDE=0"); return 1; }
#endif
    yue2_aitk_clear_cancel(); yue2_aitk_install_sigint_handler();
    std::error_code ec;
    if (!std::filesystem::create_directory(std::filesystem::u8path(config.output), ec) || ec) { fail(error, "cannot create new training output directory"); return 1; }
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_name(("CUDA" + std::to_string(config.cuda_index)).c_str());
    const char * reg_name = device ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)) : nullptr;
    if (!reg_name || std::strncmp(reg_name, "CUDA", 4) != 0) { fail(error, "requested device is not a CUDA backend"); return 1; }
    struct Backend { ggml_backend_t value = nullptr; ~Backend() { if (value) ggml_backend_free(value); } } backend{device ? ggml_backend_dev_init(device, nullptr) : nullptr};
    if (!backend.value) { fail(error, "requested CUDA backend is unavailable"); return 1; }
    std::fprintf(stderr, "[yue2-aitk] training device %s via %s; optimizer scheduler uses CUDA + CPU (no Vulkan compute)\n",
                 ggml_backend_dev_name(device), reg_name);
    using AttentionPrecision = const char * (*)(int);
    const auto attention_precision = reinterpret_cast<AttentionPrecision>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_fattn_train_last_prec"));
    try {
        event("load"); Yue2AitkModel model; Yue2AitkTrainState state;
        if (!model.load(config.checkpoint.c_str(), backend.value, yue2_aitk_load_embedding_bf16, error) ||
            !state.initialize(backend.value, static_cast<uint32_t>(config.seed), error,cursor_weight>0,config.rank,config.alpha,
                              lokr ? config.lokr_dim : 0, lokr ? config.lokr_factor : 0)) return 1;
        if (lokr) std::fprintf(stderr, "[yue2-aitk] adapter lokr: dim %d factor %d alpha %.4g, %zu trainable parameters\n",
                               config.lokr_dim, config.lokr_factor, (double) config.alpha, state.parameter_count());
        // Decoder drift probes: 3 songs spread across the dataset x 3 noise
        // levels, one fixed crop and noise per song (its own sampler, so the
        // training RNG stream is untouched). The reference is the decoder at
        // its seed-derived INITIAL adapters, taken here, before any resume
        // restore: the same code path as every later reading, so the adapted
        // path's own numerics cancel (the adapter-free base path read a
        // ~3e-4 floor against it). Rebuilt identically on resume.
        // Reconstruction (nar_recon): the same probes read against the REAL
        // latent instead of the base decoder. One-step denoise, x0_hat =
        // x_t - t * v_hat (x_t = (1-t) x0 + t eps, v = eps - x0), relative
        // squared error to x0. "How well does the decoder reproduce this
        // album's sound", which is where likeness lives; expected to fall and
        // then plateau, unlike drift, which saturates in the first steps.
        struct DriftProbe { yue2_aitk::Batch batch; std::vector<std::vector<float>> noisy, base, clean; std::vector<float> timestep, t01; };
        std::vector<DriftProbe> probes;
        const auto measure_drift = [&](const Yue2AitkExpertAdapters * nar, bool fill_base, double * drift, double * recon = nullptr) -> bool {
            double sum = 0.0, rsum = 0.0; size_t count = 0;
            for (auto & probe : probes) {
                Yue2AitkPrefixHost prefix;
                if (!yue2_aitk_joint::nar_probe_prefix(backend.value, model, probe.batch, &prefix, error)) return false;
                for (size_t k = 0; k < probe.noisy.size(); ++k) {
                    std::vector<float> prediction;
                    if (!yue2_aitk_joint::nar_probe_predict(backend.value, model, nar, prefix, probe.noisy[k], probe.timestep[k], &prediction, error)) return false;
                    if (fill_base) { probe.base.push_back(std::move(prediction)); continue; }
                    const auto & b = probe.base[k];
                    double num = 0.0, den = 0.0;
                    for (size_t i = 0; i < b.size(); ++i) { const double d = double(prediction[i]) - b[i]; num += d * d; den += double(b[i]) * b[i]; }
                    sum += den > 0.0 ? num / den : 0.0; ++count;
                    const auto & x0 = probe.clean[k]; const auto & xt = probe.noisy[k]; const double t = probe.t01[k];
                    double rnum = 0.0, rden = 0.0;
                    for (size_t i = 0; i < x0.size(); ++i) { const double hat = double(xt[i]) - t * double(prediction[i]); const double d = hat - x0[i]; rnum += d * d; rden += double(x0[i]) * x0[i]; }
                    rsum += rden > 0.0 ? rnum / rden : 0.0;
                }
            }
            if (drift) *drift = count ? sum / double(count) : 0.0;
            if (recon) *recon = count ? rsum / double(count) : 0.0;
            return true;
        };
        if (config.nar_drift) {
            const size_t n = dataset.items.size();
            const float levels[3] = {250.0f, 500.0f, 750.0f};
            for (size_t s = 0; s < std::min<size_t>(3, n); ++s) {
                const auto & item = dataset.items[s * n / std::min<size_t>(3, n)];
                DriftProbe probe;
                for (float level : levels) {
                    // Same seed per song: the crop and noise repeat, only the level moves.
                    yue2_aitk::Yue2NativeSampler probe_sampler(config.seed * 1000003ull + 7919ull * (s + 1));
                    auto sampled = probe_sampler.sample(item.song, item.prompt, 1500, std::vector<float>{level}, 0.0f, 0, 0, 0.0f);
                    if (probe.noisy.empty()) probe.batch = sampled.batch;
                    probe.noisy.push_back(std::move(sampled.noisy_bf16)); probe.timestep.push_back(sampled.timestep_bf16);
                    probe.clean.push_back(std::move(sampled.clean_f32)); probe.t01.push_back(sampled.timestep_bf16);
                }
                probes.push_back(std::move(probe));
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (!measure_drift(&state.nar_adapters(), true, nullptr)) return 1;
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // A second read of the untouched adapters is the meter's floor:
            // anything but ~0 means the forward is not deterministic.
            double floor = 0.0;
            if (!measure_drift(&state.nar_adapters(), false, &floor)) return 1;
            std::fprintf(stderr, "[yue2-aitk] decoder drift: %zu probes, %.1f s per reading, floor %.3g\n", probes.size() * 3, seconds, floor);
        }
        const bool use_lm = (config.optimizer != "adamw");
        struct LmOptimHolder {
            LmOptim opt;
            ggml_backend_sched_t osched = nullptr;
            ggml_backend_t cpu_backend = nullptr;
            ggml_context * scal_ctx = nullptr;
            ggml_backend_buffer_t scal_buf = nullptr;
            ~LmOptimHolder() {
                if (osched) ggml_backend_sched_free(osched);
                lm_optim_free(&opt);
                if (cpu_backend) ggml_backend_free(cpu_backend);
                if (scal_buf) ggml_backend_buffer_free(scal_buf);
                if (scal_ctx) ggml_free(scal_ctx);
            }
        };
        std::unique_ptr<LmOptimHolder> lm;
        std::unique_ptr<yue2_aitk::Optimizer> adamw;
        {
            const auto named = state.named_tensors();
            if (use_lm) {
                auto holder = std::make_unique<LmOptimHolder>();
                LmOptim & o = holder->opt;
                o.optimizer = config.optimizer == "adamw-lm" ? "adamw" : config.optimizer;
                o.cautious  = config.cautious;
                // LmOptim's built-in schedule is neutralised: {floor 1, total 1,
                // warmup 0} makes lm_lr_lambda identically 1.0, and base_lr is
                // set per step below, exactly as the Legacy YuE2 trainers do.
                o.lr_floor = 1.0f; o.total_steps = 1; o.warmup_steps = 0;
                o.weight_decay = config.weight_decay;
                o.grad_clip = 0.0f;  // state.clip_gradients(1.0) is the only clipper
                o.adam_beta1 = 0.9f; o.adam_beta2 = 0.999f;
                o.prodigy_d0 = config.prodigy_d0;
                o.muon.lr_scale = config.muon_lr_scale;
                o.muon.ns_steps = config.muon_ns_steps;
                o.base_lr = config.optimizer == "prodigy" ? 1.0f : config.lr;
                ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
                holder->scal_ctx = ggml_init(sp);
                if (!holder->scal_ctx) { fail(error, "optimizer scalar context failed"); return 1; }
                o.t_lossgrad = ggml_new_tensor_1d(holder->scal_ctx, GGML_TYPE_F32, 1);
                o.t_adamw = ggml_new_tensor_1d(holder->scal_ctx, GGML_TYPE_F32, 7);
                o.t_clip = ggml_new_tensor_1d(holder->scal_ctx, GGML_TYPE_F32, 1);
                o.t_eps = ggml_new_tensor_1d(holder->scal_ctx, GGML_TYPE_F32, 1);
                o.t_gnorm2 = ggml_new_tensor_1d(holder->scal_ctx, GGML_TYPE_F32, 1);
                if (!o.t_lossgrad || !o.t_adamw || !o.t_clip || !o.t_eps || !o.t_gnorm2) { fail(error, "optimizer scalar allocation failed"); return 1; }
                ggml_set_name(o.t_lossgrad, "lossgrad"); ggml_set_name(o.t_adamw, "adamw_params"); ggml_set_name(o.t_clip, "grad_clip"); ggml_set_name(o.t_eps, "eps"); ggml_set_name(o.t_gnorm2, "gnorm2");
                holder->scal_buf = ggml_backend_alloc_ctx_tensors(holder->scal_ctx, backend.value);
                if (!holder->scal_buf) { fail(error, "optimizer scalar buffer allocation failed"); return 1; }
                const float one = 1.0f, eps = 1e-6f;
                ggml_backend_tensor_set(o.t_lossgrad, &one, 0, sizeof(one));
                ggml_backend_tensor_set(o.t_clip, &one, 0, sizeof(one));
                ggml_backend_tensor_set(o.t_eps, &eps, 0, sizeof(eps));
                std::vector<ggml_tensor *> params;
                params.reserve(named.size());
                for (const auto & p : named) params.push_back(p.parameter);
                std::string lm_err;
                if (!lm_optim_init(&o, params, backend.value, &lm_err)) { fail(error, ("optimizer init: " + lm_err).c_str()); return 1; }
                // Same split the AdamW path makes below: the planner's adapters
                // are the "text_encoders." slots. lm_optim_init is what sizes
                // lr_mul, so this cannot move above it.
                if (config.planner_lr_scale != 1.0f || config.nar_lr_scale != 1.0f) {
                    size_t scaled = 0;
                    for (const auto & p : named) {
                        const bool planner = p.name.rfind("text_encoders.", 0) == 0;
                        const float mul = planner ? config.planner_lr_scale : config.nar_lr_scale;
                        if (mul == 1.0f) continue;
                        if (!lm_optim_set_lr_mul(&o, p.parameter, mul)) {
                            fail(error, "lr scale: an adapter parameter is not registered with the optimizer"); return 1;
                        }
                        ++scaled;
                    }
                    std::fprintf(stderr, "[yue2-aitk] planner lr x%.3g, decoder lr x%.3g, %zu of %zu parameters scaled (%s)\n",
                                 (double) config.planner_lr_scale, (double) config.nar_lr_scale, scaled, params.size(), config.optimizer.c_str());
                }
                // GGML's scheduler requires a CPU backend in its final slot,
                // even when all optimizer tensors are CUDA-resident.
                holder->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
                if (!holder->cpu_backend) { fail(error, "optimizer CPU scheduler fallback is unavailable"); return 1; }
                ggml_backend_t lm_backends[2] = { backend.value, holder->cpu_backend };
                ggml_backend_buffer_type_t lm_bufts[2] = {
                    ggml_backend_get_default_buffer_type(backend.value),
                    ggml_backend_get_default_buffer_type(holder->cpu_backend),
                };
                // Same headroom lm_optim_step gives its own graph cap when a modifier is
                // on: the unfused AdamW form is ~30 nodes per parameter, and the
                // scheduler's hash set must hold every node and leaf of that graph.
                const int sched_nodes = o.est_nodes + (o.cautious ? (int) params.size() * 48 + (int) o.muon_buckets.size() * 8 : 0);
                holder->osched = ggml_backend_sched_new(lm_backends, lm_bufts, 2, std::max(16384, sched_nodes), false, true);
                if (!holder->osched) { fail(error, "optimizer scheduler allocation failed"); return 1; }
                lm = std::move(holder);
                std::fprintf(stderr, "[yue2-aitk] optimizer %s%s over %zu parameters (%d on Muon)\n", config.optimizer.c_str(), config.cautious ? " (cautious)" : "", params.size(), lm->opt.n_muon);
            } else {
                std::vector<yue2_aitk::ParameterSpec> specs;
                specs.reserve(named.size());
                // The planner's adapters are the "text_encoders." slots
                // (yue2-aitk-train-state.h build_slots); everything else is the
                // decoder and trains at --lr itself.
                for (const auto & p : named) {
                    const bool planner = p.name.rfind("text_encoders.", 0) == 0;
                    specs.push_back({p.name, p.parameter, p.gradient, planner ? config.planner_lr_scale : config.nar_lr_scale});
                }
                adamw = std::make_unique<yue2_aitk::Optimizer>(backend.value, config.cuda_index, std::move(specs));
                if (config.planner_lr_scale != 1.0f || config.nar_lr_scale != 1.0f)
                    std::fprintf(stderr, "[yue2-aitk] planner lr %.3g (x%.3g), decoder lr %.3g (x%.3g)\n",
                                 (double) config.lr * config.planner_lr_scale, (double) config.planner_lr_scale,
                                 (double) config.lr * config.nar_lr_scale, (double) config.nar_lr_scale);
            }
        }
        if (!config.resume.empty()) {
            if (use_lm) {
                if (resume_plan.record.version != 2) { fail(error, ("resume record format is not compatible with --optimizer " + config.optimizer).c_str()); return 1; }
                const auto & snap = resume_plan.record.state;
                const auto named = state.named_tensors();
                if (snap.names.size() != named.size() || snap.step != resume_plan.completed) { fail(error, "resume record does not match the run shape"); return 1; }
                const bool want_v = (config.optimizer == "prodigy");
                const bool want_s = (config.optimizer == "prodigy");
                auto slot_valid = [](const std::vector<float> & fp32, const std::vector<uint8_t> & u8, const std::vector<float> & absmax, size_t n, bool required, bool * present_out) {
                    const bool f = !fp32.empty(), q = !u8.empty();
                    if (f && q) return false;
                    *present_out = f || q;
                    if (!*present_out) return !required;
                    if (f) {
                        if (n >= 4096 || fp32.size() != n) return false;
                        for (float v : fp32) if (!std::isfinite(v)) return false;
                        return true;
                    }
                    if (n < 4096 || u8.size() != n || absmax.size() != (n + 255) / 256) return false;
                    for (float v : absmax) if (!std::isfinite(v) || v < 0) return false;
                    return true;
                };
                for (size_t i = 0; i < named.size(); ++i) {
                    const size_t n = (size_t) ggml_nelements(named[i].parameter);
                    if (snap.names[i] != named[i].name || snap.elements[i] != n || snap.parameters[i].size() != n) { fail(error, "resume record shape mismatch"); return 1; }
                    for (float v : snap.parameters[i]) if (!std::isfinite(v)) { fail(error, "resume record has non-finite parameters"); return 1; }
                    bool p1 = false, p2 = false, p3 = false, p4 = false;
                    if (!slot_valid(snap.state1_fp32[i], snap.state1_u8[i], snap.absmax1[i], n, true, &p1) ||
                        !slot_valid(snap.state2_fp32[i], snap.state2_u8[i], snap.absmax2[i], n, want_v, &p2) ||
                        !slot_valid(snap.state3_fp32[i], snap.state3_u8[i], snap.absmax3[i], n, want_s, &p3) ||
                        !slot_valid(snap.state4_fp32[i], snap.state4_u8[i], snap.absmax4[i], n, want_s, &p4)) {
                        fail(error, "resume optimizer state mismatch"); return 1;
                    }
                }
                const LmOptim & o = lm->opt;
                ggml_backend_synchronize(backend.value);
                auto store_state = [&](ggml_tensor * t, const std::vector<float> & fp32, const std::vector<uint8_t> & u8, const std::vector<float> & absmax) -> bool {
                    if (!t) return true;
                    const size_t n = (size_t) ggml_nelements(t);
                    std::vector<float> values;
                    if (!u8.empty()) {
                        if (!yue2_aitk::resume_detail::dequantize_f32(u8, absmax, &values)) return false;
                    } else {
                        values = fp32;
                    }
                    if (values.size() != n) return false;
                    ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
                    return true;
                };
                for (size_t i = 0; i < named.size(); ++i) {
                    const size_t n = (size_t) ggml_nelements(named[i].parameter);
                    ggml_backend_tensor_set(named[i].parameter, snap.parameters[i].data(), 0, n * sizeof(float));
                    if (!store_state(o.mom_m[i], snap.state1_fp32[i], snap.state1_u8[i], snap.absmax1[i]) ||
                        !store_state(o.mom_v[i], snap.state2_fp32[i], snap.state2_u8[i], snap.absmax2[i]) ||
                        !store_state(o.pg_s.empty() ? nullptr : o.pg_s[i], snap.state3_fp32[i], snap.state3_u8[i], snap.absmax3[i]) ||
                        !store_state(o.pg_x0.empty() ? nullptr : o.pg_x0[i], snap.state4_fp32[i], snap.state4_u8[i], snap.absmax4[i])) {
                        fail(error, "resume optimizer state restore failed"); return 1;
                    }
                }
                lm->opt.opt_step = resume_plan.completed;
                lm->opt.opt_iter = resume_binding.opt_iter;
                if (config.optimizer == "prodigy") { lm->opt.prodigy_d = resume_binding.prodigy_d; lm->opt.prodigy_r = resume_binding.prodigy_r; }
                std::fprintf(stderr, "[yue2-aitk] resumed %s optimizer at step %d (iteration %d)\n", config.optimizer.c_str(), lm->opt.opt_step, lm->opt.opt_iter);
                resume_plan.record = {};
            } else {
                if (resume_plan.record.version != 1) { fail(error, "resume record format is not compatible with --optimizer adamw"); return 1; }
                try { adamw->restore(resume_plan.record.state); }
                catch (const std::exception & exception) { if (error) *error = exception.what(); return 1; }
                if (adamw->step() != resume_plan.completed) { fail(error, "resume optimizer step does not match completed step"); return 1; }
                resume_plan.record = {};
            }
        }
        // A frozen record resumes frozen: its planner weights, just restored,
        // are the held ones.
        int planner_frozen_at = resume_binding.planner_frozen_at;
        if (planner_frozen_at >= 0 && config.unfreeze_planner) {
            // Planner refinement: both halves train on from here; the KL
            // ceiling (--target-kl) is the new stop.
            planner_frozen_at = -1;
            std::fprintf(stderr, "[yue2-aitk] --unfreeze-planner: planner trains on from step %d\n", completed);
        }
        if (planner_frozen_at < 0 && config.freeze_planner_now) {
            // A stop decided outside the trainer: this checkpoint's planner is
            // the one kept. Same bookkeeping as the KL freeze.
            planner_frozen_at = completed;
            std::fprintf(stderr, "[yue2-aitk] --freeze-planner-now: planner frozen at the resumed step %d\n", completed);
        }
        if (planner_frozen_at >= 0) {
            state.freeze_planner();
            std::fprintf(stderr, "[yue2-aitk] planner frozen since step %d; decoder-only until step %d\n", planner_frozen_at, std::min(config.steps, planner_frozen_at + config.nar_extra_steps));
        }
        // The run's last step: --steps, or the decoder budget once frozen.
        const auto end_step = [&]() { return planner_frozen_at >= 0 ? std::min(config.steps, planner_frozen_at + config.nar_extra_steps) : config.steps; };
        if (config.meter_only) {
            if (config.resume.empty()) { fail(error, "--meter-only needs --resume"); return 1; }
            double drift = 0.0, recon = 0.0;
            if (!measure_drift(&state.nar_adapters(), false, &drift, &recon)) return 1;
            double kl = -1.0;
            if (!resume_binding.kl_history.empty()) { const auto & h = resume_binding.kl_history; const size_t n = std::min<size_t>(20, h.size()); kl = 0.0; for (size_t i = h.size() - n; i < h.size(); ++i) kl += h[i]; kl /= double(n); }
            std::ostringstream meters;
            meters << std::setprecision(9) << "{\"stage\":\"meters\",\"step\":" << completed << ",\"nar_drift\":" << drift << ",\"nar_recon\":" << recon;
            if (kl >= 0.0) meters << ",\"ar_kl_mean20\":" << kl;
            meters << ",\"planner_frozen\":" << (planner_frozen_at >= 0 ? "true" : "false") << "}\n";
            std::ofstream(std::filesystem::u8path(config.resume).parent_path() / "meters.json", std::ios::binary) << meters.str();
            std::cout << meters.str() << std::flush;
            return 0;
        }
        std::ofstream jsonl(std::filesystem::u8path(config.output) / "train.jsonl", std::ios::binary);
        if (!jsonl) { fail(error, "cannot create training JSONL"); return 1; }
        // The planner's KL at the moment a checkpoint is written, for meters.json.
        std::vector<double> kl_recent = resume_binding.kl_history;
        // An unfrozen planner puts the joint gradient norm (~10x the decoder's
        // alone) against a decoder-only baseline: every step would read as a
        // spike. Restart the guard's window, as the freeze does.
        // Only the FIRST unfreeze (the record still says frozen) restarts the
        // window; later segments keep theirs, or the guard is unarmed for 20
        // steps after every rung (a 19.9 spike at step 527 went through).
        const bool first_unfreeze_guard = config.unfreeze_planner && resume_binding.planner_frozen_at >= 0;
        std::vector<double> gnorm_window = first_unfreeze_guard ? std::vector<double>{} : resume_binding.gnorm_history;
        std::vector<double> spike_steps = first_unfreeze_guard ? std::vector<double>{} : resume_binding.spike_steps;
        std::vector<double> recon_history = config.recon_reset ? std::vector<double>{} : resume_binding.recon_history;
        int last_saved = -1;
        // Declared before save_checkpoint, which writes them into the resume
        // record; seeded from it so a resumed segment can stop at once.
        std::vector<double> loss_window = config.target_loss > 0.0f ? resume_binding.loss_history : std::vector<double>{};
        // The first unfreeze starts the KL window over: the record's window is
        // the planner's pre-freeze history, and a trend fitted across that
        // gap and the new joint steps extrapolates far past the real KL (a
        // 2.38 reading at KL ~1.4 ended a refinement at once).
        const bool first_unfreeze = config.unfreeze_planner && resume_binding.planner_frozen_at >= 0;
        int unfrozen_at = first_unfreeze ? completed : resume_binding.unfrozen_at;
        double rung_lr_mult = resume_binding.rung_lr_mult;
        std::vector<double> kl_window = config.target_kl > 0.0f && !first_unfreeze ? resume_binding.kl_history : std::vector<double>{};
        // KL rungs (--kl-checkpoint-every): the next reading that earns a
        // checkpoint, set from the first reading; the reading itself goes into
        // meters.json as kl_reading.
        double next_kl_mark = resume_binding.kl_mark_last > 0.0 ? resume_binding.kl_mark_last + config.kl_checkpoint_every : -1.0, last_kl_reading = -1.0, kl_mark_last = resume_binding.kl_mark_last;
        bool rung_checkpoint = false;  // the checkpoint being written is a KL rung
        // After the first unfreeze the window starts empty; rather than 30
        // blind steps (the KL climbed 0.92 -> 1.64 in them and skipped four
        // rungs), read the plain mean of what is there from 5 fresh steps.
        const bool partial_readings = first_unfreeze;
        auto save_checkpoint = [&](int step) -> bool {
            if (step == last_saved) return true;
            event("checkpoint_stage", step);
            const auto final_dir = std::filesystem::u8path(config.output) / ("checkpoint-step" + std::to_string(step));
            const auto temp_dir = std::filesystem::u8path(config.output) / (".checkpoint-step" + std::to_string(step) + ".tmp");
            std::error_code save_ec; if (std::filesystem::exists(final_dir, save_ec) || std::filesystem::exists(temp_dir, save_ec) || !std::filesystem::create_directory(temp_dir, save_ec) || save_ec) return false;
            const auto adapter = temp_dir / "adapter.safetensors"; const auto resume = temp_dir / "optimizer.resume";
            if (config.nar_drift) {
                double drift = 0.0, recon = 0.0;
                if (!measure_drift(&state.nar_adapters(), false, &drift, &recon)) return false;
                double kl = -1.0;
                if (!kl_recent.empty()) { kl = 0.0; for (double v : kl_recent) kl += v; kl /= double(kl_recent.size()); }
                std::ostringstream meters;
                meters << std::setprecision(9) << "{\"stage\":\"meters\",\"step\":" << step << ",\"nar_drift\":" << drift << ",\"nar_recon\":" << recon;
                if (kl >= 0.0) meters << ",\"ar_kl_mean20\":" << kl;
                if (last_kl_reading >= 0.0) meters << ",\"kl_reading\":" << last_kl_reading;
                if (rung_checkpoint) meters << ",\"kl_rung\":true";
                meters << ",\"planner_frozen\":" << (planner_frozen_at >= 0 ? "true" : "false") << "}\n";
                std::ofstream(temp_dir / "meters.json", std::ios::binary) << meters.str();
                jsonl << meters.str(); jsonl.flush();
                std::cout << meters.str() << std::flush;
                if (planner_frozen_at >= 0) recon_history.push_back(recon);
            }
            const std::string sampler_state = sampler.export_rng_state();
            if (sampler_state.empty()) return false;
            const std::string metadata = make_resume_meta(checkpoint_hash.hex(), dataset_hash.hex(), source_hash.hex(), config.seed, config.cuda_index, step, cursor, order, sampler_state, cursor_weight,
                config, use_lm ? lm->opt.opt_iter : 0, use_lm ? lm->opt.prodigy_d : 0.0, use_lm ? lm->opt.prodigy_r : 0.0, kl_window, loss_window, planner_frozen_at, gnorm_window, spike_steps, recon_history, kl_mark_last, unfrozen_at, rung_lr_mult);
            if (metadata.empty()) return false;
            if (!state.export_snapshot(adapter.u8string().c_str(), step, error, (temp_dir / "native-ar.safetensors").u8string().c_str(), (temp_dir / "native-nar.safetensors").u8string().c_str(), dataset.trigger, config.caption_dropout)) return false;
            if (use_lm) {
                yue2_aitk::HostStateSnapshot snap; snap.step = step;
                const auto named = state.named_tensors();
                snap.names.resize(named.size()); snap.elements.resize(named.size()); snap.parameters.resize(named.size());
                snap.state1_fp32.resize(named.size()); snap.state2_fp32.resize(named.size()); snap.state3_fp32.resize(named.size()); snap.state4_fp32.resize(named.size());
                snap.state1_u8.resize(named.size()); snap.state2_u8.resize(named.size()); snap.state3_u8.resize(named.size()); snap.state4_u8.resize(named.size());
                snap.absmax1.resize(named.size()); snap.absmax2.resize(named.size()); snap.absmax3.resize(named.size()); snap.absmax4.resize(named.size());
                const LmOptim & o = lm->opt;
                for (size_t i = 0; i < named.size(); ++i) {
                    const size_t n = (size_t) ggml_nelements(named[i].parameter);
                    snap.names[i] = named[i].name; snap.elements[i] = n;
                    snap.parameters[i].resize(n);
                    ggml_backend_synchronize(backend.value);
                    ggml_backend_tensor_get(named[i].parameter, snap.parameters[i].data(), 0, n * sizeof(float));
                    auto store = [&](int slot, ggml_tensor * t) -> bool {
                        if (!t) return true;
                        std::vector<float> values(n);
                        ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
                        ggml_backend_synchronize(backend.value);
                        for (float v : values) if (!std::isfinite(v)) return false;
                        if (n >= 4096) {
                            std::vector<uint8_t> q; std::vector<float> am;
                            if (!yue2_aitk::resume_detail::quantize_f32(values, &q, &am)) return false;
                            switch (slot) {
                                case 1: snap.state1_u8[i] = std::move(q); snap.absmax1[i] = std::move(am); break;
                                case 2: snap.state2_u8[i] = std::move(q); snap.absmax2[i] = std::move(am); break;
                                case 3: snap.state3_u8[i] = std::move(q); snap.absmax3[i] = std::move(am); break;
                                default: snap.state4_u8[i] = std::move(q); snap.absmax4[i] = std::move(am); break;
                            }
                        } else {
                            switch (slot) {
                                case 1: snap.state1_fp32[i] = std::move(values); break;
                                case 2: snap.state2_fp32[i] = std::move(values); break;
                                case 3: snap.state3_fp32[i] = std::move(values); break;
                                default: snap.state4_fp32[i] = std::move(values); break;
                            }
                        }
                        return true;
                    };
                    if (!store(1, o.mom_m[i]) || !store(2, o.mom_v[i]) || !store(3, o.pg_s.empty() ? nullptr : o.pg_s[i]) || !store(4, o.pg_x0.empty() ? nullptr : o.pg_x0[i])) return false;
                }
                if (!yue2_aitk::yue2_aitk_write_resume_v2(resume.u8string().c_str(), snap, metadata)) return false;
            } else {
                if (!yue2_aitk::yue2_aitk_write_resume(resume.u8string().c_str(), adamw->capture(), metadata)) return false;
            }
            std::filesystem::rename(temp_dir, final_dir, save_ec);
            if (!save_ec) { last_saved=step; event("checkpoint", step); }
            return !save_ec;
        };
        while (completed < end_step()) {
            if (yue2_aitk_cancel_requested()) {
                if (completed > 0 && !save_checkpoint(completed)) { fail(error, "cancel checkpoint publication failed"); return 1; }
                event("cancelled", completed); return 130;
            }
            const std::vector<float> schedule = sampler.sigmoid_schedule(1000);
            const auto & item = dataset.items[order[cursor]];
            auto sampled = sampler.sample(item.song, item.prompt, 1500, schedule, config.abc_dropout, 0, 999, config.caption_dropout);
            yue2_aitk_joint::Input input; input.batch = &sampled.batch; input.noisy_latents = sampled.noisy_bf16;
            input.flow_target = sampled.target_f32; input.timestep = sampled.timestep_bf16;
            input.kl_weight = config.kl_weight;
            // AdamW: constant rate after an optional linear warmup. No cosine
            // here on purpose — the LmOptim branch below has always decayed and
            // this branch has always been flat; keeping it flat means a run
            // that never passed --warmup is byte-identical to before.
            {
                double lr = (double) config.lr;
                if (config.warmup > 0 && completed < config.warmup) lr *= (double)(completed + 1) / (double)config.warmup;
                input.adamw_lr = (float) lr;
                input.adamw_weight_decay = config.weight_decay;
                // Armed once 20 applied norms are in the window.
                if (config.spike_factor > 0.0f && gnorm_window.size() >= 20) {
                    std::vector<double> sorted = gnorm_window; std::sort(sorted.begin(), sorted.end());
                    input.skip_above = (double) config.spike_factor * sorted[sorted.size() / 2];
                }
            }
            if(cursor_weight>0 && !item.prompt.cursor.instrumental && item.prompt.cursor.enabled) {
                const auto & binding=item.prompt.cursor;
                input.cursor_weight=cursor_weight;
                // Four prefixes, four lyric-start columns: the head length is
                // what moves j0, and the trigger-only heads are shorter.
                if (sampled.caption_retained) input.lyric_start=sampled.abc_retained?binding.j0_full:binding.j0_off;
                else input.lyric_start=sampled.abc_retained?binding.j0_full_nocap:binding.j0_off_nocap;
                input.lyric_count=binding.L;
                const auto & ranges=sampled.caption_retained
                    ? (sampled.abc_retained?binding.full_frame_ranges:binding.off_frame_ranges)
                    : (sampled.abc_retained?binding.full_nocap_frame_ranges:binding.off_nocap_frame_ranges);
                input.cursor_frames.reserve(ranges.size());
                for(const auto & range:ranges) input.cursor_frames.emplace_back(range.first,range.last);
            }
            yue2_aitk_joint::Metrics metrics;
            using Clock = std::chrono::steady_clock;
            const auto step_start = Clock::now();
            auto stage_start = step_start;
            std::string previous_stage;
            std::vector<std::pair<std::string,double>> stage_times;
            const auto finish_stage = [&]() {
                const auto now = Clock::now();
                if (!previous_stage.empty()) stage_times.emplace_back(previous_stage,
                    std::chrono::duration<double,std::milli>(now-stage_start).count());
                stage_start = now;
            };
            if (use_lm) {
                // Linear warmup, then cosine to zero. Prodigy's base_lr is the
                // schedule multiplier on its own d (gamma), as in the Legacy
                // trainers; with warmup 0 the cosine starts at full value.
                double lr = (config.optimizer == "prodigy") ? 1.0 : (double) config.lr;
                if (config.warmup > 0 && completed < config.warmup) {
                    lr *= (double)(completed + 1) / (double)config.warmup;
                } else {
                    const double denom = (double)std::max<int32_t>(1, config.steps - config.warmup);
                    const double ratio = std::min(1.0, (double)(completed - config.warmup) / denom);
                    lr *= 0.5 * (1.0 + std::cos(3.14159265358979 * ratio));
                }
                if (unfrozen_at >= 0) {
                    // Refinement pacing: warm up from the unfreeze, then the
                    // adaptive multiplier (halved whenever a rung was jumped).
                    const int since = completed - unfrozen_at;
                    if (config.refine_warmup > 0 && since < config.refine_warmup) lr *= (double)(since + 1) / (double) config.refine_warmup;
                    lr *= rung_lr_mult;
                }
                lm->opt.base_lr = (float)lr;
            }
            if (!yue2_aitk_joint::run(backend.value, model, state, use_lm ? nullptr : adamw.get(), input, &metrics, error,
                [&](const char * stage) { finish_stage(); previous_stage=stage; event(stage, completed + 1); },
                use_lm ? &lm->opt : nullptr, use_lm ? lm->osched : nullptr)) return 1;
            ggml_backend_synchronize(backend.value);
            finish_stage();
            const double step_ms=std::chrono::duration<double,std::milli>(Clock::now()-step_start).count();
            ++completed;
            if (metrics.step != completed) { fail(error, "optimizer update count mismatch"); return 1; }
            if (++cursor == order.size()) { sampler.rng().shuffle(order); cursor=0; }
            std::ostringstream line;
            line << std::setprecision(17) << "{\"stage\":\"joint\",\"step\":" << metrics.step;
            // Frozen steps have no planner objective, so no AR numbers: a
            // repeated or zero KL would read as a real measurement.
            if (planner_frozen_at >= 0) line << ",\"planner_frozen\":true";
            else line << ",\"ar_ce\":" << metrics.ar_ce << ",\"ar_kl\":" << metrics.ar_kl;
            line << ",\"nar_mse\":" << metrics.nar_mse << ",\"gradient_norm\":" << metrics.gradient_norm;
            if (metrics.skipped) line << ",\"spike_skipped\":true";
            if (planner_frozen_at < 0) line << ",\"cursor_ce\":" << metrics.cursor_ce << ",\"cursor_weight\":" << cursor_weight
                   << ",\"cursor_frames\":" << metrics.cursor_frames;
            line
                   << ",\"step_ms\":" << step_ms
                   << ",\"attention_forward\":\"" << (attention_precision?attention_precision(0):"unknown")
                   << "\",\"attention_backward\":\"" << (attention_precision?attention_precision(1):"unknown")
                   << "\",\"stage_ms\":{";
            for (size_t i=0;i<stage_times.size();++i) {
                if(i) line << ',';
                line << '"' << stage_times[i].first << "\":" << stage_times[i].second;
            }
            line << "}}\n";
            jsonl << line.str();
            std::cout << line.str() << std::flush;
            if (!jsonl.flush()) { fail(error, "training JSONL write failed"); return 1; }
            if (config.spike_factor > 0.0f) {
                if (metrics.skipped) {
                    spike_steps.push_back(completed);
                    std::fprintf(stderr, "[yue2-aitk] step %d: gradient norm %.3g is a spike (limit %.3g); update skipped\n", completed, metrics.gradient_norm, input.skip_above);
                } else {
                    gnorm_window.push_back(metrics.gradient_norm);
                    if (gnorm_window.size() > 50) gnorm_window.erase(gnorm_window.begin());
                }
                while (!spike_steps.empty() && spike_steps.front() <= completed - config.spike_stop_window) spike_steps.erase(spike_steps.begin());
                if (config.spike_stop > 0 && (int) spike_steps.size() >= config.spike_stop) {
                    // Skipped updates never touched the weights, so this step's
                    // weights are the last pre-spike state.
                    std::fprintf(stderr, "[yue2-aitk] %zu spikes within %d steps: stopping at step %d\n", spike_steps.size(), config.spike_stop_window, completed);
                    if (!save_checkpoint(completed)) { fail(error, "spike-stop checkpoint publication failed"); return 1; }
                    event("spike_stop", completed);
                    event("target", completed);
                    event("done", completed);
                    return 0;
                }
            }
            if (planner_frozen_at < 0) {
                kl_recent.push_back(metrics.ar_kl);
                if (kl_recent.size() > 20) kl_recent.erase(kl_recent.begin());
            }
            if (config.target_loss > 0.0f && planner_frozen_at < 0) {
                // Same composite the server reports: AR CE + weighted KL + NAR
                // MSE + weighted cursor CE. The window must be full before a
                // stop, so the earliest stop is at step target_loss_window.
                const double composite = metrics.ar_ce + (double) config.kl_weight * metrics.ar_kl + metrics.nar_mse + metrics.cursor_ce * (double) cursor_weight;
                loss_window.push_back(composite);
                while (static_cast<int>(loss_window.size()) > config.target_loss_window) loss_window.erase(loss_window.begin());
                if (static_cast<int>(loss_window.size()) == config.target_loss_window) {
                    double sum = 0.0;
                    for (double v : loss_window) sum += v;
                    if (sum / (double) config.target_loss_window <= (double) config.target_loss) {
                        if (!save_checkpoint(completed)) { fail(error, "target checkpoint publication failed"); return 1; }
                        event("target", completed);
                        event("done", completed);
                        return 0;
                    }
                }
            }
            if (config.target_kl > 0.0f && planner_frozen_at < 0) {
                // Same window as the composite stop, on the planner's KL to
                // base alone. The stop is "at or above": the adapter has moved
                // as far from the base as the recipe allows.
                kl_window.push_back(metrics.ar_kl);
                const bool trend = config.target_kl_mode == "trend";
                const int keep = trend ? kKlTrendWindow : config.target_loss_window;
                while (static_cast<int>(kl_window.size()) > keep) kl_window.erase(kl_window.begin());
                const bool partial = partial_readings && static_cast<int>(kl_window.size()) < keep && kl_window.size() >= 5;
                if (static_cast<int>(kl_window.size()) == keep || partial) {
                    double sum = 0.0;
                    for (double v : kl_window) sum += v;
                    const double reading = partial ? sum / (double) kl_window.size() : trend ? trend_at_end(kl_window) : sum / (double) keep;
                    last_kl_reading = reading;
                    if (config.kl_checkpoint_every > 0.0f) {
                        const double every = config.kl_checkpoint_every;
                        if (next_kl_mark < 0.0) next_kl_mark = (std::floor(reading / every) + 1.0) * every;
                        if (reading >= next_kl_mark && reading < (double) config.target_kl) {
                            if (config.rung_adaptive_lr && reading >= next_kl_mark + every) {
                                rung_lr_mult = std::max(0.05, rung_lr_mult * 0.5);
                                std::fprintf(stderr, "[yue2-aitk] KL %.3f jumped past rung %.2f: rate multiplier now %.3g\n", reading, next_kl_mark, rung_lr_mult);
                            }
                            kl_mark_last = next_kl_mark;  // into this checkpoint's record
                            rung_checkpoint = true;
                            const bool saved_rung = save_checkpoint(completed);
                            rung_checkpoint = false;
                            if (!saved_rung) { fail(error, "KL-rung checkpoint publication failed"); return 1; }
                            event("kl_mark", completed);
                            std::fprintf(stderr, "[yue2-aitk] KL %.3f reached at step %d: rung checkpoint\n", reading, completed);
                            while (next_kl_mark <= reading) next_kl_mark += every;
                            if (config.pause_on_kl_mark && completed < end_step()) { event("paused", completed); return 0; }
                        }
                    }
                    if (reading >= (double) config.target_kl && config.nar_extra_steps > 0) {
                        // Freeze first, so the KL checkpoint's resume record
                        // already says frozen and resumes decoder-only.
                        planner_frozen_at = completed;
                        state.freeze_planner();
                        // The decoder's gradient norm alone is ~10x below the
                        // joint one, so the spike guard's baseline starts over.
                        gnorm_window.clear(); spike_steps.clear();
                        if (!save_checkpoint(completed)) { fail(error, "planner-freeze checkpoint publication failed"); return 1; }
                        event("planner_frozen", completed);
                        std::fprintf(stderr, "[yue2-aitk] planner reached KL %.4g at step %d: frozen; decoder-only until step %d\n", reading, completed, end_step());
                    } else if (reading >= (double) config.target_kl) {
                        if (!save_checkpoint(completed)) { fail(error, "target checkpoint publication failed"); return 1; }
                        event("target", completed);
                        event("done", completed);
                        return 0;
                    }
                }
            }
            if (completed % config.save_every == 0 || completed == end_step())
                if (!save_checkpoint(completed)) { fail(error, "checkpoint publication failed"); return 1; }
            // Decoder stop: the reconstruction meter has flattened over the
            // last recon_stop_window checkpoints of the frozen phase. The
            // checkpoint just written is the one kept.
            if (config.recon_stop > 0.0f && planner_frozen_at >= 0 && completed % config.save_every == 0 && completed < end_step()
                && recon_history.size() > (size_t) config.recon_stop_window) {
                const double before = recon_history[recon_history.size() - 1 - (size_t) config.recon_stop_window], now = recon_history.back();
                const double improvement = before > 0.0 ? (before - now) / before : 0.0;
                if (improvement < (double) config.recon_stop) {
                    std::fprintf(stderr, "[yue2-aitk] reconstruction %.4f -> %.4f over %d checkpoints (%.2f%%): decoder done, stopping at step %d\n", before, now, config.recon_stop_window, 100.0 * improvement, completed);
                    event("recon_stop", completed);
                    event("target", completed);
                    event("done", completed);
                    return 0;
                }
            }
            if (config.pause_at > 0 && completed >= config.pause_at && completed < end_step()) {
                if (!save_checkpoint(completed)) { fail(error, "pause checkpoint publication failed"); return 1; }
                event("paused", completed);
                return 0;
            }
        }
        if (completed > 0 && !save_checkpoint(completed)) { fail(error, "final checkpoint publication failed"); return 1; }
        // A decoder budget that ended before --steps is an early stop, and the
        // server validates the last checkpoint off this event.
        if (planner_frozen_at >= 0 && completed < config.steps) event("target", completed);
        event("done", completed); return yue2_aitk_cancel_requested() ? 130 : 0;
    } catch (const std::exception & exception) { if (error) *error = exception.what(); return 1; }
}

int yue2_aitk_run(const Config & config, std::string * error) {
    try { return run_impl(config, error); }
    catch (const std::exception & exception) {
        if (error) *error = exception.what();
        return 1;
    }
}
} // namespace yue2_aitk_runtime


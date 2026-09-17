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
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cctype>
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
bool parse_resume_meta(const std::string & text, const std::string & checkpoint, const std::string & dataset, const std::string & source,
                       uint64_t seed, int cuda_index, size_t item_count, ResumePlan * plan, std::string * error) {
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0); if (!doc) return fail(error, "resume metadata is invalid JSON");
    struct Guard { yyjson_doc * d; ~Guard() { yyjson_doc_free(d); } } guard{doc}; yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root) || !yue2_aitk::dataset_detail::unique_keys(root)) return fail(error, "resume metadata object is malformed");
    std::string recipe, cp, ds, sm, sampler;
    yyjson_val * vrecipe=yyjson_obj_get(root,"recipe"), *vcp=yyjson_obj_get(root,"checkpoint_sha256"), *vds=yyjson_obj_get(root,"dataset_sha256"), *vsm=yyjson_obj_get(root,"source_manifest_sha256"), *vseed=yyjson_obj_get(root,"seed"), *vdev=yyjson_obj_get(root,"cuda_index"), *vstep=yyjson_obj_get(root,"completed_step"), *vcursor=yyjson_obj_get(root,"order_cursor"), *vorder=yyjson_obj_get(root,"order"), *vsampler=yyjson_obj_get(root,"sampler_state");
    if (!str_field(vrecipe,&recipe) || recipe!="yue2-aitk-runtime-v1" || !str_field(vcp,&cp) || cp!=checkpoint || !str_field(vds,&ds) || ds!=dataset || !str_field(vsm,&sm) || sm!=source || !yyjson_is_uint(vseed) || yyjson_get_uint(vseed)!=seed || !yyjson_is_int(vdev) || yyjson_get_sint(vdev)!=cuda_index || !yyjson_is_int(vstep) || yyjson_get_sint(vstep)<0 || yyjson_get_sint(vstep)>INT_MAX || !yyjson_is_uint(vcursor) || !yyjson_is_arr(vorder) || yyjson_arr_size(vorder)!=item_count || !str_field(vsampler,&sampler)) return fail(error,"resume metadata binding mismatch");
    plan->completed=static_cast<int>(yyjson_get_sint(vstep)); plan->cursor=static_cast<size_t>(yyjson_get_uint(vcursor)); plan->sampler=std::move(sampler); plan->order.clear(); std::unordered_set<size_t> seen; size_t i=0,max=0; yyjson_val * x=nullptr; yyjson_arr_foreach(vorder,i,max,x) { if(!yyjson_is_uint(x) || yyjson_get_uint(x)>=item_count || !seen.insert(static_cast<size_t>(yyjson_get_uint(x))).second) return fail(error,"resume order is not a permutation"); plan->order.push_back(static_cast<size_t>(yyjson_get_uint(x))); }
    if (plan->cursor>item_count || plan->completed<0) return fail(error,"resume cursor is out of range"); return true;
}
std::string make_resume_meta(const std::string & cp, const std::string & ds, const std::string & sm, uint64_t seed, int device, int completed, size_t cursor, const std::vector<size_t> & order, const std::string & sampler) {
    yyjson_mut_doc * doc=yyjson_mut_doc_new(nullptr);
    if (!doc) return {};
    yyjson_mut_val * root=yyjson_mut_obj(doc), * arr=yyjson_mut_arr(doc);
    if (!root || !arr) { yyjson_mut_doc_free(doc); return {}; }
    yyjson_mut_doc_set_root(doc,root);
    yyjson_mut_obj_add_strcpy(doc,root,"recipe","yue2-aitk-runtime-v1"); yyjson_mut_obj_add_strcpy(doc,root,"checkpoint_sha256",cp.c_str()); yyjson_mut_obj_add_strcpy(doc,root,"dataset_sha256",ds.c_str()); yyjson_mut_obj_add_strcpy(doc,root,"source_manifest_sha256",sm.c_str()); yyjson_mut_obj_add_uint(doc,root,"seed",seed); yyjson_mut_obj_add_int(doc,root,"cuda_index",device); yyjson_mut_obj_add_int(doc,root,"completed_step",completed); yyjson_mut_obj_add_uint(doc,root,"order_cursor",cursor); for(size_t x:order) yyjson_mut_arr_add_uint(doc,arr,x); yyjson_mut_obj_add_val(doc,root,"order",arr); yyjson_mut_obj_add_strcpy(doc,root,"sampler_state",sampler.c_str()); size_t n=0; char * raw=yyjson_mut_write(doc,0,&n); std::string out=raw?std::string(raw,n):std::string(); std::free(raw); yyjson_mut_doc_free(doc); return out;
}
}

namespace yue2_aitk_runtime {

void yue2_aitk_install_sigint_handler() { std::signal(SIGINT, on_sigint); }
bool yue2_aitk_cancel_requested() { return g_cancel != 0; }
void yue2_aitk_clear_cancel() { g_cancel = 0; }

static int run_impl(const Config & config, std::string * error) {
    if (!fresh_output(std::filesystem::u8path(config.output), error)) return 1;
    if (config.seed > UINT32_MAX) { fail(error, "native-v1 runtime seed must fit uint32_t"); return 1; }
    if (config.steps <= 0 || config.save_every <= 0 || config.cuda_index < 0 || config.cuda_index > 127) { fail(error, "invalid runtime configuration"); return 1; }
    event("preflight");
    yue2_aitk::Dataset dataset;
    if (!yue2_aitk::read_dataset(config.dataset, &dataset, error)) return 1;
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
    ResumePlan resume_plan;
    if (!config.resume.empty()) {
        if (!yue2_aitk::yue2_aitk_read_resume(config.resume.c_str(), &resume_plan.record)) { fail(error, "cannot read resume record"); return 1; }
        if (!parse_resume_meta(resume_plan.record.runner_metadata, checkpoint_hash.hex(), dataset_hash.hex(), source_hash.hex(), config.seed, config.cuda_index, dataset.items.size(), &resume_plan, error)) return 1;
        if (resume_plan.completed > config.steps || resume_plan.completed > INT_MAX || resume_plan.record.state.step != resume_plan.completed) { fail(error, "resume completed step is invalid"); return 1; }
        if (resume_plan.cursor >= dataset.items.size()) { fail(error, "resume order cursor is out of range"); return 1; }
        if (!sampler.import_rng_state(resume_plan.sampler)) { fail(error, "resume sampler state is invalid"); return 1; }
        order = resume_plan.order; cursor = resume_plan.cursor; completed = resume_plan.completed;
    } else { sampler.rng().shuffle(order); }
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
    struct Backend { ggml_backend_t value = nullptr; ~Backend() { if (value) ggml_backend_free(value); } } backend{device ? ggml_backend_dev_init(device, nullptr) : nullptr};
    if (!backend.value) { fail(error, "requested CUDA backend is unavailable"); return 1; }
    try {
        event("load"); Yue2AitkModel model; Yue2AitkTrainState state;
        if (!model.load(config.checkpoint.c_str(), backend.value, yue2_aitk_load_embedding_bf16, error) ||
            !state.initialize(backend.value, static_cast<uint32_t>(config.seed), error)) return 1;
        std::vector<yue2_aitk::ParameterSpec> specs;
        for (const auto & p : state.named_tensors()) specs.push_back({p.name, p.parameter, p.gradient});
        yue2_aitk::Optimizer optimizer(backend.value, config.cuda_index, std::move(specs));
        if (!config.resume.empty()) {
            try { optimizer.restore(resume_plan.record.state); }
            catch (const std::exception & exception) { if (error) *error = exception.what(); return 1; }
            if (optimizer.step() != resume_plan.completed) { fail(error, "resume optimizer step does not match completed step"); return 1; }
            resume_plan.record = {};
        }
        std::ofstream jsonl(std::filesystem::u8path(config.output) / "train.jsonl", std::ios::binary);
        if (!jsonl) { fail(error, "cannot create training JSONL"); return 1; }
        int last_saved = -1;
        auto save_checkpoint = [&](int step) -> bool {
            if (step == last_saved) return true;
            event("checkpoint_stage", step);
            const auto final_dir = std::filesystem::u8path(config.output) / ("checkpoint-step" + std::to_string(step));
            const auto temp_dir = std::filesystem::u8path(config.output) / (".checkpoint-step" + std::to_string(step) + ".tmp");
            std::error_code save_ec; if (std::filesystem::exists(final_dir, save_ec) || std::filesystem::exists(temp_dir, save_ec) || !std::filesystem::create_directory(temp_dir, save_ec) || save_ec) return false;
            const auto adapter = temp_dir / "adapter.safetensors"; const auto resume = temp_dir / "optimizer.resume";
            const std::string sampler_state = sampler.export_rng_state();
            if (sampler_state.empty()) return false;
            const std::string metadata = make_resume_meta(checkpoint_hash.hex(), dataset_hash.hex(), source_hash.hex(), config.seed, config.cuda_index, step, cursor, order, sampler_state);
            if (metadata.empty() || !state.export_snapshot(adapter.u8string().c_str(), step, error, (temp_dir / "native-ar.safetensors").u8string().c_str(), (temp_dir / "native-nar.safetensors").u8string().c_str()) || !yue2_aitk::yue2_aitk_write_resume(resume.u8string().c_str(), optimizer.capture(), metadata)) return false;
            std::filesystem::rename(temp_dir, final_dir, save_ec);
            if (!save_ec) { last_saved=step; event("checkpoint", step); }
            return !save_ec;
        };
        while (completed < config.steps) {
            if (yue2_aitk_cancel_requested()) {
                if (completed > 0 && !save_checkpoint(completed)) { fail(error, "cancel checkpoint publication failed"); return 1; }
                event("cancelled", completed); return 130;
            }
            const std::vector<float> schedule = sampler.sigmoid_schedule(1000);
            const auto & item = dataset.items[order[cursor]];
            auto sampled = sampler.sample(item.song, item.prompt, 1500, schedule, 0.5f, 0, 999);
            yue2_aitk_joint::Input input; input.batch = &sampled.batch; input.noisy_latents = sampled.noisy_bf16;
            input.flow_target = sampled.target_f32; input.timestep = sampled.timestep_bf16;
            yue2_aitk_joint::Metrics metrics;
            if (!yue2_aitk_joint::run(backend.value, model, state, optimizer, input, &metrics, error,
                [&](const char * stage) { event(stage, completed + 1); })) return 1;
            ++completed;
            if (metrics.step != completed) { fail(error, "optimizer update count mismatch"); return 1; }
            if (++cursor == order.size()) { sampler.rng().shuffle(order); cursor=0; }
            std::ostringstream line;
            line << std::setprecision(17) << "{\"stage\":\"joint\",\"step\":" << metrics.step
                   << ",\"ar_ce\":" << metrics.ar_ce << ",\"ar_kl\":" << metrics.ar_kl
                   << ",\"nar_mse\":" << metrics.nar_mse << ",\"gradient_norm\":" << metrics.gradient_norm << "}\n";
            jsonl << line.str();
            std::cout << line.str() << std::flush;
            if (!jsonl.flush()) { fail(error, "training JSONL write failed"); return 1; }
            if (completed % config.save_every == 0 || completed == config.steps)
                if (!save_checkpoint(completed)) { fail(error, "checkpoint publication failed"); return 1; }
        }
        if (completed > 0 && !save_checkpoint(completed)) { fail(error, "final checkpoint publication failed"); return 1; }
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

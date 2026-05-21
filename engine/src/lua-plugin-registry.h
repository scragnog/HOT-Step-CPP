#pragma once
// lua-plugin-registry.h: Runtime plugin registry replacing static C++ registries
//
// Scans engine/plugins/ and project-root/plugins/ for .lua files.
// Provides lookup functions matching the old solver-registry.h / scheduler-registry.h
// / guidance-registry.h interfaces, plus JSON serialization for the API.

#include "lua-plugin.h"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// Plugin params are stored in g_hotstep_params.plugin_params (hot-step-params.h)
// and passed to Lua call wrappers directly by the sampler.

// ═══════════════════════════════════════════════════════════════════════════
// Plugin Registry — singleton holding all loaded plugins
// ═══════════════════════════════════════════════════════════════════════════

class PluginRegistry {
public:
    static PluginRegistry & instance() {
        static PluginRegistry reg;
        return reg;
    }

    // Scan directories and load all plugins. Call once at startup.
    void init(const std::string & engine_dir, const std::string & project_dir) {
        fprintf(stderr, "[Plugins] Initializing plugin system...\n");

        // Scan engine/plugins/{solvers,schedulers,guidance,postprocess}/
        scan_dir(engine_dir + "/plugins/solvers",      PluginType::Solver);
        scan_dir(engine_dir + "/plugins/schedulers",   PluginType::Scheduler);
        scan_dir(engine_dir + "/plugins/guidance",     PluginType::Guidance);
        scan_dir(engine_dir + "/plugins/postprocess",  PluginType::Postprocess);

        // Scan project-root/plugins/{solvers,schedulers,guidance,postprocess}/
        if (!project_dir.empty() && project_dir != engine_dir) {
            scan_dir(project_dir + "/plugins/solvers",      PluginType::Solver);
            scan_dir(project_dir + "/plugins/schedulers",   PluginType::Scheduler);
            scan_dir(project_dir + "/plugins/guidance",     PluginType::Guidance);
            scan_dir(project_dir + "/plugins/postprocess",  PluginType::Postprocess);
        }

        fprintf(stderr, "[Plugins] Loaded %d solvers, %d schedulers, %d guidance, %d postprocess\n",
                (int) solvers_.size(), (int) schedulers_.size(),
                (int) guidance_.size(), (int) postprocess_.size());
    }

    // ── Lookup by name (replacing old static registries) ──

    LuaPlugin * solver_lookup(const char * name) {
        if (!name || !name[0]) {
            // Default: first solver (should be euler)
            return solvers_.empty() ? nullptr : &solvers_.begin()->second;
        }
        // Aliases
        const char * resolved = name;
        if (strcmp(name, "ode") == 0) resolved = "euler";

        auto it = solvers_.find(resolved);
        return (it != solvers_.end()) ? &it->second : nullptr;
    }

    LuaPlugin * scheduler_lookup(const char * name) {
        if (!name || !name[0]) {
            return schedulers_.empty() ? nullptr : &schedulers_.begin()->second;
        }
        const char * resolved = name;
        if (strcmp(name, "karras") == 0) resolved = "sgm_uniform";

        // Exact match
        auto it = schedulers_.find(resolved);
        if (it != schedulers_.end()) return &it->second;

        // Prefix match for parameterized: "power:4.00" -> "power"
        std::string s(resolved);
        auto colon = s.find(':');
        if (colon != std::string::npos) {
            it = schedulers_.find(s.substr(0, colon));
            if (it != schedulers_.end()) return &it->second;
        }
        return nullptr;
    }

    LuaPlugin * guidance_lookup(const char * name) {
        if (!name || !name[0]) {
            return guidance_.empty() ? nullptr : &guidance_.begin()->second;
        }
        auto it = guidance_.find(name);
        return (it != guidance_.end()) ? &it->second : nullptr;
    }

    // ── Ordered lists for API/UI ──

    std::vector<LuaPlugin *> all_solvers() {
        std::vector<LuaPlugin *> v;
        for (auto & [_, p] : solvers_) v.push_back(&p);
        return v;
    }

    std::vector<LuaPlugin *> all_schedulers() {
        std::vector<LuaPlugin *> v;
        for (auto & [_, p] : schedulers_) v.push_back(&p);
        return v;
    }

    std::vector<LuaPlugin *> all_guidance() {
        std::vector<LuaPlugin *> v;
        for (auto & [_, p] : guidance_) v.push_back(&p);
        return v;
    }

    LuaPlugin * postprocess_lookup(const char * name) {
        if (!name || !name[0]) {
            return postprocess_.empty() ? nullptr : &postprocess_.begin()->second;
        }
        auto it = postprocess_.find(name);
        return (it != postprocess_.end()) ? &it->second : nullptr;
    }

    std::vector<LuaPlugin *> all_postprocess() {
        std::vector<LuaPlugin *> v;
        for (auto & [_, p] : postprocess_) v.push_back(&p);
        return v;
    }

    // ── JSON serialization for GET /plugins ──

    std::string to_json() const {
        std::string json = "{";
        json += "\"solvers\":[";
        json += plugins_to_json_array(solvers_);
        json += "],\"schedulers\":[";
        json += plugins_to_json_array(schedulers_);
        json += "],\"guidance\":[";
        json += plugins_to_json_array(guidance_);
        json += "],\"postprocess\":[";
        json += plugins_to_json_array(postprocess_);
        json += "]}";
        return json;
    }

private:
    std::unordered_map<std::string, LuaPlugin> solvers_;
    std::unordered_map<std::string, LuaPlugin> schedulers_;
    std::unordered_map<std::string, LuaPlugin> guidance_;
    std::unordered_map<std::string, LuaPlugin> postprocess_;

    void scan_dir(const std::string & dir_path, PluginType expected_type) {
        if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) return;

        // Collect and sort files for deterministic load order
        // First pass: collect all .lua stems in this directory
        std::set<std::string> all_stems;
        for (auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".lua") {
                all_stems.insert(entry.path().stem().string());
            }
        }

        std::vector<fs::path> files;
        for (auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".lua") {
                // Skip companion data files (loaded via require, not as plugins)
                std::string stem = entry.path().stem().string();
                if (stem.find("_constants") != std::string::npos ||
                    stem.find("_math") != std::string::npos ||
                    stem.find("_data") != std::string::npos) {
                    continue;
                }
                // Skip _core files only if a corresponding non-_core plugin exists
                // e.g. skip "md_audio_tiled_core" when "md_audio_tiled" exists,
                // but keep "storm_sampler_core" when no "storm_sampler" exists.
                auto core_pos = stem.rfind("_core");
                if (core_pos != std::string::npos &&
                    core_pos == stem.size() - 5) {  // ends with _core
                    std::string base = stem.substr(0, core_pos);
                    if (all_stems.count(base)) {
                        continue;  // companion — skip
                    }
                }
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());

        for (auto & fpath : files) {
            LuaPlugin plugin;
            std::string path_str = fpath.string();
            std::string dir_str  = fs::path(dir_path).string();

            if (lua_load_plugin(plugin, path_str.c_str(), dir_str.c_str())) {
                if (plugin.type != expected_type) {
                    fprintf(stderr, "[Plugins] WARNING: %s declares wrong type, skipping\n", path_str.c_str());
                    continue;
                }

                auto & map = (expected_type == PluginType::Solver)       ? solvers_
                           : (expected_type == PluginType::Scheduler)    ? schedulers_
                           : (expected_type == PluginType::Guidance)     ? guidance_
                           :                                               postprocess_;

                if (map.count(plugin.name)) {
                    fprintf(stderr, "[Plugins] WARNING: duplicate plugin '%s' from %s (keeping first)\n",
                            plugin.name.c_str(), path_str.c_str());
                    continue;
                }

                fprintf(stderr, "[Plugins]   %-12s %-24s (%s)\n",
                        expected_type == PluginType::Solver ? "solver" :
                        expected_type == PluginType::Scheduler ? "scheduler" :
                        expected_type == PluginType::Guidance ? "guidance" : "postprocess",
                        plugin.display_name.c_str(), plugin.name.c_str());

                map.emplace(plugin.name, std::move(plugin));
            }
        }
    }

    // ── JSON helpers ──

    static std::string escape_json(const std::string & s) {
        std::string out;
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                default:   out += c;
            }
        }
        return out;
    }

    static std::string param_to_json(const ParamSchema & p) {
        std::string j = "{";
        j += "\"key\":\"" + escape_json(p.key) + "\"";
        j += ",\"label\":\"" + escape_json(p.label) + "\"";

        const char * type_str = (p.type == ParamType::Slider) ? "slider"
                              : (p.type == ParamType::Select) ? "select"
                              : (p.type == ParamType::Toggle) ? "toggle" : "text";
        j += std::string(",\"type\":\"") + type_str + "\"";

        if (!p.hint.empty()) j += ",\"hint\":\"" + escape_json(p.hint) + "\"";
        if (!p.transform.empty()) j += ",\"transform\":\"" + escape_json(p.transform) + "\"";

        switch (p.type) {
            case ParamType::Slider:
                j += ",\"default\":" + std::to_string(p.default_num);
                j += ",\"min\":" + std::to_string(p.min_val);
                j += ",\"max\":" + std::to_string(p.max_val);
                j += ",\"step\":" + std::to_string(p.step_val);
                break;
            case ParamType::Select:
                j += ",\"default\":\"" + escape_json(p.default_str) + "\"";
                j += ",\"options\":[";
                for (size_t i = 0; i < p.options.size(); i++) {
                    if (i) j += ",";
                    j += "{\"value\":\"" + escape_json(p.options[i].value) + "\"";
                    j += ",\"label\":\"" + escape_json(p.options[i].label) + "\"}";
                }
                j += "]";
                break;
            case ParamType::Toggle:
                j += std::string(",\"default\":") + (p.default_bool ? "true" : "false");
                break;
            case ParamType::Text:
                j += ",\"default\":\"" + escape_json(p.default_str) + "\"";
                break;
        }

        if (p.visible_when.active) {
            j += ",\"visible_when\":{\"key\":\"" + escape_json(p.visible_when.key) + "\"";
            j += ",\"equals\":\"" + escape_json(p.visible_when.equals) + "\"}";
        }

        j += "}";
        return j;
    }

    static std::string plugin_to_json(const LuaPlugin & p) {
        std::string j = "{";
        j += "\"name\":\"" + escape_json(p.name) + "\"";
        j += ",\"display\":\"" + escape_json(p.display_name) + "\"";
        if (!p.description.empty()) j += ",\"description\":\"" + escape_json(p.description) + "\"";
        if (!p.accent.empty()) j += ",\"accent\":\"" + escape_json(p.accent) + "\"";

        if (p.type == PluginType::Solver) {
            j += ",\"nfe\":" + std::to_string(p.nfe);
            j += ",\"order\":" + std::to_string(p.order);
            j += std::string(",\"needs_model\":") + (p.needs_model ? "true" : "false");
            j += std::string(",\"stateful\":") + (p.stateful ? "true" : "false");
            j += std::string(",\"stochastic\":") + (p.stochastic ? "true" : "false");
            j += std::string(",\"owns_loop\":") + (p.owns_loop ? "true" : "false");
        }

        j += ",\"params\":[";
        for (size_t i = 0; i < p.params.size(); i++) {
            if (i) j += ",";
            j += param_to_json(p.params[i]);
        }
        j += "]";

        j += "}";
        return j;
    }

    static std::string plugins_to_json_array(const std::unordered_map<std::string, LuaPlugin> & map) {
        // Sort by name for stable JSON output
        std::vector<const LuaPlugin *> sorted;
        for (auto & [_, p] : map) sorted.push_back(&p);
        std::sort(sorted.begin(), sorted.end(),
                  [](const LuaPlugin * a, const LuaPlugin * b) { return a->name < b->name; });

        std::string j;
        for (size_t i = 0; i < sorted.size(); i++) {
            if (i) j += ",";
            j += plugin_to_json(*sorted[i]);
        }
        return j;
    }
};

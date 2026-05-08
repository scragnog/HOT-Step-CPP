#pragma once
// lua-plugin.h: Lua plugin system for drop-in solvers, schedulers, and guidance modes
//
// Provides:
//   - Sandboxed Lua VM per plugin file
//   - Zero-copy float array bridge (C float* ↔ Lua userdata)
//   - Plugin metadata + param schema extraction
//   - Wrapper functions matching C solver/scheduler/guidance signatures

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#include "solvers/solver-interface.h"
#include "guidance/guidance-interface.h"
#include "schedulers/scheduler-interface.h"

// ═══════════════════════════════════════════════════════════════════════════
// Float array userdata — zero-copy bridge between C++ and Lua
// ═══════════════════════════════════════════════════════════════════════════

struct LuaFloatArray {
    float * data;
    int     n;
    bool    readonly;
};

static const char * LUA_FLOAT_ARRAY_MT = "FloatArray";

static int lua_floatarray_index(lua_State * L) {
    LuaFloatArray * a = (LuaFloatArray *) luaL_checkudata(L, 1, LUA_FLOAT_ARRAY_MT);
    int idx = (int) luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= a->n) {
        return luaL_error(L, "FloatArray index %d out of range [0, %d)", idx, a->n);
    }
    lua_pushnumber(L, (double) a->data[idx]);
    return 1;
}

static int lua_floatarray_newindex(lua_State * L) {
    LuaFloatArray * a = (LuaFloatArray *) luaL_checkudata(L, 1, LUA_FLOAT_ARRAY_MT);
    if (a->readonly) {
        return luaL_error(L, "FloatArray is read-only");
    }
    int idx = (int) luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= a->n) {
        return luaL_error(L, "FloatArray index %d out of range [0, %d)", idx, a->n);
    }
    a->data[idx] = (float) luaL_checknumber(L, 3);
    return 0;
}

static int lua_floatarray_len(lua_State * L) {
    LuaFloatArray * a = (LuaFloatArray *) luaL_checkudata(L, 1, LUA_FLOAT_ARRAY_MT);
    lua_pushinteger(L, a->n);
    return 1;
}

static void lua_push_floatarray(lua_State * L, float * data, int n, bool readonly) {
    LuaFloatArray * a = (LuaFloatArray *) lua_newuserdata(L, sizeof(LuaFloatArray));
    a->data     = data;
    a->n        = n;
    a->readonly = readonly;
    luaL_getmetatable(L, LUA_FLOAT_ARRAY_MT);
    lua_setmetatable(L, -2);
}

static void lua_register_floatarray(lua_State * L) {
    luaL_newmetatable(L, LUA_FLOAT_ARRAY_MT);
    lua_pushcfunction(L, lua_floatarray_index);
    lua_setfield(L, -2, "__index");
    lua_pushcfunction(L, lua_floatarray_newindex);
    lua_setfield(L, -2, "__newindex");
    lua_pushcfunction(L, lua_floatarray_len);
    lua_setfield(L, -2, "__len");
    lua_pop(L, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Param schema types
// ═══════════════════════════════════════════════════════════════════════════

enum class ParamType { Slider, Select, Toggle, Text };

struct ParamOption {
    std::string value;
    std::string label;
};

struct ParamVisibleWhen {
    std::string key;
    std::string equals;
    bool        active = false;
};

struct ParamSchema {
    std::string              key;
    ParamType                type;
    std::string              label;
    std::string              hint;
    // slider
    double                   default_num  = 0.0;
    double                   min_val      = 0.0;
    double                   max_val      = 1.0;
    double                   step_val     = 0.01;
    // select
    std::string              default_str;
    std::vector<ParamOption> options;
    // toggle
    bool                     default_bool = false;
    // conditional
    ParamVisibleWhen         visible_when;
    // transform expression (e.g., "value * 0.05")
    std::string              transform;
};

// ═══════════════════════════════════════════════════════════════════════════
// Plugin types
// ═══════════════════════════════════════════════════════════════════════════

enum class PluginType { Solver, Scheduler, Guidance };

struct LuaPlugin {
    PluginType               type;
    std::string              name;
    std::string              display_name;
    std::string              description;
    std::string              accent;       // UI accent color
    std::string              filepath;
    std::vector<ParamSchema> params;

    // Solver-specific
    int  nfe           = 1;
    int  order         = 1;
    bool needs_model   = false;
    bool stateful      = false;
    bool stochastic    = false;

    // Lua VM state
    lua_State * L = nullptr;

    ~LuaPlugin() {
        if (L) { lua_close(L); L = nullptr; }
    }

    // Non-copyable (owns lua_State)
    LuaPlugin() = default;
    LuaPlugin(LuaPlugin && o) noexcept
        : type(o.type), name(std::move(o.name)), display_name(std::move(o.display_name)),
          description(std::move(o.description)), accent(std::move(o.accent)),
          filepath(std::move(o.filepath)), params(std::move(o.params)),
          nfe(o.nfe), order(o.order), needs_model(o.needs_model),
          stateful(o.stateful), stochastic(o.stochastic), L(o.L) {
        o.L = nullptr;
    }
    LuaPlugin & operator=(LuaPlugin &&) = delete;
    LuaPlugin(const LuaPlugin &) = delete;
    LuaPlugin & operator=(const LuaPlugin &) = delete;
};

// ═══════════════════════════════════════════════════════════════════════════
// Sandbox setup — restrict Lua to safe math-only operations
// ═══════════════════════════════════════════════════════════════════════════

static void lua_setup_sandbox(lua_State * L) {
    // Whitelist: math, string, table, basic (print, type, pairs, ipairs, etc.)
    luaL_openlibs(L);

    // Remove dangerous modules
    const char * blacklist[] = {"os", "io", "debug", "package", "dofile", "loadfile"};
    for (const char * mod : blacklist) {
        lua_pushnil(L);
        lua_setglobal(L, mod);
    }

    // Register float array metatable
    lua_register_floatarray(L);
}

// ═══════════════════════════════════════════════════════════════════════════
// Schema extraction helpers
// ═══════════════════════════════════════════════════════════════════════════

static std::string lua_get_string(lua_State * L, int idx, const char * field, const char * def = "") {
    lua_getfield(L, idx, field);
    const char * s = lua_isstring(L, -1) ? lua_tostring(L, -1) : def;
    std::string result(s);
    lua_pop(L, 1);
    return result;
}

static double lua_get_number(lua_State * L, int idx, const char * field, double def = 0.0) {
    lua_getfield(L, idx, field);
    double v = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : def;
    lua_pop(L, 1);
    return v;
}

static bool lua_get_bool(lua_State * L, int idx, const char * field, bool def = false) {
    lua_getfield(L, idx, field);
    bool v = lua_isboolean(L, -1) ? (bool) lua_toboolean(L, -1) : def;
    lua_pop(L, 1);
    return v;
}

static int lua_get_int(lua_State * L, int idx, const char * field, int def = 0) {
    lua_getfield(L, idx, field);
    int v = lua_isinteger(L, -1) ? (int) lua_tointeger(L, -1) : (lua_isnumber(L, -1) ? (int) lua_tonumber(L, -1) : def);
    lua_pop(L, 1);
    return v;
}

static ParamSchema lua_extract_param(lua_State * L, int idx) {
    ParamSchema p;
    p.key   = lua_get_string(L, idx, "key");
    p.label = lua_get_string(L, idx, "label", p.key.c_str());
    p.hint  = lua_get_string(L, idx, "hint");
    p.transform = lua_get_string(L, idx, "transform");

    std::string type_str = lua_get_string(L, idx, "type", "slider");
    if (type_str == "select")      p.type = ParamType::Select;
    else if (type_str == "toggle") p.type = ParamType::Toggle;
    else if (type_str == "text")   p.type = ParamType::Text;
    else                           p.type = ParamType::Slider;

    switch (p.type) {
        case ParamType::Slider:
            p.default_num = lua_get_number(L, idx, "default", 0.0);
            p.min_val     = lua_get_number(L, idx, "min", 0.0);
            p.max_val     = lua_get_number(L, idx, "max", 1.0);
            p.step_val    = lua_get_number(L, idx, "step", 0.01);
            break;
        case ParamType::Select:
            p.default_str = lua_get_string(L, idx, "default");
            lua_getfield(L, idx, "options");
            if (lua_istable(L, -1)) {
                int n = (int) luaL_len(L, -1);
                for (int i = 1; i <= n; i++) {
                    lua_rawgeti(L, -1, i);
                    if (lua_istable(L, -1)) {
                        ParamOption opt;
                        opt.value = lua_get_string(L, -1, "value");
                        opt.label = lua_get_string(L, -1, "label", opt.value.c_str());
                        p.options.push_back(std::move(opt));
                    } else if (lua_isstring(L, -1)) {
                        ParamOption opt;
                        opt.value = lua_tostring(L, -1);
                        opt.label = opt.value;
                        p.options.push_back(std::move(opt));
                    }
                    lua_pop(L, 1);
                }
            }
            lua_pop(L, 1);
            break;
        case ParamType::Toggle:
            p.default_bool = lua_get_bool(L, idx, "default", false);
            break;
        case ParamType::Text:
            p.default_str = lua_get_string(L, idx, "default");
            break;
    }

    // visible_when
    lua_getfield(L, idx, "visible_when");
    if (lua_istable(L, -1)) {
        p.visible_when.active = true;
        p.visible_when.key    = lua_get_string(L, -1, "key");
        p.visible_when.equals = lua_get_string(L, -1, "equals");
    }
    lua_pop(L, 1);

    return p;
}

static std::vector<ParamSchema> lua_extract_params(lua_State * L, int table_idx) {
    std::vector<ParamSchema> params;
    lua_getfield(L, table_idx, "params");
    if (lua_istable(L, -1)) {
        int n = (int) luaL_len(L, -1);
        for (int i = 1; i <= n; i++) {
            lua_rawgeti(L, -1, i);
            if (lua_istable(L, -1)) {
                params.push_back(lua_extract_param(L, lua_gettop(L)));
            }
            lua_pop(L, 1);
        }
    }
    lua_pop(L, 1);
    return params;
}

// ═══════════════════════════════════════════════════════════════════════════
// Plugin loading — load a .lua file and extract metadata
// ═══════════════════════════════════════════════════════════════════════════

// Load a single plugin file. Returns true on success.
// The plugin_dir is set as a require search path for companion data files.
static bool lua_load_plugin(LuaPlugin & plugin, const char * filepath, const char * plugin_dir) {
    lua_State * L = luaL_newstate();
    if (!L) {
        fprintf(stderr, "[Plugins] ERROR: failed to create Lua state for %s\n", filepath);
        return false;
    }

    lua_setup_sandbox(L);

    // Allow require() for companion data files in the same directory
    // Set package.path to only search the plugin's directory
    lua_getglobal(L, "package");
    if (lua_istable(L, -1)) {
        std::string path = std::string(plugin_dir) + "/?.lua";
        lua_pushstring(L, path.c_str());
        lua_setfield(L, -2, "path");
        lua_pushstring(L, "");  // disable C loaders
        lua_setfield(L, -2, "cpath");
    }
    lua_pop(L, 1);
    // Re-enable require (we cleared package above but kept it for path)
    // Actually require is part of package which we didn't blacklist

    if (luaL_dofile(L, filepath) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR loading %s: %s\n", filepath, lua_tostring(L, -1));
        lua_close(L);
        return false;
    }

    plugin.filepath = filepath;
    plugin.L        = L;

    // Detect plugin type from global table name
    bool found = false;
    const char * type_names[] = {"solver", "scheduler", "guidance"};
    PluginType   types[]      = {PluginType::Solver, PluginType::Scheduler, PluginType::Guidance};

    for (int i = 0; i < 3; i++) {
        lua_getglobal(L, type_names[i]);
        if (lua_istable(L, -1)) {
            plugin.type = types[i];
            int tbl = lua_gettop(L);

            plugin.name         = lua_get_string(L, tbl, "name");
            plugin.display_name = lua_get_string(L, tbl, "display", plugin.name.c_str());
            plugin.description  = lua_get_string(L, tbl, "description");
            plugin.accent       = lua_get_string(L, tbl, "accent");
            plugin.params       = lua_extract_params(L, tbl);

            if (plugin.type == PluginType::Solver) {
                plugin.nfe         = lua_get_int(L, tbl, "nfe", 1);
                plugin.order       = lua_get_int(L, tbl, "order", 1);
                plugin.needs_model = lua_get_bool(L, tbl, "needs_model", false);
                plugin.stateful    = lua_get_bool(L, tbl, "stateful", false);
                plugin.stochastic  = lua_get_bool(L, tbl, "stochastic", false);
            }

            lua_pop(L, 1);
            found = true;
            break;
        }
        lua_pop(L, 1);
    }

    if (!found) {
        fprintf(stderr, "[Plugins] WARNING: %s has no solver/scheduler/guidance table, skipping\n", filepath);
        lua_close(L);
        plugin.L = nullptr;
        return false;
    }

    if (plugin.name.empty()) {
        fprintf(stderr, "[Plugins] WARNING: %s has empty name, skipping\n", filepath);
        lua_close(L);
        plugin.L = nullptr;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lua solver/scheduler/guidance call wrappers
// ═══════════════════════════════════════════════════════════════════════════

// Inject plugin params into Lua globals before calling step/schedule/guide
static void lua_inject_params(lua_State * L,
                              const std::unordered_map<std::string, std::string> & params,
                              const std::string & plugin_name) {
    // Create a 'params' table accessible from the step function
    lua_newtable(L);
    std::string prefix = plugin_name + ":";
    for (auto & [k, v] : params) {
        // Match keys like "jkass_fast:beat_stability"
        if (k.substr(0, prefix.size()) == prefix) {
            std::string key = k.substr(prefix.size());
            // Try number first
            char * end = nullptr;
            double num = strtod(v.c_str(), &end);
            if (end != v.c_str() && *end == '\0') {
                lua_pushnumber(L, num);
            } else if (v == "true") {
                lua_pushboolean(L, 1);
            } else if (v == "false") {
                lua_pushboolean(L, 0);
            } else {
                lua_pushstring(L, v.c_str());
            }
            lua_setfield(L, -2, key.c_str());
        }
    }
    lua_setglobal(L, "params");
}

// Call a Lua solver's step() function
static void lua_call_solver_step(LuaPlugin & plugin,
                                 float * xt, const float * vt,
                                 float t_curr, float t_prev, int n,
                                 SolverState & state,
                                 SolverModelFn model_fn,
                                 float * vt_buf,
                                 const std::unordered_map<std::string, std::string> & params) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);

    // Set state globals
    lua_pushinteger(L, state.step_index);
    lua_setglobal(L, "step_index");
    lua_pushinteger(L, state.batch_n);
    lua_setglobal(L, "batch_n");
    lua_pushinteger(L, state.n_per);
    lua_setglobal(L, "n_per");

    // Push step function
    lua_getglobal(L, "step");
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "[Plugins] ERROR: solver '%s' has no step() function\n", plugin.name.c_str());
        lua_pop(L, 1);
        return;
    }

    // Push args: xt, vt, t_curr, t_prev, n
    lua_push_floatarray(L, xt, n, false);
    lua_push_floatarray(L, const_cast<float *>(vt), n, true);
    lua_pushnumber(L, (double) t_curr);
    lua_pushnumber(L, (double) t_prev);
    lua_pushinteger(L, n);

    // For multi-eval solvers, push model_fn as a callable
    if (plugin.needs_model && model_fn) {
        // Store model_fn in a light userdata + closure
        auto * fn_ptr = new SolverModelFn(model_fn);
        lua_pushlightuserdata(L, fn_ptr);
        lua_pushcclosure(L, [](lua_State * Ls) -> int {
            auto * fn = (SolverModelFn *) lua_touserdata(Ls, lua_upvalueindex(1));
            // First arg: xt_tmp (FloatArray), second: t_val (number)
            LuaFloatArray * arr = (LuaFloatArray *) luaL_checkudata(Ls, 1, LUA_FLOAT_ARRAY_MT);
            float t_val = (float) luaL_checknumber(Ls, 2);
            (*fn)(arr->data, t_val);
            return 0;
        }, 1);
        lua_push_floatarray(L, vt_buf, n, false);
        // 7 args: xt, vt, t_curr, t_prev, n, model_fn, vt_buf
        if (lua_pcall(L, 7, 0, 0) != LUA_OK) {
            fprintf(stderr, "[Plugins] ERROR in solver '%s' step(): %s\n",
                    plugin.name.c_str(), lua_tostring(L, -1));
            lua_pop(L, 1);
        }
        delete fn_ptr;
    } else {
        // 5 args: xt, vt, t_curr, t_prev, n
        if (lua_pcall(L, 5, 0, 0) != LUA_OK) {
            fprintf(stderr, "[Plugins] ERROR in solver '%s' step(): %s\n",
                    plugin.name.c_str(), lua_tostring(L, -1));
            lua_pop(L, 1);
        }
    }
}

// Call a Lua scheduler's schedule() function
static void lua_call_scheduler(LuaPlugin & plugin,
                               float * output, int num_steps, float shift,
                               const std::unordered_map<std::string, std::string> & params) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);

    lua_getglobal(L, "schedule");
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "[Plugins] ERROR: scheduler '%s' has no schedule() function\n", plugin.name.c_str());
        lua_pop(L, 1);
        return;
    }

    lua_push_floatarray(L, output, num_steps, false);
    lua_pushinteger(L, num_steps);
    lua_pushnumber(L, (double) shift);

    if (lua_pcall(L, 3, 0, 0) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR in scheduler '%s' schedule(): %s\n",
                plugin.name.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
    }
}

// Call a Lua guidance's guide() function
static void lua_call_guidance(LuaPlugin & plugin,
                              const float * pred_cond, const float * pred_uncond,
                              float guidance_scale, APGMomentumBuffer & mbuf,
                              float * result, int Oc, int T,
                              const GuidanceCtx & ctx, float norm_threshold,
                              const std::unordered_map<std::string, std::string> & params) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);

    int n = Oc * T;

    // Set context globals
    lua_pushinteger(L, ctx.step_idx);   lua_setglobal(L, "step_idx");
    lua_pushinteger(L, ctx.total_steps); lua_setglobal(L, "total_steps");
    lua_pushnumber(L, (double) ctx.dt);  lua_setglobal(L, "dt");
    lua_pushnumber(L, (double) ctx.t_curr); lua_setglobal(L, "t_curr");

    lua_getglobal(L, "guide");
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "[Plugins] ERROR: guidance '%s' has no guide() function\n", plugin.name.c_str());
        lua_pop(L, 1);
        return;
    }

    lua_push_floatarray(L, const_cast<float *>(pred_cond), n, true);
    lua_push_floatarray(L, const_cast<float *>(pred_uncond), n, true);
    lua_pushnumber(L, (double) guidance_scale);
    lua_push_floatarray(L, result, n, false);
    lua_pushinteger(L, Oc);
    lua_pushinteger(L, T);
    lua_pushnumber(L, (double) norm_threshold);

    if (lua_pcall(L, 7, 0, 0) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR in guidance '%s' guide(): %s\n",
                plugin.name.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
    }
}

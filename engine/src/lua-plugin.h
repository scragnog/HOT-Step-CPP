#pragma once
// lua-plugin.h: Lua plugin system for drop-in solvers, schedulers, and guidance modes
//
// Provides:
//   - Sandboxed Lua VM per plugin file
//   - Zero-copy float array bridge (C float* â†” Lua userdata)
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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Float array userdata â€” zero-copy bridge between C++ and Lua
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Param schema types
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Plugin types
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
    bool owns_loop     = false;   // full-loop solver: defines sample() instead of step()

    // Guidance-specific
    bool has_post_step = false;  // guidance plugin declares post_step()

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
          stateful(o.stateful), stochastic(o.stochastic), owns_loop(o.owns_loop),
          has_post_step(o.has_post_step), L(o.L) {
        o.L = nullptr;
    }
    LuaPlugin & operator=(LuaPlugin &&) = delete;
    LuaPlugin(const LuaPlugin &) = delete;
    LuaPlugin & operator=(const LuaPlugin &) = delete;
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Sandbox setup â€” restrict Lua to safe math-only operations
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

static void lua_setup_sandbox(lua_State * L) {
    // Whitelist: math, string, table, basic (print, type, pairs, ipairs, etc.)
    luaL_openlibs(L);

    // Remove dangerous modules
    // Note: "package" is kept (needed for require() of companion data files).
    // Security: cpath is set to "" during load to block C module loading.
    const char * blacklist[] = {"os", "io", "debug", "dofile", "loadfile"};
    for (const char * mod : blacklist) {
        lua_pushnil(L);
        lua_setglobal(L, mod);
    }

    // Register float array metatable
    lua_register_floatarray(L);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Schema extraction helpers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Plugin loading â€” load a .lua file and extract metadata
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
                plugin.owns_loop   = lua_get_bool(L, tbl, "owns_loop", false);
            }

            // Detect post_step() for guidance plugins
            if (plugin.type == PluginType::Guidance) {
                lua_getglobal(L, "post_step");
                plugin.has_post_step = lua_isfunction(L, -1);
                lua_pop(L, 1);
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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Lua solver/scheduler/guidance call wrappers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Full-loop solver support â€” owns_loop = true
// The Lua plugin defines sample() instead of step() and controls the
// entire sampling iteration. Engine hooks (DCW, repaint, guidance,
// cancel, progress) are provided via an on_step_fn callback built
// by the caller (hot-step-sampler.h).
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// on_step_fn(step_idx, t_curr, t_next) â†’ bool (true = cancelled)
using LoopOnStepFn = std::function<bool(int step_idx, float t_curr, float t_next)>;
// model_fn(xt_data, t_val) â†’ writes velocity to vt
using LoopModelFn  = std::function<void(const float * xt_data, float t_val)>;

// on_step C closure â€” delegates to the LoopOnStepFn captured as upvalue
static int lua_on_step_closure(lua_State * L) {
    LoopOnStepFn * fn = (LoopOnStepFn *) lua_touserdata(L, lua_upvalueindex(1));
    int   step_idx = (int) luaL_checkinteger(L, 1);
    float t_curr   = (float) luaL_checknumber(L, 2);
    float t_next   = (float) luaL_checknumber(L, 3);
    bool cancelled = (*fn)(step_idx, t_curr, t_next);
    lua_pushboolean(L, cancelled ? 1 : 0);
    return 1;
}

// Call a full-loop Lua solver's sample(xt, vt_buf, schedule, n, model_fn)
static void lua_call_solver_loop(
    LuaPlugin &  plugin,
    float *      xt,
    float *      vt,
    const float * schedule,
    int          num_steps,
    int          n,
    int          N,         // batch size (for n_per)
    int          T,
    int          Oc,
    LoopModelFn  model_fn,
    LoopOnStepFn on_step_fn,
    const std::unordered_map<std::string, std::string> & params)
{
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);

    // Set globals
    lua_pushinteger(L, num_steps);  lua_setglobal(L, "num_steps");
    lua_pushinteger(L, N);          lua_setglobal(L, "batch_n");
    lua_pushinteger(L, T * Oc);     lua_setglobal(L, "n_per");

    // Register on_step global closure
    lua_pushlightuserdata(L, &on_step_fn);
    lua_pushcclosure(L, lua_on_step_closure, 1);
    lua_setglobal(L, "on_step");

    // Push sample() function
    lua_getglobal(L, "sample");
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "[Plugins] ERROR: full-loop solver '%s' has no sample() function\n",
                plugin.name.c_str());
        lua_pop(L, 1);
        return;
    }

    // Arg 1: xt (FloatArray, mutable)
    lua_push_floatarray(L, xt, n, false);

    // Arg 2: vt_buf (FloatArray, mutable â€” model_fn writes here)
    lua_push_floatarray(L, vt, n, false);

    // Arg 3: schedule (Lua table, 1-indexed, num_steps entries)
    lua_newtable(L);
    for (int i = 0; i < num_steps; i++) {
        lua_pushinteger(L, i + 1);
        lua_pushnumber(L, (double) schedule[i]);
        lua_settable(L, -3);
    }

    // Arg 4: n (element count)
    lua_pushinteger(L, n);

    // Arg 5: model_fn closure
    auto * mfn_ptr = new LoopModelFn(std::move(model_fn));
    lua_pushlightuserdata(L, mfn_ptr);
    lua_pushcclosure(L, [](lua_State * Ls) -> int {
        auto * fn = (LoopModelFn *) lua_touserdata(Ls, lua_upvalueindex(1));
        LuaFloatArray * arr = (LuaFloatArray *) luaL_checkudata(Ls, 1, LUA_FLOAT_ARRAY_MT);
        float t_val = (float) luaL_checknumber(Ls, 2);
        (*fn)(arr->data, t_val);
        return 0;
    }, 1);

    // 5 args: xt, vt_buf, schedule, n, model_fn
    if (lua_pcall(L, 5, 0, 0) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR in full-loop solver '%s' sample(): %s\n",
                plugin.name.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
    }

    delete mfn_ptr;
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
// â”€â”€ APG bridge for Lua guidance plugins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Registers a Lua-callable `apg(cond, uncond, scale, result, Oc, T, norm_threshold)`
// that routes through the native C++ apg_forward(), including:
//   - momentum smoothing across steps
//   - per-channel norm thresholding
//   - perpendicular projection
//
// The momentum buffer for the current batch element is injected as a global
// light userdata "_apg_mbuf" before each guide() call.

static int lua_apg_closure(lua_State * L) {
    // Args: pred_cond (FloatArray), pred_uncond (FloatArray), scale (number),
    //       result (FloatArray), Oc (int), T (int), norm_threshold (number)
    LuaFloatArray * cond   = (LuaFloatArray *) luaL_checkudata(L, 1, LUA_FLOAT_ARRAY_MT);
    LuaFloatArray * uncond = (LuaFloatArray *) luaL_checkudata(L, 2, LUA_FLOAT_ARRAY_MT);
    float scale            = (float) luaL_checknumber(L, 3);
    LuaFloatArray * result = (LuaFloatArray *) luaL_checkudata(L, 4, LUA_FLOAT_ARRAY_MT);
    int Oc                 = (int) luaL_checkinteger(L, 5);
    int T                  = (int) luaL_checkinteger(L, 6);
    float norm_threshold   = (float) luaL_optnumber(L, 7, 2.5);

    // Retrieve momentum buffer from global
    lua_getglobal(L, "_apg_mbuf");
    APGMomentumBuffer * mbuf = (APGMomentumBuffer *) lua_touserdata(L, -1);
    lua_pop(L, 1);

    if (!mbuf) {
        return luaL_error(L, "apg(): no momentum buffer available (internal error)");
    }

    apg_forward(cond->data, uncond->data, scale, *mbuf, result->data, Oc, T, norm_threshold);
    return 0;
}

// Register the apg() function in a guidance plugin's Lua state
static void lua_register_apg(lua_State * L) {
    lua_pushcfunction(L, lua_apg_closure);
    lua_setglobal(L, "apg");
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

    // Inject momentum buffer pointer for the apg() C closure
    lua_pushlightuserdata(L, &mbuf);
    lua_setglobal(L, "_apg_mbuf");

    // Set context globals
    lua_pushinteger(L, ctx.step_idx);   lua_setglobal(L, "step_idx");
    lua_pushinteger(L, ctx.total_steps); lua_setglobal(L, "total_steps");
    lua_pushnumber(L, (double) ctx.dt);  lua_setglobal(L, "dt");
    lua_pushnumber(L, (double) ctx.t_curr); lua_setglobal(L, "t_curr");

    // Register apg() on first call (idempotent check via global existence)
    lua_getglobal(L, "apg");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        lua_register_apg(L);
    } else {
        lua_pop(L, 1);
    }

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

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Post-step hook for guidance plugins that need model callbacks
// (e.g. CFG-MP manifold projection)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// Model callback type for post_step: evaluates at (xt_in, t_val), writes to bound output buffer
using PostStepModelFn = std::function<void(const float *, float)>;

static void lua_call_post_step(LuaPlugin & plugin,
                               float * xt, float t_val, int n,
                               PostStepModelFn eval_cond_fn,
                               PostStepModelFn eval_uncond_fn,
                               float * vt_cond_buf, float * vt_uncond_buf,
                               const GuidanceCtx & ctx,
                               const std::unordered_map<std::string, std::string> & params) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);

    // Set context globals (same as guide())
    lua_pushinteger(L, ctx.step_idx);    lua_setglobal(L, "step_idx");
    lua_pushinteger(L, ctx.total_steps); lua_setglobal(L, "total_steps");
    lua_pushnumber(L, (double) ctx.dt);  lua_setglobal(L, "dt");
    lua_pushnumber(L, (double) ctx.t_curr); lua_setglobal(L, "t_curr");

    lua_getglobal(L, "post_step");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    // Arg 1: xt (mutable)
    lua_push_floatarray(L, xt, n, false);
    // Arg 2: t
    lua_pushnumber(L, (double) t_val);
    // Arg 3: n
    lua_pushinteger(L, n);

    // Arg 4: eval_cond closure â€” calls model with conditioning, writes to vt_cond_buf
    auto * cond_ptr = new PostStepModelFn(eval_cond_fn);
    lua_pushlightuserdata(L, cond_ptr);
    lua_pushcclosure(L, [](lua_State * Ls) -> int {
        auto * fn = (PostStepModelFn *) lua_touserdata(Ls, lua_upvalueindex(1));
        LuaFloatArray * arr = (LuaFloatArray *) luaL_checkudata(Ls, 1, LUA_FLOAT_ARRAY_MT);
        float t = (float) luaL_checknumber(Ls, 2);
        (*fn)(arr->data, t);
        return 0;
    }, 1);

    // Arg 5: eval_uncond closure â€” calls model without conditioning, writes to vt_uncond_buf
    auto * uncond_ptr = new PostStepModelFn(eval_uncond_fn);
    lua_pushlightuserdata(L, uncond_ptr);
    lua_pushcclosure(L, [](lua_State * Ls) -> int {
        auto * fn = (PostStepModelFn *) lua_touserdata(Ls, lua_upvalueindex(1));
        LuaFloatArray * arr = (LuaFloatArray *) luaL_checkudata(Ls, 1, LUA_FLOAT_ARRAY_MT);
        float t = (float) luaL_checknumber(Ls, 2);
        (*fn)(arr->data, t);
        return 0;
    }, 1);

    // Arg 6: vt_cond output buffer
    lua_push_floatarray(L, vt_cond_buf, n, false);
    // Arg 7: vt_uncond output buffer
    lua_push_floatarray(L, vt_uncond_buf, n, false);

    // 7 args: xt, t, n, eval_cond, eval_uncond, vt_cond, vt_uncond
    if (lua_pcall(L, 7, 0, 0) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR in guidance '%s' post_step(): %s\n",
                plugin.name.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
    }

    delete cond_ptr;
    delete uncond_ptr;
}

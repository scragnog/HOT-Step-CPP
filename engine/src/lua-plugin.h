#pragma once
// lua-plugin.h: Lua plugin system for drop-in solvers, schedulers, and guidance modes
//
// Provides:
//   - Sandboxed Lua VM per plugin file
//   - Zero-copy float array bridge (C float* to a bounds-checked LuaJIT FFI view)
//   - Plugin metadata + param schema extraction
//   - Wrapper functions matching C solver/scheduler/guidance signatures

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
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

// The runtime is LuaJIT. Element loops in plugins compile to machine code only
// when the array is an FFI pointer, but a raw pointer would let a plugin read
// or write any memory. So the host builds each view in a private prelude: a
// table whose metamethods index a float* held in upvalues, with bounds and
// read-only checks. Plugins see xt[i], #xt and nothing else; the prelude then
// removes ffi and jit from the state, so no plugin can reach either.
static const char * LUA_HOST_PRELUDE = R"LUA(
local ffi = require('ffi')
local setmetatable, error, cast = setmetatable, error, ffi.cast
local ptrs = setmetatable({}, { __mode = 'k' })
local function view(p, n, ro)
  local d = cast('float*', p)
  local v = setmetatable({}, {
    __index = function(_, i)
      if i < 0 or i >= n then error('float array index ' .. tostring(i) .. ' out of range [0, ' .. n .. ')', 2) end
      return d[i]
    end,
    __newindex = function(_, i, x)
      if ro then error('float array is read-only', 2) end
      if i < 0 or i >= n then error('float array index ' .. tostring(i) .. ' out of range [0, ' .. n .. ')', 2) end
      d[i] = x
    end,
    __len = function() return n end,
    __metatable = false,
  })
  ptrs[v] = p
  return v
end
local function unwrap(v)
  local p = ptrs[v]
  if p == nil then error('expected a float array', 3) end
  return p
end
return view, unwrap
)LUA";

static const char * LUA_VIEW_KEY   = "hot_step.float_view";
static const char * LUA_UNWRAP_KEY = "hot_step.float_unwrap";

// Push a view of n floats at data. The view is valid for the duration of the
// call it is passed to; the host owns the memory.
static void lua_push_floatarray(lua_State * L, float * data, int n, bool readonly) {
    lua_getfield(L, LUA_REGISTRYINDEX, LUA_VIEW_KEY);
    lua_pushlightuserdata(L, data);
    lua_pushinteger(L, n);
    lua_pushboolean(L, readonly ? 1 : 0);
    lua_call(L, 3, 1);
}

// The float* behind a view the host pushed; raises a Lua error for anything else.
static float * lua_check_floatarray(lua_State * L, int idx) {
    lua_pushvalue(L, idx);
    lua_getfield(L, LUA_REGISTRYINDEX, LUA_UNWRAP_KEY);
    lua_insert(L, -2);
    lua_call(L, 1, 1);
    float * data = (float *) lua_touserdata(L, -1);
    lua_pop(L, 1);
    return data;
}

// math.random / math.randomseed with Lua 5.4.7's generator (xoshiro256**,
// lmathlib.c), so a plugin draws the same numbers for the same job seed as it
// did before the move to LuaJIT. The state lives in a userdata upvalue.
struct LuaRandState { uint64_t s[4]; };

static uint64_t lua_rand_rotl(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

static uint64_t lua_rand_next(LuaRandState * st) {
    uint64_t * s = st->s;
    const uint64_t s0 = s[0], s1 = s[1], s2 = s[2] ^ s0, s3 = s[3] ^ s1;
    const uint64_t res = lua_rand_rotl(s1 * 5, 7) * 9;
    s[0] = s0 ^ s3;
    s[1] = s1 ^ s2;
    s[2] = s2 ^ (s1 << 17);
    s[3] = lua_rand_rotl(s3, 45);
    return res;
}

static void lua_rand_seed(LuaRandState * st, uint64_t n1, uint64_t n2) {
    st->s[0] = n1;
    st->s[1] = 0xff;
    st->s[2] = n2;
    st->s[3] = 0;
    for (int i = 0; i < 16; i++) lua_rand_next(st);
}

static int lua_math_random(lua_State * L) {
    LuaRandState * st = (LuaRandState *) lua_touserdata(L, lua_upvalueindex(1));
    uint64_t rv = lua_rand_next(st);
    int64_t low, up;
    switch (lua_gettop(L)) {
        case 0: {
            // Top 53 bits as a double in [0, 1), as lmathlib's I2d.
            lua_pushnumber(L, (double) (rv >> 11) * (0.5 / (double) (1ull << 52)));
            return 1;
        }
        case 1:
            low = 1;
            up  = (int64_t) luaL_checknumber(L, 1);
            if (up == 0) { lua_pushnumber(L, (double) (int64_t) rv); return 1; }
            break;
        case 2:
            low = (int64_t) luaL_checknumber(L, 1);
            up  = (int64_t) luaL_checknumber(L, 2);
            break;
        default: return luaL_error(L, "wrong number of arguments");
    }
    luaL_argcheck(L, low <= up, 1, "interval is empty");
    const uint64_t n = (uint64_t) up - (uint64_t) low;
    uint64_t ran = rv;
    if ((n & (n + 1)) == 0) ran &= n;
    else {
        uint64_t lim = n;
        lim |= lim >> 1; lim |= lim >> 2; lim |= lim >> 4;
        lim |= lim >> 8; lim |= lim >> 16; lim |= lim >> 32;
        while ((ran &= lim) > n) ran = lua_rand_next(st);
    }
    lua_pushnumber(L, (double) (int64_t) (ran + (uint64_t) low));
    return 1;
}

static int lua_math_randomseed(lua_State * L) {
    LuaRandState * st = (LuaRandState *) lua_touserdata(L, lua_upvalueindex(1));
    const uint64_t n1 = lua_isnoneornil(L, 1) ? (uint64_t) time(nullptr) : (uint64_t) (int64_t) luaL_checknumber(L, 1);
    const uint64_t n2 = lua_isnoneornil(L, 1) ? (uint64_t) (size_t) L : (uint64_t) (int64_t) luaL_optnumber(L, 2, 0);
    lua_rand_seed(st, n1, n2);
    return 0;
}

static const char * LUA_RAND_KEY = "hot_step.rand_state";

static void lua_register_random(lua_State * L) {
    LuaRandState * st = (LuaRandState *) lua_newuserdata(L, sizeof(LuaRandState));
    lua_rand_seed(st, (uint64_t) time(nullptr), (uint64_t) (size_t) L);
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_REGISTRYINDEX, LUA_RAND_KEY);
    lua_getglobal(L, "math");
    lua_pushvalue(L, -2);
    lua_pushcclosure(L, lua_math_random, 1);
    lua_setfield(L, -2, "random");
    lua_pushvalue(L, -2);
    lua_pushcclosure(L, lua_math_randomseed, 1);
    lua_setfield(L, -2, "randomseed");
    lua_pop(L, 2);
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

enum class PluginType { Solver, Scheduler, Guidance, Postprocess };

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
    const char * blacklist[] = {"os", "io", "debug", "dofile", "loadfile", "ffi"};
    for (const char * mod : blacklist) {
        lua_pushnil(L);
        lua_setglobal(L, mod);
    }

    // Host prelude: needs ffi, so it runs before ffi is removed below.
    if (luaL_loadstring(L, LUA_HOST_PRELUDE) != 0 || lua_pcall(L, 0, 2, 0) != 0) {
        fprintf(stderr, "[Plugins] ERROR: host prelude failed: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
    } else {
        lua_setfield(L, LUA_REGISTRYINDEX, LUA_UNWRAP_KEY);
        lua_setfield(L, LUA_REGISTRYINDEX, LUA_VIEW_KEY);
    }
    lua_register_random(L);

    // Removing the globals is not enough: require() also finds modules through
    // package.loaded and package.preload, and package.loadlib opens any DLL.
    const char * hidden[] = {"os", "io", "debug", "ffi", "jit"};
    lua_getglobal(L, "package");
    if (lua_istable(L, -1)) {
        for (const char * field : {"loaded", "preload"}) {
            lua_getfield(L, -1, field);
            if (lua_istable(L, -1)) {
                for (const char * mod : hidden) {
                    lua_pushnil(L);
                    lua_setfield(L, -2, mod);
                }
            }
            lua_pop(L, 1);
        }
        lua_pushnil(L);
        lua_setfield(L, -2, "loadlib");
    }
    lua_pop(L, 1);
    lua_pushnil(L);
    lua_setglobal(L, "jit");
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
    int v = lua_isnumber(L, -1) ? (int) lua_tonumber(L, -1) : def;
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
                int n = (int) lua_objlen(L, -1);
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
        int n = (int) lua_objlen(L, -1);
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
    const char * type_names[] = {"solver", "scheduler", "guidance", "postprocess"};
    PluginType   types[]      = {PluginType::Solver, PluginType::Scheduler, PluginType::Guidance, PluginType::Postprocess};

    for (int i = 0; i < 4; i++) {
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
        fprintf(stderr, "[Plugins] WARNING: %s has no solver/scheduler/guidance/postprocess table, skipping\n", filepath);
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

// -- Model context injection ------------------------------------------------
// Injects a 'model_context' global table before every plugin call so Lua
// plugins can adapt to the active model's native sample rate and latent
// geometry instead of hardcoding ACE-Step's. The caller constructs the values:
// hot-step-sampler.h leaves the defaults for ACE, mm3-plugins.h passes MM3's,
// sa3-refine.h passes SA3's.
//
// Lua-side usage:
//   local sr = (model_context and model_context.native_sr) or 48000
//
// The `or 48000` fallback keeps every existing plugin working unchanged on
// ACE. Only plugins that want to be model-aware need to read it.

struct LuaModelContext {
    int          native_sr       = 48000;  // audio sample rate (48000 ACE, 44100 MM3/SA3)
    int          latent_fps      = 25;     // latent frames per second
    int          latent_channels = 64;     // Oc (64 ACE, 128 MM3, 256 SA3)
    const char * model_id        = "ace";  // "ace", "mm3", "sa3"
};

static void lua_inject_model_context(lua_State * L, const LuaModelContext & ctx) {
    lua_newtable(L);
    lua_pushinteger(L, ctx.native_sr);       lua_setfield(L, -2, "native_sr");
    lua_pushinteger(L, ctx.latent_fps);      lua_setfield(L, -2, "latent_fps");
    lua_pushinteger(L, ctx.latent_channels); lua_setfield(L, -2, "latent_channels");
    lua_pushstring(L, ctx.model_id);         lua_setfield(L, -2, "model_id");
    lua_setglobal(L, "model_context");
}

// Seed a plugin's Lua RNG deterministically from the job seed.
//
// WHY: Lua 5.4 seeds math.random from the clock + an address at lua_State
// creation ("a weak attempt at randomness"), and each plugin's lua_State lives
// for the whole ace-server process. A plugin that calls math.random() therefore
// draws from a stream that depends on process start time AND on how many numbers
// every previous generation consumed — so two renders of an identical payload
// with an identical seed produced different audio (md_storm_V4's look-back
// jitter, plugins/solvers/md_storm_core_V4.lua). Re-seeding from the job seed at
// the start of every generation makes math.random() a pure function of the seed
// while leaving its distribution (xoshiro256**) untouched.
static void lua_seed_plugin_rng(lua_State * L, int64_t seed, const char * plugin_name) {
    if (!L) return;
    lua_getfield(L, LUA_REGISTRYINDEX, LUA_RAND_KEY);
    LuaRandState * st = (LuaRandState *) lua_touserdata(L, -1);
    lua_pop(L, 1);
    if (!st) return;
    lua_rand_seed(st, (uint64_t) seed, 0);  // math.randomseed(seed) in Lua 5.4
    fprintf(stderr, "[Plugins] solver '%s' RNG seeded from job seed %lld\n",
            plugin_name ? plugin_name : "?", (long long) seed);
}

// Call a Lua solver's step() function
static void lua_call_solver_step(LuaPlugin & plugin,
                                 float * xt, const float * vt,
                                 float t_curr, float t_prev, int n,
                                 SolverState & state,
                                 SolverModelFn model_fn,
                                 float * vt_buf,
                                 const std::unordered_map<std::string, std::string> & params,
                                 const LuaModelContext & model_ctx = LuaModelContext{}) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

    // Job seed: batch item 0 (one Lua state serves the whole flattened batch).
    const int64_t job_seed = state.seeds ? state.seeds[0] : 0;

    // Re-seed once per generation, then let the stream run across the steps.
    if (state.step_index == 0) lua_seed_plugin_rng(L, job_seed, plugin.name.c_str());

    // Set state globals
    lua_pushinteger(L, state.step_index);
    lua_setglobal(L, "step_index");
    lua_pushinteger(L, state.batch_n);
    lua_setglobal(L, "batch_n");
    lua_pushinteger(L, state.n_per);
    lua_setglobal(L, "n_per");
    lua_pushinteger(L, (lua_Integer) job_seed);
    lua_setglobal(L, "seed");

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
            float * data = lua_check_floatarray(Ls, 1);
            float t_val = (float) luaL_checknumber(Ls, 2);
            (*fn)(data, t_val);
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
    const std::unordered_map<std::string, std::string> & params,
    const LuaModelContext & model_ctx = LuaModelContext{},
    int64_t      seed = 0)
{
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

    // Deterministic RNG for this generation (see lua_seed_plugin_rng).
    lua_seed_plugin_rng(L, seed, plugin.name.c_str());

    // Set globals
    lua_pushinteger(L, num_steps);  lua_setglobal(L, "num_steps");
    lua_pushinteger(L, N);          lua_setglobal(L, "batch_n");
    lua_pushinteger(L, T * Oc);     lua_setglobal(L, "n_per");
    lua_pushinteger(L, (lua_Integer) seed); lua_setglobal(L, "seed");

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
        float * data = lua_check_floatarray(Ls, 1);
        float t_val = (float) luaL_checknumber(Ls, 2);
        (*fn)(data, t_val);
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
                               const std::unordered_map<std::string, std::string> & params,
                               const LuaModelContext & model_ctx = LuaModelContext{}) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

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
    float * cond           = lua_check_floatarray(L, 1);
    float * uncond         = lua_check_floatarray(L, 2);
    float scale            = (float) luaL_checknumber(L, 3);
    float * result         = lua_check_floatarray(L, 4);
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

    apg_forward(cond, uncond, scale, *mbuf, result, Oc, T, norm_threshold);
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
                              const std::unordered_map<std::string, std::string> & params,
                              const LuaModelContext & model_ctx = LuaModelContext{}) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

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
                               const std::unordered_map<std::string, std::string> & params,
                               const LuaModelContext & model_ctx = LuaModelContext{}) {
    lua_State * L = plugin.L;
    if (!L) return;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

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
        float * data = lua_check_floatarray(Ls, 1);
        float t = (float) luaL_checknumber(Ls, 2);
        (*fn)(data, t);
        return 0;
    }, 1);

    // Arg 5: eval_uncond closure â€” calls model without conditioning, writes to vt_uncond_buf
    auto * uncond_ptr = new PostStepModelFn(eval_uncond_fn);
    lua_pushlightuserdata(L, uncond_ptr);
    lua_pushcclosure(L, [](lua_State * Ls) -> int {
        auto * fn = (PostStepModelFn *) lua_touserdata(Ls, lua_upvalueindex(1));
        float * data = lua_check_floatarray(Ls, 1);
        float t = (float) luaL_checknumber(Ls, 2);
        (*fn)(data, t);
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

// ═══════════════════════════════════════════════════════════════════════════
// Postprocess plugin support — replaces built-in tiled VAE decode
// The Lua plugin defines process() which receives latent data as a Lua table,
// a vae_decode callback, and returns decoded audio as a Lua table.
// ═══════════════════════════════════════════════════════════════════════════

// VAE decode callback type for postprocess plugins:
//   decode_fn(latent_data, T_latent) → (audio_data, T_audio)
// latent_data is float* [T_latent, 64] time-major, audio_data is float* [2, T_audio]
using PostprocessVaeDecodeFn = std::function<int(const float * latent, int T_latent, float * audio_out, int max_T_audio)>;

// Call a Lua postprocess plugin's process() function.
// latents: [B * T * 64] time-major flat array (B batch items × T frames × 64 channels)
// For each batch item, calls process() which internally calls vae_decode_fn.
// Returns decoded audio via audio_out (caller-allocated [2 * T_audio_max] per batch item).
static int lua_call_postprocess(
    LuaPlugin &   plugin,
    const float * latents,     // [T * 64] time-major for one batch item
    int           T_latent,    // number of latent frames
    int           C_lat,       // latent channels (64)
    int           C_aud,       // audio channels (2)
    float *       audio_out,   // output: [2 * T_audio] interleaved by channel
    int           max_T_audio,
    PostprocessVaeDecodeFn vae_decode_fn,
    const std::unordered_map<std::string, std::string> & params,
    const LuaModelContext & model_ctx = LuaModelContext{})
{
    lua_State * L = plugin.L;
    if (!L) return -1;

    lua_inject_params(L, params, plugin.name);
    lua_inject_model_context(L, model_ctx);

    // Push process() function
    lua_getglobal(L, "process");
    if (!lua_isfunction(L, -1)) {
        fprintf(stderr, "[Plugins] ERROR: postprocess '%s' has no process() function\n",
                plugin.name.c_str());
        lua_pop(L, 1);
        return -1;
    }

    int upscale  = 1920;
    int final_samples = T_latent * upscale;

    // Arg 1: latents as Lua table (1-indexed, channel-major [C_lat, T_latent])
    // C++ layout is time-major [T_latent, C_lat] — transpose for Lua core module
    lua_newtable(L);
    for (int c = 0; c < C_lat; c++) {
        for (int t = 0; t < T_latent; t++) {
            int lua_idx = c * T_latent + t + 1;          // [C, T] channel-major
            int cpp_idx = t * C_lat + c;                  // [T, C] time-major
            lua_pushinteger(L, lua_idx);
            lua_pushnumber(L, (double) latents[cpp_idx]);
            lua_settable(L, -3);
        }
    }

    // Arg 2: B (batch dimension — always 1 here, called per-batch-item)
    lua_pushinteger(L, 1);

    // Arg 3: C_lat (latent channels)
    lua_pushinteger(L, C_lat);

    // Arg 4: W (latent width = T_latent)
    lua_pushinteger(L, T_latent);

    // Arg 5: C_aud (audio channels)
    lua_pushinteger(L, C_aud);

    // Arg 6: final_samples (expected audio length per channel)
    lua_pushinteger(L, final_samples);

    // Arg 7: upscale_factor
    lua_pushinteger(L, upscale);

    // Arg 8: vae_decode_fn closure
    // Captures the C++ decode function via light userdata.
    // Lua signature: vae_decode_fn(latent_table, T_latent) → audio_table, T_audio
    auto * fn_ptr = new PostprocessVaeDecodeFn(std::move(vae_decode_fn));
    lua_pushlightuserdata(L, fn_ptr);
    lua_pushcclosure(L, [](lua_State * Ls) -> int {
        auto * fn = (PostprocessVaeDecodeFn *) lua_touserdata(Ls, lua_upvalueindex(1));

        // Read latent table from Lua (arg 1) — 1-indexed, channel-major [C, T]
        luaL_checktype(Ls, 1, LUA_TTABLE);
        int T_lat = (int) luaL_checkinteger(Ls, 2);
        int C_l   = 64;
        int n_lat = T_lat * C_l;

        // Read channel-major [C, T] from Lua and transpose to time-major [T, C] for C++
        std::vector<float> lat_buf(n_lat);
        for (int c = 0; c < C_l; c++) {
            for (int t = 0; t < T_lat; t++) {
                int lua_idx = c * T_lat + t + 1;     // [C, T] channel-major
                lua_pushinteger(Ls, lua_idx);
                lua_gettable(Ls, 1);
                lat_buf[t * C_l + c] = (float) lua_tonumber(Ls, -1);  // [T, C] time-major
                lua_pop(Ls, 1);
            }
        }

        // Decode
        int max_T = T_lat * 1920;
        std::vector<float> aud_buf(2 * max_T);
        int T_audio = (*fn)(lat_buf.data(), T_lat, aud_buf.data(), max_T);

        if (T_audio < 0) {
            lua_pushnil(Ls);
            lua_pushinteger(Ls, 0);
            return 2;
        }

        // Return audio as Lua table (1-indexed, [B * C_aud * T_audio])
        lua_newtable(Ls);
        int total = 2 * T_audio;
        for (int i = 0; i < total; i++) {
            lua_pushinteger(Ls, i + 1);
            lua_pushnumber(Ls, (double) aud_buf[i]);
            lua_settable(Ls, -3);
        }
        lua_pushinteger(Ls, T_audio);
        return 2;
    }, 1);

    // 8 args: latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn
    // Returns: audio_table, T_audio
    if (lua_pcall(L, 8, 2, 0) != LUA_OK) {
        fprintf(stderr, "[Plugins] ERROR in postprocess '%s' process(): %s\n",
                plugin.name.c_str(), lua_tostring(L, -1));
        lua_pop(L, 1);
        delete fn_ptr;
        return -1;
    }

    // Read result: audio table at -2, T_audio at -1
    int T_audio = (int) lua_tointeger(L, -1);
    if (T_audio <= 0 || !lua_istable(L, -2)) {
        fprintf(stderr, "[Plugins] ERROR: postprocess '%s' returned invalid audio (T=%d)\n",
                plugin.name.c_str(), T_audio);
        lua_pop(L, 2);
        delete fn_ptr;
        return -1;
    }

    // Copy audio from Lua table to output buffer (1-indexed)
    int total = 2 * T_audio;
    if (T_audio > max_T_audio) {
        T_audio = max_T_audio;
        total   = 2 * T_audio;
    }
    for (int i = 0; i < total; i++) {
        lua_pushinteger(L, i + 1);
        lua_gettable(L, -3);
        audio_out[i] = (float) lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    lua_pop(L, 2);
    delete fn_ptr;
    return T_audio;
}

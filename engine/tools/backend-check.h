#pragma once
// ace-server --backend-check: model-free GGML backend diagnostic for release smoke tests.
//
// Loads backends the way backend_init does (ggml_backend_load_all, then
// ggml_backend_init_best) and prints one JSON report on stdout: each registered
// backend with its module and devices, the device the engine would pick, a CPU
// backend init, every ggml-*.dll beside the exe with whether it registered, and
// the GGML log lines emitted on the way. Release GGML loads backends silently,
// so a missing dependency looks exactly like "no GPU"; for a DLL that did not
// register the report names the Windows load error and the first import that
// cannot be loaded, following the dependency chain to the missing leaf.
// Pass/fail policy lives in release/smoke-artifact.mjs, not here.

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "yyjson.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace backend_check {

// Compute directly on the selected backend, with no scheduler CPU fallback.
static bool self_test(ggml_backend_t backend) {
    const auto started = std::chrono::steady_clock::now();
    const char * error = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    if (!backend) {
        error = "device initialization failed";
    } else {
        ggml_init_params params = {2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true};
        ctx = ggml_init(params);
        if (!ctx) {
            error = "graph allocation failed";
        } else {
            ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
            ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
            ggml_build_forward_expand(graph, output);
            if (!ggml_backend_supports_op(backend, output)) {
                error = "device does not support the probe kernel";
            } else if (!(buffer = ggml_backend_alloc_ctx_tensors(ctx, backend))) {
                error = "device buffer allocation failed";
            } else {
                float values[32];
                for (int i = 0; i < 32; ++i) values[i] = (float) (i + 1);
                ggml_backend_tensor_set(input, values, 0, sizeof(values));
                ggml_status status = ggml_backend_graph_compute(backend, graph);
#ifdef ACE_ENABLE_FAULT_INJECTION
                const char * fault = std::getenv("ACE_FAULT_BACKEND_ERROR");
                if (fault && !strcmp(fault, "startup")) status = GGML_STATUS_FAILED;
#endif
                if (status != GGML_STATUS_SUCCESS) {
                    error = "probe kernel execution failed";
                } else {
                    ggml_backend_tensor_get(output, values, 0, sizeof(values));
                    for (int i = 0; i < 32; ++i) {
                        if (!std::isfinite(values[i]) || values[i] != 2.0f * (i + 1)) {
                            error = "probe kernel returned incorrect values";
                            break;
                        }
                    }
                }
            }
        }
    }
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (error) {
        fprintf(stderr, "[Backend self-test] FATAL: %s on %s. Check the device driver and backend libraries.\n",
                error, backend ? ggml_backend_name(backend) : "unavailable device");
    } else {
        fprintf(stderr, "[Backend self-test] PASS on %s (%.2f ms)\n", ggml_backend_name(backend), ms);
    }
    return error == nullptr;
}

static std::vector<std::string> g_log;

static void log_cb(ggml_log_level level, const char * text, void *) {
    fputs(text, stderr);
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    if (level == GGML_LOG_LEVEL_CONT && !g_log.empty()) {
        g_log.back() += line;
    } else if (!line.empty()) {
        g_log.push_back(line);
    }
}

static const char * dev_type(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
    }
    return "unknown";
}

#ifdef _WIN32
static std::string narrow(const std::wstring & w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int) w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

static std::wstring lower(std::wstring s) {
    for (auto & c : s) {
        c = towlower(c);
    }
    return s;
}

static std::string win_error(DWORD err) {
    wchar_t buf[512] = {};
    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err, 0, buf, 512, nullptr);
    std::wstring msg(buf);
    while (!msg.empty() && (msg.back() == L'\n' || msg.back() == L'\r' || msg.back() == L' ')) {
        msg.pop_back();
    }
    return std::to_string((unsigned long) err) + ": " + narrow(msg);
}

// Full path of the module holding addr. A backend reg is a static object inside
// the DLL that registered it, so this names the file GGML actually loaded.
static std::wstring module_path(const void * addr) {
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) addr, &h)) {
        return L"";
    }
    wchar_t path[MAX_PATH];
    DWORD   n = GetModuleFileNameW(h, path, MAX_PATH);
    return std::wstring(path, n);
}

// Static import names of a PE file, read from an image mapping that runs no
// code and resolves nothing. Delay-loaded imports cannot fail a load, so they
// are not listed.
static std::vector<std::wstring> imports(const std::wstring & path) {
    std::vector<std::wstring> out;
    HMODULE h = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!h) {
        return out;
    }
    auto * base = (const BYTE *) ((ULONG_PTR) h & ~(ULONG_PTR) 3);
    auto * dos  = (const IMAGE_DOS_HEADER *) base;
    auto * nt   = (const IMAGE_NT_HEADERS *) (base + dos->e_lfanew);
    const auto & dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dos->e_magic == IMAGE_DOS_SIGNATURE && nt->Signature == IMAGE_NT_SIGNATURE && dir.VirtualAddress) {
        for (auto * d = (const IMAGE_IMPORT_DESCRIPTOR *) (base + dir.VirtualAddress); d->Name; d++) {
            const char * name = (const char *) (base + d->Name);
            out.emplace_back(name, name + strlen(name));
        }
    }
    FreeLibrary(h);
    return out;
}

struct Missing {
    std::string chain;  // "cublas64_13.dll -> cublasLt64_13.dll"
    std::string leaf;   // the import that will not load
    std::string error;  // Windows error for the leaf
    DWORD       code   = 0;
    bool        exists = false;  // the leaf is on the loader's search path
};

// Depth-first: the first import of path that will not load. A dependency that
// exists but fails with "module not found" is itself missing something, so
// follow it down to the file that is really absent (or the real error).
static Missing missing_import(const std::wstring & path, std::set<std::wstring> & seen, int depth) {
    for (const auto & name : imports(path)) {
        if (!seen.insert(lower(name)).second) {
            continue;
        }
        HMODULE h = LoadLibraryW(name.c_str());
        if (h) {
            FreeLibrary(h);
            continue;
        }
        DWORD   err    = GetLastError();
        wchar_t found[MAX_PATH];
        bool    exists = SearchPathW(nullptr, name.c_str(), nullptr, MAX_PATH, found, nullptr) != 0;
        if (err == ERROR_MOD_NOT_FOUND && depth < 6 && exists) {
            Missing deeper = missing_import(found, seen, depth + 1);
            if (!deeper.leaf.empty()) {
                deeper.chain = narrow(name) + " -> " + deeper.chain;
                return deeper;
            }
        }
        return { narrow(name), narrow(name), win_error(err), err, exists };
    }
    return {};
}

// ggml-vulkan returns a NULL reg and swallows the reason when instance creation
// fails. Ask the loader directly: VK_ERROR_INCOMPATIBLE_DRIVER (-9) means no
// installable client driver. Hand-declared so CPU and CUDA builds need no SDK.
static int vk_create_instance_result() {
    struct VkInstanceCreateInfo {
        int                 sType;  // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
        const void *        pNext;
        unsigned            flags;
        const void *        pApplicationInfo;
        unsigned            enabledLayerCount;
        const char * const * ppEnabledLayerNames;
        unsigned            enabledExtensionCount;
        const char * const * ppEnabledExtensionNames;
    };
    using create_fn  = int(__stdcall *)(const VkInstanceCreateInfo *, const void *, void **);
    using destroy_fn = void(__stdcall *)(void *, const void *);
    HMODULE vk       = LoadLibraryW(L"vulkan-1.dll");
    if (!vk) {
        return 1;  // not a VkResult; the import walk reports the missing loader
    }
    auto                 create  = (create_fn) GetProcAddress(vk, "vkCreateInstance");
    auto                 destroy = (destroy_fn) GetProcAddress(vk, "vkDestroyInstance");
    VkInstanceCreateInfo info    = { 1, nullptr, 0, nullptr, 0, nullptr, 0, nullptr };
    void *               inst    = nullptr;
    int                  res     = create ? create(&info, nullptr, &inst) : 1;
    if (res == 0 && destroy) {
        destroy(inst, nullptr);
    }
    FreeLibrary(vk);
    return res;
}
#endif

static int run() {
#ifdef _WIN32
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS);
#endif
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

#ifdef _WIN32
    std::set<std::wstring> loaded_modules;
#endif
    yyjson_mut_val * regs = yyjson_mut_arr(doc);
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        yyjson_mut_val *   r   = yyjson_mut_arr_add_obj(doc, regs);
        yyjson_mut_obj_add_strcpy(doc, r, "name", ggml_backend_reg_name(reg));
#ifdef _WIN32
        std::wstring mod = module_path(reg);
        yyjson_mut_obj_add_strcpy(doc, r, "module", narrow(mod).c_str());
        loaded_modules.insert(lower(mod));
#endif
        yyjson_mut_val * devs = yyjson_mut_obj_add_arr(doc, r, "devices");
        for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); d++) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
            yyjson_mut_val *   o   = yyjson_mut_arr_add_obj(doc, devs);
            yyjson_mut_obj_add_strcpy(doc, o, "name", ggml_backend_dev_name(dev));
            yyjson_mut_obj_add_strcpy(doc, o, "description", ggml_backend_dev_description(dev));
            yyjson_mut_obj_add_str(doc, o, "type", dev_type(dev));
        }
    }
    yyjson_mut_obj_add_val(doc, root, "registered", regs);

    // What backend_init would run on.
    ggml_backend_t best = ggml_backend_init_best();
    yyjson_mut_obj_add_strcpy(doc, root, "selected", best ? ggml_backend_name(best) : "");
    const bool kernel_ok = self_test(best);
    yyjson_mut_obj_add_bool(doc, root, "kernel_ok", kernel_ok);
    if (best) {
        ggml_backend_free(best);
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    yyjson_mut_val *   c       = yyjson_mut_obj_add_obj(doc, root, "cpu");
    const bool         cpu_ok  = cpu != nullptr;
    yyjson_mut_obj_add_bool(doc, c, "initialised", cpu_ok);
    if (cpu_dev) {
        yyjson_mut_obj_add_strcpy(doc, c, "description", ggml_backend_dev_description(cpu_dev));
#ifdef _WIN32
        yyjson_mut_obj_add_strcpy(doc, c, "module", narrow(module_path(ggml_backend_dev_backend_reg(cpu_dev))).c_str());
#endif
    }
    if (cpu) {
        ggml_backend_free(cpu);
    }

#ifdef _WIN32
    // Every backend DLL beside the exe; diagnose the ones that did not register.
    wchar_t exe[MAX_PATH];
    DWORD   n   = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir(exe, n);
    dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
    yyjson_mut_val * files = yyjson_mut_obj_add_arr(doc, root, "files");
    WIN32_FIND_DATAW fd;
    HANDLE           fh = FindFirstFileW((dir + L"ggml-*.dll").c_str(), &fd);
    for (BOOL more = fh != INVALID_HANDLE_VALUE; more; more = FindNextFileW(fh, &fd)) {
        std::wstring name(fd.cFileName);
        if (lower(name) == L"ggml-base.dll") {
            continue;
        }
        std::wstring     full = dir + name;
        yyjson_mut_val * f    = yyjson_mut_arr_add_obj(doc, files);
        yyjson_mut_obj_add_strcpy(doc, f, "file", narrow(name).c_str());
        bool registered = loaded_modules.count(lower(full)) > 0;
        yyjson_mut_obj_add_bool(doc, f, "registered", registered);
        if (registered) {
            continue;
        }
        HMODULE h = LoadLibraryW(full.c_str());
        if (h) {
            // Loads, so GGML passed it over: a CPU variant that lost on score,
            // or a backend whose score says this machine cannot run it.
            auto score = (int (*)()) GetProcAddress(h, "ggml_backend_score");
            auto init  = (ggml_backend_reg_t (*)()) GetProcAddress(h, "ggml_backend_init");
            if (score) {
                yyjson_mut_obj_add_int(doc, f, "score", score());
            } else if (init) {
                // No score to lose on, so a NULL reg is why GGML dropped it.
                yyjson_mut_obj_add_bool(doc, f, "init_ok", init() != nullptr);
            }
            if (lower(name) == L"ggml-vulkan.dll") {
                yyjson_mut_obj_add_int(doc, f, "vk_create_instance", vk_create_instance_result());
            }
            FreeLibrary(h);
            continue;
        }
        DWORD load_err = GetLastError();
        yyjson_mut_obj_add_strcpy(doc, f, "load_error", win_error(load_err).c_str());
        yyjson_mut_obj_add_uint(doc, f, "load_error_code", load_err);
        std::set<std::wstring> seen;
        Missing                m = missing_import(full, seen, 0);
        if (!m.leaf.empty()) {
            yyjson_mut_val * mo = yyjson_mut_obj_add_obj(doc, f, "missing");
            yyjson_mut_obj_add_strcpy(doc, mo, "dll", m.leaf.c_str());
            yyjson_mut_obj_add_strcpy(doc, mo, "chain", m.chain.c_str());
            yyjson_mut_obj_add_strcpy(doc, mo, "error", m.error.c_str());
            yyjson_mut_obj_add_uint(doc, mo, "code", m.code);
            yyjson_mut_obj_add_bool(doc, mo, "exists", m.exists);
        }
    }
    if (fh != INVALID_HANDLE_VALUE) {
        FindClose(fh);
    }
#endif

    yyjson_mut_val * log = yyjson_mut_obj_add_arr(doc, root, "log");
    for (const auto & line : g_log) {
        yyjson_mut_arr_add_strcpy(doc, log, line.c_str());
    }

    char * json = yyjson_mut_write(doc, 0, nullptr);
    printf("%s\n", json);
    fflush(stdout);
    free(json);
    yyjson_mut_doc_free(doc);
    return cpu_ok && kernel_ok ? 0 : 1;
}

}  // namespace backend_check

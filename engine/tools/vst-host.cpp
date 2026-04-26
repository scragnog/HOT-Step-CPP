// vst-host.cpp — HOT-Step VST3 Host Tool
//
// Modes:
//   --scan                          Scan for VST3 plugins, output JSON
//   --gui    --plugin <path>        Open plugin GUI in a native window
//   --process --plugin <p> --input <i> --output <o>  Offline processing
//   --process-chain --chain <json> --input <i> --output <o>  Chain mode
//
// Part of the HOT-Step 9000 CPP engine.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

// Windows headers MUST come before VST3 SDK to avoid IConnectionPoint collision
// (ocidl.h defines COM IConnectionPoint, VST3 has Steinberg::Vst::IConnectionPoint)
#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#endif

// VST3 SDK hosting
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "public.sdk/source/vst/hosting/processdata.h"
#include "public.sdk/source/vst/hosting/parameterchanges.h"
#include "public.sdk/source/vst/hosting/eventlist.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/vsttypes.h"
#include "pluginterfaces/gui/iplugview.h"
#include "pluginterfaces/base/funknown.h"
#include "public.sdk/source/common/memorystream.h"
#include "pluginterfaces/base/ibstream.h"

// yyjson for JSON output
#include "vendor/yyjson/yyjson.h"

// WAV I/O (our existing header)
#include "audio-io.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

// Global host context
static FUnknown* gHostContext = nullptr;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void init_host_context() {
    if (!gHostContext) {
        gHostContext = new HostApplication();
        // Set the global plugin context so PlugProvider passes it to IComponent::initialize()
        PluginContextFactory::instance().setPluginContext(gHostContext);
#ifdef _WIN32
        // Suppress crash/error dialogs during plugin loading
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
        // COM is needed by some plugins and by the module scanner (shell links)
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#endif
    }
}

// ── Scan Mode ────────────────────────────────────────────────────────────────

// Flush stderr before each module load attempt so crash diagnosis is possible
static int cmd_scan() {
    init_host_context();
    auto paths = VST3::Hosting::Module::getModulePaths();
    fprintf(stderr, "[vst-host] Scan found %zu module path(s)\n", paths.size());

    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, root);

    for (const auto & path : paths) {
        std::string error;
        VST3::Hosting::Module::Ptr module;

        try {
            module = VST3::Hosting::Module::create(path, error);
        } catch (...) {
            continue;
        }

        if (!module) continue;


        auto & factory = module->getFactory();
        auto classInfos = factory.classInfos();
        int audio_effects = 0;

        for (auto & classInfo : classInfos) {
            if (classInfo.category() != kVstAudioEffectClass) continue;
            audio_effects++;

            yyjson_mut_val * obj = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, obj, "name", classInfo.name().c_str());
            yyjson_mut_obj_add_strcpy(doc, obj, "vendor", classInfo.vendor().c_str());
            yyjson_mut_obj_add_strcpy(doc, obj, "version", classInfo.version().c_str());
            yyjson_mut_obj_add_strcpy(doc, obj, "path", path.c_str());
            yyjson_mut_obj_add_strcpy(doc, obj, "uid", classInfo.ID().toString().c_str());
            yyjson_mut_obj_add_strcpy(doc, obj, "subcategories",
                                   classInfo.subCategoriesString().c_str());
            yyjson_mut_arr_append(root, obj);
        }
        fprintf(stderr, "OK (%d effects)\n", audio_effects);
    }

    size_t json_len = 0;
    char * json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, &json_len);
    if (json && json_len > 0) {
#ifdef _WIN32
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD written = 0;
        WriteFile(hStdout, json, (DWORD)json_len, &written, nullptr);
        WriteFile(hStdout, "\n", 1, &written, nullptr);
        FlushFileBuffers(hStdout);
#else
        fputs(json, stdout);
        fputs("\n", stdout);
        fflush(stdout);
#endif
        free(json);
    }
    yyjson_mut_doc_free(doc);
    return 0;
}

// ── Process Mode ─────────────────────────────────────────────────────────────

struct PluginInstance {
    VST3::Hosting::Module::Ptr module;
    IPtr<PlugProvider>         provider;
    IComponent *               component  = nullptr;
    IAudioProcessor *          processor  = nullptr;
    bool                       active     = false;

    ~PluginInstance() {
        if (active && processor) {
            processor->setProcessing(false);
            component->setActive(false);
        }
    }
};

static bool load_plugin(const std::string & path, PluginInstance & inst) {
    std::string error;
    inst.module = VST3::Hosting::Module::create(path, error);
    if (!inst.module) {
        fprintf(stderr, "[vst-host] Failed to load module: %s\n  Error: %s\n",
                path.c_str(), error.c_str());
        return false;
    }

    auto & factory = inst.module->getFactory();
    for (auto & classInfo : factory.classInfos()) {
        if (classInfo.category() == kVstAudioEffectClass) {
            inst.provider = owned(new PlugProvider(factory, classInfo, true));
            break;
        }
    }

    if (!inst.provider) {
        fprintf(stderr, "[vst-host] No audio effect class found in: %s\n", path.c_str());
        return false;
    }

    inst.component = inst.provider->getComponent();
    if (!inst.component) {
        fprintf(stderr, "[vst-host] Failed to get component\n");
        return false;
    }

    if (inst.component->queryInterface(IAudioProcessor::iid,
                                        (void **)&inst.processor) != kResultOk) {
        fprintf(stderr, "[vst-host] Component does not support IAudioProcessor\n");
        return false;
    }

    return true;
}

static bool load_state(PluginInstance & inst, const std::string & state_path) {
    std::ifstream f(state_path, std::ios::binary);
    if (!f.is_open()) return false;

    f.seekg(0, std::ios::end);
    size_t size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    f.read(data.data(), (std::streamsize)size);

    // Use Steinberg's stream to pass state to the component
    auto * stream = new Steinberg::MemoryStream(data.data(), (int32)size);
    tresult res = inst.component->setState(stream);
    if (res != kResultOk) {
        fprintf(stderr, "[vst-host] WARNING: setState returned %d\n", res);
    }

    // Also restore controller state if separate controller exists
    auto controller = inst.provider->getController();
    if (controller) {
        stream->seek(0, IBStream::kIBSeekSet, nullptr);
        controller->setComponentState(stream);
    }
    stream->release();

    return true;
}

static bool save_state(PluginInstance & inst, const std::string & state_path) {
    auto * stream = new Steinberg::MemoryStream();
    tresult res = inst.component->getState(stream);
    if (res != kResultOk) {
        fprintf(stderr, "[vst-host] getState failed: %d\n", res);
        stream->release();
        return false;
    }

    std::ofstream f(state_path, std::ios::binary);
    if (!f.is_open()) { stream->release(); return false; }

    f.write((const char *)stream->getData(), (std::streamsize)stream->getSize());
    stream->release();
    return true;
}

static bool setup_processing(PluginInstance & inst, int sample_rate, int block_size,
                              int32 process_mode = kOffline) {
    ProcessSetup setup;
    setup.processMode       = process_mode;
    setup.symbolicSampleSize = kSample32;
    setup.maxSamplesPerBlock = block_size;
    setup.sampleRate         = (double)sample_rate;

    if (inst.processor->setupProcessing(setup) != kResultOk) {
        fprintf(stderr, "[vst-host] setupProcessing failed\n");
        return false;
    }

    // Activate all audio buses
    int32 numInputBuses  = inst.component->getBusCount(kAudio, kInput);
    int32 numOutputBuses = inst.component->getBusCount(kAudio, kOutput);
    for (int32 i = 0; i < numInputBuses; i++)
        inst.component->activateBus(kAudio, kInput, i, true);
    for (int32 i = 0; i < numOutputBuses; i++)
        inst.component->activateBus(kAudio, kOutput, i, true);

    inst.component->setActive(true);
    inst.processor->setProcessing(true);
    inst.active = true;

    return true;
}

static int cmd_process(const char * plugin_path, const char * input_path,
                       const char * output_path, const char * state_path) {
    init_host_context();

    // Load audio
    int T = 0, sr = 0;
    float * audio = audio_io_read_wav(input_path, &T, &sr);
    if (!audio || T <= 0) {
        fprintf(stderr, "[vst-host] Failed to read input: %s\n", input_path);
        return 1;
    }
    fprintf(stderr, "[vst-host] Input: %d samples @ %d Hz (%.1f sec)\n",
            T, sr, (float)T / sr);

    // Load plugin
    PluginInstance inst;
    if (!load_plugin(plugin_path, inst)) return 1;

    // Restore state
    if (state_path && state_path[0]) {
        if (load_state(inst, state_path)) {
            fprintf(stderr, "[vst-host] State loaded from: %s\n", state_path);
        }
    }

    // Setup processing
    const int block_size = 4096;
    if (!setup_processing(inst, sr, block_size)) return 1;

    // audio_io_read_wav returns planar [L:T][R:T]
    float * left_in  = audio;
    float * right_in = audio + T;

    // Allocate output buffers (planar)
    std::vector<float> left_out(T, 0.0f);
    std::vector<float> right_out(T, 0.0f);

    // Process in blocks
    int pos = 0;
    while (pos < T) {
        int n = std::min(block_size, T - pos);

        // Set up process data manually
        float * in_bufs[2]  = { left_in + pos,       right_in + pos };
        float * out_bufs[2] = { left_out.data() + pos, right_out.data() + pos };

        AudioBusBuffers input_bus;
        input_bus.numChannels = 2;
        input_bus.silenceFlags = 0;
        input_bus.channelBuffers32 = in_bufs;

        AudioBusBuffers output_bus;
        output_bus.numChannels = 2;
        output_bus.silenceFlags = 0;
        output_bus.channelBuffers32 = out_bufs;

        ProcessData data;
        data.processMode         = kOffline;
        data.symbolicSampleSize  = kSample32;
        data.numSamples          = n;
        data.numInputs           = 1;
        data.numOutputs          = 1;
        data.inputs              = &input_bus;
        data.outputs             = &output_bus;
        data.inputParameterChanges  = nullptr;
        data.outputParameterChanges = nullptr;
        data.inputEvents         = nullptr;
        data.outputEvents        = nullptr;
        data.processContext      = nullptr;

        inst.processor->process(data);
        pos += n;
    }

    // Write output as WAV: reassemble planar [L:T][R:T] and encode
    std::vector<float> output_planar(T * 2);
    memcpy(output_planar.data(),     left_out.data(),  T * sizeof(float));
    memcpy(output_planar.data() + T, right_out.data(), T * sizeof(float));

    std::string wav_data = audio_encode_wav(output_planar.data(), T, sr, WAV_F32);
    {
        std::ofstream f(output_path, std::ios::binary);
        if (!f.is_open()) {
            fprintf(stderr, "[vst-host] Failed to write output: %s\n", output_path);
            free(audio);
            return 1;
        }
        f.write(wav_data.data(), (std::streamsize)wav_data.size());
    }

    fprintf(stderr, "[vst-host] Output written: %s\n", output_path);
    free(audio);
    return 0;
}

// ── GUI Mode (Windows) ──────────────────────────────────────────────────────

#ifdef _WIN32
static IPlugView * g_plugView = nullptr;
static PluginInstance * g_guiInst = nullptr;
static const char * g_guiStatePath = nullptr;
static bool g_guiRunning = true;

static LRESULT CALLBACK VstWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CLOSE:
            g_guiRunning = false;
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

static int cmd_gui(const char * plugin_path, const char * state_path) {
    init_host_context();

    PluginInstance inst;
    if (!load_plugin(plugin_path, inst)) return 1;

    // Load state if exists
    if (state_path && state_path[0]) {
        load_state(inst, state_path);
    }

    // Setup processing (needed for some plugins to show GUI correctly)
    setup_processing(inst, 48000, 4096);

    // Get edit controller and create view
    auto controller = inst.provider->getController();
    if (!controller) {
        fprintf(stderr, "[vst-host] No edit controller available\n");
        return 1;
    }

    IPlugView * view = controller->createView(ViewType::kEditor);
    if (!view) {
        fprintf(stderr, "[vst-host] Plugin has no editor view\n");
        return 1;
    }

    // Get preferred size
    ViewRect rect;
    if (view->getSize(&rect) != kResultOk) {
        rect.left = 0; rect.top = 0;
        rect.right = 800; rect.bottom = 600;
    }

    int w = rect.right - rect.left;
    int h = rect.bottom - rect.top;

    // Register window class
    WNDCLASSA wc = {};
    wc.lpfnWndProc   = VstWndProc;
    wc.hInstance      = GetModuleHandleA(nullptr);
    wc.lpszClassName  = "VstHostWindow";
    wc.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    // Adjust window rect for non-client area
    RECT wr = { 0, 0, w, h };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX), FALSE);

    HWND hwnd = CreateWindowA("VstHostWindow", "HOT-Step VST3 Plugin Editor",
                              WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX),
                              CW_USEDEFAULT, CW_USEDEFAULT,
                              wr.right - wr.left, wr.bottom - wr.top,
                              nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);

    if (!hwnd) {
        fprintf(stderr, "[vst-host] Failed to create window\n");
        view->release();
        return 1;
    }

    // Attach plugin view to window
    if (view->attached(hwnd, kPlatformTypeHWND) != kResultOk) {
        fprintf(stderr, "[vst-host] Failed to attach plugin view\n");
        view->release();
        DestroyWindow(hwnd);
        return 1;
    }

    g_plugView = view;
    g_guiInst = &inst;
    g_guiStatePath = state_path;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    fprintf(stderr, "[vst-host] Plugin GUI opened (%dx%d). Close window to save state.\n", w, h);

    // Message loop
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Detach view
    view->removed();
    view->release();

    // Save state on close
    if (state_path && state_path[0]) {
        if (save_state(inst, state_path)) {
            fprintf(stderr, "[vst-host] State saved to: %s\n", state_path);
        }
    }

    return 0;
}
#endif

// ── Chain Mode ───────────────────────────────────────────────────────────────

static int cmd_process_chain(const char * chain_json_path, const char * input_path,
                             const char * output_path) {
    // Read chain JSON
    std::ifstream f(chain_json_path);
    if (!f.is_open()) {
        fprintf(stderr, "[vst-host] Failed to read chain: %s\n", chain_json_path);
        return 1;
    }
    std::string json_str((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    yyjson_doc * doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!doc) {
        fprintf(stderr, "[vst-host] Failed to parse chain JSON\n");
        return 1;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * plugins_arr = yyjson_obj_get(root, "plugins");
    if (!plugins_arr || !yyjson_is_arr(plugins_arr)) {
        fprintf(stderr, "[vst-host] Chain JSON must have a 'plugins' array\n");
        yyjson_doc_free(doc);
        return 1;
    }

    // Build list of enabled plugins
    struct ChainEntry {
        std::string path;
        std::string state;
    };
    std::vector<ChainEntry> entries;

    size_t idx, max;
    yyjson_val * val;
    yyjson_arr_foreach(plugins_arr, idx, max, val) {
        yyjson_val * enabled_val = yyjson_obj_get(val, "enabled");
        if (enabled_val && !yyjson_get_bool(enabled_val)) continue;

        yyjson_val * path_val  = yyjson_obj_get(val, "path");
        yyjson_val * state_val = yyjson_obj_get(val, "state");
        if (!path_val) continue;

        ChainEntry e;
        e.path  = yyjson_get_str(path_val);
        e.state = state_val ? yyjson_get_str(state_val) : "";
        entries.push_back(e);
    }
    yyjson_doc_free(doc);

    if (entries.empty()) {
        fprintf(stderr, "[vst-host] No enabled plugins in chain, copying input to output\n");
        // Just copy input to output
        std::ifstream src(input_path, std::ios::binary);
        std::ofstream dst(output_path, std::ios::binary);
        dst << src.rdbuf();
        return 0;
    }

    fprintf(stderr, "[vst-host] Processing chain with %zu plugin(s)\n", entries.size());

    // Process sequentially: input → plugin1 → plugin2 → ... → output
    std::string current_input = input_path;
    std::string temp_path;

    for (size_t i = 0; i < entries.size(); i++) {
        bool is_last = (i == entries.size() - 1);
        std::string current_output;

        if (is_last) {
            current_output = output_path;
        } else {
            // Use temp file for intermediate results
            temp_path = std::string(output_path) + ".vst_temp_" + std::to_string(i) + ".wav";
            current_output = temp_path;
        }

        fprintf(stderr, "[vst-host]   [%zu/%zu] %s\n",
                i + 1, entries.size(), entries[i].path.c_str());

        int ret = cmd_process(entries[i].path.c_str(),
                              current_input.c_str(),
                              current_output.c_str(),
                              entries[i].state.c_str());
        if (ret != 0) {
            fprintf(stderr, "[vst-host] Plugin %zu failed, aborting chain\n", i);
            return ret;
        }

        current_input = current_output;
    }

    // Clean up temp files
    for (size_t i = 0; i + 1 < entries.size(); i++) {
        std::string tf = std::string(output_path) + ".vst_temp_" + std::to_string(i) + ".wav";
        remove(tf.c_str());
    }

    return 0;
}

// ── Monitor Mode (Windows WASAPI) ────────────────────────────────────────────

#ifdef _WIN32

struct MonitorState {
    // Audio data (planar [L:T][R:T])
    float * audio = nullptr;
    int     T     = 0;
    int     sr    = 0;
    int     pos   = 0;
    bool    loop  = true;

    // Chain
    std::vector<PluginInstance> plugins;
    std::vector<std::string>   state_paths;

    // Control
    std::string control_file;
    std::string status_file;
    std::string current_track;
    FILETIME    control_mtime = {};
    bool        running = true;

    // Processing buffers (reused per block)
    std::vector<float> buf_a_L, buf_a_R, buf_b_L, buf_b_R;
};

static bool monitor_load_track(MonitorState & ms, const std::string & path) {
    int T = 0, sr = 0;
    float * audio = audio_io_read_wav(path.c_str(), &T, &sr);
    if (!audio || T <= 0) {
        fprintf(stderr, "[monitor] Failed to load: %s\n", path.c_str());
        return false;
    }
    if (ms.audio) free(ms.audio);
    ms.audio = audio;
    ms.T = T;
    ms.sr = sr;
    ms.pos = 0;
    ms.current_track = path;
    fprintf(stderr, "[monitor] Loaded: %s (%d frames, %d Hz, %.1fs)\n",
            path.c_str(), T, sr, (float)T / sr);
    return true;
}

static void monitor_check_control(MonitorState & ms) {
    if (ms.control_file.empty()) return;
    HANDLE hFile = CreateFileA(ms.control_file.c_str(), GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, 0, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return;

    FILETIME ft;
    GetFileTime(hFile, nullptr, nullptr, &ft);
    bool changed = (ft.dwHighDateTime != ms.control_mtime.dwHighDateTime ||
                    ft.dwLowDateTime  != ms.control_mtime.dwLowDateTime);
    if (!changed) { CloseHandle(hFile); return; }
    ms.control_mtime = ft;

    DWORD size = GetFileSize(hFile, nullptr);
    if (size == 0 || size == INVALID_FILE_SIZE) { CloseHandle(hFile); return; }
    std::vector<char> buf(size + 1, 0);
    DWORD bytesRead = 0;
    ReadFile(hFile, buf.data(), size, &bytesRead, nullptr);
    CloseHandle(hFile);

    yyjson_doc * doc = yyjson_read(buf.data(), bytesRead, 0);
    if (!doc) return;
    yyjson_val * root = yyjson_doc_get_root(doc);

    yyjson_val * action_val = yyjson_obj_get(root, "action");
    if (action_val) {
        const char * action = yyjson_get_str(action_val);
        if (action && !strcmp(action, "stop")) {
            ms.running = false;
            yyjson_doc_free(doc);
            return;
        }
    }

    yyjson_val * track_val = yyjson_obj_get(root, "track");
    if (track_val) {
        const char * track = yyjson_get_str(track_val);
        if (track && ms.current_track != track) {
            monitor_load_track(ms, track);
        }
    }

    // Seek support: { "seek": 42.5 } = jump to 42.5 seconds
    yyjson_val * seek_val = yyjson_obj_get(root, "seek");
    if (seek_val && yyjson_is_num(seek_val)) {
        double seek_sec = yyjson_get_real(seek_val);
        int new_pos = (int)(seek_sec * ms.sr);
        if (new_pos < 0) new_pos = 0;
        if (new_pos >= ms.T) new_pos = ms.T - 1;
        ms.pos = new_pos;
    }

    yyjson_doc_free(doc);
}

static void monitor_process_block(MonitorState & ms, float * out_L, float * out_R, int n) {
    if (!ms.audio || ms.T <= 0) {
        memset(out_L, 0, n * sizeof(float));
        memset(out_R, 0, n * sizeof(float));
        return;
    }

    // Ensure buffers are large enough
    if ((int)ms.buf_a_L.size() < n) {
        ms.buf_a_L.resize(n); ms.buf_a_R.resize(n);
        ms.buf_b_L.resize(n); ms.buf_b_R.resize(n);
    }

    // Read from WAV (with loop)
    float * src_L = ms.buf_a_L.data();
    float * src_R = ms.buf_a_R.data();
    for (int i = 0; i < n; i++) {
        int p = ms.pos + i;
        if (p >= ms.T) {
            if (ms.loop) { p = p % ms.T; }
            else         { src_L[i] = 0; src_R[i] = 0; continue; }
        }
        src_L[i] = ms.audio[p];
        src_R[i] = ms.audio[ms.T + p];
    }
    ms.pos += n;
    if (ms.loop && ms.pos >= ms.T) ms.pos = ms.pos % ms.T;

    // Process through each plugin
    float * cur_L = src_L, * cur_R = src_R;
    float * dst_L = ms.buf_b_L.data(), * dst_R = ms.buf_b_R.data();

    for (size_t pi = 0; pi < ms.plugins.size(); pi++) {
        float * in_bufs[2]  = { cur_L, cur_R };
        float * ob[2]       = { dst_L, dst_R };
        AudioBusBuffers ib; ib.numChannels = 2; ib.silenceFlags = 0; ib.channelBuffers32 = in_bufs;
        AudioBusBuffers ob_bus; ob_bus.numChannels = 2; ob_bus.silenceFlags = 0; ob_bus.channelBuffers32 = ob;

        ProcessData pd;
        pd.processMode = kRealtime; pd.symbolicSampleSize = kSample32;
        pd.numSamples = n; pd.numInputs = 1; pd.numOutputs = 1;
        pd.inputs = &ib; pd.outputs = &ob_bus;
        pd.inputParameterChanges = nullptr; pd.outputParameterChanges = nullptr;
        pd.inputEvents = nullptr; pd.outputEvents = nullptr; pd.processContext = nullptr;

        ms.plugins[pi].processor->process(pd);

        // Swap buffers for next plugin
        std::swap(cur_L, dst_L); std::swap(cur_R, dst_R);
    }

    memcpy(out_L, cur_L, n * sizeof(float));
    memcpy(out_R, cur_R, n * sizeof(float));
}

// Window proc for plugin GUIs in monitor mode
static MonitorState * g_monitorState = nullptr;
static int g_monitorWindowCount = 0;

static LRESULT CALLBACK MonitorWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            g_monitorWindowCount--;
            if (g_monitorWindowCount <= 0 && g_monitorState) {
                g_monitorState->running = false;
                PostQuitMessage(0);
            }
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

static int cmd_monitor(const char * chain_json_path, const char * input_path,
                        const char * control_path, const char * status_path) {
    init_host_context();

    // Parse chain JSON
    std::ifstream cf(chain_json_path);
    if (!cf.is_open()) { fprintf(stderr, "[monitor] Failed to read chain: %s\n", chain_json_path); return 1; }
    std::string json_str((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    cf.close();

    yyjson_doc * doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!doc) { fprintf(stderr, "[monitor] Failed to parse chain JSON\n"); return 1; }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * plugins_arr = yyjson_obj_get(root, "plugins");

    struct ChainDef { std::string path, state; };
    std::vector<ChainDef> defs;
    if (plugins_arr && yyjson_is_arr(plugins_arr)) {
        size_t idx, max; yyjson_val * val;
        yyjson_arr_foreach(plugins_arr, idx, max, val) {
            yyjson_val * ev = yyjson_obj_get(val, "enabled");
            if (ev && !yyjson_get_bool(ev)) continue;
            yyjson_val * pv = yyjson_obj_get(val, "path");
            yyjson_val * sv = yyjson_obj_get(val, "state");
            if (!pv) continue;
            defs.push_back({ yyjson_get_str(pv), sv ? yyjson_get_str(sv) : "" });
        }
    }
    yyjson_doc_free(doc);

    if (defs.empty()) { fprintf(stderr, "[monitor] No enabled plugins\n"); return 1; }

    // Setup monitor state
    MonitorState ms;
    ms.control_file = control_path ? control_path : "";
    ms.status_file  = status_path  ? status_path  : "";

    // Load initial track
    if (!monitor_load_track(ms, input_path)) return 1;

    // Load plugins
    ms.plugins.resize(defs.size());
    ms.state_paths.resize(defs.size());
    const int block_size = 512;  // Low latency for real-time

    for (size_t i = 0; i < defs.size(); i++) {
        ms.state_paths[i] = defs[i].state;
        if (!load_plugin(defs[i].path, ms.plugins[i])) {
            fprintf(stderr, "[monitor] Failed to load plugin %zu: %s\n", i, defs[i].path.c_str());
            return 1;
        }
        if (!defs[i].state.empty()) load_state(ms.plugins[i], defs[i].state);
        if (!setup_processing(ms.plugins[i], ms.sr, block_size, kRealtime)) {
            fprintf(stderr, "[monitor] Failed to setup plugin %zu\n", i);
            return 1;
        }
    }

    // Register window class for plugin GUIs
    WNDCLASSA wc = {};
    wc.lpfnWndProc  = MonitorWndProc;
    wc.hInstance     = GetModuleHandleA(nullptr);
    wc.lpszClassName = "VstMonitorWindow";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    g_monitorState = &ms;
    g_monitorWindowCount = 0;

    // Open GUI for each plugin
    std::vector<IPlugView*> views;
    for (size_t i = 0; i < ms.plugins.size(); i++) {
        auto ctrl = ms.plugins[i].provider->getController();
        if (!ctrl) continue;
        IPlugView * view = ctrl->createView(ViewType::kEditor);
        if (!view) continue;

        ViewRect rect;
        if (view->getSize(&rect) != kResultOk) { rect = {0, 0, 800, 600}; }
        int w = rect.right - rect.left, h = rect.bottom - rect.top;
        RECT wr = { 0, 0, w, h };
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX), FALSE);

        std::string title = "Monitor: " + defs[i].path.substr(defs[i].path.find_last_of("/\\") + 1);
        HWND hwnd = CreateWindowA("VstMonitorWindow", title.c_str(),
                                   WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX),
                                   CW_USEDEFAULT, CW_USEDEFAULT,
                                   wr.right - wr.left, wr.bottom - wr.top,
                                   nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
        if (!hwnd) { view->release(); continue; }
        if (view->attached(hwnd, kPlatformTypeHWND) != kResultOk) {
            view->release(); DestroyWindow(hwnd); continue;
        }
        ShowWindow(hwnd, SW_SHOW);
        g_monitorWindowCount++;
        views.push_back(view);
    }

    fprintf(stderr, "[monitor] %d plugin GUI(s) opened\n", g_monitorWindowCount);

    // ── WASAPI init ──
    IMMDeviceEnumerator * pEnum = nullptr;
    IMMDevice * pDevice = nullptr;
    IAudioClient * pAudioClient = nullptr;
    IAudioRenderClient * pRenderClient = nullptr;
    WAVEFORMATEX * pwfx = nullptr;
    HANDLE hAudioEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT32 bufferFrames = 0;

    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                  CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&pEnum);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] Failed to create device enumerator\n"); return 1; }
    hr = pEnum->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] No default audio device\n"); return 1; }
    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] Failed to activate audio client\n"); return 1; }
    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] Failed to get mix format\n"); return 1; }

    fprintf(stderr, "[monitor] WASAPI: %d Hz, %d ch, %d bits\n",
            (int)pwfx->nSamplesPerSec, (int)pwfx->nChannels, (int)pwfx->wBitsPerSample);

    REFERENCE_TIME bufDuration = 200000; // 20ms buffer
    hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                   AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                   bufDuration, 0, pwfx, nullptr);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] WASAPI Initialize failed: 0x%08X\n", (unsigned)hr); return 1; }
    pAudioClient->SetEventHandle(hAudioEvent);
    pAudioClient->GetBufferSize(&bufferFrames);
    hr = pAudioClient->GetService(__uuidof(IAudioRenderClient), (void**)&pRenderClient);
    if (FAILED(hr)) { fprintf(stderr, "[monitor] Failed to get render client\n"); return 1; }

    fprintf(stderr, "[monitor] Buffer: %u frames (%.1f ms)\n",
            bufferFrames, 1000.0f * bufferFrames / pwfx->nSamplesPerSec);

    // Pre-fill buffer
    {   BYTE * data; pRenderClient->GetBuffer(bufferFrames, &data);
        memset(data, 0, bufferFrames * pwfx->nBlockAlign);
        pRenderClient->ReleaseBuffer(bufferFrames, 0); }

    pAudioClient->Start();
    fprintf(stderr, "[monitor] Playing... Close all plugin windows to stop.\n");

    // Temp buffers for processed audio
    std::vector<float> proc_L(bufferFrames), proc_R(bufferFrames);
    DWORD controlCheckTick = GetTickCount();

    // ── Main loop: WASAPI + Win32 messages ──
    while (ms.running) {
        DWORD wait = MsgWaitForMultipleObjects(1, &hAudioEvent, FALSE, 100, QS_ALLINPUT);

        if (wait == WAIT_OBJECT_0) {
            // WASAPI wants data
            UINT32 padding = 0;
            pAudioClient->GetCurrentPadding(&padding);
            UINT32 available = bufferFrames - padding;
            if (available > 0) {
                BYTE * data = nullptr;
                if (SUCCEEDED(pRenderClient->GetBuffer(available, &data))) {
                    monitor_process_block(ms, proc_L.data(), proc_R.data(), (int)available);
                    // Interleave into WASAPI buffer (float32, potentially >2 channels)
                    float * fdata = (float *)data;
                    int ch = pwfx->nChannels;
                    for (UINT32 i = 0; i < available; i++) {
                        fdata[i * ch + 0] = proc_L[i];
                        fdata[i * ch + 1] = (ch >= 2) ? proc_R[i] : 0;
                        for (int c = 2; c < ch; c++) fdata[i * ch + c] = 0;
                    }
                    pRenderClient->ReleaseBuffer(available, 0);
                }
            }
        } else if (wait == WAIT_OBJECT_0 + 1) {
            // Win32 messages
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) { ms.running = false; break; }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }

        // Check control file periodically (~500ms)
        DWORD now = GetTickCount();
        if (now - controlCheckTick > 500) {
            controlCheckTick = now;
            monitor_check_control(ms);
        }

        // Write status file periodically (~200ms) for UI position display
        static DWORD statusTick = 0;
        if (!ms.status_file.empty() && now - statusTick > 200) {
            statusTick = now;
            double pos_sec = (ms.sr > 0) ? (double)ms.pos / ms.sr : 0;
            double dur_sec = (ms.sr > 0) ? (double)ms.T / ms.sr : 0;
            char sbuf[256];
            snprintf(sbuf, sizeof(sbuf),
                     "{\"position\":%.2f,\"duration\":%.2f,\"loop\":%s}",
                     pos_sec, dur_sec, ms.loop ? "true" : "false");
            HANDLE hStatus = CreateFileA(ms.status_file.c_str(), GENERIC_WRITE,
                                         FILE_SHARE_READ, nullptr,
                                         CREATE_ALWAYS, 0, nullptr);
            if (hStatus != INVALID_HANDLE_VALUE) {
                DWORD written;
                WriteFile(hStatus, sbuf, (DWORD)strlen(sbuf), &written, nullptr);
                CloseHandle(hStatus);
            }
        }
    }

    // ── Cleanup ──
    pAudioClient->Stop();
    if (pRenderClient) pRenderClient->Release();
    if (pAudioClient)  pAudioClient->Release();
    if (pDevice)       pDevice->Release();
    if (pEnum)         pEnum->Release();
    if (pwfx)          CoTaskMemFree(pwfx);
    CloseHandle(hAudioEvent);

    // Detach and release views
    for (auto * v : views) { v->removed(); v->release(); }

    // Save all plugin states
    for (size_t i = 0; i < ms.plugins.size(); i++) {
        if (!ms.state_paths[i].empty()) {
            if (save_state(ms.plugins[i], ms.state_paths[i])) {
                fprintf(stderr, "[monitor] State saved: %s\n", ms.state_paths[i].c_str());
            }
        }
    }

    if (ms.audio) free(ms.audio);
    fprintf(stderr, "[monitor] Done.\n");
    return 0;
}
#endif // _WIN32

// ── Usage + Main ─────────────────────────────────────────────────────────────

static void usage(const char * prog) {
    fprintf(stderr, "vst-host — HOT-Step VST3 Plugin Host\n\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --scan\n", prog);
    fprintf(stderr, "  %s --gui --plugin <path.vst3> [--state <file>]\n", prog);
    fprintf(stderr, "  %s --process --plugin <path.vst3> --input <in.wav> --output <out.wav> [--state <file>]\n", prog);
    fprintf(stderr, "  %s --process-chain --chain <chain.json> --input <in.wav> --output <out.wav>\n", prog);
    fprintf(stderr, "  %s --monitor --chain <chain.json> --input <in.wav> [--control <file>] [--status <file>]\n", prog);
}

int main(int argc, char * argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char * mode    = nullptr;
    const char * plugin  = nullptr;
    const char * input   = nullptr;
    const char * output  = nullptr;
    const char * state   = nullptr;
    const char * chain   = nullptr;
    const char * control = nullptr;
    const char * status  = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--scan"))          mode = "scan";
        else if (!strcmp(argv[i], "--gui"))      mode = "gui";
        else if (!strcmp(argv[i], "--process"))  mode = "process";
        else if (!strcmp(argv[i], "--process-chain")) mode = "chain";
        else if (!strcmp(argv[i], "--monitor"))  mode = "monitor";
        else if (!strcmp(argv[i], "--plugin")  && i+1 < argc)  plugin  = argv[++i];
        else if (!strcmp(argv[i], "--input")   && i+1 < argc)  input   = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1 < argc)  output  = argv[++i];
        else if (!strcmp(argv[i], "--state")   && i+1 < argc)  state   = argv[++i];
        else if (!strcmp(argv[i], "--chain")   && i+1 < argc)  chain   = argv[++i];
        else if (!strcmp(argv[i], "--control") && i+1 < argc)  control = argv[++i];
        else if (!strcmp(argv[i], "--status")  && i+1 < argc)  status  = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    if (!mode) { usage(argv[0]); return 1; }

    if (!strcmp(mode, "scan")) {
        return cmd_scan();
    }
    else if (!strcmp(mode, "gui")) {
#ifdef _WIN32
        if (!plugin) { fprintf(stderr, "Error: --plugin required\n"); return 1; }
        return cmd_gui(plugin, state);
#else
        fprintf(stderr, "Error: GUI mode only supported on Windows\n");
        return 1;
#endif
    }
    else if (!strcmp(mode, "process")) {
        if (!plugin || !input || !output) {
            fprintf(stderr, "Error: --plugin, --input, --output required\n");
            return 1;
        }
        return cmd_process(plugin, input, output, state);
    }
    else if (!strcmp(mode, "chain")) {
        if (!chain || !input || !output) {
            fprintf(stderr, "Error: --chain, --input, --output required\n");
            return 1;
        }
        return cmd_process_chain(chain, input, output);
    }
    else if (!strcmp(mode, "monitor")) {
#ifdef _WIN32
        if (!chain || !input) {
            fprintf(stderr, "Error: --chain, --input required for monitor\n");
            return 1;
        }
        return cmd_monitor(chain, input, control, status);
#else
        fprintf(stderr, "Error: Monitor mode only supported on Windows\n");
        return 1;
#endif
    }

    usage(argv[0]);
    return 1;
}


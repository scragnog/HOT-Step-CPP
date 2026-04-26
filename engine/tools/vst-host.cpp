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

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#endif

using namespace Steinberg;
using namespace Steinberg::Vst;

// Global host context
static FUnknown* gHostContext = nullptr;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void init_host_context() {
    if (!gHostContext) {
        gHostContext = new HostApplication();
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
        fprintf(stderr, "[vst-host]   Probing: %s ... ", path.c_str());
        fflush(stderr);

        std::string error;
        VST3::Hosting::Module::Ptr module;

        try {
            module = VST3::Hosting::Module::create(path, error);
        } catch (...) {
            fprintf(stderr, "EXCEPTION\n");
            continue;
        }

        if (!module) {
            fprintf(stderr, "FAIL (%s)\n", error.c_str());
            continue;
        }

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

    fprintf(stderr, "[vst-host] JSON array has %zu entries\n",
            yyjson_mut_arr_size(root));

    size_t json_len = 0;
    char * json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, &json_len);
    fprintf(stderr, "[vst-host] JSON write: ptr=%p len=%zu\n", (void*)json, json_len);
    if (json && json_len > 0) {
        // Write directly using low-level I/O to avoid plugin stdout corruption
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD written = 0;
        WriteFile(hStdout, json, (DWORD)json_len, &written, nullptr);
        WriteFile(hStdout, "\n", 1, &written, nullptr);
        FlushFileBuffers(hStdout);
        fprintf(stderr, "[vst-host] Wrote %lu bytes to stdout\n", written);
        free(json);
    } else {
        fprintf(stderr, "[vst-host] ERROR: yyjson_mut_write returned null!\n");
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

static bool setup_processing(PluginInstance & inst, int sample_rate, int block_size) {
    ProcessSetup setup;
    setup.processMode       = kOffline;
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

// ── Usage + Main ─────────────────────────────────────────────────────────────

static void usage(const char * prog) {
    fprintf(stderr, "vst-host — HOT-Step VST3 Plugin Host\n\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --scan\n", prog);
    fprintf(stderr, "  %s --gui --plugin <path.vst3> [--state <file>]\n", prog);
    fprintf(stderr, "  %s --process --plugin <path.vst3> --input <in.wav> --output <out.wav> [--state <file>]\n", prog);
    fprintf(stderr, "  %s --process-chain --chain <chain.json> --input <in.wav> --output <out.wav>\n", prog);
}

int main(int argc, char * argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char * mode    = nullptr;
    const char * plugin  = nullptr;
    const char * input   = nullptr;
    const char * output  = nullptr;
    const char * state   = nullptr;
    const char * chain   = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--scan"))          mode = "scan";
        else if (!strcmp(argv[i], "--gui"))      mode = "gui";
        else if (!strcmp(argv[i], "--process"))  mode = "process";
        else if (!strcmp(argv[i], "--process-chain")) mode = "chain";
        else if (!strcmp(argv[i], "--plugin") && i+1 < argc)  plugin = argv[++i];
        else if (!strcmp(argv[i], "--input")  && i+1 < argc)  input  = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1 < argc)  output = argv[++i];
        else if (!strcmp(argv[i], "--state")  && i+1 < argc)  state  = argv[++i];
        else if (!strcmp(argv[i], "--chain")  && i+1 < argc)  chain  = argv[++i];
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

    usage(argv[0]);
    return 1;
}

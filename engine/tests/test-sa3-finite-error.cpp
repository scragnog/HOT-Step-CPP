// CPU-only error propagation through the real SA3 orchestration and Lua bridge.
// Link with ggml, ggml-base, yyjson and LuaJIT. No weights or device are used.
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include "sa3-refine.h"

struct FakeSa3 final : Sa3Backend {
    bool poison = false;
    const char * name() const override { return "CPU-double"; }
    bool text_encode(const int64_t *, int, float * out) override {
        std::fill(out, out + SA3_TOK_LEN * SA3_COND_DIM, 0.0f); return true;
    }
    bool seconds_embed(float, float * out) override {
        std::fill(out, out + SA3_COND_DIM, 0.0f); return true;
    }
    bool same_encode_chunk(const float *, float * out) override {
        std::fill(out, out + SA3_LAT_CH * SA3_CHUNK_LAT, 0.0f); return true;
    }
    bool same_decode_chunk(const float *, float * out) override {
        std::fill(out, out + 2 * SA3_CHUNK_SAMPLES, poison ? NAN : 0.25f); return true;
    }
    bool dit_forward(const float *, float, const float *, const float *,
                     const float *, const char *, int T, float * out) override {
        std::fill(out, out + SA3_LAT_CH * T, 0.0f); return true;
    }
};

int main() {
    const std::string fixtures = ".hotstep-1127/sa3-finite-fixtures";
    std::filesystem::create_directories(fixtures + "/postprocess");
    std::ofstream(fixtures + "/postprocess/callback.lua") << R"(
postprocess = { name = "finite_callback" }
function process(latents, B, C, W, channels, samples, upscale, decode)
    pcall(decode, latents, 1)
    return {}, 0
end
)";
    std::ofstream(fixtures + "/postprocess/output.lua") << R"(
postprocess = { name = "finite_output" }
function process() return { 0/0, 0.25 }, 1 end
)";
    PluginRegistry::instance().init(fixtures, fixtures, fixtures);
    assert(PluginRegistry::instance().postprocess_lookup("finite_callback"));
    assert(PluginRegistry::instance().postprocess_lookup("finite_output"));
    FakeSa3 be;
    std::vector<float> input(882, 0.0f), out;
    int64_t tokens[SA3_TOK_LEN] = {};
    Sa3PluginParams params;
    std::string err;
    auto run = [&](bool noise_only) {
        return sa3_refine_run_backend(&be, input.data(), 441, tokens, 1,
                                      0.5f, 1, false, 42, true, out, &params,
                                      noise_only, nullptr, nullptr, &err);
    };
    for (bool noise_only : {false, true}) {
        params.postprocess_plugin.clear();
        be.poison = true;
        assert(!run(noise_only));
        assert(err.find("SA3 SAME decode: ") == 0);
        assert(err.find("non-finite values out of ") != std::string::npos);

        params.postprocess_plugin = "finite_callback";
        assert(!run(noise_only));
        assert(err == "SA3 SAME decode: " + std::to_string(2 * SA3_DS) +
                      " non-finite values out of " + std::to_string(2 * SA3_DS));

        be.poison = false;
        params.postprocess_plugin = "finite_output";
        assert(!run(noise_only));
        assert(err == "SA3 final PCM: 1 non-finite values out of 882");

        params.postprocess_plugin.clear();
        assert(run(noise_only));
        assert(err.empty());
        assert(out.size() == 882);
        for (float sample : out) assert(sample == 0.25f);
    }
    puts("SA3 finite error propagation: PASS (native, swallowed callback, plugin output, synth/refine, finite recovery)");
}

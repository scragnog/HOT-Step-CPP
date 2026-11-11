"""CPU fault probe for the decoder assembly functions, with fake graph outputs.

Run from the repo root in a C++ developer shell:
  python engine/tests/test-synth-finite.py [--baseline <git-revision>]
No model weights or GPU are used. The compiled C++ bodies come from the source
under test; only the graph execution and model configuration are replaced.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
out = root / ".hotstep-1127"
out.mkdir(exist_ok=True)
baseline = sys.argv[2] if len(sys.argv) == 3 and sys.argv[1] == "--baseline" else None


def source(path):
    if baseline:
        return subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=root, encoding="utf-8")
    return (root / path).read_text(encoding="utf-8")


def function(text, name):
    start = text.rfind("\n", 0, text.index(name + "(")) + 1
    opening = text.index("{", start)
    depth = 1
    i = opening + 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start:i] + "\n"


mm3 = source("engine/src/minimax/mm3-vocoder-graph.h")
yue2 = source("engine/src/yue2/yue2-vae-graph.h")
sa3 = source("engine/src/sa3-refine.h")
cpp = r'''
#include "synth-finite.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
static bool poison = false;
static float sample() { return poison ? std::numeric_limits<float>::quiet_NaN() : 0.25f; }
struct MM3VocGraph {};
struct MM3VocConfig { int total_upsample=2, fold_channels=64; };
struct MM3Model { struct { MM3VocConfig voc; } synth_cfg; };
static const int MM3_VOC_CHUNK=8, MM3_VOC_OVERLAP=1;
static bool mm3_vocoder_prepare(const MM3Model &, MM3VocGraph *, std::string *) { return true; }
static bool mm3_voc_run(const MM3Model &, MM3VocGraph *, const float *, int64_t L, float * dst, std::string *) {
    std::fill(dst, dst + 2*L, sample()); return true;
}
struct Yue2VaeGraph {};
struct Yue2Model { struct { int latent_dim=64, downsampling_ratio=2, required_halo=1; } vae_cfg; };
static Yue2VaeGraph graph;
static Yue2VaeGraph * yue2_vae_decode_cache(Yue2Model &) { return &graph; }
static bool yue2_vae_prepare(Yue2Model &, Yue2VaeGraph *, std::string *) { return true; }
static bool yue2_vae_run(Yue2Model &, Yue2VaeGraph *, const float *, int64_t T,
                       std::vector<float> * out, int64_t * samples, std::string *) {
    out->assign(4*T, sample()); if (samples) *samples=2*T; return true;
}
struct Yue2VaeTileBoundary { int64_t tile_index, start, end, left, right, out_start, out_end, crop_start; };
static const int SA3_DS=2, SA3_LAT_CH=256, SA3_CHUNK_LAT=8, SA3_OVERLAP_LAT=2, SA3_CHUNK_SAMPLES=16;
struct Sa3Backend {
    bool same_decode_chunk(const float *, float * dst) {
        std::fill(dst, dst + 2*SA3_CHUNK_SAMPLES, sample()); return true;
    }
};
'''
cpp += function(mm3, "mm3_vocoder_decode_with")
cpp += function(yue2, "yue2_vae_decode")
cpp += function(yue2, "yue2_vae_decode_tiled")
cpp += function(sa3, "sa3_chunk_starts")
cpp += function(sa3, "sa3_decode_tiled")
cpp += "\nstatic const bool baseline = " + ("true" if baseline else "false") + ";\n"
cpp += r'''
int main() {
    float good[] = {0.0f, -0.0f, 1.0f, -1.0f, std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::denorm_min()};
    float copy[6]; memcpy(copy, good, sizeof good);
    assert(synth_finite(good, 6, "finite")); assert(!memcmp(good, copy, sizeof good));
    assert(synth_finite(nullptr, 0, "empty"));
    float bad[] = {0, NAN, INFINITY, -INFINITY};
    std::string err;
    assert(!synth_finite(bad, 4, "HeartMuLa codec decode", &err));
    assert(err == "HeartMuLa codec decode: 3 non-finite values out of 4");
    MM3Model m; MM3VocGraph g; Yue2Model y; Sa3Backend s;
    std::vector<float> input(256*20, 0), output;
    for (bool inject : {false, true}) {
        poison = inject;
        const bool expected = !inject || baseline;
        for (int length : {4, 20}) {
            err.clear();
            assert(mm3_vocoder_decode_with(m, &g, input.data(), length, output, &err) == expected);
            if (!expected) assert(err == "MM3 vocoder decode: " + std::to_string(4*length) +
                                         " non-finite values out of " + std::to_string(4*length));
            if (!inject) for (float v : output) assert(v == 0.25f);
            assert(sa3_decode_tiled(&s, input.data(), length, output) == expected);
            if (!inject) for (float v : output) assert(v == 0.25f);
        }
        int64_t samples=0;
        err.clear();
        assert(yue2_vae_decode(y, input.data(), 20, &output, &samples, &err) == expected);
        if (!expected) assert(err == "YuE2 VAE decode: 80 non-finite values out of 80");
        err.clear();
        assert(yue2_vae_decode_tiled(y, input.data(), 20, 8, 1, &output, &samples, nullptr, &err) == expected);
        if (!expected) assert(err == "YuE2 VAE decode: 80 non-finite values out of 80");
        if (!inject) for (float v : output) assert(v == 0.25f);
    }
    puts(baseline ? "REPRO: MM3/YuE2/SA3 accepted non-finite decoded PCM" :
                    "PASS: MM3 single/windowed, YuE2 full/tiled, SA3 padded/tiled reject invalid PCM; finite PCM unchanged");
}
'''
path = out / ("finite-baseline.cpp" if baseline else "finite-probe.cpp")
path.write_text(cpp, encoding="utf-8")
exe = path.with_suffix(".exe" if sys.platform == "win32" else "")
compiler = ["cl", "/nologo", "/EHsc", "/std:c++17", "/Od", f"/I{root / 'engine/src'}", str(path), f"/Fe:{exe}", f"/Fo:{path.with_suffix('.obj')}"] if sys.platform == "win32" else ["c++", "-std=c++17", "-O0", "-I", str(root / "engine/src"), str(path), "-o", str(exe)]
subprocess.run(compiler, cwd=root, check=True, timeout=120)
subprocess.run([str(exe)], cwd=root, check=True, timeout=30)

// The held-out planner gate's resume binding (GH #316), through the
// trainer's own record writer and reader (make_resume_meta /
// parse_resume_meta, file-local to yue2-aitk-runtime.cpp, so that file is
// included here). No model, no GPU. Build like test-yue2-planner-freeze.cpp:
//   cl /std:c++17 /EHsc /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /DGGML_MAX_NAME=128 /DYUE2_AITK_CUDA_RUNTIME=0
//      /I..\src /I..\src\train /I..\ggml\include /I..\vendor\yyjson
//      test-yue2-planner-gate-resume.cpp ..\vendor\yyjson\yyjson.c
//      /link ..\build\ggml\src\Release\ggml.lib ..\build\ggml\src\Release\ggml-base.lib
#include "../src/train/yue2-aitk-runtime.cpp"

#include <cstdio>

using yue2_aitk_runtime::Config;
using yue2_aitk_runtime::PlannerGate;

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)

// Six songs, 0 and 1 held out, a record at step 20.
static Config gated() {
    Config c; c.planner_gate_every = 10; c.planner_gate_min_epochs = 5; c.planner_gate_patience = 3; c.planner_gate_epsilon = 0.001f;
    c.holdout_items = {0, 1}; c.nar_extra_steps = 100; c.eval_seed = 42; c.eval_seed_set = true;
    return c;
}
static std::string record(const Config & c, const PlannerGate & g) {
    return make_resume_meta("cp", "ds", "sm", 7, 0, 20, 1, {2, 3, 4, 5}, "rng", 0.0f, c, 0, 0.0, 0.0, {}, {}, -1, {}, {}, {}, 0.0, -1, 1.0, -1, g);
}
static bool read(const std::string & text, const Config & c, ResumeBinding * b, std::string * why) {
    ResumePlan plan; float cursor = 0.0f;
    return parse_resume_meta(text, "cp", "ds", "sm", 7, 0, 6, &plan, why, &cursor, false, c, b);
}

int main() {
    const Config c = gated();
    PlannerGate g; g.best = 2.25; g.best_step = 20; g.stale = 1; g.draws = 80;
    const std::string text = record(c, g);
    std::string why;

    { ResumeBinding b; CHECK(read(text, c, &b, &why));
      CHECK(b.gate.best == 2.25 && b.gate.best_step == 20 && b.gate.stale == 1 && b.gate.draws == 80 && !b.gate.pending(c)); }

    // The evaluation window binds in both forms: --eval-nar-window itself,
    // and --nar-crop-frames when it is the fallback. Same effective window
    // spelled the other way still resumes.
    { Config k = c; k.eval_nar_window = 100; const std::string t = record(k, g);
      Config w = k; w.eval_nar_window = 200; ResumeBinding b;
      CHECK(!read(t, w, &b, &why) && why.find("window") != std::string::npos);
      CHECK(read(t, k, &b, &why)); }
    { Config w = c; w.nar_crop_frames = 1000; ResumeBinding b;
      CHECK(!read(text, w, &b, &why) && why.find("window") != std::string::npos);
      Config same = c; same.eval_nar_window = 1500; CHECK(read(text, same, &b, &why)); }

    // A grad-accum change is allowed on resume; the epochs ride in draws, so
    // the floor is where the run left it (80 draws / 4 training songs = 20).
    { Config a = c; a.grad_accum = 4; ResumeBinding b;
      CHECK(read(text, a, &b, &why) && b.gate.draws == 80 && PlannerGate::epochs(b.gate.draws, 4) == 20.0); }

    // Settings, seed and presence bind; scoring a gated checkpoint does not.
    { Config p = c; p.planner_gate_patience = 4; ResumeBinding b; CHECK(!read(text, p, &b, &why));
      Config s = c; s.eval_seed = 43; CHECK(!read(text, s, &b, &why));
      Config off; off.holdout_items = {0, 1}; off.eval_seed = 42; off.eval_seed_set = true; CHECK(!read(text, off, &b, &why));
      Config scoring = off; scoring.eval_adapted = true; CHECK(read(text, scoring, &b, &why));
      CHECK(!read(record(off, {}), c, &b, &why)); }

    // A verdict a spike stop ended the step on comes back pending.
    { PlannerGate f = g; f.stale = 3; ResumeBinding b; CHECK(read(record(c, f), c, &b, &why) && b.gate.pending(c)); }

    // Malformed state is refused: a negative draw count.
    { std::string t = text; const auto at = t.find("\"draws\":80"); CHECK(at != std::string::npos);
      t.replace(at, 10, "\"draws\":-1"); ResumeBinding b; CHECK(!read(t, c, &b, &why)); }

    std::puts(failures ? "FAIL test-yue2-planner-gate-resume" : "PASS test-yue2-planner-gate-resume");
    return failures ? 1 : 0;
}

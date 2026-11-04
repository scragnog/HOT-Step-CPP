// Self-check for the YuE2 held-out planner gate (--planner-gate-every, GH
// #316): the plateau verdict on synthetic CE curves, the token-weighted
// reduction, resume equivalence, the flags and the configuration rules.
// Header-only, no CUDA, no model files:
//   cl /std:c++17 /EHsc /I..\src\train test-yue2-planner-gate.cpp
//   c++ -std=c++17 -I../src/train test-yue2-planner-gate.cpp
#include "yue2-aitk-runtime.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using yue2_aitk_runtime::Config;
using yue2_aitk_runtime::PlannerGate;
using yue2_aitk_runtime::validate_planner_gate;
using Verdict = PlannerGate::Verdict;

// A gated run: 10 training songs, a reading every 10 updates, so a reading is
// one epoch and reading k (1-based) lands at step 10k, epoch k.
static Config gated() {
    Config c;
    c.planner_gate_every = 10; c.planner_gate_min_epochs = 5; c.planner_gate_patience = 3; c.planner_gate_epsilon = 0.001f;
    c.holdout_items = {0, 1, 2}; c.nar_extra_steps = 100;
    return c;
}
constexpr size_t kTrain = 10;

// Feeds the curve from `from` on; returns the step the gate fires at, or -1.
static int run(const Config & c, PlannerGate & gate, const std::vector<double> & curve, size_t from = 0) {
    for (size_t k = from; k < curve.size(); ++k) {
        const int step = int(k + 1) * c.planner_gate_every;
        assert(PlannerGate::due(c, step));
        gate.draws += std::int64_t(c.planner_gate_every) * c.grad_accum;  // the trainer adds grad-accum per update
        const Verdict v = gate.observe(c, step, PlannerGate::epochs(gate.draws, kTrain), curve[k]);
        assert(v != Verdict::nonfinite);
        if (v == Verdict::fire) return step;
    }
    return -1;
}

int main() {
    const Config c = gated();

    // Improvement: every reading beats the best by more than epsilon.
    { PlannerGate g; std::vector<double> curve; for (int k = 0; k < 40; ++k) curve.push_back(3.0 - 0.01 * k);
      assert(run(c, g, curve) == -1 && g.stale == 0 && g.best_step == 400); }

    // Plateau from the start: nothing counts before epoch 5, then three stale
    // readings (epochs 5, 6, 7) fire at step 70. The first reading is the best.
    { PlannerGate g; assert(run(c, g, std::vector<double>(20, 2.5)) == 70 && g.best_step == 10 && g.stale == 3); }

    // Regression (the reporter's U-curve): best at epoch 8, rising after.
    // Fires three readings later, at epoch 11; no rollback, so the gate
    // reports where the best was and the planner freezes where it is.
    { PlannerGate g; std::vector<double> curve = {3.0, 2.8, 2.6, 2.5, 2.45, 2.42, 2.41, 2.40, 2.43, 2.47, 2.52, 2.6};
      assert(run(c, g, curve) == 110 && g.best_step == 80 && std::fabs(g.best - 2.40) < 1e-12); }

    // Improvements smaller than epsilon are not improvements.
    { PlannerGate g; std::vector<double> curve = {2.0, 2.0, 2.0, 2.0, 2.0, 1.9995, 1.9992, 1.9985};
      assert(run(c, g, curve) == 70 && g.best == 2.0); }

    // One real improvement resets the count.
    { PlannerGate g; std::vector<double> curve = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5};
      assert(run(c, g, curve) == 100 && g.best_step == 70); }

    // Patience 1, floor 0: the first non-improving reading fires.
    { Config c1 = c; c1.planner_gate_min_epochs = 0; c1.planner_gate_patience = 1;
      PlannerGate g; assert(run(c1, g, {2.0, 1.0, 1.0}) == 30); }

    // Epochs count song draws over TRAINING songs (the hold-out is not in the
    // denominator), summed as the run goes: grad-accum 4 is four draws an
    // update, and a resume that changes it keeps the epochs already run.
    { assert(PlannerGate::epochs(25, kTrain) == 2.5 && PlannerGate::epochs(25, 0) == 0.0);
      Config c4 = c; c4.grad_accum = 4;
      PlannerGate g; run(c, g, {3.0});                     // 10 updates at accum 1: epoch 1
      assert(g.draws == 10);
      PlannerGate r = g;                                   // the record, resumed at accum 4
      run(c4, r, {3.0, 2.9}, 1);                           // 10 more updates: 40 draws, epoch 5
      assert(r.draws == 50 && PlannerGate::epochs(r.draws, kTrain) == 5.0); }  // steps x new accum would say 8

    // Fire verdicts a spike stop kept the run from acting on stay pending in
    // the record, so the resumed segment freezes at once.
    { PlannerGate g; assert(run(c, g, std::vector<double>(7, 2.0)) == 70 && g.pending(c));
      PlannerGate live; run(c, live, std::vector<double>(6, 2.0)); assert(!live.pending(c));
      Config off; assert(!g.pending(off)); }

    // Schedule: multiples of the interval, never step 0, never when off.
    { assert(!PlannerGate::due(c, 0) && !PlannerGate::due(c, 9) && PlannerGate::due(c, 10) && PlannerGate::due(c, 20));
      Config off; assert(!PlannerGate::due(off, 10)); }

    // Nonfinite readings are refused and change nothing.
    { PlannerGate g; g.observe(c, 10, 1.0, 2.0);
      const double inf = std::numeric_limits<double>::infinity();
      for (double bad : {std::nan(""), inf, -inf}) {
          assert(g.observe(c, 20, 6.0, bad) == Verdict::nonfinite);
          assert(g.best == 2.0 && g.best_step == 10 && g.stale == 0);
      }
      assert(std::isnan(PlannerGate::token_mean({}))); }

    // Unequal token counts: CE over all tokens, not the mean of song means.
    { const double ce = PlannerGate::token_mean({{2.0, 100}, {4.0, 300}, {1.0, 0}});
      assert(std::fabs(ce - 3.5) < 1e-12);
      assert(std::fabs(PlannerGate::token_mean({{2.0, 1}, {4.0, 1}}) - 3.0) < 1e-12); }

    // Continuous vs resumed: the state the record carries (best through the
    // record's %.17g-precision real, best_step, stale), restored at every
    // split point, fires at the step a continuous run does.
    { const std::vector<double> curve = {3.0, 2.8, 2.6, 2.5, 2.45, 2.42, 2.41, 2.40, 2.43, 2.47, 2.52, 2.6};
      PlannerGate whole; const int fired = run(c, whole, curve);
      for (size_t split = 1; split < curve.size(); ++split) {
          PlannerGate first; const int early = run(c, first, std::vector<double>(curve.begin(), curve.begin() + split));
          if (early >= 0) { assert(early == fired); continue; }
          char text[64]; std::snprintf(text, sizeof text, "%.17g", first.best);
          PlannerGate resumed; resumed.best = std::strtod(text, nullptr); resumed.best_step = first.best_step; resumed.stale = first.stale; resumed.draws = first.draws;
          assert(run(c, resumed, curve, split) == fired);
          assert(resumed.best == whole.best && resumed.best_step == whole.best_step);
      } }

    // --unfreeze-planner keeps the best and counts patience afresh.
    { PlannerGate g; assert(run(c, g, std::vector<double>(8, 2.0)) == 70);
      g.stale = 0;
      assert(g.observe(c, 80, 8.0, 2.0) == Verdict::keep && g.observe(c, 90, 9.0, 2.0) == Verdict::keep);
      assert(g.observe(c, 100, 10.0, 2.0) == Verdict::fire); }

    // Flags: names, defaults, bad values.
    { const char * argv[] = {"x", "--checkpoint", "c", "--dataset", "d", "--output", "o", "--steps", "500", "--save-every", "50",
                             "--holdout-items", "3,1", "--nar-extra-steps", "100", "--eval-seed", "42",
                             "--planner-gate-every", "10", "--planner-gate-min-epochs", "4", "--planner-gate-patience", "2", "--planner-gate-epsilon", "0.005"};
      Config p; std::string error;
      assert(yue2_aitk_runtime::parse(int(sizeof argv / sizeof *argv), const_cast<char **>(argv), &p, &error) == yue2_aitk_runtime::ParseResult::ok);
      assert(p.planner_gate_every == 10 && p.planner_gate_min_epochs == 4 && p.planner_gate_patience == 2 && p.planner_gate_epsilon == 0.005f);
      assert(p.holdout_items == (std::vector<size_t>{3, 1}) && p.eval_seed == 42 && p.eval_seed_set && validate_planner_gate(p, &error));
      Config d; assert(d.planner_gate_every == 0 && d.planner_gate_min_epochs == 5 && d.planner_gate_patience == 3 && d.planner_gate_epsilon == 0.001f);
      const char * bad[][2] = {{"--planner-gate-every", "-1"}, {"--planner-gate-patience", "0"}, {"--planner-gate-epsilon", "-0.1"},
                               {"--planner-gate-epsilon", "nan"}, {"--planner-gate-min-epochs", "x"}, {"--holdout-items", "1,1"}};
      for (auto & b : bad) {
          const char * a[] = {"x", "--checkpoint", "c", "--dataset", "d", "--output", "o", "--steps", "5", "--save-every", "5", b[0], b[1]};
          assert(yue2_aitk_runtime::parse(13, const_cast<char **>(a), &p, &error) == yue2_aitk_runtime::ParseResult::error);
      } }

    // Configuration: every adapter on every optimizer takes the gate (the
    // freeze it reuses zeroes the planner's gradients before the clip and
    // writes its weights back after the update, whatever the update rule).
    // What it cannot run with is refused, whatever the adapter or optimizer.
    for (const char * adapter : {"lora", "dora", "lokr"})
        for (const char * optimizer : {"adamw", "adamw-lm", "prodigy", "muon"}) {
            Config k = c; k.adapter_type = adapter; k.optimizer = optimizer; std::string error;
            assert(validate_planner_gate(k, &error));
            const auto refused = [&](void (*edit)(Config &), const char * needle) {
                Config bad = k; edit(bad); std::string why;
                const bool ok = validate_planner_gate(bad, &why);
                if (ok || why.find(needle) == std::string::npos) { std::fprintf(stderr, "expected refusal containing '%s', got '%s'\n", needle, why.c_str()); return false; }
                return true;
            };
            assert(refused([](Config & b) { b.holdout_items.clear(); }, "--holdout-items"));
            assert(refused([](Config & b) { b.eval_dataset = "held.json"; }, "--eval-dataset"));
            assert(refused([](Config & b) { b.nar_extra_steps = 0; }, "--nar-extra-steps"));
            assert(refused([](Config & b) { b.lr_schedule = "wsd"; }, "wsd"));
            assert(refused([](Config & b) { b.eval_adapted = true; }, "training runs"));
            assert(refused([](Config & b) { b.eval_base_loss = true; }, "training runs"));
            assert(refused([](Config & b) { b.meter_only = true; }, "training runs"));
            assert(refused([](Config & b) { b.fixture_replay = "dir"; }, "training runs"));
            assert(refused([](Config & b) { b.planner_gate_patience = 0; }, "patience"));
            assert(refused([](Config & b) { b.planner_gate_epsilon = NAN; }, "epsilon"));
        }
    { Config off; off.lr_schedule = "wsd"; std::string error; assert(validate_planner_gate(off, &error)); }  // off = no rules

    // Splits: the gate's songs never train. An empty split leaves the gate
    // refused above; a leaking one (a held-out song in the order), an
    // out-of-range or repeated index, or no training song left are refused.
    { using namespace yue2_aitk_runtime::detail; std::vector<size_t> order; std::string error;
      assert(training_order(5, {0, 3}, &order, &error) && order == (std::vector<size_t>{1, 2, 4}));
      assert(!validate_resume_order(5, {0, 3}, {0, 3}, {1, 2, 4, 3}, 0, &error));
      assert(!validate_resume_order(5, {0, 3}, {0, 2}, {1, 3, 4}, 0, &error));  // changed split on resume
      assert(!training_order(5, {5}, &order, &error) && !training_order(5, {1, 1}, &order, &error));
      assert(!training_order(2, {0, 1}, &order, &error)); }

    std::puts("ok");
}

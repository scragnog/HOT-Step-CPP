#pragma once
// hot-step-families.h — Registry of in-process model families riding on
// ace-server (MiniMax-Music3 today, YuE2 next).
//
// The two boot gates in hot-step-server.cpp used to ask "is MM3 present?" by
// naming mm3_weights_present() directly. That question generalises to "can
// ANY in-process family still serve?" — this header answers it without the
// two gate call sites having to know how many families exist or what any of
// them are named. Each family keeps its own cheap probe (directory/filename
// scan only, no weight reads, no GGUF header parse — see mm3_weights_present
// itself for the shape a probe should take).
//
// INCLUDE ORDER (load-bearing, do not "fix"): this header does NOT
// `#include "minimax/mm3-model.h"` or "minimax/mm3-server.h". mm3-model.h's
// own header comment is explicit that nothing outside engine/src/minimax/ may
// include it — the single approved wire-in is hot-step-server.cpp's
// `#include "minimax/mm3-server.h"` hook (checked by verify-hooks.ps1). This
// header instead relies on that include having already happened EARLIER in
// the same translation unit, so `mm3_weights_present` (declared `static` in
// mm3-model.h) is already visible by name when g_hot_step_families below is
// initialised. Concretely: in hot-step-server.cpp this header must be
// `#include`d AFTER "minimax/mm3-server.h" (:86), in the same file. It is not
// safe to include from anywhere else, or before that line, today — it has
// exactly one consumer. A second consumer TU would need mm3_weights_present
// to stop being `static` first (drop `static`, or add an `inline` non-static
// shim in mm3-model.h); that is a minimax/ change, out of scope here, and
// NOT made by this patch (see APPLY.md).
#include <filesystem>
#include <string>

// sa3_weights_present lives in its own header (sa3-families.h) rather than
// riding a hook include the way mm3_weights_present and yue2_weights_present
// do: SA3 has no "sa3-server.h" subtree hook in hot-step-server.cpp for this
// header to rely on having been included earlier in the TU (see sa3-families.h's
// own comment for why), so it is safe — and necessary — to include directly
// here instead of forward-declaring it.
#include "sa3-families.h"

struct HotStepFamily {
    const char * name;                                  // for the two [Server] log lines only
    // nullptr = "handled by the ACE registry scan itself" (reserved for a
    // family folded into registry_scan() rather than probed separately; no
    // current family uses this — both rows below have a real probe).
    bool (*weights_present)(const char * models_dir);
};

// yue2_weights_present is declared `static` in engine/src/yue2/yue2-model.h,
// which hot-step-server.cpp includes at the TOP of the file (alongside
// minimax/mm3-server.h, before this header), so it is already visible by
// name here — same "earlier #include in the same TU" contract this file's
// own header comment documents for mm3_weights_present. No forward
// declaration here (a non-static one would conflict with yue2-model.h's own
// `static` definition) — same reason mm3_weights_present has none either.

// HeartMuLa/MuLaCover: filename scan only, same two dirs and name prefixes as
// heartmula_discover (heartmula/pipeline.h), which this header cannot call
// (that file is included later in the TU). Any heartmula-*.gguf LM file other
// than the tokenizer counts; compatibility is checked per request, not at boot.
static bool heartmula_weights_present(const char * models_dir) {
    if (!models_dir || !models_dir[0]) {
        return false;
    }
    namespace fs = std::filesystem;
    for (const fs::path dir : { fs::path(models_dir) / "heartmula", fs::path(models_dir) }) {
        std::error_code ec;
        for (const auto & e : fs::directory_iterator(dir, ec)) {
            const std::string n = e.path().filename().string();
            if (e.is_regular_file(ec) && n.rfind("heartmula-", 0) == 0 && n.rfind("heartmula-tokenizer", 0) != 0 &&
                e.path().extension() == ".gguf") {
                return true;
            }
        }
    }
    return false;
}

// C++17 inline variable: exactly one definition across however many TUs end
// up including this header (today: one), so g_hot_step_families itself is
// never an ODR hazard even though the array is `constexpr` and one of its
// entries points at a function with internal (`static`) linkage — taking the
// address of an internal-linkage function is a valid constant expression.
inline constexpr HotStepFamily g_hot_step_families[] = {
    { "ace-step", nullptr },           // registry_scan() already covers this row; kept here so the
                                        // table — not the two call sites — is the map of "who can serve".
    { "MiniMax-Music3", mm3_weights_present },
    { "YuE2", yue2_weights_present },
    { "StableAudio3", sa3_weights_present },
    { "HeartMuLa", heartmula_weights_present },
};

inline bool hot_step_any_family_weights_present(const char * models_dir) {
    for (const auto & f : g_hot_step_families) {
        if (f.weights_present && f.weights_present(models_dir)) {
            return true;
        }
    }
    return false;
}

inline std::string hot_step_present_family_names(const char * models_dir) {
    std::string out;
    for (const auto & f : g_hot_step_families) {
        if (!f.weights_present || !f.weights_present(models_dir)) {
            continue;
        }
        if (!out.empty()) {
            out += ", ";
        }
        out += f.name;
    }
    return out;
}

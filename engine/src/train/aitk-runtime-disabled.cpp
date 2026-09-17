#include "yue2-aitk-runtime.h"

#include <csignal>

namespace {
volatile std::sig_atomic_t g_cancel = 0;
void on_sigint(int) { g_cancel = 1; }
}

namespace yue2_aitk_runtime {

void yue2_aitk_install_sigint_handler() { std::signal(SIGINT, on_sigint); }
bool yue2_aitk_cancel_requested() { return g_cancel != 0; }
void yue2_aitk_clear_cancel() { g_cancel = 0; }

int yue2_aitk_run(const Config &, std::string * error) {
    if (error) *error = "yue2-joint-train requires a CUDA build; this binary has no CUDA runtime";
    return 1;
}

} // namespace yue2_aitk_runtime

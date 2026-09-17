// Capability-0 translation unit for non-CUDA builds. The AITK CUDA APIs are
// intentionally unavailable here; this keeps the runtime target a static
// library without inventing a CPU or HIP implementation.
namespace yue2_aitk_runtime_disabled {
const int capability = 0;
}

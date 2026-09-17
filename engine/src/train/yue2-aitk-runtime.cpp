// Compile the native joint path in CUDA engine builds even before UI exposure.
#include "yue2-aitk-joint-step.h"
#include "yue2-aitk-resume.h"
namespace yue2_aitk_runtime { bool compiled_cuda_joint_step() { return true; } }
#pragma once
#include "ggml-backend.h"
#include <cstring>

// CUDA is a loadable GGML module in portable builds. Identify it through the
// backend registry, without directly linking symbols from that module.
inline bool yue2_aitk_is_cuda(ggml_backend_t backend) {
    if (!backend) return false;
    const auto device = ggml_backend_get_device(backend);
    if (!device) return false;
    const auto registry = ggml_backend_dev_backend_reg(device);
    const char * name = registry ? ggml_backend_reg_name(registry) : nullptr;
    return name && std::strcmp(name, "CUDA") == 0;
}
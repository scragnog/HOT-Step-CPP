#include "convrot_cuda_api.h"

#include <cassert>
#include <cstddef>

int main() {
    using namespace aitk_reference;
    const ConvRotCudaShape shape{3, 2048, 4096, 256, true};
    assert(convrot_cuda_validate_shape(shape));
    assert(convrot_cuda_forward_workspace_bytes(shape) != 0);
    assert(convrot_cuda_input_backward_workspace_bytes(shape) != 0);
    assert(!convrot_cuda_validate_shape(ConvRotCudaShape{0, 2048, 4096, 256, true}));
    assert(!convrot_cuda_validate_shape(ConvRotCudaShape{3, 2048, 4096, 3, true}));
    assert(convrot_cuda_validate_shape(ConvRotCudaShape{1500, 2048, 4096, 256, true}));
    assert(convrot_cuda_validate_shape(ConvRotCudaShape{24576, 2048, 12288, 256, true}));
    assert(!convrot_cuda_validate_shape(ConvRotCudaShape{24577, 2048, 4096, 256, true}));
    assert(!convrot_cuda_validate_shape(ConvRotCudaShape{3, 17, 32, 1, true}));
    assert(!convrot_cuda_validate_shape(ConvRotCudaShape{3, 64, 24, 16, true}));
    assert(convrot_cuda_status_string(ConvRotCudaStatus::success)[0] == 's');
    return 0;
}

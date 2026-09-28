# Native joint training runtime. Built on every backend; the CUDA build adds
# the 8-bit AdamW and fused head-loss kernels (YUE2_AITK_CUDA_RUNTIME=1), and
# the others run the same trainer through ggml ops (LmOptim, portable loss).

add_library(yue2-aitk-kernels STATIC ${CMAKE_CURRENT_LIST_DIR}/yue2-aitk-runtime.cpp)
target_include_directories(yue2-aitk-kernels PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR}/../../vendor/yyjson)
if(GGML_CUDA AND TARGET ggml-cuda)
    enable_language(CUDA)
    target_sources(yue2-aitk-kernels PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/adamw8bit_cuda.cu
        ${CMAKE_CURRENT_LIST_DIR}/joint_loss_cuda.cu
    )
    set_target_properties(yue2-aitk-kernels PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(yue2-aitk-kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    )
    target_link_libraries(yue2-aitk-kernels PUBLIC CUDA::cudart PRIVATE ggml)
    target_compile_definitions(yue2-aitk-kernels PUBLIC YUE2_AITK_CUDA_RUNTIME=1)
else()
    target_link_libraries(yue2-aitk-kernels PRIVATE ggml)
    target_compile_definitions(yue2-aitk-kernels PUBLIC YUE2_AITK_CUDA_RUNTIME=0)
endif()

target_link_libraries(ace-train PRIVATE yue2-aitk-kernels)

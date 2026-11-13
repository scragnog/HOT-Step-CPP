# Standalone spike: links the HOT-ggml libraries the engine build already made,
# so no ggml/CUDA compile happens here. Configure with -DENGINE_BUILD=<engine/build>.
cmake_minimum_required(VERSION 3.20)
project(lead-sheet-port-spike CXX)
set(CMAKE_CXX_STANDARD 17)

set(ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../engine")
set(ENGINE_BUILD "${ENGINE_DIR}/build" CACHE PATH "engine build directory")

add_executable(yourmt3-spike spike.cpp)
target_include_directories(yourmt3-spike PRIVATE "${ENGINE_DIR}/ggml/include")
target_link_libraries(yourmt3-spike PRIVATE
    "${ENGINE_BUILD}/ggml/src/Release/ggml.lib"
    "${ENGINE_BUILD}/ggml/src/Release/ggml-base.lib")

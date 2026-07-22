cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

# CI cache generation: 2 (2026-07-16). This file is hashed into the GitHub
# Actions build-cache key — bump this comment to force cold builds when the
# cached objects themselves are suspect (v1.1.3 stale-cache mixed-ABI crash,
# issues #82/#83). Routine drift is handled by the .built-commit stamp guard
# in release.yml/cache-warm.yml; this is the manual override.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings and Windows.h macro pollution.
# NOMINMAX: prevents Windows.h from defining min/max macros that collide with
#           std::min/std::max (causes C2589 errors in solvers/schedulers).
# WIN32_LEAN_AND_MEAN: reduces Windows.h header bloat.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Static MSVC runtime (/MT) for portable release builds.
# Eliminates the VC++ Redistributable dependency for end users.
# Only enable during release builds: -DHOT_STEP_STATIC_RUNTIME=ON
option(HOT_STEP_STATIC_RUNTIME "Use static MSVC runtime (/MT) for portable builds" OFF)
if(HOT_STEP_STATIC_RUNTIME AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    message(STATUS "MSVC runtime: static (/MT)")
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# macOS rpath: make binaries relocatable (portable release support).
# Without this, CMake bakes the absolute build directory into LC_RPATH,
# which breaks on any machine other than the one that built it.
# @executable_path tells dyld to look for dylibs next to the binary.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_MACOSX_RPATH TRUE)
endif()

# Linux rpath: make binaries relocatable (portable release support).
# $ORIGIN tells the dynamic linker to search for .so files next to the binary.
if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
if(NOT MSVC)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries (CI: CUDA 13.1 / 12.8).
# The CUDA 12.8 build additionally targets legacy Pascal/Volta (see below).
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    # Base arch list: Turing through Lovelace/Ada
    set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;90-real")
    if(CUDAToolkit_FOUND)
        # Legacy GPUs — Pascal only (sm_60 P100, sm_61 GTX 10xx / P40 / P4).
        # CUDA 12.x can still compile real SASS for these; CUDA 13.0 removed
        # offline compilation support, so they go only into the cuda12.8 variant.
        # Real SASS is required — the 75-virtual PTX above can't JIT backwards
        # onto pre-Turing cards.
        #
        # NOTE: Volta (sm_70) is deliberately EXCLUDED. ggml's mma-based MMQ and
        # flash-attention kernels have no device code for sm_70 (they need
        # Turing+), so a sm_70 build crashes on Volta. Worse, sm_70 SASS is
        # binary-compatible upward to Turing (sm_75), so the driver loads it on
        # 75 cards in preference to JIT-ing the compute_75 PTX — which regressed
        # all Turing users on the cuda12.8 bundle in v1.1.1 (#63). Pascal (major
        # 6) is unaffected since Turing can't load major-6 SASS.
        if(CUDAToolkit_VERSION VERSION_LESS "13.0")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "60-real;61-real")
        endif()
        # Blackwell sm_120a: CUDA 12.8+
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "120a-real")
        endif()
        # Blackwell sm_121a: CUDA 12.9+ (same as upstream GGML)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "121a-real")
        endif()
    endif()
endif()

# Disable flash attention (cuda12-volta variant). ggml's mma flash-attention has
# no device code for Volta (sm_70); with this defined the engine takes the manual
# attention path instead. Pair with -DGGML_CUDA_FORCE_CUBLAS=ON (ggml) which
# likewise replaces the mma MMQ kernels. See engine/src/hot-step-build-flags.h.
option(HOT_STEP_DISABLE_FA "Disable flash attention (Volta / pre-Turing GPUs)" OFF)
if(HOT_STEP_DISABLE_FA)
    add_compile_definitions(HOT_STEP_DISABLE_FA)
    message(STATUS "[HOT-Step] Flash attention DISABLED (HOT_STEP_DISABLE_FA)")
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
# CUDA graphs default on: standalone ggml ships them off. Overridable with
# -DGGML_CUDA_GRAPHS=OFF or at runtime with GGML_CUDA_DISABLE_GRAPHS=1.
if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
add_subdirectory(ggml)

# cpp-httplib (HTTP server library, used by ace-server)
add_subdirectory(vendor/cpp-httplib)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()
    add_dependencies(${target} version)
endmacro()

# yyjson (MIT, fast JSON parser/writer)
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Lua 5.4 (MIT, embedded scripting for plugin system)
# All .c files except lua.c (standalone interpreter) and luac.c (compiler)
file(GLOB LUA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vendor/lua/*.c")
list(FILTER LUA_SOURCES EXCLUDE REGEX "(lua|luac)\\.c$")
add_library(lua54 STATIC ${LUA_SOURCES})
target_include_directories(lua54 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/lua)
if(MSVC)
    target_compile_options(lua54 PRIVATE /W0)
else()
    target_compile_options(lua54 PRIVATE -w)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime (SuperSep stem separation + VAE-ORT TensorRT acceleration)
# ─────────────────────────────────────────────────────────────────────────────
# Pre-built ORT GPU package. Resolution order:
#   1. ORT_ROOT cmake variable
#   2. ONNXRUNTIME_ROOT environment variable
#   3. Auto-detect from engine/deps/onnxruntime/ (populated by buildall.cmd)
#
# CUDAToolkit detection: the find_package at L73 is conditional on
# CMAKE_CUDA_ARCHITECTURES, so on cached re-configures CUDAToolkit_FOUND
# may be unset. Ensure it's always available for CUDA EP support.
find_package(CUDAToolkit QUIET)
option(HOT_STEP_SUPERSEP "Enable SuperSep stem separation (requires ONNX Runtime)" ON)

set(ORT_ROOT "" CACHE PATH "Path to ONNX Runtime pre-built package")
if(NOT ORT_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
# Auto-detect from deps directory (buildall.cmd downloads here)
if(NOT ORT_ROOT)
    set(_ORT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")
    if(EXISTS "${_ORT_DEPS_DIR}/include/onnxruntime_cxx_api.h")
        set(ORT_ROOT "${_ORT_DEPS_DIR}")
        message(STATUS "ORT auto-detected at ${ORT_ROOT}")
    endif()
endif()

set(SUPERSEP_ENABLED FALSE)
if(HOT_STEP_SUPERSEP AND ORT_ROOT)
    if(EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
        set(SUPERSEP_ENABLED TRUE)
    else()
        message(WARNING "ORT_ROOT set but onnxruntime_cxx_api.h not found at ${ORT_ROOT}/include")
    endif()
endif()

# SuperSep library (ONNX Runtime inference + STFT)
add_library(supersep STATIC src/supersep.cpp)
target_include_directories(supersep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/pocketfft
    ${CMAKE_CURRENT_BINARY_DIR}
)
if(MSVC)
    target_compile_options(supersep PRIVATE /W4 /wd4100 /wd4505 /wd4244 /wd4267)
else()
    target_compile_options(supersep PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-sign-conversion)
endif()

if(SUPERSEP_ENABLED)
    target_compile_definitions(supersep PUBLIC HOT_STEP_SUPERSEP)
    target_include_directories(supersep PUBLIC "${ORT_ROOT}/include")
    target_link_directories(supersep PUBLIC "${ORT_ROOT}/lib")
    target_link_libraries(supersep PUBLIC onnxruntime)
    message(STATUS "SuperSep: ENABLED (ORT at ${ORT_ROOT})")
else()
    message(STATUS "SuperSep: DISABLED (set ORT_ROOT to enable)")
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(supersep PUBLIC GGML_USE_CUDA)
endif()

# macOS: CoreML EP requires Apple frameworks
if(APPLE AND SUPERSEP_ENABLED)
    target_link_libraries(supersep PUBLIC "-framework CoreML" "-framework Foundation")
endif()
# TensorRT Native SDK (DiT TRT acceleration + LoRA refitting)
# ─────────────────────────────────────────────────────────────────────────────
# Platform-specific TRT detection:
#   Windows: vendored SDK in engine/deps/tensorrt/ (versioned import libs)
#   Linux:   system-installed TRT packages via find_path/find_library
# The runtime DLLs/SOs are expected on the library search path at runtime.
set(TRT_ENABLED FALSE)

# --- Windows: vendored SDK in engine/deps/tensorrt/ ---
set(_TRT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tensorrt")
if(WIN32 AND EXISTS "${_TRT_DEPS_DIR}/include/NvInfer.h" AND EXISTS "${_TRT_DEPS_DIR}/lib/nvinfer_10.lib")
    set(TRT_ENABLED TRUE)
    set(_TRT_INCLUDE_DIR "${_TRT_DEPS_DIR}/include")
    set(_TRT_LIB_DIR "${_TRT_DEPS_DIR}/lib")
    # Windows vendored SDK ships versioned import libs
    set(_TRT_NVINFER_LIB nvinfer_10)
    set(_TRT_NVONNXPARSER_LIB nvonnxparser_10)
    message(STATUS "[TRT] Found vendored SDK at ${_TRT_DEPS_DIR}")

# --- Linux: system-installed TRT packages ---
elseif(NOT WIN32)
    find_path(_TRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS /usr/include/x86_64-linux-gnu /usr/local/include /usr/include
    )
    find_library(_TRT_NVINFER_LIB
        NAMES nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib
    )
    find_library(_TRT_NVONNXPARSER_LIB
        NAMES nvonnxparser
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib
    )
    if(_TRT_INCLUDE_DIR AND _TRT_NVINFER_LIB AND _TRT_NVONNXPARSER_LIB)
        set(TRT_ENABLED TRUE)
        message(STATUS "[TRT] Found system TRT: ${_TRT_NVINFER_LIB}")
    else()
        message(STATUS "[TRT] Not found (install libnvinfer-dev + libnvonnxparsers-dev to enable)")
    endif()
endif()

if(NOT TRT_ENABLED AND WIN32)
    message(STATUS "[TRT] Not found (set engine/deps/tensorrt/ to enable DiT TRT)")
endif()

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson lua54 supersep)
link_ggml_backends(acestep-core)

# TRT native linkage for DiT acceleration (shared across platforms)
if(TRT_ENABLED)
    target_compile_definitions(acestep-core PUBLIC HOT_STEP_TRT)
    target_include_directories(acestep-core PUBLIC "${_TRT_INCLUDE_DIR}")
    if(_TRT_LIB_DIR)
        target_link_directories(acestep-core PUBLIC "${_TRT_LIB_DIR}")
    endif()
    target_link_libraries(acestep-core PUBLIC ${_TRT_NVINFER_LIB} ${_TRT_NVONNXPARSER_LIB})
    # TRT headers include cuda_runtime_api.h — on Linux g++ needs the CUDA
    # include path explicitly (nvcc gets it automatically, but acestep-core
    # compiles as plain C++). Windows CUDA Toolkit puts headers on PATH.
    if(NOT WIN32)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            target_include_directories(acestep-core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(acestep-core PUBLIC CUDA::cudart)
            message(STATUS "[TRT] CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
        endif()
    endif()
    message(STATUS "[TRT] DiT TRT acceleration: ENABLED")
endif()

# acestep-core compiles as plain C++ (not nvcc), but pipeline-synth.cpp includes
# <cuda_runtime.h> for VRAM instrumentation whenever GGML_USE_CUDA is defined
# (propagated from ggml). The host compiler needs the CUDA Toolkit include path
# explicitly on EVERY CUDA build — Linux g++ and Windows MSVC (under Ninja) both
# fail to find it otherwise. (The TRT block above also adds this, but only when
# TRT is enabled, so plain CUDA release builds were missing it.)
if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_include_directories(acestep-core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(acestep-core PUBLIC CUDA::cudart)
        message(STATUS "[CUDA] acestep-core include: ${CUDAToolkit_INCLUDE_DIRS}")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# TRT-LLM Executor (C++ Executor API for LM inference)
# ─────────────────────────────────────────────────────────────────────────────
# Pre-built TRT-LLM SDK (tensorrt_llm.dll + plugin DLL).
# Auto-detect from engine/trtllm-libs/.
# This is SEPARATE from HOT_STEP_TRT (raw NvInfer for DiT). Both can coexist.
#
# STATUS: DISABLED (2026-06-02). Native Windows TRT-LLM is not viable:
#   - FMHA/XQA cubin embedding requires GCC inline asm (INCBIN), impossible on MSVC
#   - Docker-built engines have Linux platform tags, can't deserialize on Windows
#   - TRT version mismatch between Docker (10.14) and Windows SDK (10.16)
#   - ONNX-rebuilt engines lack TRT-LLM tensor bindings (kv_cache_block_offsets etc.)
# The code remains intact behind #ifdef HOT_STEP_TRTLLM for future WSL2 or
# cross-platform engine support. To re-enable, set HOT_STEP_TRTLLM_ENABLE=ON.
option(HOT_STEP_TRTLLM_ENABLE "Enable TRT-LLM Executor (currently broken on native Windows)" OFF)
set(TRTLLM_ENABLED FALSE)
set(_TRTLLM_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtllm-libs")
set(_TRTLLM_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtllm-include")
if(HOT_STEP_TRTLLM_ENABLE AND WIN32
         AND EXISTS "${_TRTLLM_LIBS_DIR}/tensorrt_llm.lib"
         AND EXISTS "${_TRTLLM_INC_DIR}/tensorrt_llm/executor/executor.h")
    set(TRTLLM_ENABLED TRUE)
    message(STATUS "[TRT-LLM] Found at ${_TRTLLM_LIBS_DIR}")
    target_compile_definitions(acestep-core PUBLIC HOT_STEP_TRTLLM)
    target_include_directories(acestep-core PUBLIC "${_TRTLLM_INC_DIR}")
    target_link_directories(acestep-core PUBLIC "${_TRTLLM_LIBS_DIR}")
    target_link_libraries(acestep-core PUBLIC tensorrt_llm)
    # NOTE: nvinfer_plugin_tensorrt_llm is loaded dynamically via LoadLibrary
    # in lm-trtllm.h to avoid pulling its dependency chain at process startup.
    message(STATUS "[TRT-LLM] LM Executor: ENABLED")
else()
    if(HOT_STEP_TRTLLM_ENABLE)
        message(STATUS "[TRT-LLM] Not found (set engine/trtllm-libs/ + trtllm-include/ to enable)")
    else()
        message(STATUS "[TRT-LLM] DISABLED (set -DHOT_STEP_TRTLLM_ENABLE=ON to re-enable)")
    endif()
endif()


# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)
# CUDA runtime for TRT DiT path (cudaMalloc, cudaMemcpy, etc.)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(ace-synth PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-synth PRIVATE CUDA::cudart)
    endif()
endif()

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)
# CUDA runtime for TRT LM path (cudaMalloc, cudaMemcpy, etc.)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(ace-lm PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-lm PRIVATE CUDA::cudart)
    endif()
endif()

# webui: convert tools/webui/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild ace-server.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# hot-step-server: HOT-Step HTTP server (LM + synth endpoints + embedded webui)
# NOTE: upstream ace-server.cpp is kept as reference but NOT compiled.
#       Our server binary is hot-step-server.cpp with extension layer support.
add_executable(ace-server tools/hot-step-server.cpp ${WEBUI_OUTPUT})
target_include_directories(ace-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ace-server PRIVATE acestep-core httplib supersep)
link_ggml_backends(ace-server)
# CUDA runtime for GET /vram (cudaMemGetInfo)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(ace-server PRIVATE GGML_USE_CUDA)
    if(GGML_STATIC)
        target_link_libraries(ace-server PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-server PRIVATE CUDA::cudart)
    endif()
    # Copy CUDA runtime DLL to build dir (needed for portable release builds —
    # end users don't have the CUDA Toolkit, so cudart64_*.dll must ship with the binary)
    if(WIN32)
        get_target_property(_CUDART_LOC CUDA::cudart IMPORTED_LOCATION)
        if(_CUDART_LOC)
            add_custom_command(TARGET ace-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_CUDART_LOC}"
                    "$<TARGET_FILE_DIR:ace-server>"
                COMMENT "Copying cudart DLL for portable release"
            )
        endif()
    endif()
    # Write CUDA version marker next to ace-server (server reads this to
    # select the correct runtime DLLs — cuBLAS/cudart/cuDNN for 12 vs 13)
    file(GENERATE OUTPUT "$<TARGET_FILE_DIR:ace-server>/.cuda-version"
         CONTENT "${CUDAToolkit_VERSION_MAJOR}")
endif()

# Copy ONNX Runtime DLLs next to ace-server at build time
if(SUPERSEP_ENABLED AND WIN32)
    set(_ORT_DLL_DIR "${ORT_ROOT}/lib")
    foreach(_dll onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll onnxruntime_providers_tensorrt.dll)
        if(EXISTS "${_ORT_DLL_DIR}/${_dll}")
            add_custom_command(TARGET ace-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_ORT_DLL_DIR}/${_dll}"
                    "$<TARGET_FILE_DIR:ace-server>/${_dll}"
                COMMENT "Copying ${_dll}"
            )
        endif()
    endforeach()
endif()

# Copy ONNX Runtime dylibs next to ace-server at build time (macOS)
if(SUPERSEP_ENABLED AND APPLE)
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    file(GLOB _ORT_DYLIBS "${_ORT_LIB_DIR}/libonnxruntime*.dylib")
    foreach(_dylib ${_ORT_DYLIBS})
        add_custom_command(TARGET ace-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dylib}" "$<TARGET_FILE_DIR:ace-server>"
            COMMENT "Copying ${_dylib}"
        )
    endforeach()
endif()

# Copy ONNX Runtime shared libs next to ace-server at build time (Linux)
if(SUPERSEP_ENABLED AND UNIX AND NOT APPLE)
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    file(GLOB _ORT_SOLIBS "${_ORT_LIB_DIR}/libonnxruntime*.so*")
    foreach(_solib ${_ORT_SOLIBS})
        add_custom_command(TARGET ace-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_solib}" "$<TARGET_FILE_DIR:ace-server>"
            COMMENT "Copying ${_solib}"
        )
    endforeach()
endif()

add_executable(ace-understand tools/ace-understand.cpp)
target_link_libraries(ace-understand PRIVATE acestep-core)
link_ggml_backends(ace-understand)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)

# sa3-ggml-test: parity tests for the StableStep GGML SA3 modules vs goldens
add_executable(sa3-ggml-test tools/sa3-ggml-test.cpp)
target_link_libraries(sa3-ggml-test PRIVATE yyjson)
link_ggml_backends(sa3-ggml-test)

# ace-midi: MuScriptor audio->MIDI transcription (GGML port, in development —
# docs/plans/muscriptor-cpp-port.md). Standalone tool, no acestep-core needed.
add_executable(ace-midi tools/ace-midi.cpp)
link_ggml_backends(ace-midi)

# mp3-codec: MP3 encoder/decoder (standalone, no ggml needed)
# The mp3/ headers are header-only and usable by ace-synth too via #include "mp3/mp3enc.h"
add_executable(mp3-codec tools/mp3-codec.cpp)
target_include_directories(mp3-codec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mp3-codec version)
if(MSVC)
    target_compile_options(mp3-codec PRIVATE /W4 /wd4100 /wd4505)
else()
    target_compile_options(mp3-codec PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mp3-codec PRIVATE m)
endif()
target_link_libraries(mp3-codec PRIVATE Threads::Threads)

# mastering: reference-based audio mastering (standalone, no ggml needed)
# Implements the matchering algorithm using pocketfft for FFT.
add_executable(mastering tools/mastering.cpp)
target_include_directories(mastering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/pocketfft
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mastering version)
if(MSVC)
    target_compile_options(mastering PRIVATE /W4 /wd4100 /wd4505 /wd4244 /wd4267)
else()
    target_compile_options(mastering PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mastering PRIVATE m)
endif()
target_link_libraries(mastering PRIVATE Threads::Threads)

# ─────────────────────────────────────────────────────────────────────────────
# VST3 Hosting Library + vst-host tool
# ─────────────────────────────────────────────────────────────────────────────
# We compile only the necessary VST3 SDK source files for hosting (loading
# plugins, processing audio, state save/restore). We do NOT use the SDK's
# own CMakeLists.txt because it requires cmake 3.25+ and pulls in the full
# build system including plugin examples and VSTGUI.

set(VST3SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vst3sdk")

# Static library: vst3-hosting
# Platform-specific hosting sources:
#   Windows: module_win32.cpp + threadchecker_win32.cpp
#   macOS:   module_mac.mm (ARC) + threadchecker_mac.mm
#   Linux:   module_linux.cpp + threadchecker_linux.cpp
if(APPLE)
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_mac.mm
    )
    # module_mac.mm requires Objective-C ARC
    set_source_files_properties(
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_mac.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
elseif(WIN32)
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_win32.cpp
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_win32.cpp
    )
else()
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_linux.cpp
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_linux.cpp
    )
endif()

add_library(vst3-hosting STATIC
    # Base library
    ${VST3SDK_ROOT}/base/source/baseiids.cpp
    ${VST3SDK_ROOT}/base/source/fobject.cpp
    ${VST3SDK_ROOT}/base/source/fdebug.cpp
    ${VST3SDK_ROOT}/base/source/fstreamer.cpp
    ${VST3SDK_ROOT}/base/source/fbuffer.cpp
    ${VST3SDK_ROOT}/base/source/updatehandler.cpp
    ${VST3SDK_ROOT}/base/source/timer.cpp
    ${VST3SDK_ROOT}/base/source/fstring.cpp

    # Pluginterfaces
    ${VST3SDK_ROOT}/pluginterfaces/base/funknown.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/ustring.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/coreiids.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/conststringtable.cpp

    # Public SDK hosting (cross-platform + platform-specific)
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module.cpp
    ${VST3_PLATFORM_SOURCES}
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/hostclasses.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/plugprovider.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/processdata.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/parameterchanges.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/pluginterfacesupport.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/eventlist.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/connectionproxy.cpp

    # Public SDK common (memory streams for state I/O)
    ${VST3SDK_ROOT}/public.sdk/source/common/memorystream.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/commoniids.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/pluginview.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/commonstringconvert.cpp

    # VST interface ID definitions (all DEF_CLASS_IID symbols)
    ${VST3SDK_ROOT}/public.sdk/source/vst/vstinitiids.cpp

    # VST utilities
    ${VST3SDK_ROOT}/public.sdk/source/vst/utility/stringconvert.cpp
)

target_include_directories(vst3-hosting PUBLIC
    ${VST3SDK_ROOT}
    ${VST3SDK_ROOT}/pluginterfaces
    ${VST3SDK_ROOT}/public.sdk
)

# The SDK defines DEVELOPMENT=1 for debug builds
target_compile_definitions(vst3-hosting PRIVATE
    $<$<CONFIG:Debug>:DEVELOPMENT=1>
    $<$<CONFIG:Release>:RELEASE=1>
)

if(MSVC)
    # Suppress noisy SDK warnings
    target_compile_options(vst3-hosting PRIVATE /W0)
else()
    target_compile_options(vst3-hosting PRIVATE -w)
endif()

# macOS: VST3 hosting needs Cocoa + CoreFoundation frameworks
if(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    target_link_libraries(vst3-hosting PRIVATE
        ${COCOA_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
    )
endif()

# vst-host: standalone VST3 host tool (GUI + offline processing + chain)
add_executable(vst-host tools/vst-host.cpp)
target_include_directories(vst-host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${VST3SDK_ROOT}
)
target_link_libraries(vst-host PRIVATE vst3-hosting yyjson Threads::Threads)
if(MSVC)
    target_compile_options(vst-host PRIVATE /W4 /wd4100 /wd4505)
    target_compile_definitions(vst-host PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(vst-host PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
endif()
add_dependencies(vst-host version)

# macOS: vst-host also needs Apple frameworks
if(APPLE)
    target_link_libraries(vst-host PRIVATE ${COCOA_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK})
endif()

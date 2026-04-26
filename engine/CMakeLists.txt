cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

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

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
if(NOT MSVC)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
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

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson)
link_ggml_backends(acestep-core)

# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)

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
target_link_libraries(ace-server PRIVATE acestep-core httplib)
link_ggml_backends(ace-server)
# CUDA runtime for GET /vram (cudaMemGetInfo)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(ace-server PRIVATE GGML_USE_CUDA)
    target_link_libraries(ace-server PRIVATE CUDA::cudart)
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

    # Public SDK hosting
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_win32.cpp
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
    ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_win32.cpp
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

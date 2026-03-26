cmake_minimum_required(VERSION 3.22.1)
project(accv_llama)

set(LLAMA_ROOT ${CMAKE_SOURCE_DIR}/../../../../external/llama.cpp)
set(MTMD_ROOT  ${LLAMA_ROOT}/tools/mtmd)

# Disable unneeded llama.cpp components
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL  OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# Vulkan requires host-side shader compilation via vulkan-shaders-gen.
# NDK 27 + CMake 4.0 generates a broken host-toolchain.cmake that mixes
# MSVC (compile) with ld.lld (link), so Vulkan build fails on Windows.
set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)

# Build llama core (ggml + llama)
add_subdirectory(${LLAMA_ROOT} ${CMAKE_BINARY_DIR}/llama_core)

# Build mtmd directly (avoids MACHO_CURRENT_VERSION issue on Android)
find_package(Threads REQUIRED)
add_library(mtmd STATIC
    ${MTMD_ROOT}/mtmd.cpp
    ${MTMD_ROOT}/mtmd-audio.cpp
    ${MTMD_ROOT}/mtmd-helper.cpp
    ${MTMD_ROOT}/clip.cpp
    ${MTMD_ROOT}/models/cogvlm.cpp
    ${MTMD_ROOT}/models/conformer.cpp
    ${MTMD_ROOT}/models/glm4v.cpp
    ${MTMD_ROOT}/models/internvl.cpp
    ${MTMD_ROOT}/models/kimivl.cpp
    ${MTMD_ROOT}/models/kimik25.cpp
    ${MTMD_ROOT}/models/nemotron-v2-vl.cpp
    ${MTMD_ROOT}/models/llama4.cpp
    ${MTMD_ROOT}/models/llava.cpp
    ${MTMD_ROOT}/models/minicpmv.cpp
    ${MTMD_ROOT}/models/paddleocr.cpp
    ${MTMD_ROOT}/models/pixtral.cpp
    ${MTMD_ROOT}/models/qwen2vl.cpp
    ${MTMD_ROOT}/models/qwen3vl.cpp
    ${MTMD_ROOT}/models/siglip.cpp
    ${MTMD_ROOT}/models/whisper-enc.cpp
    ${MTMD_ROOT}/models/mobilenetv5.cpp
    ${MTMD_ROOT}/models/youtuvl.cpp
)
target_include_directories(mtmd PUBLIC  ${MTMD_ROOT})
target_include_directories(mtmd PRIVATE ${LLAMA_ROOT} ${LLAMA_ROOT}/vendor)
target_link_libraries(mtmd PUBLIC ggml llama)
target_link_libraries(mtmd PRIVATE Threads::Threads)
target_compile_features(mtmd PRIVATE cxx_std_17)
if (NOT MSVC)
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

# JNI bridge
add_library(accv_llama SHARED llama_bridge.cpp)

target_include_directories(accv_llama PRIVATE
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
    ${MTMD_ROOT}
)

target_link_libraries(accv_llama
    llama
    mtmd
    android
    log
)

target_compile_features(accv_llama PRIVATE cxx_std_17)

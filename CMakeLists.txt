cmake_minimum_required(VERSION 3.31)
project(FluidiTy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLUIDITY_ENABLE_CUDA "Build the GPU implementations with the CUDA compiler." OFF)
option(BUILD_TESTING "Build the tests." ON)

# Before Visual Studio 2022 (MSVC 19.30), /fp:precise still allowed the compiler to contract a * b + c into one fused
# multiply-add, which rounds once instead of twice; from 19.30 it does not. This is per Microsoft's documentation of
# /fp, not verified here, so an older MSVC is refused rather than trusted. The check reads the compiler id, not the
# MSVC variable: clang-cl reports the id Clang (simulating MSVC) and a Clang version, which says nothing about MSVC's
# /fp, so clang-cl is not checked here.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.30)
  message(FATAL_ERROR "MSVC ${CMAKE_CXX_COMPILER_VERSION} is older than 19.30 (Visual Studio 2022): its /fp:precise "
                      "may contract multiply-adds, which changes results.")
endif()

# GPU code is C++20: the CUDA compiler does not build C++23. The library links the shared CUDA runtime
# (CUDA::cudart), so the CUDA compiler links the same one; otherwise a program would carry two runtimes.
if(FLUIDITY_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  # Carried over from main: NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS add flags to every nvcc call outside CMake, so
  # they could undo the numerical flags; configuration fails while either is set.
  if(NOT "$ENV{NVCC_PREPEND_FLAGS}" STREQUAL "" OR NOT "$ENV{NVCC_APPEND_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Unset NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS before configuring CUDA: they add flags to "
                        "every nvcc call, outside the numerical flags of cmake/fluidity_targets.cmake.")
  endif()
  # Windows, carried over from the configuration main ran on the owner's machine: nvcc uses the same cl as the C++
  # code, and CMake's default /EHsc for nvcc is removed so each target chooses its exception flag.
  if(WIN32 AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(MSVC)
    string(REPLACE "-Xcompiler=\" /EHsc\"" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  endif()
endif()

# Host code returns std::expected, so a standard library without it cannot build FluidiTy.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles([=[
#include <expected>
int main() {
  const std::expected<int, int> value = 1;
  return value.has_value() ? 0 : 1;
}
]=] FLUIDITY_HAS_STD_EXPECTED)
if(NOT FLUIDITY_HAS_STD_EXPECTED)
  message(FATAL_ERROR "This compiler's standard library has no std::expected")
endif()

include(cmake/fluidity_targets.cmake)

# The library. Each module folder adds its public headers, its sources and its tests from its own CMakeLists.txt.
add_library(fluidity STATIC)
fluidity_apply_compile_flags(fluidity EXCEPTIONS OFF)
if(FLUIDITY_ENABLE_CUDA)
  # PUBLIC, so tests see it too: a test that expects kGpuNotBuilt exists only in a build without GPU code.
  target_compile_definitions(fluidity PUBLIC FLUIDITY_HAS_GPU)
  target_link_libraries(fluidity PUBLIC CUDA::cudart)
endif()

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED CONFIG)
  include(GoogleTest)
endif()

add_subdirectory(native/fluidity/base)

# Last: every target now exists, so the check sees them all.
fluidity_check_compile_flags()
fluidity_write_gpu_only_sources()

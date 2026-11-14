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

# CUDA: the GPU build is added by a later commit. Until then, stop rather than build a CPU-only library under a
# GPU preset.
if(FLUIDITY_ENABLE_CUDA)
  message(FATAL_ERROR "FLUIDITY_ENABLE_CUDA is not written yet. Configure with FLUIDITY_ENABLE_CUDA=OFF.")
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

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED CONFIG)
  include(GoogleTest)
endif()

add_subdirectory(native/fluidity/base)

# Last: every target now exists, so the check sees them all.
fluidity_check_compile_flags()

cmake_minimum_required(VERSION 3.31)
project(FluidiTy LANGUAGES CXX)

option(FLUIDITY_ENABLE_CUDA "Enable the CUDA C++20 toolchain" OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
include(CTest)
include(cmake/NativeOptions.cmake)

if(WIN32)
  fluidity_check_windows_toolchain()
endif()

if(FLUIDITY_ENABLE_CUDA)
  if(NOT "$ENV{NVCC_PREPEND_FLAGS}" STREQUAL "" OR NOT "$ENV{NVCC_APPEND_FLAGS}" STREQUAL "")
    message(FATAL_ERROR "Unset NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS before configuring CUDA.")
  endif()
  if(WIN32 AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
  if(WIN32)
    fluidity_check_windows_cuda_toolchain()
  endif()
endif()

if(BUILD_TESTING)
  find_package(absl 20250814.2 EXACT CONFIG REQUIRED)
  find_package(GTest 1.17.0 EXACT CONFIG REQUIRED)
  include(GoogleTest)

  if("$ENV{VIRTUAL_ENV}" STREQUAL "")
    message(FATAL_ERROR "Configure native tests through uv run --project reference --locked --group native.")
  endif()
  set(Python3_FIND_VIRTUALENV ONLY)
  find_package(Python3 3.13...<3.14 REQUIRED COMPONENTS Interpreter)
  cmake_path(CONVERT "$ENV{VIRTUAL_ENV}" TO_CMAKE_PATH_LIST fixture_venv NORMALIZE)
  cmake_path(IS_PREFIX fixture_venv "${Python3_EXECUTABLE}" NORMALIZE fixture_python_in_venv)
  if(NOT fixture_python_in_venv)
    message(FATAL_ERROR "Native tests require Python 3.13 from the active uv environment.")
  endif()

  set(FLUIDITY_FIXTURE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures" CACHE PATH
      "Directory for exported reference fixtures")
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(fixture_header "${generated_dir}/fixture_data.h")
  set(fixture_export_stamp "${CMAKE_CURRENT_BINARY_DIR}/reference-fixtures.stamp")
  set(reference_src "${PROJECT_SOURCE_DIR}/reference/src/fluidity_reference")
  # Track the export separately so an edited or missing JSON file is validated as-is.
  add_custom_command(
    OUTPUT "${fixture_export_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${fixture_header}"
    COMMAND "${Python3_EXECUTABLE}" -m fluidity_reference.cli export "${FLUIDITY_FIXTURE_DIRECTORY}"
            --overwrite
    COMMAND "${CMAKE_COMMAND}" -E touch "${fixture_export_stamp}"
    DEPENDS "${reference_src}/__init__.py" "${reference_src}/cli.py" "${reference_src}/fixtures.py"
            "${PROJECT_SOURCE_DIR}/reference/pyproject.toml" "${PROJECT_SOURCE_DIR}/reference/uv.lock"
    COMMENT "Exporting independent reference fixtures"
    VERBATIM)

  set(fixture_generator "${PROJECT_SOURCE_DIR}/tools/native/compile_fixtures.py")
  function(fluidity_compile_fixtures target input header)
    # Validate on every build to catch missing, extra and changed JSON files, including edits
    # that preserve timestamps. The generator leaves an identical header's timestamp alone.
    add_custom_target(${target}
      COMMAND "${Python3_EXECUTABLE}" "${fixture_generator}" --input "${input}" --output "${header}"
      BYPRODUCTS "${header}"
      DEPENDS "${fixture_generator}" ${ARGN}
      COMMENT "Validating hashed JSON for ${target}"
      VERBATIM)
  endfunction()

  fluidity_compile_fixtures(fluidity_fixture_headers "${FLUIDITY_FIXTURE_DIRECTORY}"
                           "${fixture_header}" "${fixture_export_stamp}")
  add_executable(fluidity_native_tests native/tests/test_build.cc native/tests/test_fixture_bridge.cc)
  add_dependencies(fluidity_native_tests fluidity_fixture_headers)

  # Use a separate test executable for the synthetic case so each executable owns one data set.
  set(escaping_input "${PROJECT_SOURCE_DIR}/native/tests/fixtures/escaping")
  set(escaping_header "${generated_dir}/fixture_escaping_data.h")
  fluidity_compile_fixtures(fluidity_fixture_escaping_header "${escaping_input}" "${escaping_header}"
                           "${escaping_input}/escaping.json" "${escaping_input}/manifest.json")
  add_executable(fluidity_fixture_bridge_escaping native/tests/test_fixture_escaping.cc)
  add_dependencies(fluidity_fixture_bridge_escaping fluidity_fixture_escaping_header)

  foreach(target IN ITEMS fluidity_native_tests fluidity_fixture_bridge_escaping)
    target_include_directories(${target} PRIVATE "${generated_dir}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    target_link_libraries(${target} PRIVATE absl::status absl::statusor GTest::gtest_main)
    gtest_discover_tests(${target} PROPERTIES LABELS cpu)
  endforeach()
endif()

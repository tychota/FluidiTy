# Compile flags and the helpers that create every FluidiTy target. They apply to our targets only, never to the
# Conan packages. Every C++ or CUDA target of the project is created through these helpers, so it gets the
# numerical flags: fluidity_check_compile_flags, called at the end of the top-level CMakeLists.txt, fails the
# configuration for any target that did not.

if(MSVC)
  # CMake adds /EHsc to every MSVC target. Remove it so each target chooses its own exception flag below.
  string(REGEX REPLACE "(^|[ \t])/EHsc($|[ \t])" "\\1\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# fluidity_apply_compile_flags(<target> EXCEPTIONS <ON|OFF>)
#
# Warnings are errors and floating-point contraction and fast-math are off. RTTI is off. EXCEPTIONS OFF is for the
# library; EXCEPTIONS ON is for test programs, because GoogleTest needs exceptions.
# CUDA sources get the device flags: --fmad=false forbids fused multiply-add, and --ftz=false, --prec-div=true and
# --prec-sqrt=true govern single precision only, because an NVIDIA GPU always keeps subnormal doubles and rounds
# double division and square root correctly. CUDA sources also pass the host flags to the host compiler with
# -Xcompiler, because the CPU half of a .cu file, such as the CPU loop of the GPU math check, must round as the .cc
# files do.
# Windows (MSVC host compiler): the CUDA flags follow the configuration that the proof of concept on main ran on the
# owner's RTX 5060 Ti (nvcc 13.4, MSVC 19.44; main:docs/native/toolchains.md). Carried over from it: the four device
# flags, --no-exceptions, -Xcompiler=/GR-,/fp:precise,/Zc:__cplusplus, the shared runtime, no CUDA warnings as
# errors, because nvcc 13.4 on Windows warns that the `--cudart=shared` spelling CMake passes is deprecated, and (on
# every platform) the refusal to configure while NVCC_PREPEND_FLAGS or NVCC_APPEND_FLAGS is set, in the top-level
# CMakeLists.txt. New here
# and not yet compiled on Windows: -Xcompiler=/EHsc for the .cu files of test programs. On Linux CUDA warnings are
# errors and the GCC host flags are passed one by one, as the container build proves.
# It also sets the target property FLUIDITY_COMPILE_FLAGS_APPLIED, which fluidity_check_compile_flags reads.
function(fluidity_apply_compile_flags target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "EXCEPTIONS" "")
  if(NOT DEFINED arg_EXCEPTIONS)
    message(FATAL_ERROR "fluidity_apply_compile_flags(${target}) needs EXCEPTIONS ON or EXCEPTIONS OFF.")
  endif()
  set_property(TARGET ${target} PROPERTY FLUIDITY_COMPILE_FLAGS_APPLIED TRUE)

  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/WX;/GR-;/fp:precise;/permissive-;/Zc:__cplusplus>"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror;-fno-rtti>"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fno-fast-math;-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:--Werror=all-warnings>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/GR-,/fp:precise,/Zc:__cplusplus>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang>>:-Xcompiler=-Wall;-Xcompiler=-Wextra>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang>>:-Xcompiler=-Werror>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang>>:-Xcompiler=-fno-rtti;-Xcompiler=-fno-fast-math>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU,Clang>>:-Xcompiler=-ffp-contract=off>"
  )
  if(arg_EXCEPTIONS)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>"
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fexceptions>"
      "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/EHsc>"
    )
  else()
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHs-c->"
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fno-exceptions>"
      "$<$<COMPILE_LANGUAGE:CUDA>:--no-exceptions>"
    )
  endif()
endfunction()

# fluidity_add_headers(<files...>)
#
# Adds public headers of the calling module folder to the library, as the HEADERS file set, with src/ as the
# include root. For each header it also writes one .cc file in the build folder that contains only
# `#include "<header>"` and adds it to the library. Each header is therefore compiled alone, with the library flags
# (no exceptions, no RTTI, warnings as errors), and a header that forgets one of its own includes fails to build.
# With FLUIDITY_ENABLE_CUDA ON, each shared header is also compiled alone as a .cu file, with the device flags. A
# header that includes base/cpu_only.h is CPU-only and is skipped, and so is cpu_only.h itself, which stops any GPU
# build by design. A .cuh header needs the CUDA compiler, so it gets only the .cu file, and only with CUDA ON.
function(fluidity_add_headers)
  set(include_root "${PROJECT_SOURCE_DIR}/src")
  target_sources(fluidity PUBLIC FILE_SET HEADERS BASE_DIRS "${include_root}" FILES ${ARGN})
  foreach(header IN LISTS ARGN)
    cmake_path(ABSOLUTE_PATH header BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE header_path)
    cmake_path(RELATIVE_PATH header_path BASE_DIRECTORY "${include_root}" OUTPUT_VARIABLE include_path)
    cmake_path(GET header_path EXTENSION LAST_ONLY header_extension)
    if(NOT header_extension STREQUAL ".cuh")
      set(check_source "${PROJECT_BINARY_DIR}/header_check/${include_path}.cc")
      file(CONFIGURE OUTPUT "${check_source}" CONTENT "#include \"${include_path}\"\n")
      target_sources(fluidity PRIVATE "${check_source}")
    endif()

    if(FLUIDITY_ENABLE_CUDA)
      file(READ "${header_path}" header_text)
      string(FIND "${header_text}" "#include \"fluidity/base/cpu_only.h\"" cpu_only_include_position)
      set(is_cpu_only_guard FALSE)
      if(include_path STREQUAL "fluidity/base/cpu_only.h")
        set(is_cpu_only_guard TRUE)
      endif()
      if(cpu_only_include_position EQUAL -1 AND NOT is_cpu_only_guard)
        set(gpu_check_source "${PROJECT_BINARY_DIR}/header_check/${include_path}.cu")
        file(CONFIGURE OUTPUT "${gpu_check_source}" CONTENT "#include \"${include_path}\"\n")
        target_sources(fluidity PRIVATE "${gpu_check_source}")
      endif()
    endif()
  endforeach()
endfunction()

# fluidity_add_gpu_sources(<files...>)
#
# Adds the GPU-only sources of the calling module folder (.cu files, or .cc files that only a CUDA build needs) to the
# library when FLUIDITY_ENABLE_CUDA is ON; a .cc file among them is recorded for fluidity_write_gpu_only_sources, so
# the gate tidies it. Does nothing otherwise, so a CPU build never needs the CUDA compiler.
function(fluidity_add_gpu_sources)
  if(NOT FLUIDITY_ENABLE_CUDA)
    return()
  endif()
  target_sources(fluidity PRIVATE ${ARGN})
  fluidity_record_gpu_only_sources(${ARGN})
endfunction()

# fluidity_record_gpu_only_sources(<files...>)
#
# Records the .cc files among <files> (relative to the calling folder) as compiled only by a build with
# FLUIDITY_ENABLE_CUDA ON, for fluidity_write_gpu_only_sources. Other files are ignored: the .cu files are not tidied.
function(fluidity_record_gpu_only_sources)
  foreach(source IN LISTS ARGN)
    cmake_path(GET source EXTENSION LAST_ONLY source_extension)
    if(source_extension STREQUAL ".cc")
      cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE source_path)
      set_property(GLOBAL APPEND PROPERTY FLUIDITY_GPU_ONLY_SOURCES "${source_path}")
    endif()
  endforeach()
endfunction()

# fluidity_add_test(<name> SOURCES <files...> LABEL <cpu|gpu|no_gpu> [LINKS <targets...>])
#
# Builds one GoogleTest program linked to the library and registers each of its tests with CTest under the label,
# so `ctest -L '^cpu$'` runs the CPU tests only. Each test also gets the name of its module, the name of the folder of
# the calling CMakeLists.txt (base, grid, contact), so `ctest --preset macos-debug-cpu -L grid` runs the grid tests
# alone. A label filter is a regular expression: a bare `-L gpu` would also
# select the no_gpu tests, so `ctest -L '^gpu$'` runs the tests that need a GPU.
#   cpu     runs on the CPU only.
#   gpu     needs a CUDA GPU at run time.
#   no_gpu  checks behaviour on a machine without a usable GPU (for example, that choosing the GPU fails).
# Tests are listed when ctest runs (PRE_TEST), not after the build, so building never runs a test program: a GPU
# test program on a machine without the CUDA runtime cannot break the build. Does nothing when BUILD_TESTING is OFF.
# The gpu and no_gpu tests need GPU code, so they are built only when FLUIDITY_ENABLE_CUDA is ON. SOURCES may list .cu
# files; they are compiled with the device flags.
function(fluidity_add_test name)
  if(NOT BUILD_TESTING)
    return()
  endif()
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "LABEL" "SOURCES;LINKS")
  if(NOT arg_SOURCES)
    message(FATAL_ERROR "fluidity_add_test(${name}) needs SOURCES.")
  endif()
  if(NOT arg_LABEL MATCHES "^(cpu|gpu|no_gpu)$")
    message(FATAL_ERROR "fluidity_add_test(${name}) needs LABEL cpu, gpu or no_gpu, not '${arg_LABEL}'.")
  endif()
  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "fluidity_add_test(${name}) got unknown arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT arg_LABEL STREQUAL "cpu" AND NOT FLUIDITY_ENABLE_CUDA)
    return()
  endif()
  if(NOT arg_LABEL STREQUAL "cpu")
    # Only the CUDA build compiles these .cc files, so the gate's cuda step tidies them (gpu_only_sources.txt).
    fluidity_record_gpu_only_sources(${arg_SOURCES})
  endif()

  cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME module_name)

  add_executable(${name} ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE fluidity GTest::gtest_main ${arg_LINKS})
  fluidity_apply_compile_flags(${name} EXCEPTIONS ON)
  gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST PROPERTIES LABELS "${arg_LABEL};${module_name}")
endfunction()

# fluidity_add_test_library(<name> SOURCES <files...> [GPU_SOURCES <files...>] [LINKS <targets...>])
#
# Builds a static library of code that several test programs share, such as test-support functions, linked to the
# library, to GoogleTest (so it may use EXPECT_ and ASSERT_) and to LINKS. A test program names it in the LINKS of
# fluidity_add_test. It gets the flags of a test program: exceptions on, because GoogleTest needs them. GPU_SOURCES
# are added only when FLUIDITY_ENABLE_CUDA is ON; a .cc file among them is recorded for
# fluidity_write_gpu_only_sources. Does nothing when BUILD_TESTING is OFF.
function(fluidity_add_test_library name)
  if(NOT BUILD_TESTING)
    return()
  endif()
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;GPU_SOURCES;LINKS")
  if(NOT arg_SOURCES)
    message(FATAL_ERROR "fluidity_add_test_library(${name}) needs SOURCES.")
  endif()
  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "fluidity_add_test_library(${name}) got unknown arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()

  add_library(${name} STATIC ${arg_SOURCES})
  if(FLUIDITY_ENABLE_CUDA AND arg_GPU_SOURCES)
    target_sources(${name} PRIVATE ${arg_GPU_SOURCES})
    fluidity_record_gpu_only_sources(${arg_GPU_SOURCES})
  endif()
  target_link_libraries(${name} PUBLIC fluidity GTest::gtest ${arg_LINKS})
  fluidity_apply_compile_flags(${name} EXCEPTIONS ON)
endfunction()

# fluidity_write_gpu_only_sources()
#
# Writes gpu_only_sources.txt in the build folder: the .cc files that only a build with FLUIDITY_ENABLE_CUDA ON
# compiles (the sources of the gpu and no_gpu tests, and the .cc files given to fluidity_add_gpu_sources or to the
# GPU_SOURCES of fluidity_add_test_library), one absolute path per line; empty in a CPU build. The gate's
# cuda step runs clang-tidy over them, since the CPU build that the tidy step reads never compiles them. Called once,
# at the end of the top-level CMakeLists.txt.
function(fluidity_write_gpu_only_sources)
  get_property(sources GLOBAL PROPERTY FLUIDITY_GPU_ONLY_SOURCES)
  list(JOIN sources "\n" sources_text)
  if(sources_text)
    string(APPEND sources_text "\n")
  endif()
  file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/gpu_only_sources.txt" CONTENT "${sources_text}")
endfunction()

# fluidity_check_compile_flags()
#
# Fails the configuration, naming each target, when a C++ or CUDA target (a library or a program) of a folder of this
# project was not given the flags by fluidity_apply_compile_flags. Without this check a target created with a plain
# add_library or add_executable would build without the numerical flags (no contraction, no fast-math), and nothing
# would say so. Imported targets (GoogleTest from Conan) are not targets of a folder, and folders outside the source
# tree (third-party code added as a subfolder) are skipped. Called once, at the end of the top-level CMakeLists.txt.
function(fluidity_check_compile_flags)
  set(unmarked_targets "")
  set(folders "${PROJECT_SOURCE_DIR}")
  while(folders)
    list(POP_FRONT folders folder)
    get_property(subfolders DIRECTORY "${folder}" PROPERTY SUBDIRECTORIES)
    foreach(subfolder IN LISTS subfolders)
      cmake_path(IS_PREFIX PROJECT_SOURCE_DIR "${subfolder}" NORMALIZE is_inside_project)
      if(is_inside_project)
        list(APPEND folders "${subfolder}")
      endif()
    endforeach()
    get_property(targets DIRECTORY "${folder}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
      get_target_property(target_type ${target} TYPE)
      if(NOT target_type MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
        continue()
      endif()
      get_target_property(flags_applied ${target} FLUIDITY_COMPILE_FLAGS_APPLIED)
      if(NOT flags_applied)
        list(APPEND unmarked_targets ${target})
      endif()
    endforeach()
  endwhile()
  if(unmarked_targets)
    list(JOIN unmarked_targets ", " unmarked_text)
    message(FATAL_ERROR "These targets were not created through the helpers of cmake/fluidity_targets.cmake, so "
                        "they lack the numerical flags: ${unmarked_text}. Use fluidity_add_test, "
                        "fluidity_add_test_library, or call fluidity_apply_compile_flags.")
  endif()
endfunction()

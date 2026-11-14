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
  )
  if(arg_EXCEPTIONS)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>"
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fexceptions>"
    )
  else()
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHs-c->"
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fno-exceptions>"
    )
  endif()
endfunction()

# fluidity_add_headers(<files...>)
#
# Adds public headers of the calling module folder to the library, as the HEADERS file set, with native/ as the
# include root. For each header it also writes one .cc file in the build folder that contains only
# `#include "<header>"` and adds it to the library. Each header is therefore compiled alone, with the library flags
# (no exceptions, no RTTI, warnings as errors), and a header that forgets one of its own includes fails to build.
function(fluidity_add_headers)
  set(include_root "${PROJECT_SOURCE_DIR}/native")
  target_sources(fluidity PUBLIC FILE_SET HEADERS BASE_DIRS "${include_root}" FILES ${ARGN})
  foreach(header IN LISTS ARGN)
    cmake_path(ABSOLUTE_PATH header BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE header_path)
    cmake_path(RELATIVE_PATH header_path BASE_DIRECTORY "${include_root}" OUTPUT_VARIABLE include_path)
    set(check_source "${PROJECT_BINARY_DIR}/header_check/${include_path}.cc")
    file(CONFIGURE OUTPUT "${check_source}" CONTENT "#include \"${include_path}\"\n")
    target_sources(fluidity PRIVATE "${check_source}")
  endforeach()
endfunction()

# fluidity_add_test(<name> SOURCES <files...> LABEL <cpu|gpu|no_gpu> [LINKS <targets...>])
#
# Builds one GoogleTest program linked to the library and registers each of its tests with CTest under the label,
# so `ctest -L cpu` runs the CPU tests only.
#   cpu     runs on the CPU only.
#   gpu     needs a CUDA GPU at run time.
#   no_gpu  checks behaviour on a machine without a usable GPU (for example, that choosing the GPU fails).
# Tests are listed when ctest runs (PRE_TEST), not after the build, so building never runs a test program: a GPU
# test program on a machine without the CUDA runtime cannot break the build. Does nothing when BUILD_TESTING is OFF.
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

  # CUDA: a later commit compiles .cu test sources here when FLUIDITY_ENABLE_CUDA is ON.
  add_executable(${name} ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE fluidity GTest::gtest_main ${arg_LINKS})
  fluidity_apply_compile_flags(${name} EXCEPTIONS ON)
  gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST PROPERTIES LABELS ${arg_LABEL})
endfunction()

# fluidity_add_test_library(<name> SOURCES <files...> [GPU_SOURCES <files...>] [LINKS <targets...>])
#
# Builds a static library of code that several test programs share, such as test-support functions, linked to the
# library, to GoogleTest (so it may use EXPECT_ and ASSERT_) and to LINKS. A test program names it in the LINKS of
# fluidity_add_test. It gets the flags of a test program: exceptions on, because GoogleTest needs them. GPU_SOURCES
# (.cu files) are added only when FLUIDITY_ENABLE_CUDA is ON. Does nothing when BUILD_TESTING is OFF.
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
  endif()
  target_link_libraries(${name} PUBLIC fluidity GTest::gtest ${arg_LINKS})
  fluidity_apply_compile_flags(${name} EXCEPTIONS ON)
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

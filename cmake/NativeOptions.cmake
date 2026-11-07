add_library(fluidity_native_options INTERFACE)
target_compile_features(fluidity_native_options INTERFACE cxx_std_20)

if(MSVC)
  # CMake initializes /EHsc for all C++ targets. Enable it explicitly on test targets.
  string(REGEX REPLACE "(^|[ \t])/EHsc($|[ \t])" "\\1\\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

target_compile_options(fluidity_native_options INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/WX;/GR-;/EHs-c-;/fp:precise;/permissive-;/Zc:__cplusplus>"
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror;-fno-rtti;-fno-exceptions;-fno-fast-math;-ffp-contract=off>"
)

function(fluidity_match_qualified_executable compiler recorded)
  file(REAL_PATH "${compiler}" actual_path)
  file(REAL_PATH "${recorded}" recorded_path)
  string(TOLOWER "${actual_path}" actual_path)
  string(TOLOWER "${recorded_path}" recorded_path)
  if(NOT actual_path STREQUAL recorded_path)
    message(FATAL_ERROR "Resolved ${compiler} differs from qualified ${recorded}. Requalify.")
  endif()
endfunction()

function(fluidity_check_windows_toolchain)
  if(NOT EXISTS "${FLUIDITY_TOOLCHAIN_RECEIPT}")
    message(FATAL_ERROR "Run the Windows toolchain qualification before configuring. "
                        "Expected receipt: ${FLUIDITY_TOOLCHAIN_RECEIPT}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FLUIDITY_TOOLCHAIN_RECEIPT}")
  file(READ "${FLUIDITY_TOOLCHAIN_RECEIPT}" receipt)
  string(JSON schema GET "${receipt}" schema_version)
  string(JSON architecture GET "${receipt}" host architecture)
  string(JSON compiler GET "${receipt}" tools cl executable)
  string(JSON version GET "${receipt}" host configurations "${CMAKE_BUILD_TYPE}" compiler_version)
  string(JSON crt GET "${receipt}" host configurations "${CMAKE_BUILD_TYPE}" crt)

  if(NOT schema EQUAL 1 OR NOT architecture STREQUAL "x64" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8
     OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "The Windows build requires a qualified MSVC x64 toolchain.")
  endif()
  fluidity_match_qualified_executable("${CMAKE_CXX_COMPILER}" "${compiler}")
  if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL version)
    message(FATAL_ERROR "MSVC version differs from the qualification receipt. Requalify.")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND crt STREQUAL "MDd")
    set(runtime MultiThreadedDebugDLL)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release" AND crt STREQUAL "MD")
    set(runtime MultiThreadedDLL)
  else()
    message(FATAL_ERROR "The receipt does not qualify the ${CMAKE_BUILD_TYPE} DLL CRT.")
  endif()
  if(NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "$<$<CONFIG:${CMAKE_BUILD_TYPE}>:${runtime}>"
     AND NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "${runtime}")
    message(FATAL_ERROR "Use the Conan dynamic ${CMAKE_BUILD_TYPE} CRT profile.")
  endif()
endfunction()

function(fluidity_check_windows_cuda_toolchain)
  file(READ "${FLUIDITY_TOOLCHAIN_RECEIPT}" receipt)
  string(JSON mode GET "${receipt}" mode)
  if(NOT mode STREQUAL "cuda")
    message(FATAL_ERROR "Requalify the toolchain with --mode cuda.")
  endif()
  string(JSON compiler GET "${receipt}" tools nvcc executable)
  string(JSON version_output GET "${receipt}" tools nvcc version)
  string(REGEX MATCH "V([0-9]+[.][0-9]+[.][0-9]+)" nvcc_version "${version_output}")
  if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR NOT nvcc_version
     OR NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CMAKE_MATCH_1)
    message(FATAL_ERROR "nvcc version differs from the qualification receipt. Requalify.")
  endif()
  string(JSON host_compiler GET "${receipt}" cuda host_compiler)
  string(JSON architecture GET "${receipt}" cuda architecture)
  string(JSON runtime GET "${receipt}" cuda runtime)
  fluidity_match_qualified_executable("${CMAKE_CUDA_COMPILER}" "${compiler}")
  fluidity_match_qualified_executable("${CMAKE_CUDA_HOST_COMPILER}" "${host_compiler}")
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${architecture}-real;${architecture}-virtual"
     OR NOT CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "${runtime}"
     OR NOT runtime STREQUAL "Shared")
    message(FATAL_ERROR "CUDA architecture or runtime differs from the qualification receipt.")
  endif()
endfunction()

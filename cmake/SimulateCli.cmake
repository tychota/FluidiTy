find_package(Threads REQUIRED)
find_package(CLI11 2.7.2 EXACT CONFIG REQUIRED)
# Embed the exact runtime schema at configure time; reconfigure when it changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs")
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" simulate_common_schema)
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs" simulate_resolved_schema)
string(REPLACE "include \"common.fbs\";" "" simulate_resolved_schema "${simulate_resolved_schema}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/simulate_schema_text.h.in"
  "${FLUIDITY_CACHE_GENERATED_DIR}/simulate_schema_text.h" @ONLY)
add_library(fluidity_blender_resolve STATIC native/cli/blender_resolve.cc)
target_link_libraries(fluidity_blender_resolve PRIVATE fluidity_native fluidity_native_options)
function(fluidity_add_simulate_executable target)
  add_executable(${target}
    native/cli/fluidity_simulate.cc native/cli/simulate_execution.cc native/cli/simulate_job.cc native/cli/effects_execution.cc)
  target_include_directories(${target} PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
  target_link_libraries(${target} PRIVATE fluidity_native fluidity_native_options
    fluidity_blender_resolve flatbuffers::flatbuffers BLAKE3::blake3 Threads::Threads)
  if(FLUIDITY_ENABLE_CUDA)
    target_compile_definitions(${target} PRIVATE FLUIDITY_SIMULATE_CUDA=1)
    target_sources(${target} PRIVATE native/cli/simulate_cuda_compatibility.cc native/cli/simulate_cuda_identity.cc)
    target_link_libraries(${target} PRIVATE fluidity_cuda_probe fluidity_cuda_operators CUDA::cudart ${CMAKE_DL_LIBS})
  endif()
endfunction()
# The adapter translates CLI11 exceptions to Status before dispatch.
add_library(fluidity_public_cli OBJECT native/cli/public_cli.cc)
set_property(TARGET fluidity_public_cli PROPERTY FLUIDITY_CLI_EXCEPTIONS ON)
target_link_libraries(fluidity_public_cli PRIVATE absl::statusor CLI11::CLI11 fluidity_native_options)
fluidity_add_simulate_executable(fluidity_simulate)
target_compile_definitions(fluidity_simulate PRIVATE FLUIDITY_PUBLIC_CLI)
target_link_libraries(fluidity_simulate PRIVATE fluidity_public_cli)

# Both entry points must hash to the same executable identity for FLRS scenes
# and existing caches. The executable selects the interface by its filename.
set(fluidity_cli_executable
  "${CMAKE_CURRENT_BINARY_DIR}/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>/>fluidity${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(TARGET fluidity_simulate POST_BUILD
  BYPRODUCTS "${fluidity_cli_executable}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:fluidity_simulate>" "${fluidity_cli_executable}"
  VERBATIM)
if(BUILD_TESTING)
  add_executable(fluidity_simulate_cuda_identity_tests
    native/tests/test_simulate_cuda_identity.cc native/cli/simulate_cuda_identity.cc)
  target_link_libraries(fluidity_simulate_cuda_identity_tests PRIVATE fluidity_native GTest::gtest_main)
  target_compile_options(fluidity_simulate_cuda_identity_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
  gtest_discover_tests(fluidity_simulate_cuda_identity_tests PROPERTIES LABELS cpu)

  add_executable(fluidity_liquid_acceptance_fixture native/tests/liquid_acceptance_fixture.cc)
  target_link_libraries(fluidity_liquid_acceptance_fixture PRIVATE fluidity_native fluidity_native_options)
  set(liquid_acceptance_backends cpu)
  if(FLUIDITY_ENABLE_CUDA)
    list(APPEND liquid_acceptance_backends cuda)
  endif()
  foreach(backend IN LISTS liquid_acceptance_backends)
    foreach(scene IN ITEMS free-fall-pic free-fall-default free-fall-retained tank-layout)
      add_test(NAME LiquidAcceptance.${backend}.${scene}
        COMMAND "${Python3_EXECUTABLE}" -E -X utf8
          "${PROJECT_SOURCE_DIR}/tools/native/tests/run_liquid_acceptance.py"
          --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_liquid_acceptance_fixture>
          --backend ${backend} --case ${scene}
          --temporary-output-parent "${PROJECT_SOURCE_DIR}/out/m1/acceptance")
      set_tests_properties(LiquidAcceptance.${backend}.${scene}
        PROPERTIES LABELS liquid-acceptance TIMEOUT 420)
    endforeach()
  endforeach()
  add_executable(fluidity_simulate_fixture native/tests/simulate_fixture.cc)
  target_link_libraries(fluidity_simulate_fixture PRIVATE fluidity_native fluidity_native_options)
  add_test(NAME SimulateCli.public_interface
    COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_fluidity_cli.py"
      "${fluidity_cli_executable}" "$<TARGET_FILE:fluidity_simulate>" "$<TARGET_FILE:fluidity_simulate_fixture>")
  set_tests_properties(SimulateCli.public_interface PROPERTIES LABELS cli)
  add_test(NAME SimulateCli.rigid_frame
    COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/run_rigid_frame_smoke.py"
      "$<TARGET_FILE:fluidity_simulate>" "$<TARGET_FILE:fluidity_simulate_fixture>")
  set_tests_properties(SimulateCli.rigid_frame PROPERTIES LABELS cli)
  set(simulate_cuda_option)
  if(FLUIDITY_ENABLE_CUDA)
    set(simulate_cuda_option --cuda)
  endif()
  # Ignore inherited PYTHONOPTIMIZE/PYTHONPATH while keeping Unicode paths and
  # diagnostics in UTF-8 explicitly; the driver also rejects direct -O invocation.
  foreach(case IN ITEMS arguments schema identity inspection cpu_device cpu_execution cpu_replay execution_validation cpu_lifecycle cpu_effects cpu_mesh cpu_ocean cpu_graded_width)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} ${simulate_cuda_option})
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cli)
  endforeach()
  if(FLUIDITY_ENABLE_CUDA)
    add_test(NAME SimulateCli.public_graded_cuda
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_fluidity_cli.py"
        "${fluidity_cli_executable}" "$<TARGET_FILE:fluidity_simulate>" "$<TARGET_FILE:fluidity_simulate_fixture>"
        --graded-cuda)
    set_tests_properties(SimulateCli.public_graded_cuda PROPERTIES LABELS cuda-cli)
    # Only this test-linked executable can arm typed runtime faults.
    fluidity_add_simulate_executable(fluidity_simulate_fault_fixture)
    target_sources(fluidity_simulate_fault_fixture PRIVATE native/tests/simulate_fault_fixture.cc)
    target_compile_definitions(fluidity_simulate_fault_fixture PRIVATE FLUIDITY_SIMULATE_FAULT_FIXTURE=1)
    add_test(NAME SimulateReliability.InterruptionCampaign
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8
        "${PROJECT_SOURCE_DIR}/tools/native/tests/run_liquid_fault_campaign.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fault-executable $<TARGET_FILE:fluidity_simulate_fault_fixture>
        --fixture $<TARGET_FILE:fluidity_simulate_fixture>)
    set_tests_properties(SimulateReliability.InterruptionCampaign PROPERTIES LABELS cuda-reliability TIMEOUT 300)
    foreach(case IN ITEMS cuda_device cuda_execution cuda_replay cuda_mixed cuda_lifecycle cuda_effects cuda_ocean cuda_graded_width)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} --cuda)
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cuda-cli)
    endforeach()
  endif()
endif()

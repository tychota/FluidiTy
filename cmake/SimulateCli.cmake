find_package(Threads REQUIRED)
# Embed the exact runtime schema at configure time; reconfigure when it changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs")
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" simulate_common_schema)
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs" simulate_resolved_schema)
string(REPLACE "include \"common.fbs\";" "" simulate_resolved_schema "${simulate_resolved_schema}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/simulate_schema_text.h.in"
  "${FLUIDITY_CACHE_GENERATED_DIR}/simulate_schema_text.h" @ONLY)
function(fluidity_add_simulate_executable target)
  add_executable(${target}
    native/cli/fluidity_simulate.cc native/cli/simulate_execution.cc native/cli/simulate_job.cc)
  target_include_directories(${target} PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
  target_link_libraries(${target} PRIVATE fluidity_native fluidity_native_options
    flatbuffers::flatbuffers BLAKE3::blake3 Threads::Threads)
  if(FLUIDITY_ENABLE_CUDA)
    target_compile_definitions(${target} PRIVATE FLUIDITY_SIMULATE_CUDA=1)
    target_sources(${target} PRIVATE native/cli/simulate_cuda_compatibility.cc)
    target_link_libraries(${target} PRIVATE fluidity_cuda_probe fluidity_cuda_operators CUDA::cudart ${CMAKE_DL_LIBS})
  endif()
endfunction()
fluidity_add_simulate_executable(fluidity_simulate)
if(BUILD_TESTING)
  add_executable(fluidity_liquid_acceptance_fixture native/tests/liquid_acceptance_fixture.cc)
  target_link_libraries(fluidity_liquid_acceptance_fixture PRIVATE fluidity_native fluidity_native_options)
  set(liquid_acceptance_backends cpu)
  if(FLUIDITY_ENABLE_CUDA)
    list(APPEND liquid_acceptance_backends cuda)
  endif()
  foreach(backend IN LISTS liquid_acceptance_backends)
    foreach(scene IN ITEMS free-fall-pic free-fall-default free-fall-retained)
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
  set(simulate_cuda_option)
  if(FLUIDITY_ENABLE_CUDA)
    set(simulate_cuda_option --cuda)
  endif()
  # Ignore inherited PYTHONOPTIMIZE/PYTHONPATH while keeping Unicode paths and
  # diagnostics in UTF-8 explicitly; the driver also rejects direct -O invocation.
  foreach(case IN ITEMS arguments schema identity inspection cpu_device cpu_execution cpu_replay execution_validation cpu_lifecycle)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} ${simulate_cuda_option})
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cli)
  endforeach()
  if(FLUIDITY_ENABLE_CUDA)
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
    foreach(case IN ITEMS cuda_device cuda_execution cuda_replay cuda_lifecycle)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} --cuda)
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cuda-cli)
    endforeach()
  endif()
endif()

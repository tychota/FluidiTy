add_executable(fluidity_simulate native/cli/fluidity_simulate.cc native/cli/simulate_execution.cc)
# Embed the exact runtime schema at configure time; reconfigure when it changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs")
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/common.fbs" simulate_common_schema)
file(READ "${PROJECT_SOURCE_DIR}/schemas/simulate/resolved_scene.fbs" simulate_resolved_schema)
string(REPLACE "include \"common.fbs\";" "" simulate_resolved_schema "${simulate_resolved_schema}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/simulate_schema_text.h.in"
  "${FLUIDITY_CACHE_GENERATED_DIR}/simulate_schema_text.h" @ONLY)
target_include_directories(fluidity_simulate PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
target_link_libraries(fluidity_simulate PRIVATE fluidity_native fluidity_native_options flatbuffers::flatbuffers BLAKE3::blake3)
if(FLUIDITY_ENABLE_CUDA)
  target_compile_definitions(fluidity_simulate PRIVATE FLUIDITY_SIMULATE_CUDA=1)
  target_link_libraries(fluidity_simulate PRIVATE fluidity_cuda_probe fluidity_cuda_operators)
endif()
if(BUILD_TESTING)
  add_executable(fluidity_simulate_fixture native/tests/simulate_fixture.cc)
  target_link_libraries(fluidity_simulate_fixture PRIVATE fluidity_native fluidity_native_options)
  set(simulate_cuda_option)
  if(FLUIDITY_ENABLE_CUDA)
    set(simulate_cuda_option --cuda)
  endif()
  # Ignore inherited PYTHONOPTIMIZE/PYTHONPATH while keeping Unicode paths and
  # diagnostics in UTF-8 explicitly; the driver also rejects direct -O invocation.
  foreach(case IN ITEMS arguments schema identity inspection cpu_device cpu_execution cpu_replay execution_validation)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} ${simulate_cuda_option})
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cli)
  endforeach()
  if(FLUIDITY_ENABLE_CUDA)
    foreach(case IN ITEMS cuda_device cuda_execution cuda_replay)
    add_test(NAME SimulateCli.${case}
      COMMAND "${Python3_EXECUTABLE}" -E -X utf8 "${PROJECT_SOURCE_DIR}/tools/native/tests/test_simulate_replay.py"
        --executable $<TARGET_FILE:fluidity_simulate> --fixture $<TARGET_FILE:fluidity_simulate_fixture>
        --case ${case} --cuda)
    set_tests_properties(SimulateCli.${case} PROPERTIES LABELS cuda-cli)
    endforeach()
  endif()
endif()

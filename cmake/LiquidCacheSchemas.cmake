# The Conan build context supplies flatc; never discover an unrelated PATH binary.
if(NOT IS_ABSOLUTE "${FLUIDITY_FLATC_EXECUTABLE}" OR NOT EXISTS "${FLUIDITY_FLATC_EXECUTABLE}")
  message(FATAL_ERROR "Conan must supply an absolute FLUIDITY_FLATC_EXECUTABLE")
endif()
execute_process(COMMAND "${FLUIDITY_FLATC_EXECUTABLE}" --version
  OUTPUT_VARIABLE flatc_version OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT flatc_version STREQUAL "flatc version 25.12.19")
  message(FATAL_ERROR "Expected pinned flatc 25.12.19, got ${flatc_version}")
endif()
set(FLUIDITY_CACHE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/cache")
set(cache_schema_dir "${PROJECT_SOURCE_DIR}/schemas/simulate")
set(cache_generated_headers)
foreach(schema IN ITEMS common liquid_state point_frame frame_manifest resolved_scene)
  set(header "${FLUIDITY_CACHE_GENERATED_DIR}/${schema}_generated.h")
  add_custom_command(OUTPUT "${header}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FLUIDITY_CACHE_GENERATED_DIR}"
    COMMAND "${FLUIDITY_FLATC_EXECUTABLE}" --cpp --scoped-enums --gen-mutable
            -I "${cache_schema_dir}" -o "${FLUIDITY_CACHE_GENERATED_DIR}" "${cache_schema_dir}/${schema}.fbs"
    DEPENDS "${cache_schema_dir}/${schema}.fbs" "${cache_schema_dir}/common.fbs" "${FLUIDITY_FLATC_EXECUTABLE}"
    VERBATIM)
  list(APPEND cache_generated_headers "${header}")
  if(BUILD_TESTING)
    add_test(NAME LiquidCacheSchema.${schema}
      COMMAND "${FLUIDITY_FLATC_EXECUTABLE}" --conform "${cache_schema_dir}/v1/${schema}.fbs"
              --conform-includes "${cache_schema_dir}/v1" -I "${cache_schema_dir}"
              "${cache_schema_dir}/${schema}.fbs")
    set_tests_properties(LiquidCacheSchema.${schema} PROPERTIES LABELS "native;schema")
  endif()
endforeach()
add_custom_target(fluidity_cache_schemas DEPENDS ${cache_generated_headers})
if(BUILD_TESTING)
  add_test(NAME LiquidCacheSchema.RejectIncompatibleField
    COMMAND "${CMAKE_COMMAND}" "-DFLUIDITY_FLATC_EXECUTABLE=${FLUIDITY_FLATC_EXECUTABLE}"
            "-DSCHEMA_DIR=${cache_schema_dir}" -P "${PROJECT_SOURCE_DIR}/cmake/RejectIncompatibleLiquidSchema.cmake")
  set_tests_properties(LiquidCacheSchema.RejectIncompatibleField PROPERTIES LABELS "native;schema")
endif()

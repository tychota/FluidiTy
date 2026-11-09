execute_process(
  COMMAND "${FLUIDITY_FLATC_EXECUTABLE}" --conform "${SCHEMA_DIR}/v1/checkpoint_reference.fbs"
          --conform-includes "${SCHEMA_DIR}/v1" -I "${SCHEMA_DIR}"
          "${SCHEMA_DIR}/incompatible/checkpoint_reference.fbs"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "types differ for field: fluidity[.]cache[.]v1[.]CheckpointReference[.]schema_version")
  message(FATAL_ERROR "Expected incompatible schema_version type rejection; result=${result}: ${output}${error}")
endif()

# A missing executable/include or parser failure must not count as a negative proof.
execute_process(
  COMMAND "${FLUIDITY_FLATC_EXECUTABLE}" --conform "${SCHEMA_DIR}/v1/liquid_state.fbs"
          --conform-includes "${SCHEMA_DIR}/v1" -I "${SCHEMA_DIR}"
          "${SCHEMA_DIR}/incompatible/liquid_state.fbs"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "types differ for field: fluidity[.]cache[.]v1[.]LiquidState[.]random_seed")
  message(FATAL_ERROR "Expected incompatible random_seed type rejection; result=${result}: ${output}${error}")
endif()

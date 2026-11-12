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
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Set CMAKE_CUDA_ARCHITECTURES to the explicit target architectures.")
  endif()
  foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(NOT architecture MATCHES "^[0-9]+[af]?(-real|-virtual)?$")
      message(FATAL_ERROR "CUDA architecture '${architecture}' must name an explicit target.")
    endif()
  endforeach()
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
  find_package(CUDAToolkit REQUIRED)
  if(WIN32)
    fluidity_check_windows_cuda_toolchain()
    # CMake's nvcc defaults enable MSVC exception unwinding. CUDA probe code uses Status.
    string(REPLACE "-Xcompiler=\" /EHsc\"" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  endif()
endif()

find_package(absl 20250814.2 EXACT CONFIG REQUIRED)
find_package(flatbuffers 25.12.19 EXACT CONFIG REQUIRED)
find_package(blake3 1.8.5 EXACT CONFIG REQUIRED)
include(cmake/LiquidCacheSchemas.cmake)
add_library(fluidity_native STATIC
    native/src/contact_face_bodies_3d.cc
    native/src/mixed_contact_3d.cc
    native/src/dry_solid_step_3d.cc
    native/src/plastic_solid_checkpoint_3d.cc
    native/src/effects_source_3d.cc
    native/src/surface_closest_point_3d.cc
    native/src/surface_reconstruction_3d.cc
    native/src/surface_settings_3d.cc
    native/src/surface_stage_3d.cc
    native/src/surface_component_lineage_3d.cc
    native/src/surface_material_components_3d.cc
    native/src/whitewater_sphere_source_3d.cc
    native/src/whitewater_transport_3d.cc
    native/src/whitewater_frame_3d.cc
    native/src/whitewater_cache_3d.cc
    native/src/whitewater_stage_3d.cc
    native/src/whitewater_volume_cache_3d.cc
    native/src/ocean_boundary_3d.cc
    native/src/volumes_stage_3d.cc
    native/src/whitewater_volume_3d.cc
    native/src/whitewater_3d.cc
    native/src/surface_cache_3d.cc
    native/src/surface_tracking_3d.cc
    native/src/surface_field_3d.cc
  native/src/surface_mesh_3d.cc
  native/src/surface_occupancy_3d.cc
  native/src/motion_carrier_3d.cc
  native/src/motion_capture_3d.cc
  native/src/effects_cache_3d.cc
  native/src/motion_interval_3d.cc
  native/src/liquid_boundary_workspace_3d.cc
  native/src/liquid_step_workspace_3d.cc
  native/src/liquid_isolated_drop_3d.cc
  native/src/liquid_remap_3d.cc
  native/src/liquid_band_handoff_3d.cc
  native/src/liquid_substep_3d.cc
  native/src/wetted_piston_3d.cc
  native/src/liquid_substep_3d_cpu.cc
  native/src/liquid_substep_3d_staging.cc
  native/src/liquid_resolved_scene_3d.cc
  native/src/liquid_cache_3d.cc
  native/src/liquid_cache_store_3d.cc
  native/src/liquid_cache_files_3d.cc
  native/src/liquid_state_3d.cc
  native/src/liquid_resampling_3d.cc
  native/src/grid.cc
  native/src/adaptive_family_change_3d.cc
  native/src/adaptive_liquid_remap_3d.cc
  native/src/adaptive_mac_topology_3d.cc
  native/src/sparse_mac_grid_3d.cc
  native/src/basis.cc
  native/src/coarse_fine_transfer_3d.cc
  native/src/small_solve.cc
  native/src/transfers.cc
  native/src/transfers_3d.cc
  native/src/elastic_blade_deformation_3d.cc
  native/src/rigid_body_3d.cc
  native/src/solid_interaction_checkpoint_3d.cc
  native/src/pressure.cc
  native/src/mac_pressure_3d_system.cc
  native/src/liquid_boundary_geometry_3d.cc
  native/src/liquid_boundary_transfer_3d.cc
  native/src/liquid_wall_contact_3d.cc
  native/src/mac_pressure_3d_bounded_hierarchy.cc
  native/src/mac_pressure_workspace_3d.cc
  native/src/mac_pressure_3d.cc
  native/src/mac_pressure_3d_multigrid.cc
  native/src/adaptive_width_policy_3d.cc
  native/src/mac_viscous_pressure_3d.cc
  native/src/liquid_air_components.cc
  native/src/liquid_cavity.cc
  native/src/liquid_eos.cc
  native/src/liquid_eos_sensor_3d.cc
  native/src/liquid_mixture_3d.cc
  native/src/viscosity.cc)
target_include_directories(fluidity_native PUBLIC "${PROJECT_SOURCE_DIR}/native/include")
target_compile_features(fluidity_native PUBLIC cxx_std_20)
add_dependencies(fluidity_native fluidity_cache_schemas)
target_include_directories(fluidity_native PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
target_link_libraries(fluidity_native
  PUBLIC absl::function_ref absl::status absl::statusor
  PRIVATE flatbuffers::flatbuffers BLAKE3::blake3 fluidity_native_options)

if(FLUIDITY_ENABLE_CUDA)
  add_library(fluidity_cuda_probe_host OBJECT native/cuda/probe_host.cc)
  target_include_directories(fluidity_cuda_probe_host PRIVATE "${PROJECT_SOURCE_DIR}/native/include")
  target_include_directories(fluidity_cuda_probe_host SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(fluidity_cuda_probe_host PRIVATE absl::status fluidity_native_options)

  add_library(fluidity_cuda_probe STATIC native/cuda/probe.cu $<TARGET_OBJECTS:fluidity_cuda_probe_host>)
  target_include_directories(fluidity_cuda_probe PUBLIC "${PROJECT_SOURCE_DIR}/native/include")
  target_compile_features(fluidity_cuda_probe PUBLIC cxx_std_20)
  target_link_libraries(fluidity_cuda_probe INTERFACE absl::status PRIVATE CUDA::cudart)
  add_library(fluidity_cuda_operators STATIC
    native/cuda/dry_solid_step_3d_host.cc
    native/cuda/liquid_transfer_stages_3d_host.cc
    native/cuda/liquid_pressure_stage_3d_host.cc
    native/cuda/wetted_piston_3d.cu
    native/cuda/wetted_piston_3d_host.cc
    native/cuda/liquid_geometry_stages_3d_host.cc
    native/cuda/liquid_geometry_stages_3d.cu
    native/cuda/liquid_arena_3d_packing.cc
    native/cuda/liquid_snapshot_3d_host.cc
    native/cuda/liquid_substep_3d_host.cc
    native/cuda/liquid_arena_3d_host.cc
    native/cuda/liquid_arena_3d.cu
    native/cuda/transfer_host.cc
    native/cuda/transfers.cu
    native/cuda/transfer_3d_host.cc
    native/cuda/transfers_3d.cu
    native/cuda/coarse_fine_transfer_3d.cu
    native/cuda/liquid_boundary_geometry_3d_host.cc
    native/cuda/liquid_boundary_geometry_3d.cu
    native/cuda/liquid_boundary_transfer_3d.cu
    native/cuda/liquid_boundary_transfer_3d_host.cc
    native/cuda/liquid_wall_contact_3d_host.cc
    native/cuda/liquid_wall_contact_3d.cu
    native/cuda/viscosity_velocity_block_3d_host.cc
    native/cuda/viscosity_velocity_block_3d.cu
    native/cuda/mac_pressure_3d_host.cc
    native/cuda/mac_pressure_3d.cu
    native/cuda/pressure_host.cc
    native/cuda/pressure.cu)
  target_link_libraries(fluidity_cuda_operators PUBLIC fluidity_native PRIVATE CUDA::cudart fluidity_native_options)
  if(MSVC)
    set_source_files_properties(
      native/cuda/coarse_fine_transfer_3d.cu
      native/cuda/wetted_piston_3d.cu
      native/cuda/liquid_geometry_stages_3d.cu
      native/cuda/liquid_arena_3d.cu
      native/cuda/liquid_boundary_transfer_3d.cu
      native/cuda/liquid_boundary_geometry_3d.cu
      native/cuda/liquid_wall_contact_3d.cu
      PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor" )
  endif()

  foreach(target IN ITEMS fluidity_cuda_probe fluidity_cuda_operators)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--no-exceptions;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>")
    if(MSVC)
      target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/GR-,/fp:precise,/Zc:__cplusplus>")
    endif()
  endforeach()
endif()

if(BUILD_TESTING)
  add_executable(fluidity_m5_dry_scene_tests native/tests/test_m5_dry_scenes_3d.cc)
  target_link_libraries(fluidity_m5_dry_scene_tests PRIVATE fluidity_native flatbuffers::flatbuffers)
  add_dependencies(fluidity_m5_dry_scene_tests fluidity_cache_schemas)
  target_include_directories(fluidity_m5_dry_scene_tests PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
  add_test(NAME M5DryScenes.native COMMAND fluidity_m5_dry_scene_tests)
  add_executable(fluidity_m5_planar_coupling_tests native/tests/test_m5_planar_coupling_3d.cc)
  target_link_libraries(fluidity_m5_planar_coupling_tests PRIVATE fluidity_native)
  add_test(NAME M5PlanarCoupling.native COMMAND fluidity_m5_planar_coupling_tests)
  add_executable(fluidity_m5_plastic_checkpoint_tests native/tests/test_m5_plastic_checkpoint_3d.cc)
  target_link_libraries(fluidity_m5_plastic_checkpoint_tests PRIVATE fluidity_native flatbuffers::flatbuffers)
  add_dependencies(fluidity_m5_plastic_checkpoint_tests fluidity_cache_schemas)
  target_include_directories(fluidity_m5_plastic_checkpoint_tests PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
  add_test(NAME M5PlasticCheckpoint.native COMMAND fluidity_m5_plastic_checkpoint_tests)
  add_executable(fluidity_m5_snow_compaction_tests native/tests/test_m5_snow_compaction_3d.cc)
  target_link_libraries(fluidity_m5_snow_compaction_tests PRIVATE fluidity_native)
  add_test(NAME M5SnowCompaction.native COMMAND fluidity_m5_snow_compaction_tests)
  add_executable(fluidity_m5_contact_scatter_tests native/tests/test_m5_contact_scatter_3d.cc)
  target_link_libraries(fluidity_m5_contact_scatter_tests PRIVATE fluidity_native)
  add_test(NAME M5ContactScatter.native COMMAND fluidity_m5_contact_scatter_tests)
  add_executable(fluidity_m5_constitutive_tests native/tests/test_m5_constitutive_3d.cc)
  target_include_directories(fluidity_m5_constitutive_tests PRIVATE "${PROJECT_SOURCE_DIR}/native/include")
  target_compile_features(fluidity_m5_constitutive_tests PRIVATE cxx_std_20)
  add_test(NAME M5Constitutive.native COMMAND fluidity_m5_constitutive_tests)
  if(FLUIDITY_ENABLE_CUDA)
    add_executable(fluidity_cuda_m5_dry_scene_tests native/tests/test_m5_dry_scenes_3d.cc)
    target_compile_definitions(fluidity_cuda_m5_dry_scene_tests PRIVATE FLUIDITY_M5_CUDA_SCENES=1)
    target_link_libraries(fluidity_cuda_m5_dry_scene_tests PRIVATE fluidity_cuda_operators flatbuffers::flatbuffers)
    add_dependencies(fluidity_cuda_m5_dry_scene_tests fluidity_cache_schemas)
    target_include_directories(fluidity_cuda_m5_dry_scene_tests PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
    add_test(NAME M5DryScenes.cuda COMMAND fluidity_cuda_m5_dry_scene_tests)
    add_executable(fluidity_cuda_m5_planar_coupling_tests native/tests/test_m5_planar_coupling_3d.cc)
    target_compile_definitions(fluidity_cuda_m5_planar_coupling_tests PRIVATE FLUIDITY_M5_CUDA_SCENES=1)
    target_link_libraries(fluidity_cuda_m5_planar_coupling_tests PRIVATE fluidity_cuda_operators)
    add_test(NAME M5PlanarCoupling.cuda COMMAND fluidity_cuda_m5_planar_coupling_tests)
    add_executable(fluidity_cuda_m5_plastic_checkpoint_tests native/tests/test_m5_plastic_checkpoint_3d.cc)
    target_compile_definitions(fluidity_cuda_m5_plastic_checkpoint_tests PRIVATE FLUIDITY_M5_CUDA_SCENES=1)
    target_link_libraries(fluidity_cuda_m5_plastic_checkpoint_tests PRIVATE fluidity_cuda_operators flatbuffers::flatbuffers)
    add_dependencies(fluidity_cuda_m5_plastic_checkpoint_tests fluidity_cache_schemas)
    target_include_directories(fluidity_cuda_m5_plastic_checkpoint_tests PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
    add_test(NAME M5PlasticCheckpoint.cuda COMMAND fluidity_cuda_m5_plastic_checkpoint_tests)
    add_executable(fluidity_cuda_m5_snow_compaction_tests native/tests/test_m5_snow_compaction_3d.cc)
    target_compile_definitions(fluidity_cuda_m5_snow_compaction_tests PRIVATE FLUIDITY_M5_CUDA_SCENES=1)
    target_link_libraries(fluidity_cuda_m5_snow_compaction_tests PRIVATE fluidity_cuda_operators)
    add_test(NAME M5SnowCompaction.cuda COMMAND fluidity_cuda_m5_snow_compaction_tests)
    add_executable(fluidity_cuda_m5_constitutive_tests native/tests/test_cuda_m5_constitutive_3d.cu)
    target_include_directories(fluidity_cuda_m5_constitutive_tests PRIVATE "${PROJECT_SOURCE_DIR}/native/include")
    target_compile_options(fluidity_cuda_m5_constitutive_tests PRIVATE
      "$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:preprocessor>")
    set_target_properties(fluidity_cuda_m5_constitutive_tests PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME M5Constitutive.cuda COMMAND fluidity_cuda_m5_constitutive_tests)
  endif()
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
            "${reference_src}/transfer_3d_fixtures.py" "${reference_src}/pressure_3d_fixtures.py"
            "${reference_src}/liquid_boundary_3d_fixtures.py"
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
  # Shared decoded pressure fixtures for evaluator, CPU solver and future CUDA tests.
  add_library(fluidity_pressure_3d_test_support STATIC native/tests/mac_pressure_3d_fixture.cc)
  add_dependencies(fluidity_pressure_3d_test_support fluidity_fixture_headers)
  target_include_directories(fluidity_pressure_3d_test_support
    PUBLIC "${PROJECT_SOURCE_DIR}/native/tests"
    PRIVATE "${generated_dir}")
  target_compile_features(fluidity_pressure_3d_test_support PUBLIC cxx_std_20)
  target_compile_options(fluidity_pressure_3d_test_support PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
  target_link_libraries(fluidity_pressure_3d_test_support PUBLIC fluidity_native GTest::gtest)

  add_executable(fluidity_native_tests
    native/tests/test_surface_settings_3d.cc
    native/tests/test_surface_stage_3d.cc
    native/tests/test_surface_component_lineage_3d.cc
    native/tests/test_surface_material_components_3d.cc
    native/tests/test_whitewater_sphere_source_3d.cc
    native/tests/test_whitewater_transport_3d.cc
    native/tests/test_whitewater_cache_3d.cc
    native/tests/test_whitewater_volume_cache_3d.cc
    native/tests/test_ocean_boundary_3d.cc
    native/tests/test_effects_source_3d.cc
    native/tests/test_surface_closest_point_3d.cc
    native/tests/test_surface_reconstruction_3d.cc
    native/tests/test_whitewater_volume_3d.cc
    native/tests/test_whitewater_3d.cc
    native/tests/test_surface_cache_3d.cc
    native/tests/test_surface_tracking_3d.cc
    native/tests/test_surface_field_3d.cc
    native/tests/test_surface_mesh_3d.cc
    native/tests/test_surface_occupancy_3d.cc
    native/tests/test_motion_carrier_3d.cc
    native/tests/test_motion_capture_3d.cc
    native/tests/test_effects_cache_3d.cc
    native/tests/test_motion_interval_3d.cc
    native/tests/test_liquid_boundary_workspace_3d.cc
    native/tests/test_liquid_step_workspace_3d.cc
    native/tests/test_wetted_piston_3d.cc
    native/tests/test_liquid_isolated_drop_3d.cc
    native/tests/test_liquid_remap_3d.cc
    native/tests/test_liquid_band_handoff_3d.cc
    native/tests/test_liquid_substep_3d.cc
    native/tests/test_liquid_cache_3d.cc
    native/tests/test_liquid_cache_store_3d.cc
    native/tests/test_liquid_state_3d.cc
    native/tests/test_liquid_resampling_3d.cc
    native/tests/test_liquid_merge_3d.cc
    native/tests/test_build.cc
    native/tests/test_fixture_bridge.cc
    native/tests/test_grid_basis.cc
    native/tests/test_coarse_fine_transfer_3d.cc
    native/tests/test_sparse_mac_grid_3d.cc
    native/tests/test_small_solve.cc
    native/tests/test_transfers.cc
    native/tests/test_transfers_3d.cc
    native/tests/test_rigid_body_3d.cc
    native/tests/test_pressure.cc
    native/tests/test_mac_pressure_3d_system.cc
    native/tests/test_liquid_boundary_geometry_3d.cc
    native/tests/test_liquid_boundary_transfer_3d.cc
    native/tests/test_liquid_wall_contact_3d.cc
    native/tests/test_mac_pressure_workspace_3d.cc
    native/tests/test_adaptive_mac_topology_3d.cc
    native/tests/test_adaptive_liquid_remap_3d.cc
    native/tests/test_mac_pressure_3d.cc
    native/tests/test_adaptive_width_policy_3d.cc
    native/tests/test_liquid_cavity.cc
    native/tests/test_liquid_eos.cc
    native/tests/test_liquid_mixture_3d.cc
    native/tests/test_viscosity.cc)
  target_link_libraries(fluidity_native_tests PRIVATE fluidity_native fluidity_pressure_3d_test_support flatbuffers::flatbuffers)
  target_include_directories(fluidity_native_tests PRIVATE "${FLUIDITY_CACHE_GENERATED_DIR}")
  target_compile_definitions(fluidity_native_tests PRIVATE
    FLUIDITY_CACHE_SCRATCH_PATH="${CMAKE_CURRENT_BINARY_DIR}/cache-test-scratch"
    FLUIDITY_CACHE_GOLDEN_PATH="${PROJECT_SOURCE_DIR}/native/tests/fixtures/liquid_cache/resolved_defaults.hex")
  add_dependencies(fluidity_native_tests fluidity_fixture_headers)

  # Use a separate test executable for the synthetic case so each executable owns one data set.
  set(escaping_input "${PROJECT_SOURCE_DIR}/native/tests/fixtures/escaping")
  set(escaping_header "${generated_dir}/fixture_escaping_data.h")
  # Let the validator handle missing JSON files so it can remove a stale header.
  fluidity_compile_fixtures(fluidity_fixture_escaping_header "${escaping_input}" "${escaping_header}")
  add_executable(fluidity_fixture_bridge_escaping native/tests/test_fixture_escaping.cc)
  add_dependencies(fluidity_fixture_bridge_escaping fluidity_fixture_escaping_header)
  target_link_libraries(fluidity_fixture_bridge_escaping PRIVATE absl::status absl::statusor)

  foreach(target IN ITEMS fluidity_native_tests fluidity_fixture_bridge_escaping)
    target_include_directories(${target} PRIVATE "${generated_dir}")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
    gtest_discover_tests(${target} PROPERTIES LABELS cpu)
  endforeach()

  add_executable(fluidity_m3_matched_pressure native/tests/matched_error_pressure_3d.cc)
  target_include_directories(fluidity_m3_matched_pressure PRIVATE "${PROJECT_SOURCE_DIR}/native/src")
  target_link_libraries(fluidity_m3_matched_pressure PRIVATE fluidity_native)
  target_compile_options(fluidity_m3_matched_pressure PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
  add_test(NAME M3MatchedPressure.manufactured_sealed_cube COMMAND fluidity_m3_matched_pressure)
  set_tests_properties(M3MatchedPressure.manufactured_sealed_cube PROPERTIES LABELS pressure-benchmark)

  # Historical-cache research needs an external authenticated archive. Keep it
  # outside the mandatory CPU inventory, which never permits skipped tests.
  add_executable(fluidity_surface_fixed_cache_tests native/tests/test_surface_fixed_cache_3d.cc)
  target_link_libraries(fluidity_surface_fixed_cache_tests PRIVATE fluidity_native GTest::gtest_main)
  target_compile_options(fluidity_surface_fixed_cache_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")

  if(FLUIDITY_ENABLE_CUDA)
    add_executable(fluidity_cuda_probe_tests native/tests/test_cuda_probe.cc)
    target_link_libraries(fluidity_cuda_probe_tests PRIVATE fluidity_cuda_probe CUDA::cudart GTest::gtest_main)
    target_compile_options(fluidity_cuda_probe_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    gtest_discover_tests(fluidity_cuda_probe_tests PROPERTIES LABELS gpu)

    add_executable(fluidity_cuda_operator_tests
      native/tests/test_cuda_transfers.cc
      native/tests/test_cuda_transfers_3d.cc
      native/tests/test_cuda_pressure.cc
      native/tests/test_cuda_liquid_snapshot_3d.cc
      native/tests/test_cuda_wetted_piston_3d.cc
      native/tests/test_cuda_liquid_substep_3d.cc
      native/tests/test_cuda_liquid_boundary_3d.cc
      native/tests/test_cuda_mac_pressure_3d.cc
      native/tests/test_cuda_viscosity.cc)
    add_dependencies(fluidity_cuda_operator_tests fluidity_fixture_headers)
    target_include_directories(fluidity_cuda_operator_tests PRIVATE "${generated_dir}")
    target_link_libraries(fluidity_cuda_operator_tests
      PRIVATE fluidity_cuda_operators fluidity_pressure_3d_test_support CUDA::cudart GTest::gtest_main)
    target_compile_options(fluidity_cuda_operator_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    if(WIN32)
      foreach(target IN ITEMS fluidity_cuda_probe_tests fluidity_cuda_operator_tests)
        target_compile_definitions(${target}
          PRIVATE "FLUIDITY_QUALIFIED_CUDA_DEVICE=\"${FLUIDITY_QUALIFIED_CUDA_DEVICE}\"")
      endforeach()
    endif()
    gtest_discover_tests(fluidity_cuda_operator_tests PROPERTIES LABELS gpu)

    # Keep bounded adaptivity parity separate from the production CUDA validation inventory.
    add_executable(fluidity_cuda_coarse_fine_tests native/tests/test_cuda_coarse_fine_transfer_3d.cc)
    target_include_directories(fluidity_cuda_coarse_fine_tests PRIVATE "${PROJECT_SOURCE_DIR}/native/cuda")
    target_link_libraries(fluidity_cuda_coarse_fine_tests
      PRIVATE fluidity_cuda_operators CUDA::cudart GTest::gtest_main)
    target_compile_options(fluidity_cuda_coarse_fine_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    gtest_discover_tests(fluidity_cuda_coarse_fine_tests PROPERTIES LABELS cuda-adaptivity)

    add_executable(fluidity_cuda_graph_tests
      native/tests/test_cuda_graph_deposit.cc
      native/tests/test_cuda_graph_pressure.cc)
    target_include_directories(fluidity_cuda_graph_tests PRIVATE "${PROJECT_SOURCE_DIR}/native/cuda")
    target_link_libraries(fluidity_cuda_graph_tests PRIVATE fluidity_cuda_operators CUDA::cudart GTest::gtest_main)
    target_compile_options(fluidity_cuda_graph_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    gtest_discover_tests(fluidity_cuda_graph_tests PROPERTIES LABELS gpu)

    # Focused Task 6B/6C1/6C2/6C3a qualification. Retain in final mandatory 6D verification;
    # Debug: 43 cases (42 real-device, 1 host-only); Release: 39 cases.
    # The separate mandatory gpu-label inventory includes the two integrated scheduler cases.
    add_executable(fluidity_cuda_arena_tests
      native/tests/test_cuda_liquid_arena_3d.cc
      native/tests/test_cuda_liquid_pressure_stage_3d.cc)
    target_link_libraries(fluidity_cuda_arena_tests
      PRIVATE fluidity_cuda_operators fluidity_cuda_probe fluidity_pressure_3d_test_support CUDA::cudart GTest::gtest_main)
    target_compile_options(fluidity_cuda_arena_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")

    # Link the same host wrapper to CUDA API doubles for deterministic failure-path tests.
    add_executable(fluidity_cuda_probe_error_tests
      native/tests/test_cuda_probe_errors.cc $<TARGET_OBJECTS:fluidity_cuda_probe_host>)
    target_include_directories(fluidity_cuda_probe_error_tests
      PRIVATE "${PROJECT_SOURCE_DIR}/native/include" "${PROJECT_SOURCE_DIR}/native/cuda")
    target_include_directories(fluidity_cuda_probe_error_tests SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_features(fluidity_cuda_probe_error_tests PRIVATE cxx_std_20)
    target_link_libraries(fluidity_cuda_probe_error_tests PRIVATE absl::status GTest::gtest_main)
    target_compile_options(fluidity_cuda_probe_error_tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>")
    gtest_discover_tests(fluidity_cuda_probe_error_tests PROPERTIES LABELS cuda-mock)
  endif()
endif()

include(cmake/SimulateCli.cmake)

option(FLUIDITY_BUILD_BLENDER_BRIDGE "Build the Blender cache and scene bridge (requires Alembic)" OFF)
if(FLUIDITY_BUILD_BLENDER_BRIDGE)
  find_package(Alembic 1.8.8 EXACT CONFIG REQUIRED)
  add_library(fluidity_blender_resolve STATIC native/cli/blender_resolve.cc)
  target_link_libraries(fluidity_blender_resolve PRIVATE fluidity_native fluidity_native_options)
  add_executable(fluidity_blender_bridge tools/blender/point_cache_writer/blender_bridge.cc)
  target_include_directories(fluidity_blender_bridge PRIVATE native/cli)
  target_compile_features(fluidity_blender_bridge PRIVATE cxx_std_20)
  target_compile_options(fluidity_blender_bridge PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/EHsc>")
  target_link_libraries(fluidity_blender_bridge PRIVATE fluidity_blender_resolve fluidity_native Alembic::Alembic)
endif()

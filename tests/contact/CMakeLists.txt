fluidity_add_test(
  contact_response_on_generated_cases_functional_test
  SOURCES
  contact_response_on_generated_cases.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  find_first_hit_on_generated_sweeps_functional_test
  SOURCES
  find_first_hit_on_generated_sweeps.test.cc
  LABEL
  cpu
  TIER
  functional
)

fluidity_add_test(
  move_particle_far_from_origin_functional_test
  SOURCES
  move_particle_far_from_origin.test.cc
  LABEL
  cpu
  TIER
  functional
)

# Counts the reads of each solid (FLUIDITY_COUNT_SOLID_READS, contact/solid_shapes.h) and compares with a copy built to
# read every solid, as the unit tests of the rule do.
fluidity_add_test(
  move_particle_left_out_every_set_functional_test
  SOURCES
  move_particle_left_out_every_set.test.cc
  ${PROJECT_SOURCE_DIR}/src/fluidity/contact/testing/left_out_reference.cc
  LABEL
  cpu
  TIER
  functional
)
if(TARGET move_particle_left_out_every_set_functional_test)
  target_compile_definitions(move_particle_left_out_every_set_functional_test PRIVATE FLUIDITY_COUNT_SOLID_READS)
endif()

fluidity_add_test(
  move_particle_on_generated_paths_functional_test
  SOURCES
  move_particle_on_generated_paths.test.cc
  LABEL
  cpu
  TIER
  functional
)

fluidity_add_test(
  move_particle_shorter_step_on_catalogue_functional_test
  SOURCES
  move_particle_shorter_step_on_catalogue.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particle_slides_over_many_steps_functional_test
  SOURCES
  move_particle_slides_over_many_steps.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particles_against_move_particle_functional_test
  SOURCES
  move_particles_against_move_particle.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particles_in_every_thread_count_functional_test
  SOURCES
  move_particles_in_every_thread_count.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particles_on_gpu_functional_test
  SOURCES
  move_particles_on_gpu.test.cc
  LABEL
  gpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particles_scenes_functional_test
  SOURCES
  move_particles_scenes.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  move_particles_steps_over_every_thread_functional_test
  SOURCES
  move_particles_steps_over_every_thread.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

# Its two generated sets compare 20,000 paths with a double-double reference that searches 160 times per path: about
# 11 s unoptimised. -O1 in a debug build keeps every assert and every IEEE result (no fast math, no contraction) and
# runs it in under a second, with the same printed counts and worst cases. MSVC is left unoptimised: its debug builds
# use /RTC1, which no /O option may be combined with.
fluidity_add_test(
  round_solid_hit_on_generated_sweeps_functional_test
  SOURCES
  round_solid_hit_on_generated_sweeps.test.cc
  LABEL
  cpu
  TIER
  functional
)
if(TARGET round_solid_hit_on_generated_sweeps_functional_test)
  target_compile_options(round_solid_hit_on_generated_sweeps_functional_test PRIVATE
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-O1>"
  )
endif()

fluidity_add_test(
  smallest_correction_on_generated_cases_functional_test
  SOURCES
  smallest_correction_on_generated_cases.test.cc
  LABEL
  cpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

fluidity_add_test(
  solid_shapes_shared_formulas_on_gpu_functional_test
  SOURCES
  solid_shapes_shared_formulas_on_gpu.test.cc
  ${PROJECT_SOURCE_DIR}/src/fluidity/contact/testing/shared_formulas_runner_gpu.cu
  LABEL
  gpu
  TIER
  functional
  LINKS
  fluidity_base_testing
)

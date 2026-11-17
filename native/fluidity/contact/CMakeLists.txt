fluidity_add_headers(
  contact_error.h
  contact_error_message.h
  contact_tolerance.h
  move_particles.h
  move_particles_gpu.h
  move_particles_steps.h
  solid_contact.h
  solid_contact_steps.h
  solid_contact_types.h
  solid_shapes.h
)

target_sources(fluidity PRIVATE contact_error_message.cc move_particles_cpu.cc)
fluidity_add_gpu_sources(move_particles_gpu.cu)

fluidity_add_test(contact_error_test SOURCES contact_error_test.cc LABEL cpu)
fluidity_add_test(contact_response_test SOURCES contact_response_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(contact_tolerance_test SOURCES contact_tolerance_test.cc LABEL cpu)
fluidity_add_test(find_first_hit_test SOURCES find_first_hit_test.cc LABEL cpu)
fluidity_add_test(move_particle_test SOURCES move_particle_test.cc LABEL cpu)
fluidity_add_test(move_particles_test SOURCES move_particles_test.cc LABEL cpu LINKS fluidity_base_test_support)
# The test of the rule that leaves out a solid a particle cannot come near counts the reads of each solid
# (FLUIDITY_COUNT_SOLID_READS, contact/solid_shapes.h) and compares with a copy built to read every solid.
fluidity_add_test(move_particle_left_out_test SOURCES move_particle_left_out_test.cc
                  move_particle_left_out_reference_test.cc LABEL cpu)
if(TARGET move_particle_left_out_test)
  target_compile_definitions(move_particle_left_out_test PRIVATE FLUIDITY_COUNT_SOLID_READS)
endif()
fluidity_add_test(move_particle_tolerance_test SOURCES move_particle_tolerance_test.cc LABEL cpu)
fluidity_add_test(round_solid_hit_rounding_test SOURCES round_solid_hit_rounding_test.cc LABEL cpu)
fluidity_add_test(solid_shapes_test SOURCES solid_shapes_test.cc LABEL cpu)

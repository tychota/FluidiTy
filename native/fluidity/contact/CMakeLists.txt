fluidity_add_headers(
  contact_error.h
  contact_error_message.h
  contact_tolerance.h
  solid_contact.h
  solid_contact_steps.h
  solid_contact_types.h
  solid_shapes.h
)

target_sources(fluidity PRIVATE contact_error_message.cc)

fluidity_add_test(contact_error_test SOURCES contact_error_test.cc LABEL cpu)
fluidity_add_test(contact_response_test SOURCES contact_response_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(contact_tolerance_test SOURCES contact_tolerance_test.cc LABEL cpu)
fluidity_add_test(find_first_hit_test SOURCES find_first_hit_test.cc LABEL cpu)
fluidity_add_test(move_particle_test SOURCES move_particle_test.cc LABEL cpu)
fluidity_add_test(solid_shapes_test SOURCES solid_shapes_test.cc LABEL cpu)

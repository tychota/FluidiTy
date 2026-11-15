fluidity_add_headers(
  contact_error.h
  contact_error_message.h
)

target_sources(fluidity PRIVATE contact_error_message.cc)

fluidity_add_test(contact_error_test SOURCES contact_error_test.cc LABEL cpu)

fluidity_add_headers(
  grid_error.h
  grid_error_message.h
)

target_sources(fluidity PRIVATE grid_error_message.cc)

fluidity_add_test(grid_error_test SOURCES grid_error_test.cc LABEL cpu)

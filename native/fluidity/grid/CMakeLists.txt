fluidity_add_headers(
  grid_error.h
  grid_error_message.h
  grid_ids.h
  grid_tables.h
  weight_curve.h
)

target_sources(fluidity PRIVATE grid_error_message.cc)

fluidity_add_test(grid_error_test SOURCES grid_error_test.cc LABEL cpu)
fluidity_add_test(grid_ids_test SOURCES grid_ids_test.cc LABEL cpu)
fluidity_add_test(grid_shared_gpu_compile_test SOURCES grid_shared_gpu_compile_test.cu LABEL no_gpu)
fluidity_add_test(weight_curve_test SOURCES weight_curve_test.cc LABEL cpu)

fluidity_add_headers(
  cell_size_rule.h
  grid_build_status.h
  grid_build_steps.h
  grid_buffers_internal.h
  grid_cell_checks.h
  grid_error.h
  grid_error_message.h
  grid_ids.h
  grid_memory_estimate.h
  grid_tables.h
  staggered_grid.h
  staggered_grid_gpu.h
  weight_curve.h
)

target_sources(
  fluidity PRIVATE grid_cell_checks.cc grid_error_message.cc grid_memory_estimate.cc staggered_grid_cpu.cc
)
fluidity_add_gpu_sources(staggered_grid_gpu.cu)
# ptxas prints the stack frame and the spills of each GPU function for each architecture, so every compile of the file
# with CUDA shows the stack each one keeps against the driver's default (method paper <sec-grid-memory>).
# Source properties are seen only in the folder that made the target, so the line names the library's folder.
set_source_files_properties(staggered_grid_gpu.cu TARGET_DIRECTORY fluidity PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")

# Test-only: the geometric checker and the random grids that grid tests compare the grid against. Never linked by
# the library.
fluidity_add_test_library(fluidity_grid_test_support SOURCES grid_test_support.cc LINKS fluidity_base_test_support)
# The gate runs one process per test, and each process that uses them builds the 200 random grids and their checkers
# again. Unoptimised, that took about 1 s per process; at -O2 it takes about 0.1 s, and serial ctest in the debug
# preset dropped from 19.6 to 9.5 s for grid_build_steps_test and from 32.9 to 25.9 s for grid_cell_count_test. Only
# this test-only target gets the flag, in Debug, and only with GCC, Clang and AppleClang on their usual command line:
# asserts stay on (no NDEBUG), and the library and the test programs are still built for the debugger. Compilers with
# the MSVC command line (MSVC itself and clang-cl, for both of which CMake sets MSVC) keep their debug flags, whose
# run-time checks (/RTC1) cannot be combined with optimisation.
if(NOT MSVC)
  target_compile_options(
    fluidity_grid_test_support PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-O2>"
  )
endif()

fluidity_add_test(cell_size_rule_test SOURCES cell_size_rule_test.cc LABEL cpu)
fluidity_add_test(find_cell_at_test SOURCES find_cell_at_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(grid_buffers_test SOURCES grid_buffers_test.cc LABEL cpu)
fluidity_add_test(grid_build_steps_test SOURCES grid_build_steps_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(grid_cell_checks_test SOURCES grid_cell_checks_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(
  grid_cell_checks_gpu_test
  SOURCES
  grid_cell_checks_gpu_test.cc
  LABEL
  gpu
  LINKS
  fluidity_base_test_support
  fluidity_grid_test_support
)
# Counts every allocation of the program, so its allocation counter is part of the program, never of a library.
fluidity_add_test(
  staggered_grid_gpu_test
  SOURCES
  staggered_grid_gpu_test.cc
  allocation_counter_test_support.cc
  LABEL
  gpu
  LINKS
  fluidity_base_test_support
  fluidity_grid_test_support
)
fluidity_add_test(grid_cell_count_test SOURCES grid_cell_count_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(grid_buffers_no_gpu_test SOURCES grid_buffers_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(grid_error_test SOURCES grid_error_test.cc LABEL cpu)
fluidity_add_test(grid_ids_test SOURCES grid_ids_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(grid_memory_estimate_test SOURCES grid_memory_estimate_test.cc LABEL cpu)
# Replaces the nothrow allocation functions for the whole program, so it shares its program with no other test.
fluidity_add_test(grid_out_of_memory_test SOURCES grid_out_of_memory_test.cc LABEL cpu)
fluidity_add_test(grid_shared_gpu_compile_test SOURCES grid_shared_gpu_compile_test.cu LABEL no_gpu)
# Counts every allocation of the program, so its allocation counter is part of the program, never of a library.
fluidity_add_test(
  staggered_grid_test
  SOURCES
  staggered_grid_test.cc
  allocation_counter_test_support.cc
  LABEL
  cpu
  LINKS
  fluidity_grid_test_support
)
# Counts every allocation of the program, so its allocation counter is part of the program, never of a library.
fluidity_add_test(
  grid_counts_on_buffers_test
  SOURCES
  grid_counts_on_buffers_test.cc
  allocation_counter_test_support.cc
  LABEL
  cpu
  LINKS
  fluidity_grid_test_support
)
# Counts every allocation of the program, as grid_counts_on_buffers_test does.
fluidity_add_test(
  read_grid_test
  SOURCES
  read_grid_test.cc
  allocation_counter_test_support.cc
  LABEL
  cpu
  LINKS
  fluidity_grid_test_support
)
fluidity_add_test(
  staggered_grid_no_gpu_test SOURCES staggered_grid_no_gpu_test.cc LABEL no_gpu LINKS fluidity_grid_test_support
)
fluidity_add_test(grid_tables_test SOURCES grid_tables_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(grid_test_support_test SOURCES grid_test_support_test.cc LABEL cpu LINKS fluidity_grid_test_support)
fluidity_add_test(weight_curve_test SOURCES weight_curve_test.cc LABEL cpu LINKS fluidity_grid_test_support)

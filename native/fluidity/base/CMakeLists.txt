fluidity_add_headers(
  axis.h
  check_can_run_on.h
  cpu_and_gpu.h
  cpu_only.h
  mat3.h
  run_on.h
  scalar_math.h
  vec3.h
)

target_sources(fluidity PRIVATE check_can_run_on_cpu.cc)

fluidity_add_test(vec3_test SOURCES vec3_test.cc LABEL cpu)
fluidity_add_test(mat3_test SOURCES mat3_test.cc LABEL cpu)
fluidity_add_test(scalar_math_test SOURCES scalar_math_test.cc LABEL cpu)
fluidity_add_test(axis_test SOURCES axis_test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_test SOURCES check_can_run_on_test.cc LABEL cpu)

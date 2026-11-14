fluidity_add_headers(
  axis.h
  check_can_run_on.h
  check_can_run_on_gpu.h
  check_can_run_on_math.h
  cpu_and_gpu.h
  cpu_only.h
  gpu_launch.cuh
  mat3.h
  run_on.h
  run_on_call.h
  scalar_math.h
  vec3.h
)

target_sources(fluidity PRIVATE check_can_run_on_cpu.cc)
fluidity_add_gpu_sources(check_can_run_on_gpu.cu)

fluidity_add_test_library(fluidity_base_test_support SOURCES test_support.cc)

fluidity_add_test(vec3_test SOURCES vec3_test.cc LABEL cpu)
fluidity_add_test(mat3_test SOURCES mat3_test.cc LABEL cpu)
fluidity_add_test(scalar_math_test SOURCES scalar_math_test.cc LABEL cpu)
fluidity_add_test(axis_test SOURCES axis_test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_test SOURCES check_can_run_on_test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_math_test SOURCES check_can_run_on_math_test.cc LABEL cpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(test_support_test SOURCES test_support_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(run_on_call_test SOURCES run_on_call_test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_gpu_test SOURCES check_can_run_on_gpu_test.cc LABEL gpu)
fluidity_add_test(shared_math_gpu_test SOURCES shared_math_gpu_test.cc shared_math_test_gpu.cu LABEL gpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(check_can_run_on_no_gpu_test SOURCES check_can_run_on_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(base_constants_gpu_compile_test SOURCES base_constants_gpu_compile_test.cu LABEL no_gpu)

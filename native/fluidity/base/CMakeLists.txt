fluidity_add_headers(
  axis.h
  buffer.h
  buffer_bytes.h
  buffer_gpu.h
  check_can_run_on.h
  check_can_run_on_gpu.h
  check_can_run_on_math.h
  cpu_and_gpu.h
  cpu_only.h
  gpu_launch.cuh
  mat3.h
  ordered_sums.h
  ordered_sums_gpu.h
  processor_description.h
  processor_description_gpu.h
  run_on.h
  run_on_call.h
  scalar_math.h
  stable_order.h
  stable_order_digit.h
  stable_order_gpu.h
  stable_order_steps.h
  tree_sum.h
  tree_sum_layers.h
  vec3.h
)

target_sources(fluidity PRIVATE buffer_cpu.cc check_can_run_on_cpu.cc ordered_sums_cpu.cc processor_description_cpu.cc
               stable_order_cpu.cc)
fluidity_add_gpu_sources(buffer_gpu.cu check_can_run_on_gpu.cu ordered_sums_gpu.cu processor_description_gpu.cu
                         stable_order_gpu.cu)

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
fluidity_add_test(buffer_test SOURCES buffer_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(processor_description_test SOURCES processor_description_test.cc LABEL cpu)
fluidity_add_test(tree_sum_test SOURCES tree_sum_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(tree_sum_layers_test SOURCES tree_sum_layers_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(ordered_sums_test SOURCES ordered_sums_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(stable_order_test SOURCES stable_order_test.cc LABEL cpu LINKS fluidity_base_test_support)
fluidity_add_test(stable_order_steps_test SOURCES stable_order_steps_test.cc LABEL cpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(check_can_run_on_gpu_test SOURCES check_can_run_on_gpu_test.cc LABEL gpu)
fluidity_add_test(shared_math_gpu_test SOURCES shared_math_gpu_test.cc shared_math_test_gpu.cu LABEL gpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(buffer_gpu_test SOURCES buffer_gpu_test.cc LABEL gpu LINKS fluidity_base_test_support)
fluidity_add_test(ordered_sums_gpu_test SOURCES ordered_sums_gpu_test.cc LABEL gpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(stable_order_gpu_test SOURCES stable_order_gpu_test.cc LABEL gpu
                  LINKS fluidity_base_test_support)
fluidity_add_test(check_can_run_on_no_gpu_test SOURCES check_can_run_on_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(buffer_no_gpu_test SOURCES buffer_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(ordered_sums_no_gpu_test SOURCES ordered_sums_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(processor_description_no_gpu_test SOURCES processor_description_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(stable_order_no_gpu_test SOURCES stable_order_no_gpu_test.cc LABEL no_gpu)
fluidity_add_test(base_constants_gpu_compile_test SOURCES base_constants_gpu_compile_test.cu LABEL no_gpu)

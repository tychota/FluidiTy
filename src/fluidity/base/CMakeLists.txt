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
  memory_count.h
  ordered_sums.h
  ordered_sums_gpu.h
  per_particle_velocity.h
  processor_description.h
  processor_description_gpu.h
  processor_description_text.h
  run_on.h
  run_memory.h
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

target_sources(fluidity PRIVATE buffer_cpu.cc check_can_run_on_cpu.cc memory_count.cc ordered_sums_cpu.cc
               processor_description_cpu.cc run_memory_cpu.cc stable_order_cpu.cc)
fluidity_add_gpu_sources(buffer_gpu.cu check_can_run_on_gpu.cu ordered_sums_gpu.cu processor_description_gpu.cu
                         stable_order_gpu.cu)

fluidity_add_test_library(fluidity_base_testing SOURCES testing/base_support.cc)

fluidity_add_test(vec3_test SOURCES vec3.test.cc LABEL cpu)
fluidity_add_test(mat3_test SOURCES mat3.test.cc LABEL cpu)
fluidity_add_test(scalar_math_test SOURCES scalar_math.test.cc LABEL cpu)
fluidity_add_test(axis_test SOURCES axis.test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_test SOURCES check_can_run_on.test.cc LABEL cpu)
fluidity_add_test(check_can_run_on_math_test SOURCES check_can_run_on_math.test.cc LABEL cpu
                  LINKS fluidity_base_testing)
fluidity_add_test(base_support_test SOURCES testing/base_support.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(run_on_call_test SOURCES run_on_call.test.cc LABEL cpu)
fluidity_add_test(buffer_test SOURCES buffer.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(processor_description_test SOURCES processor_description.test.cc LABEL cpu)
fluidity_add_test(memory_count_test SOURCES memory_count.test.cc LABEL cpu)
fluidity_add_test(run_memory_test SOURCES run_memory.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(per_particle_velocity_test SOURCES per_particle_velocity.test.cc LABEL cpu)
fluidity_add_test(tree_sum_test SOURCES tree_sum.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(tree_sum_layers_test SOURCES tree_sum_layers.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(ordered_sums_test SOURCES ordered_sums.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(stable_order_test SOURCES stable_order.test.cc LABEL cpu LINKS fluidity_base_testing)
fluidity_add_test(stable_order_steps_test SOURCES stable_order_steps.test.cc LABEL cpu
                  LINKS fluidity_base_testing)
fluidity_add_test(check_can_run_on_gpu_test SOURCES check_can_run_on.gpu.test.cc LABEL gpu)
fluidity_add_test(buffer_gpu_test SOURCES buffer.gpu.test.cc LABEL gpu LINKS fluidity_base_testing)
fluidity_add_test(processor_description_gpu_test SOURCES processor_description.gpu.test.cc LABEL gpu)
fluidity_add_test(run_memory_gpu_test SOURCES run_memory.gpu.test.cc LABEL gpu LINKS fluidity_base_testing)
fluidity_add_test(check_can_run_on_no_gpu_test SOURCES check_can_run_on.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(buffer_no_gpu_test SOURCES buffer.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(run_memory_no_gpu_test SOURCES run_memory.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(ordered_sums_no_gpu_test SOURCES ordered_sums.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(processor_description_text_test SOURCES processor_description_text.test.cc LABEL cpu)
fluidity_add_test(processor_description_no_gpu_test SOURCES processor_description.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(stable_order_no_gpu_test SOURCES stable_order.no_gpu.test.cc LABEL no_gpu)
fluidity_add_test(base_constants_gpu_compile_test SOURCES scalar_math.shared.gpu_compile.test.cu LABEL no_gpu)

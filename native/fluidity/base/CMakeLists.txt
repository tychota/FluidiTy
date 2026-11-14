fluidity_add_headers(
  cpu_and_gpu.h
  vec3.h
)

fluidity_add_test(vec3_test SOURCES vec3_test.cc LABEL cpu)
